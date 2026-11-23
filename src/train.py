"""Build musubi-tuner TOML + cache/train commands, stream logs, collect outputs."""

from __future__ import annotations

import glob
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any

from .config import MAX_BLOCKS_1_3B, MAX_BLOCKS_14B, default_sample_prompts, render_prompt
from .secret_utils import redact

log = logging.getLogger(__name__)

MUSUBI = os.environ.get("MUSUBI_TUNER_DIR", "/opt/musubi-tuner")
LOSS_RE = re.compile(r"avr_loss=([0-9.eE+-]+)")


class TrainingError(RuntimeError):
    pass


def gpu_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            major, minor = torch.cuda.get_device_capability(0)
            archs = torch.cuda.get_arch_list()
            supported = f"sm_{major}{minor}" in archs or any(
                a.startswith(f"sm_{major}") and int(a[3:]) <= major * 10 + minor for a in archs)
            return {"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 1),
                    "compute_capability": f"{major}.{minor}", "supported": bool(supported)}
    except Exception:  # pragma: no cover
        pass
    return {"gpu": None, "vram_gb": 0.0}


def toml_escape(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_dataset_toml(cfg, videos_dir: str, images_dir: str, cache_root: str, path: str,
                       n_videos: int, n_images: int) -> dict[str, str]:
    """Write musubi dataset TOML. Each [[datasets]] gets its own cache_directory."""
    w, h = cfg.resolution_wh
    # Repeat tiny datasets so a step is not a full "epoch" of 2-3 items with a restart each time.
    n = max(1, n_videos + n_images)
    repeats = max(1, min(50, 100 // n))
    lines = [
        "[general]",
        f"resolution = [{w}, {h}]",
        'caption_extension = ".txt"',
        f"batch_size = {cfg.batch_size}",
        "enable_bucket = true",
        "bucket_no_upscale = false",
        "",
    ]
    caches: dict[str, str] = {}
    if n_videos:
        caches["videos"] = os.path.join(cache_root, "cache_videos")
        os.makedirs(caches["videos"], exist_ok=True)
        frames = ", ".join(str(x) for x in cfg.target_frames)
        lines += [
            "[[datasets]]",
            f"video_directory = {toml_escape(videos_dir)}",
            f"cache_directory = {toml_escape(caches['videos'])}",
            f"target_frames = [{frames}]",
            f'frame_extraction = "{cfg.frame_extraction}"',
            f"source_fps = {cfg.source_fps:g}",
            f"num_repeats = {repeats}",
            "",
        ]
    if n_images:
        caches["images"] = os.path.join(cache_root, "cache_images")
        os.makedirs(caches["images"], exist_ok=True)
        lines += [
            "[[datasets]]",
            f"image_directory = {toml_escape(images_dir)}",
            f"cache_directory = {toml_escape(caches['images'])}",
            f"num_repeats = {repeats}",
            "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return caches


def build_prompts(cfg) -> list[str]:
    raw = cfg.sample_prompts or (default_sample_prompts(cfg) if cfg.num_samples else [])
    return [render_prompt(p, cfg.trigger_word, cfg.class_word or "") for p in raw][: cfg.num_samples]


def auto_blocks_to_swap(cfg, vram_gb: float) -> int:
    """Pick --blocks_to_swap. 0 means disabled. Never combined with --offload_inactive_dit."""
    if cfg.blocks_to_swap != "auto":
        return int(cfg.blocks_to_swap)
    cap = MAX_BLOCKS_14B if cfg.is_14b else MAX_BLOCKS_1_3B
    if not cfg.is_14b:
        # 1.3B fits in 24 GB without swap; use a little swap on very small cards.
        val = 0 if vram_gb >= 20 else 8
    elif cfg.is_wan22 and cfg.wan22_mode == "both":
        # 24 GB: swap, no offload (they are mutually exclusive). 48 GB+: offload, no swap.
        val = 28 if vram_gb < 40 else 0
    else:
        # Single 14B DiT.
        val = 24 if vram_gb < 40 else 0
    return max(0, min(cap, val))


def use_offload_inactive(cfg, blocks: int, vram_gb: float) -> bool:
    return bool(cfg.is_wan22 and cfg.wan22_mode == "both" and blocks == 0 and vram_gb >= 40)


def use_fp8_t5(vram_gb: float) -> bool:
    return bool(vram_gb) and vram_gb <= 24


def use_vae_cache_cpu(vram_gb: float) -> bool:
    return bool(vram_gb) and vram_gb < 28


def _script(*parts: str) -> str:
    return os.path.join(MUSUBI, *parts)


def cache_latents_cmd(cfg, models: dict[str, str], dataset_toml: str, vram_gb: float) -> list[str]:
    cmd = [sys.executable, _script("wan_cache_latents.py"),
           "--dataset_config", dataset_toml,
           "--vae", models["vae"],
           "--skip_broken"]
    if cfg.is_i2v:
        cmd.append("--i2v")
    if "clip" in models:
        cmd += ["--clip", models["clip"]]
    if use_vae_cache_cpu(vram_gb):
        cmd.append("--vae_cache_cpu")
    return cmd


def cache_text_cmd(cfg, models: dict[str, str], dataset_toml: str, vram_gb: float) -> list[str]:
    del cfg
    cmd = [sys.executable, _script("wan_cache_text_encoder_outputs.py"),
           "--dataset_config", dataset_toml,
           "--t5", models["t5"],
           "--batch_size", "1"]
    if use_fp8_t5(vram_gb):
        cmd.append("--fp8_t5")
    return cmd


def build_train_command(cfg, models: dict[str, str], dataset_toml: str, out_dir: str,
                        output_name: str, vram_gb: float, blocks: int) -> list[str]:
    script = _script("src", "musubi_tuner", "wan_train_network.py")
    if not os.path.isfile(script):
        # pip install -e . layout fallback
        script = _script("wan_train_network.py")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_cpu_threads_per_process=1",
        "--mixed_precision=fp16",
        "--num_processes=1",
        "--num_machines=1",
        "--dynamo_backend=no",
        script,
        f"--task={cfg.task}",
        f"--dit={models['dit']}",
        f"--dataset_config={dataset_toml}",
        f"--output_dir={out_dir}",
        f"--output_name={output_name}",
        "--sdpa",
        "--mixed_precision=fp16",
        "--gradient_checkpointing",
        "--optimizer_type=adamw8bit",
        f"--learning_rate={cfg.learning_rate:g}",
        "--network_module=networks.lora_wan",
        f"--network_dim={cfg.rank}",
        f"--network_alpha={cfg.alpha:g}",
        "--timestep_sampling=shift",
        "--discrete_flow_shift=3.0",
        f"--max_train_steps={cfg.steps}",
        f"--seed={cfg.seed}",
        "--max_data_loader_n_workers=2",
        "--persistent_data_loader_workers",
    ]
    if cfg.fp8_base:
        args.append("--fp8_base")
    if cfg.save_every_n_steps:
        args.append(f"--save_every_n_steps={cfg.save_every_n_steps}")
    if "dit_high" in models and cfg.wan22_mode == "both":
        args.append(f"--dit_high_noise={models['dit_high']}")
        boundary = 0.9 if cfg.is_i2v else 0.875
        args.append(f"--timestep_boundary={boundary}")
        if use_offload_inactive(cfg, blocks, vram_gb):
            args.append("--offload_inactive_dit")
    elif cfg.is_wan22 and cfg.wan22_mode in ("low", "high"):
        args.append("--preserve_distribution_shape")
        if cfg.is_i2v:
            lo, hi = (0, 900) if cfg.wan22_mode == "low" else (900, 1000)
        else:
            lo, hi = (0, 875) if cfg.wan22_mode == "low" else (875, 1000)
        args += [f"--min_timestep={lo}", f"--max_timestep={hi}"]
    if blocks:
        # Mutually exclusive with --offload_inactive_dit (we only set offload when blocks == 0).
        args.append(f"--blocks_to_swap={blocks}")
    return args


def convert_lora_cmd(src: str, dest: str) -> list[str]:
    script = _script("convert_lora.py")
    return [sys.executable, script, "--input", src, "--output", dest, "--target", "other"]


def sample_cmd(cfg, models: dict[str, str], lora_path: str, prompt: str, save_path: str,
               seed: int) -> list[str]:
    script = _script("wan_generate_video.py")
    if not os.path.isfile(script):
        script = _script("src", "musubi_tuner", "wan_generate_video.py")
    w, h = cfg.resolution_wh
    cmd = [sys.executable, script,
           "--task", cfg.task,
           "--dit", models["dit"],
           "--vae", models["vae"],
           "--t5", models["t5"],
           "--attn_mode", "sdpa",
           "--prompt", prompt,
           "--video_size", str(h), str(w),  # musubi takes H W in some versions; also try --w/--h via prompt file
           "--video_length", str(cfg.sample_frames),
           "--infer_steps", str(cfg.sample_steps),
           "--seed", str(seed),
           "--save_path", save_path,
           "--lora_weight", lora_path,
           "--fp8"]
    if "dit_high" in models:
        cmd += ["--dit_high_noise", models["dit_high"]]
    if "clip" in models:
        cmd += ["--clip", models["clip"]]
    return cmd


def run_command(cmd: list[str], log_path: str, progress, stage: str, total_steps: int | None = None,
                timeout_s: int | None = None, cwd: str | None = None) -> dict[str, Any]:
    """Run a musubi process, stream its log, optionally parse training steps."""
    log.info("launching: %s", redact(" ".join(shlex.quote(c) for c in cmd)))
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    state = {"step": 0, "loss": None, "last_report": 0.0, "first_step_t": None, "tail": []}
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n# --- {stage} ---\n")
        proc = subprocess.Popen(cmd, cwd=cwd or MUSUBI, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1, errors="replace")

        def watchdog():
            if timeout_s:
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    log.error("%s exceeded %ss, terminating", stage, timeout_s)
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill()

        threading.Thread(target=watchdog, daemon=True).start()
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if ch == "" and proc.poll() is not None:
                break
            if ch not in ("\n", "\r"):
                buf += ch
                continue
            line, buf = redact(buf), ""
            if not line.strip():
                continue
            logf.write(line + "\n")
            logf.flush()
            state["tail"] = (state["tail"] + [line])[-80:]
            m = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
            if m and ("steps" in line or "step" in line.lower() or "it/" in line or "s/it" in line):
                step = int(m.group(1))
                if step > 0 and state["first_step_t"] is None:
                    state["first_step_t"] = time.time()
                state["step"] = step
                lm = LOSS_RE.search(line)
                if lm:
                    try:
                        state["loss"] = float(lm.group(1))
                    except ValueError:
                        pass
                now = time.time()
                if now - state["last_report"] > 15:
                    state["last_report"] = now
                    payload = {"stage": stage, "step": step}
                    if total_steps:
                        payload["total_steps"] = total_steps
                    if state["loss"] is not None:
                        payload["loss"] = state["loss"]
                    progress(payload)
            elif not m:
                log.info("musubi[%s]: %s", stage, line[:500])
        rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        tail = "\n".join(state["tail"][-25:])
        raise TrainingError(f"{stage} exited with code {rc}. Last log lines:\n{tail[-3000:]}")
    train_s = (time.time() - state["first_step_t"]) if state["first_step_t"] else None
    return {"global_step": state["step"], "final_avr_loss": state["loss"],
            "total_seconds": round(elapsed, 1),
            "train_loop_seconds": round(train_s, 1) if train_s else None}


def find_outputs(out_dir: str, output_name: str) -> dict[str, Any]:
    final = os.path.join(out_dir, f"{output_name}.safetensors")
    ckpts = sorted(glob.glob(os.path.join(out_dir, f"{output_name}-*.safetensors")))
    samples = []
    for pattern in ("*.mp4", "*.webm", "*.png", "*.gif"):
        samples += sorted(glob.glob(os.path.join(out_dir, "sample", pattern)))
        samples += sorted(glob.glob(os.path.join(out_dir, "samples", pattern)))
    return {"lora_path": final if os.path.isfile(final) else None, "checkpoints": ckpts, "samples": samples}
