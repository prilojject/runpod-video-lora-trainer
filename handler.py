"""Runpod Serverless entry point: Wan 2.1 / 2.2 LoRA training with kohya-ss/musubi-tuner.

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_url": "https://.../data.zip"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset  # noqa: E402
from src.paths import configure_caches, has_volume, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("video-lora-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Always start the next job on a fresh worker (frees GPU memory, drops partial downloads).
    result["refresh_worker"] = True
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    output_name = cfg.output_name or "_".join(
        x for x in ((cfg.trigger_word or "lora").replace(" ", "_"), cfg.base_model.replace("-", "_"), "lora") if x)
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/video-lora/work", run_name)
    videos_dir = os.path.join(work, "videos")
    images_dir = os.path.join(work, "images")
    shutil.rmtree(work, ignore_errors=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "base_model": cfg.base_model,
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append(
            "no network volume: base models are re-downloaded on every cold start "
            "(attach one to cache them under /runpod-volume/video-lora/models)"
        )

    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, work, videos_dir, images_dir)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    from src.train import build_prompts
    prompts = build_prompts(cfg)
    result["sample_prompts"] = prompts

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if all(v.get("ok") for v in access.values()) else "error",
            "model_access": access,
            "model_download_gb": round(sum(v.get("size_gb") or 0 for v in access.values()), 2),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "base model files not reachable: " + "; ".join(
                f"{k}: {v.get('error')}" for k, v in access.items() if not v.get("ok"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    from src.models import ModelError, ensure_models
    from src.train import (TrainingError, auto_blocks_to_swap, build_train_command, cache_latents_cmd,
                           cache_text_cmd, convert_lora_cmd, find_outputs, gpu_info, run_command,
                           sample_cmd, use_offload_inactive, write_dataset_toml)

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu["gpu"]:
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}
    if gpu.get("supported") is False:
        return {"error": f"GPU {gpu['gpu']} (compute {gpu.get('compute_capability')}) is not supported by this "
                         "image's PyTorch/CUDA build (Blackwell MIG GPUs are excluded in hub.json); "
                         "redeploy on Ampere/Ada (RTX 4090, L40S, A6000 Ada, A6000, A40)",
                **{k: result[k] for k in ("run_name", "config")}, "gpu": gpu}
    if cfg.is_14b and gpu["vram_gb"] < 40:
        result["warnings"].append(
            f"{cfg.base_model} on {gpu['vram_gb']} GB VRAM will use block swap and a lower resolution is safer; "
            "a 48 GB GPU is recommended"
        )

    try:
        models, minfo = ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}
    result["models"] = minfo

    os.makedirs(run_dir, exist_ok=True)
    toml_path = os.path.join(work, "dataset.toml")
    write_dataset_toml(cfg, videos_dir, images_dir, os.path.join(work, "cache"), toml_path,
                       report.videos_used, report.images_used)
    log_path = os.path.join(run_dir, "train.log")
    timeout_s = int(os.environ.get("TRAIN_TIMEOUT_SECONDS", "0")) or None
    # Leave some budget for download (already done) + cache + optional convert.
    cache_timeout = None if not timeout_s else max(120, timeout_s // 3)

    try:
        progress({"stage": "cache_latents"})
        run_command(cache_latents_cmd(cfg, models, toml_path, gpu["vram_gb"]),
                    log_path, progress, "cache_latents", timeout_s=cache_timeout)
        progress({"stage": "cache_text"})
        run_command(cache_text_cmd(cfg, models, toml_path, gpu["vram_gb"]),
                    log_path, progress, "cache_text", timeout_s=cache_timeout)
    except TrainingError as exc:
        return {"error": redact(str(exc))[:4000], "run_name": run_name, "gpu": gpu, "log_path": log_path}

    blocks = auto_blocks_to_swap(cfg, gpu["vram_gb"])
    result["blocks_to_swap"] = blocks
    result["offload_inactive_dit"] = use_offload_inactive(cfg, blocks, gpu["vram_gb"])
    cmd = build_train_command(cfg, models, toml_path, run_dir, output_name, gpu["vram_gb"], blocks)
    try:
        progress({"stage": "training", "step": 0, "total_steps": cfg.steps})
        train = run_command(cmd, log_path, progress, "training", total_steps=cfg.steps, timeout_s=timeout_s)
    except TrainingError as exc:
        outs = find_outputs(run_dir, output_name)
        return {"error": redact(str(exc))[:4000], "run_name": run_name, "gpu": gpu,
                "checkpoints": outs["checkpoints"], "log_path": log_path}
    except Exception as exc:
        log.error("training crashed: %s", redact(traceback.format_exc()))
        return {"error": f"training crashed: {redact(str(exc)).splitlines()[0][:500]}", "run_name": run_name}
    result["training"] = train

    outs = find_outputs(run_dir, output_name)
    if not outs["lora_path"]:
        return {"error": "training finished but no LoRA file was written", "run_name": run_name,
                "training": train, "log_path": log_path}

    converted_path = None
    if cfg.convert_lora:
        converted_path = os.path.join(run_dir, f"{output_name}_comfy.safetensors")
        try:
            progress({"stage": "convert_lora"})
            run_command(convert_lora_cmd(outs["lora_path"], converted_path),
                        log_path, progress, "convert_lora", timeout_s=300)
        except TrainingError as exc:
            result["warnings"].append(f"convert_lora failed: {redact(str(exc))[:300]}")
            converted_path = None

    sample_files: list[str] = []
    if cfg.num_samples and prompts:
        sample_dir = os.path.join(run_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        for i, prompt in enumerate(prompts):
            try:
                progress({"stage": "sampling", "index": i + 1, "total": len(prompts)})
                run_command(sample_cmd(cfg, models, outs["lora_path"], prompt, sample_dir, cfg.seed + i),
                            log_path, progress, "sampling", timeout_s=timeout_s)
            except TrainingError as exc:
                result["warnings"].append(f"sample {i} failed: {redact(str(exc))[:300]}")
        outs = find_outputs(run_dir, output_name)
        sample_files = outs["samples"]

    from src.outputs import deliver, encode_file_base64
    progress({"stage": "uploading"})
    artifacts, warns = deliver(cfg, run_name, output_name, outs["lora_path"], sample_files,
                               prompts, train, token, persistent, converted_path=converted_path)
    result["warnings"] += warns
    artifacts["checkpoints"] = [os.path.basename(c) for c in outs["checkpoints"]]
    if cfg.return_lora_base64:
        b64 = encode_file_base64(outs["lora_path"])
        if b64:
            artifacts["lora_base64"] = b64
        else:
            result["warnings"].append("return_lora_base64 skipped: LoRA is larger than 80 MiB")
    result["artifacts"] = artifacts
    samples_out = []
    for i, s in enumerate(sample_files):
        item: dict[str, Any] = {"prompt": prompts[i] if i < len(prompts) else None, "file": os.path.basename(s)}
        if cfg.return_samples_base64:
            b64 = encode_file_base64(s, max_bytes=15 * 2**20)
            if b64:
                item["video_base64"] = b64
                item["mime"] = "video/mp4" if s.endswith(".mp4") else "application/octet-stream"
        samples_out.append(item)
    result["samples"] = samples_out
    result.update({"status": "completed", "elapsed_seconds": round(time.time() - started, 1)})
    shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
