"""Wan checkpoints: download at job start, cache on the network volume if attached.

Public fp16/bf16 weights only. Official musubi does not support ready-made Comfy
``fp8_scaled`` files — we refuse those filenames and never download them.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any

import requests

from .paths import has_volume, model_cache_dir
from .secret_utils import redact

log = logging.getLogger(__name__)

# (repo_id, filename, expected_size_bytes or None)
T5 = ("Wan-AI/Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth", 11_361_920_418)
VAE = ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", 253_815_318)
CLIP = ("Wan-AI/Wan2.1-I2V-14B-720P",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", 4_772_359_047)

# DiT files under Comfy-Org split_files/diffusion_models/
WAN21_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
WAN22_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
DIT_PREFIX = "split_files/diffusion_models"

# name -> list of component keys used by this base_model
# component values: (role, repo, filename, expected_size)

def _dit(repo: str, name: str, size: int) -> tuple[str, str, int]:
    return (repo, f"{DIT_PREFIX}/{name}", size)


MODEL_COMPONENTS: dict[str, dict[str, tuple[str, str, int]]] = {
    "wan-t2v-1.3b": {
        "t5": T5,
        "vae": VAE,
        "dit": _dit(WAN21_REPO, "wan2.1_t2v_1.3B_fp16.safetensors", 2_838_303_560),
    },
    "wan-t2v-14b": {
        "t5": T5,
        "vae": VAE,
        "dit": _dit(WAN21_REPO, "wan2.1_t2v_14B_fp16.safetensors", 28_577_761_176),
    },
    "wan-i2v-14b": {
        "t5": T5,
        "vae": VAE,
        "clip": CLIP,
        "dit": _dit(WAN21_REPO, "wan2.1_i2v_720p_14B_fp16.safetensors", 32_791_377_504),
    },
    "wan22-t2v-a14b": {
        "t5": T5,
        "vae": VAE,
        "dit": _dit(WAN22_REPO, "wan2.2_t2v_low_noise_14B_fp16.safetensors", 28_577_095_592),
        "dit_high": _dit(WAN22_REPO, "wan2.2_t2v_high_noise_14B_fp16.safetensors", 28_577_095_592),
    },
    "wan22-i2v-a14b": {
        "t5": T5,
        "vae": VAE,
        "dit": _dit(WAN22_REPO, "wan2.2_i2v_low_noise_14B_fp16.safetensors", 28_577_914_792),
        "dit_high": _dit(WAN22_REPO, "wan2.2_i2v_high_noise_14B_fp16.safetensors", 28_577_914_792),
    },
}

FORBIDDEN_SUBSTR = ("fp8_scaled", "fp8scaled")


class ModelError(RuntimeError):
    pass


def components_for(cfg) -> dict[str, tuple[str, str, int]]:
    files = dict(MODEL_COMPONENTS[cfg.base_model])
    if cfg.is_wan22 and cfg.wan22_mode in ("low", "high"):
        # Single-expert training: only one DiT is required.
        if cfg.wan22_mode == "low":
            files.pop("dit_high", None)
        else:
            # high expert is passed as --dit
            high = files.pop("dit_high")
            files["dit"] = high
    return files


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _assert_not_fp8_scaled(filename: str) -> None:
    low = filename.lower()
    if any(s in low for s in FORBIDDEN_SUBSTR):
        raise ModelError(
            f"refusing {filename}: musubi-tuner does not support ready-made Comfy fp8_scaled weights "
            "(use the fp16/bf16 files; the worker enables --fp8_base at training time)"
        )


def _head_url(url: str) -> dict[str, Any]:
    r = requests.head(url, allow_redirects=True, timeout=20)
    # some HF endpoints reject HEAD; fall back to a range GET
    if r.status_code in (400, 403, 405):
        r = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=20)
    size = int(r.headers.get("content-length") or r.headers.get("content-range", "/0").rsplit("/", 1)[-1] or 0)
    return {"ok": r.status_code < 400, "status": r.status_code, "size": size,
            "size_gb": round(size / 1e9, 2) if size else None}


def check_access(cfg, token: str | None = None) -> dict[str, Any]:
    """Metadata-only check (no weights) that every required file is reachable. No token needed for defaults."""
    del token  # default Wan weights are public
    out: dict[str, Any] = {}
    for name, (repo, fn, expected) in components_for(cfg).items():
        _assert_not_fp8_scaled(fn)
        url = hf_url(repo, fn)
        try:
            meta = _head_url(url)
            if expected and meta.get("size") and abs(meta["size"] - expected) > 1024:
                meta["warning"] = f"size {meta['size']} != expected {expected}"
            meta["repo"] = repo
            meta["file"] = fn
            meta["url"] = url
            out[name] = meta
        except Exception as exc:
            out[name] = {"ok": False, "error": redact(str(exc)).splitlines()[0][:240],
                         "repo": repo, "file": fn, "url": url}
    return out


def _looks_complete(path: str, expected: int | None) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    if expected and os.path.getsize(path) != expected:
        return False
    return True


def _hf_download(repo: str, filename: str, dest_dir: str, expected: int | None, token: str | None) -> str:
    from huggingface_hub import hf_hub_download

    _assert_not_fp8_scaled(filename)
    os.makedirs(dest_dir, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            path = os.path.realpath(hf_hub_download(
                repo_id=repo, filename=filename, token=token, local_dir=dest_dir,
            ))
            if expected and os.path.getsize(path) != expected:
                # Incomplete / truncated: delete and retry.
                log.warning("size mismatch for %s: got %s expected %s (attempt %s)",
                            filename, os.path.getsize(path), expected, attempt)
                try:
                    os.remove(path)
                except OSError:
                    pass
                last_err = ModelError(f"{filename}: size {os.path.getsize(path) if os.path.exists(path) else 'missing'} "
                                      f"!= expected {expected}")
                continue
            return path
        except Exception as exc:
            last_err = exc
            log.warning("download of %s failed (attempt %s): %s", filename, attempt, redact(str(exc))[:200])
            time.sleep(min(8, 2 ** attempt))
    raise ModelError(f"download of {repo}/{filename} failed: {redact(str(last_err)).splitlines()[0][:300]}")


def ensure_models(cfg, token: str | None, progress=None) -> tuple[dict[str, str], dict[str, Any]]:
    """Download every required component (cached). Returns name -> local path and timing info."""
    cache = model_cache_dir()
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    paths: dict[str, str] = {}
    for name, (repo, fn, expected) in components_for(cfg).items():
        if progress:
            progress({"stage": "downloading_model", "component": name, "file": os.path.basename(fn)})
        local_dir = os.path.join(cache, repo.replace("/", "--"))
        dest_name = os.path.join(local_dir, fn)
        if _looks_complete(dest_name, expected):
            log.info("cache hit %s (%s)", name, dest_name)
            paths[name] = os.path.realpath(dest_name)
            continue
        paths[name] = _hf_download(repo, fn, local_dir, expected, token)
    sizes = {k: round(os.path.getsize(v) / 1e9, 2) for k, v in paths.items()}
    free_gb = round(shutil.disk_usage(cache).free / 1e9, 1)
    info = {"download_seconds": round(time.time() - t0, 1), "sizes_gb": sizes,
            "cache_dir": cache, "disk_free_gb": free_gb, "on_volume": has_volume()}
    if not has_volume():
        info["warning"] = "no network volume: weights were stored under /tmp and will be re-downloaded on the next worker"
    return paths, info
