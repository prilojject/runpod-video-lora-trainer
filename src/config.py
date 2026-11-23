"""Input schema for the video LoRA trainer (musubi-tuner / Wan).

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
TRIGGER_RE = re.compile(r"^[^\n\r,]{1,64}$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

# v1 must-support Wan tasks. Coming-soon IDs are recognized so the error is explicit.
WAN_MODELS = (
    "wan-t2v-1.3b",
    "wan-t2v-14b",
    "wan-i2v-14b",
    "wan22-t2v-a14b",
    "wan22-i2v-a14b",
)
COMING_SOON_MODELS = ("hunyuan-t2v", "framepack-i2v")
BASE_MODELS = WAN_MODELS + COMING_SOON_MODELS

# musubi --task values
MODEL_TASK = {
    "wan-t2v-1.3b": "t2v-1.3B",
    "wan-t2v-14b": "t2v-14B",
    "wan-i2v-14b": "i2v-14B",
    "wan22-t2v-a14b": "t2v-A14B",
    "wan22-i2v-a14b": "i2v-A14B",
}

# Max --blocks_to_swap musubi documents.
MAX_BLOCKS_1_3B = 29
MAX_BLOCKS_14B = 39

I2V_MODELS = {"wan-i2v-14b", "wan22-i2v-a14b"}
WAN22_MODELS = {"wan22-t2v-a14b", "wan22-i2v-a14b"}
IS_14B = {"wan-t2v-14b", "wan-i2v-14b", "wan22-t2v-a14b", "wan22-i2v-a14b"}

# Preset defaults applied only when the field was not set by input or env.
PRESETS: dict[str, dict[str, Any]] = {
    "wan-t2v-1.3b": {
        "steps": 2000, "rank": 16, "learning_rate": 2e-4, "resolution": (832, 480),
    },
    "wan-t2v-14b": {
        "steps": 2000, "rank": 16, "learning_rate": 2e-4, "resolution": (832, 480),
    },
    "wan-i2v-14b": {
        "steps": 2000, "rank": 16, "learning_rate": 2e-4, "resolution": (832, 480),
    },
    "wan22-t2v-a14b": {
        "steps": 2000, "rank": 16, "learning_rate": 2e-4, "resolution": (832, 480),
    },
    "wan22-i2v-a14b": {
        "steps": 2000, "rank": 16, "learning_rate": 2e-4, "resolution": (832, 480),
    },
    "quick": {
        "steps": 20, "rank": 8, "learning_rate": 2e-4, "resolution": (512, 512),
    },
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


# Sentinel type for fields with custom coercion.
class ResolutionType:
    pass


class FramesType:
    pass


class BlocksType:
    pass


PARAMS: list[Param] = [
    Param("base_model", str, "wan-t2v-1.3b", "BASE_MODEL", choices=BASE_MODELS,
          help="Wan 2.1 T2V 1.3B (default, 24 GB), T2V/I2V 14B, or Wan 2.2 T2V/I2V A14B (48 GB recommended)."),
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL to a .zip / .tar(.gz) of videos and/or images with optional same-name .txt captions."),
    Param("trigger_word", str, "", "TRIGGER_WORD", help="Unique token for the concept, e.g. 'zxowl'."),
    Param("class_word", str, "", "CLASS_WORD", help="Generic class, e.g. 'figurine'."),
    Param("caption_mode", str, "auto", "CAPTION_MODE", choices=("auto", "existing", "trigger"),
          help="auto = use .txt captions when present, else trigger word; trigger = ignore captions; "
               "existing = require .txt (missing ones fall back to the trigger word with a warning)."),
    Param("prepend_trigger", bool, True, "PREPEND_TRIGGER",
          help="Make sure every caption starts with the trigger word."),
    Param("preset", str, "wan-t2v-1.3b", "PRESET", choices=tuple(PRESETS),
          help="Named defaults for steps/rank/lr/resolution. Explicit settings win."),
    Param("steps", int, None, "STEPS", min=10, max=20000),
    Param("rank", int, None, "LORA_RANK", min=1, max=256),
    Param("alpha", float, 0.0, "LORA_ALPHA", min=0.0, max=1024, help="0 = same as rank."),
    Param("learning_rate", float, None, "LEARNING_RATE", min=1e-7, max=1e-2),
    Param("resolution", ResolutionType, None, "RESOLUTION",
          help="Training resolution: 512 or '832x480' or [w, h]. Must be a multiple of 16."),
    Param("batch_size", int, 1, "BATCH_SIZE", min=1, max=8),
    Param("frames", FramesType, None, None,
          help="Alias of target_frames: int or list, each length must be N*4+1 (e.g. 1, 17, 25, 81)."),
    Param("target_frames", FramesType, None, "TARGET_FRAMES",
          help="Video frame lengths to extract (N×4+1). Default [17]. Ignored for image-only datasets."),
    Param("source_fps", float, 16.0, "SOURCE_FPS", min=1.0, max=120.0),
    Param("frame_extraction", str, "head", "FRAME_EXTRACTION",
          choices=("head", "chunk", "full", "slide", "uniform")),
    Param("wan22_mode", str, "both", "WAN22_MODE", choices=("both", "low", "high"),
          help="Wan 2.2 only: train both experts, or only the low- or high-noise DiT."),
    Param("blocks_to_swap", BlocksType, "auto", "BLOCKS_TO_SWAP",
          help="'auto' picks a value from VRAM, 0 disables swap, or an integer (max 29 for 1.3B, 39 for 14B)."),
    Param("fp8_base", bool, True, "FP8_BASE", help="Train the base DiT in fp8 (recommended)."),
    Param("seed", int, 42, "SEED", min=0),
    Param("save_every_n_steps", int, 0, "SAVE_EVERY_N_STEPS", min=0,
          help="Intermediate checkpoints (kept on the network volume). 0 = only the final LoRA."),
    Param("num_samples", int, 0, "NUM_SAMPLES", min=0, max=4,
          help="Short sample videos after training (expensive; default 0)."),
    Param("sample_steps", int, 12, "SAMPLE_STEPS", min=4, max=50),
    Param("sample_frames", int, 17, "SAMPLE_FRAMES", min=1, max=81),
    Param("sample_prompts", list, None, "SAMPLE_PROMPTS",
          help="List (or '|'-separated string) of prompts; '{trigger}' is replaced."),
    Param("return_lora_base64", bool, False, "RETURN_LORA_BASE64",
          help="Include the LoRA as base64 in the job output (only if the file is small)."),
    Param("return_samples_base64", bool, False, "RETURN_SAMPLES_BASE64"),
    Param("convert_lora", bool, False, "CONVERT_LORA",
          help="Also write a ComfyUI-compatible LoRA via convert_lora.py --target other."),
    Param("output_name", str, "", "OUTPUT_NAME", help="LoRA file name (default: '<trigger>_<base_model>_lora')."),
    Param("run_name", str, "", "RUN_NAME", help="Folder name for outputs. Default: the Runpod job id."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO", help="Upload the LoRA here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the .safetensors file."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET"),
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate the dataset, check model URLs, return the plan. No GPU work."),
    Param("max_media", int, 200, "MAX_MEDIA", min=1, max=5000,
          help="Max number of videos + images to use from the archive."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "model": "base_model",
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "trigger": "trigger_word",
    "lr": "learning_rate",
    "max_train_steps": "steps",
    "network_dim": "rank",
    "lora_rank": "rank",
    "network_alpha": "alpha",
    "lora_alpha": "alpha",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def task(self) -> str:
        return MODEL_TASK[self.values["base_model"]]

    @property
    def is_i2v(self) -> bool:
        return self.values["base_model"] in I2V_MODELS

    @property
    def is_wan22(self) -> bool:
        return self.values["base_model"] in WAN22_MODELS

    @property
    def is_14b(self) -> bool:
        return self.values["base_model"] in IS_14B

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.values["resolution"]

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        # tuples → lists for JSON
        if isinstance(out.get("resolution"), tuple):
            out["resolution"] = list(out["resolution"])
        return out


def _parse_resolution(raw: Any) -> tuple[int, int]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        w, h = int(raw[0]), int(raw[1])
        return w, h
    if isinstance(raw, bool):
        raise ValueError("resolution: expected a number or WxH, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"resolution: expected integer or WxH, got {raw!r}")
        n = int(raw)
        return n, n
    s = str(raw).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+)[x,;](\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    if re.fullmatch(r"\d+", s):
        n = int(s)
        return n, n
    raise ValueError(f"resolution: expected a number or WxH (e.g. 832x480), got {raw!r}")


def _parse_frames(raw: Any) -> list[int]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, bool):
        raise ValueError("target_frames: expected int or list, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"target_frames: expected integers, got {raw!r}")
        return [int(raw)]
    if isinstance(raw, (list, tuple)):
        out = []
        for x in raw:
            out.extend(_parse_frames(x))
        return out
    s = str(raw).strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    try:
        return [int(p) for p in parts if p]
    except ValueError as exc:
        raise ValueError(f"target_frames: expected integers (N×4+1), got {raw!r}") from exc


def _parse_blocks(raw: Any) -> Any:
    if raw is None or raw == "":
        return "auto"
    if isinstance(raw, bool):
        raise ValueError("blocks_to_swap: expected 'auto' or an integer, got boolean")
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            raise ValueError(f"blocks_to_swap: expected integer or 'auto', got {raw!r}")
        return int(raw)
    s = str(raw).strip().lower()
    if s == "auto":
        return "auto"
    try:
        return int(s)
    except ValueError as exc:
        raise ValueError(f"blocks_to_swap: expected 'auto' or an integer, got {raw!r}") from exc


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is ResolutionType:
        return _parse_resolution(raw)
    if t is FramesType:
        return _parse_frames(raw)
    if t is BlocksType:
        return _parse_blocks(raw)
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is list:
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            return [x.strip() for x in raw.split("|") if x.strip()]
        raise ValueError(f"{param.name}: expected a list of strings")
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def _valid_frame_len(n: int) -> bool:
    # N×4+1, N >= 0 → 1, 5, 9, 13, 17, ...
    return n >= 1 and (n - 1) % 4 == 0


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = os.environ if env is None else env
    job_input = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []

    for alias, canonical in ALIASES.items():
        if alias in job_input:
            if canonical in job_input:
                errors.append(f"both '{alias}' and '{canonical}' given")
            else:
                job_input[canonical] = job_input.pop(alias)

    # frames is an alias of target_frames with the same "both given" rule.
    if "frames" in job_input and "target_frames" in job_input:
        errors.append("both 'frames' and 'target_frames' given")
    elif "frames" in job_input and "target_frames" not in job_input:
        job_input["target_frames"] = job_input["frames"]

    for secret_like in ("hf_token", "HF_TOKEN", "token", "aws_secret_access_key", "secret_key"):
        if secret_like in job_input:
            errors.append("do not send tokens/keys in the job input; set HF_TOKEN / BUCKET_* as Runpod secrets")
            job_input.pop(secret_like, None)

    unknown = sorted(k for k in job_input if k not in PARAM_BY_NAME)
    if unknown:
        errors.append(f"unknown input field(s): {', '.join(unknown)}")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for p in PARAMS:
        raw, src = None, "default"
        if p.name in job_input and job_input[p.name] is not None:
            raw, src = job_input[p.name], "input"
        elif p.env and env.get(p.env) not in (None, ""):
            raw, src = env.get(p.env), "env"
        val = p.default
        if src != "default":
            try:
                val = _coerce(p, raw)
            except ValueError as exc:
                errors.append(str(exc))
        values[p.name], sources[p.name] = val, src

    preset_name = values.get("preset") or "wan-t2v-1.3b"
    preset = PRESETS.get(preset_name)
    if preset is None:
        errors.append(f"preset: must be one of {list(PRESETS)}, got {preset_name!r}")
        preset = PRESETS["wan-t2v-1.3b"]
    for k, v in preset.items():
        if values.get(k) is None:
            values[k], sources[k] = (list(v) if isinstance(v, list) else v), "preset"

    # Built-in fallbacks if still unset (e.g. a typo preset already reported).
    if values.get("steps") is None:
        values["steps"], sources["steps"] = 2000, "default"
    if values.get("rank") is None:
        values["rank"], sources["rank"] = 16, "default"
    if values.get("learning_rate") is None:
        values["learning_rate"], sources["learning_rate"] = 2e-4, "default"
    if values.get("resolution") is None:
        values["resolution"], sources["resolution"] = (832, 480), "default"
    if values.get("target_frames") is None:
        values["target_frames"], sources["target_frames"] = [17], "default"
    if not values.get("target_frames"):
        values["target_frames"] = [17]

    for p in PARAMS:
        val = values[p.name]
        if p.type in (int, float) and isinstance(val, (int, float)) and not isinstance(val, bool):
            if p.min is not None and val < p.min:
                errors.append(f"{p.name}: must be >= {p.min}, got {val}")
            if p.max is not None and val > p.max:
                errors.append(f"{p.name}: must be <= {p.max}, got {val}")
        if p.choices and val not in p.choices and val is not None and p.type is not BlocksType:
            errors.append(f"{p.name}: must be one of {list(p.choices)}, got {val!r}")

    bm = values["base_model"]
    if bm in COMING_SOON_MODELS:
        errors.append(f"{bm} is coming soon in a later release; v1 trains Wan 2.1 / Wan 2.2 only "
                      f"({', '.join(WAN_MODELS)})")

    url = values["dataset_url"]
    if not url:
        errors.append("dataset: set 'dataset_url' to an https URL of a .zip / .tar(.gz) of videos and/or images")
    elif not URL_RE.match(url):
        errors.append("dataset_url: expected an http(s) URL")

    tw = values["trigger_word"]
    if tw and not TRIGGER_RE.match(tw):
        errors.append("trigger_word: one short phrase, no commas or newlines (max 64 chars)")
    if not tw:
        if values["caption_mode"] == "trigger":
            errors.append("caption_mode 'trigger' needs trigger_word")
        else:
            warnings.append("no trigger_word: the LoRA will be driven by your captions only")

    if values["alpha"] in (0, 0.0, None):
        values["alpha"] = float(values["rank"])

    res = values["resolution"]
    if not (isinstance(res, tuple) and len(res) == 2):
        errors.append("resolution: expected a number or [width, height]")
    else:
        w, h = res
        if min(w, h) < 256 or max(w, h) > 1920:
            errors.append(f"resolution: each side must be 256–1920, got {w}x{h}")
        if w % 16 or h % 16:
            errors.append(f"resolution must be a multiple of 16 (Wan latent size), got {w}x{h}")

    frames = values["target_frames"]
    if not isinstance(frames, list) or not frames:
        errors.append("target_frames: provide at least one length (N×4+1, e.g. 17)")
    else:
        bad = [n for n in frames if not isinstance(n, int) or not _valid_frame_len(n)]
        if bad:
            errors.append(f"target_frames must be N×4+1 (e.g. 1, 17, 25, 45, 81); invalid: {bad}")
        if any(n > 129 for n in frames):
            errors.append("target_frames: each length must be <= 129")

    if not _valid_frame_len(values["sample_frames"]):
        errors.append("sample_frames must be N×4+1 (e.g. 17)")

    blocks = values["blocks_to_swap"]
    if blocks != "auto":
        if not isinstance(blocks, int) or blocks < 0:
            errors.append("blocks_to_swap: must be 'auto' or an integer >= 0")
        else:
            cap = MAX_BLOCKS_14B if bm in IS_14B else MAX_BLOCKS_1_3B
            if blocks > cap:
                errors.append(f"blocks_to_swap: max {cap} for {bm}, got {blocks}")

    prompts = [p for p in (values["sample_prompts"] or []) if p.strip()]
    if any(len(p) > 1000 for p in prompts):
        errors.append("sample_prompts: each prompt max 1000 characters")
    values["sample_prompts"] = prompts[: values["num_samples"]] if values["num_samples"] else []

    if values["hf_repo_id"] and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id: expected 'owner/name'")
    if values["upload_url"] and not values["upload_url"].startswith("https://"):
        errors.append("upload_url: expected an https presigned PUT URL")
    if values["run_name"] and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name: letters, digits, '.', '_' or '-' (max 100 chars)")
    if values["output_name"] and not RUN_NAME_RE.match(values["output_name"]):
        errors.append("output_name: letters, digits, '.', '_' or '-' (max 100 chars)")

    has_token = bool((env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
                     and not (env.get("HF_TOKEN") or "").startswith("{{"))
    if values["hf_repo_id"] and not has_token:
        msg = "hf_repo_id is set but HF_TOKEN is not; add HF_TOKEN (write access) as a Runpod secret"
        (warnings if values["dry_run"] else errors).append(msg)

    if bm in IS_14B:
        warnings.append(f"{bm} is a large model (24–53 GiB of weights); a 48 GB GPU is recommended. "
                        "The default smoke path is wan-t2v-1.3b on 24 GB.")

    # Drop the frames alias from the public values (kept only as input alias).
    values.pop("frames", None)

    if errors:
        raise ConfigError(errors)
    return ResolvedConfig(values=values, sources=sources, warnings=warnings)


def render_prompt(template: str, trigger: str, class_word: str) -> str:
    subject = " ".join(x for x in (trigger, class_word) if x).strip()
    out = template.replace("{trigger}", subject or "the subject")
    return re.sub(r"\s+", " ", out).strip()


def default_sample_prompts(cfg: ResolvedConfig) -> list[str]:
    tw = " ".join(x for x in (cfg.trigger_word, cfg.class_word) if x).strip() or "the subject"
    if cfg.is_i2v:
        return [f"{tw} slowly turning, cinematic lighting, highly detailed"]
    return [
        f"a video of {tw}, cinematic lighting, highly detailed, smooth motion",
        f"{tw} on a wooden table, soft daylight, gentle camera move",
    ]
