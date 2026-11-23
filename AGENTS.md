# AGENTS.md

Instructions for coding agents working in `prilojject/runpod-video-lora-trainer`.

## What this repo is

A Runpod Serverless worker (Hub listing, `type: serverless`) that trains Wan 2.1 / Wan 2.2 LoRAs with
kohya-ss/musubi-tuner (pinned tag `v0.3.6` in the `Dockerfile`) and returns a `.safetensors` LoRA.
It is not a UI and not a Pod template.

## Layout

- `handler.py` — Runpod entry point; orchestrates dataset → models → cache → training → delivery
- `src/config.py` — job input > env (Hub form) > preset > defaults; validation
- `src/dataset.py` — download/unpack (zip-slip safe) videos + images + captions
- `src/models.py` — public Wan checkpoints (download at start, cache on `/runpod-volume`), access check
- `src/train.py` — musubi dataset TOML (separate `cache_directory` per `[[datasets]]`), cache/train commands, log streaming
- `src/outputs.py` — HF upload + model card, S3 (BUCKET_*), presigned PUT
- `src/secret_utils.py` — tokens from env only; log redaction
- `.runpod/hub.json`, `.runpod/tests.json`, `.runpod/icon.png` — Hub metadata, test, icon
- `examples/` — request bodies and a 3-image example dataset (`tiny_wan.zip`)
- `tests/` — unit tests (no GPU, no weight downloads)

## Local checks (no GPU, no paid cloud)

```bash
pip install -r requirements-dev.txt
pytest -q
python -c "import json,sys; sys.path.insert(0,'.'); from handler import handler; \
print(handler({'id':'local-dry','input': json.load(open('examples/dry_run.json'))['input']}))"
```

Do not add tests that download model weights or call paid APIs.

## Secrets

- `HF_TOKEN` and `BUCKET_*` come from env / Runpod secrets only. The worker rejects tokens in job input — keep it that way.
- Never print tokens, presigned URLs or `.env` files in logs, README examples or issues.

## Conventions

- Never bake model weights into the image. Downloads go to the network volume cache when present (`/runpod-volume/video-lora/models`).
- Clone musubi-tuner **by tag** (`--branch v0.3.6`), never by SHA. Record `git rev-parse HEAD` in `/opt/musubi-tuner.commit` for logs only.
- Do not download Comfy `fp8_scaled` weights. Train with `--fp8_base` (and `--fp8_t5` on small GPUs).
- Each `[[datasets]]` must have its own `cache_directory`. `target_frames` must be N×4+1.
- `--offload_inactive_dit` is mutually exclusive with `--blocks_to_swap`.
- Validate the dataset before any GPU work.
- Keep `tests.json` to a single 1.3B / 20-step / 512 / `num_samples=0` job. Never add 14B or Wan 2.2 to the official Hub test.
- Public docs are English.

## Paid GPU jobs

- One worker max, short 1.3B config, timeouts set (`TRAIN_TIMEOUT_SECONDS` and endpoint `executionTimeout`).
- Delete endpoints, pods, test templates and registry auths right after the test and verify nothing is left running.
- Training uses async `POST /run` with `policy.executionTimeout`; the 600 s default kills real jobs (cold start can download ~13.5 GiB).

## Releases

A GitHub release is required for the Hub to (re)index the repo. Bump `src/__init__.py` `__version__` and tag `vX.Y.Z`.
**`targetCommitish` must be `main`, never a SHA.** Hub clones with `git clone -b <targetCommitish>` and fails (exit 128) when the target is a commit SHA.
