# Video LoRA Trainer (Wan 2.1 & Wan 2.2) — Runpod Serverless worker

Train your own **Wan 2.1** or **Wan 2.2** LoRA — a character, a product, an art style — on a
**Runpod Serverless** GPU with [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) **v0.3.6**.
You send a URL to a zip of your videos and/or images (with optional `.txt` captions) and a trigger word;
you get back a **`.safetensors` LoRA** (kohya / musubi format) on your Hugging Face repo, S3 bucket,
presigned URL or network volume. Optional conversion to a ComfyUI-friendly file via `convert_lora.py --target other`.
No Pod, no UI, pay per second of training.

Useful search terms: *train Wan 2.1 LoRA on Runpod serverless*, *Wan 2.2 LoRA training API*,
*musubi-tuner serverless worker*, *Wan T2V 1.3B LoRA trainer Runpod Hub*, *image-to-video LoRA training*.

<!-- The badge and Hub URL go live after Runpod reviews the listing. -->
[![Runpod](https://api.runpod.io/badge/prilojject/runpod-video-lora-trainer)](https://console.runpod.io/hub/prilojject/runpod-video-lora-trainer)

**Deploy on Runpod Hub:** [console.runpod.io/hub/prilojject/runpod-video-lora-trainer](https://console.runpod.io/hub/prilojject/runpod-video-lora-trainer)

> **Listing status:** TODO — paste the live Hub URL here once moderation approves the repo.

Source: [github.com/prilojject/runpod-video-lora-trainer](https://github.com/prilojject/runpod-video-lora-trainer) (MIT).
For agents: [llms.txt](./llms.txt) · [AGENTS.md](./AGENTS.md) · [openapi.yaml](./openapi.yaml) · [CITATION.cff](./CITATION.cff)

---

## What you send / what you get

**You send** (job `input`; defaults can come from the Hub deploy form):

| Field | Example | Notes |
|---|---|---|
| `base_model` | `wan-t2v-1.3b` (default) | Also `wan-t2v-14b`, `wan-i2v-14b`, `wan22-t2v-a14b`, `wan22-i2v-a14b`. |
| `dataset_url` | `https://.../clips.zip` | `.zip` or `.tar(.gz)` of videos (`.mp4/.webm/.mov/.mkv`) and/or images (`.jpg/.png/.webp`), optional same-name `.txt` captions. |
| `trigger_word` | `zxowl` | Rare token that activates the LoRA. Prepended to every caption. |
| `class_word` | `figurine` | Optional generic class used in trigger-only captions. |
| `steps`, `rank`, `learning_rate` | `2000`, `16`, `2e-4` | musubi-tuner defaults for Wan LoRAs. |
| `resolution` | `832x480` or `512` | Multiple of 16. A single number is square. |
| `target_frames` / `frames` | `[17]` | Video lengths; each must be **N×4+1** (1, 17, 25, 45, 81…). Ignored for image-only datasets. |
| `wan22_mode` | `both` / `low` / `high` | Wan 2.2 only: both experts, or a single DiT. |
| `caption_mode` | `auto` / `trigger` / `existing` | `auto` uses `.txt` when present, otherwise the trigger word. |
| `convert_lora` | `false` | Also write a ComfyUI file via `convert_lora.py --target other`. |
| `hf_repo_id` | `your-name/my-lora` | Upload LoRA + model card (needs `HF_TOKEN` with write access). |
| `upload_url` | presigned `PUT` URL | The `.safetensors` is `PUT` there. |
| `num_samples` | `0` (default) | Short sample clips after training; video sampling is expensive. |
| `dry_run` | `true` | Validate the dataset and check that model URLs are reachable. No GPU work. |

**You get** (job `output`, trimmed):

```json
{
  "status": "completed",
  "gpu": {"gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 23.5},
  "training": {"global_step": 20, "final_avr_loss": 0.12, "train_loop_seconds": 90.0},
  "artifacts": {
    "lora_file": "zxowl_wan_t2v_1_3b_lora.safetensors",
    "lora_size_mb": 12.4,
    "hf_repo_url": "https://huggingface.co/your-name/zxowl-wan13-lora",
    "volume_path": "/runpod-volume/video-lora/runs/<run_name>/..."
  },
  "dataset": {"videos_used": 0, "images_used": 3, "captions_from_files": 3, "warnings": []}
}
```

The LoRA is in kohya / musubi format. For ComfyUI, set `convert_lora: true` or convert the file yourself with musubi's `convert_lora.py --target other`.

> **Deliver the LoRA somewhere.** Serverless workers are ephemeral. Set at least one of: a network volume
> (files land in `/runpod-volume/video-lora/runs/<run_name>/`), `hf_repo_id` + `HF_TOKEN`, `upload_url`, or the
> `BUCKET_*` env vars. Without them the response carries a warning (and the LoRA only if you set `return_lora_base64`).

Always use async `POST /run` and raise `policy.executionTimeout` — the 600 s default is too short (a cold start may download ~13.5 GiB for the default 1.3B stack).

## Supported models

| `base_model` | musubi `--task` | Notes |
|---|---|---|
| `wan-t2v-1.3b` **(default)** | `t2v-1.3B` | No token, fits 24 GB, fastest smoke test. Weights ≈ 13.5 GiB (T5 + VAE + DiT). |
| `wan-t2v-14b` | `t2v-14B` | 48 GB recommended. DiT alone ≈ 26.6 GiB. |
| `wan-i2v-14b` | `i2v-14B` | Needs **video** in the dataset (first frame is the I2V control). Also downloads CLIP. |
| `wan22-t2v-a14b` | `t2v-A14B` | Two DiTs (high + low) or one expert via `wan22_mode`. 48 GB recommended. |
| `wan22-i2v-a14b` | `i2v-A14B` | Same; CLIP is not required for Wan 2.2. |

**Coming soon (not in v1):** HunyuanVideo T2V and FramePack I2V. The worker returns a clear error if you request them so the Wan path stays stable. Fun-Control / VACE, Wan 2.2 5B, FLUX.1 Kontext, Qwen-Image and full fine-tunes are out of scope.

Default Wan weights are **public** (Comfy-Org fp16 / official T5+CLIP) and do **not** need `HF_TOKEN`. We never download Comfy `fp8_scaled` files — musubi-tuner does not support them. Training uses `--fp8_base` instead.

Attach a **network volume** so checkpoints are cached under `/runpod-volume/video-lora/models/` and are not re-downloaded on every cold start. Weights are **not** baked into the image.

## Dataset format

```
clips.zip
├── videos/                 # optional folder
│   ├── take_001.mp4
│   └── take_001.txt
├── images/                 # optional folder (or a flat zip of images only)
│   ├── still_001.jpg
│   └── still_001.txt
└── ...
```

A flat archive (images and/or videos in one folder) is also fine. Captions are the same file name with `.txt`.
Missing captions fall back to `trigger_word [class_word]`. The trigger word is prepended when it is not already in the caption.

* Image-only datasets are officially supported by musubi and are the cheapest way to smoke-test (this repo ships `examples/datasets/tiny_wan.zip`).
* I2V models (`wan-i2v-14b`, `wan22-i2v-a14b`) need at least one **video**.
* `target_frames` must be **N×4+1**. Default `[17]`, `frame_extraction=head`, `source_fps=16`.
* Archives are zip-slip safe; size limit is 8 GiB.

## GPUs, VRAM and cost

* **Minimum 24 GB** (RTX 4090, L4, A5000, …) for the default 1.3B T2V job. Blackwell MIG SKUs are excluded (this image is a CUDA 12.6 build).
* **48 GB** (L40S, A6000 Ada, A6000, H100) is recommended for 14B and Wan 2.2 (two 14B experts ≈ 53 GiB of DiT weights alone).
* The worker always uses `--sdpa --mixed_precision fp16 --gradient_checkpointing --optimizer_type adamw8bit`. Official Comfy-Org DiT files are fp16; musubi-tuner requires `fp16` (or `no`) mixed precision for those weights. Default `--fp8_base`. On 24 GB cards it enables `--fp8_t5` / `--vae_cache_cpu` and automatic `--blocks_to_swap` (max 29 for 1.3B, 39 for 14B). `--offload_inactive_dit` (Wan 2.2 both experts on 48 GB+) is mutually exclusive with block swap.
* Recommended default resolution: **832×480**. On 24 GB with 14B or long clips, drop to 640×480 or 480×480.
* **Do not** run 14B / Wan 2.2 on the official Hub test — that test is a 20-step 1.3B job at 512 px with no samples.

**Measured smoke test (23 Nov 2026, Asia/Bangkok, Runpod Serverless, RTX 4090 / `ADA_24`, image `v1.0.1`, no network volume):**
Wan 2.1 T2V 1.3B, 20 steps, 512 px, 3 images, no samples — job execution 94.8 s (model download 43 s, train loop 13.5 s), LoRA 41.82 MB, final avr_loss 0.077. Queue time 46.7 s. Check [Runpod pricing](https://www.runpod.io/pricing) for current rates.

## Quick start

```bash
# Validate dataset + model URLs (no GPU work)
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"dry_run": true, "dataset_url": "https://.../clips.zip", "trigger_word": "zxowl"}}'

# 20-step smoke test (Wan 2.1 T2V 1.3B)
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d @examples/smoke_test_request.json
```

See `examples/` for more request bodies. Poll `GET /status/{id}`. Raise `policy.executionTimeout` (30+ minutes for a cold 1.3B job).

## Secrets

Put tokens in **Runpod secrets / endpoint env**, never in the job `input`. The worker rejects `hf_token` / `HF_TOKEN` in the payload. `HF_TOKEN` is only required to **upload** the LoRA to `hf_repo_id`. Default base weights are public.

## License

The worker code is **MIT**. Upstream [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) is Apache-2.0. Wan 2.1 and Wan 2.2 weights are Apache-2.0. If HunyuanVideo support is added later, Tencent's license applies to those weights and will be documented in the model card.

## FAQ

**Why is my job still downloading after several minutes?** The default stack is ~13.5 GiB (T5 10.6 GiB + VAE + 1.3B DiT). Attach a network volume so the next job is a cache hit.

**Can I train on images only?** Yes, for T2V models. I2V models need video.

**Will this train Hunyuan or FramePack?** Not in v1. Those IDs return a "coming soon" error.

**Why was my 14B job slow / OOM on a 4090?** Use a 48 GB GPU, or lower resolution/frames and keep automatic block swap. Do not point the worker at `fp8_scaled` Comfy files.

## Disclaimer

This is an unofficial worker around musubi-tuner. Training quality depends on your data, captions, steps and rank. The 20-step Hub test only checks that the pipeline runs; it does not produce a usable LoRA.
