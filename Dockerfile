# Video LoRA trainer (Wan 2.1 / Wan 2.2) for Runpod Serverless, built on kohya-ss/musubi-tuner.
# No model weights are baked in: checkpoints are downloaded at job start and cached on the
# network volume (/runpod-volume/video-lora/models) when one is attached.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG MUSUBI_TUNER_REF=v0.3.6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    MUSUBI_TUNER_DIR=/opt/musubi-tuner

# git: clone musubi-tuner; ffmpeg: video decode; libgl1/libglib2.0-0: opencv; build-essential: bnb/av.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ffmpeg ca-certificates libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# kohya-ss/musubi-tuner pinned to a release tag (clone by tag, not SHA). Keep the
# base image's torch stack frozen so we do not upgrade CUDA/torch at install time.
RUN git clone --depth 1 --branch ${MUSUBI_TUNER_REF} https://github.com/kohya-ss/musubi-tuner.git ${MUSUBI_TUNER_DIR} \
    && cd ${MUSUBI_TUNER_DIR} && git rev-parse HEAD > /opt/musubi-tuner.commit \
    && pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install -e . -c /tmp/constraints.txt \
    && rm -rf .git

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && pip freeze > /app/requirements.lock \
    && python -c "import torch, musubi_tuner, accelerate, bitsandbytes, runpod, av; \
print('torch', torch.__version__, '| musubi commit', open('/opt/musubi-tuner.commit').read().strip(), \
'| accelerate', accelerate.__version__, '| bnb', bitsandbytes.__version__, '| runpod', runpod.__version__, '| av', av.__version__)"

# accelerate default config: single GPU, no prompts at runtime.
RUN mkdir -p /root/.cache/huggingface/accelerate && printf '%s\n' \
    'compute_environment: LOCAL_MACHINE' 'distributed_type: "NO"' 'downcast_bf16: "no"' \
    'machine_rank: 0' 'main_training_function: main' 'mixed_precision: fp16' 'num_machines: 1' \
    'num_processes: 1' 'rdzv_backend: static' 'same_network: true' 'use_cpu: false' \
    > /root/.cache/huggingface/accelerate/default_config.yaml

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
CMD ["python", "-u", "/app/handler.py"]
