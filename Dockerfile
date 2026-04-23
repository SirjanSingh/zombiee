# Dockerfile for DGX / GPU training environment
# This image includes all training dependencies (torch, unsloth, trl)
# plus the env server for self-play training.
#
# Build:
#   docker build -f Dockerfile.dgx -t survivecity-train .
#
# Run (mount for checkpoint output):
#   docker run --gpus all -v $(pwd)/lora_v1:/app/lora_v1 survivecity-train

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1) — must come first so unsloth detects it
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Install core ML deps first (smaller graph, pinned versions)
RUN pip install --no-cache-dir \
    "transformers==4.40.2" \
    "accelerate==0.30.1" \
    "peft==0.10.0" \
    "datasets==2.19.1" \
    "trl==0.8.6"

# Install bitsandbytes separately (has its own CUDA build logic)
RUN pip install --no-cache-dir "bitsandbytes>=0.41"

# Install unsloth last — after all its deps are already satisfied
RUN pip install --no-cache-dir \
    "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Install monitoring / util deps
RUN pip install --no-cache-dir wandb "matplotlib>=3.8"

# Install env server deps
RUN pip install --no-cache-dir \
    "pydantic>=2.0" \
    "fastapi>=0.104" \
    "uvicorn[standard]>=0.24"

# Copy source
COPY survivecity_env/ survivecity_env/
COPY server/ server/
COPY training/ training/
COPY openenv.yaml pyproject.toml ./

# Create plots directory
RUN mkdir -p plots lora_v1

EXPOSE 7860

# Default: start env server in background, then run training
CMD bash -c "\
    echo '=== Starting SurviveCity env server ===' && \
    uvicorn server.app:app --host 0.0.0.0 --port 7860 & \
    sleep 3 && \
    echo '=== Starting GRPO training ===' && \
    python -m training.train \
    --env-url http://localhost:7860 \
    --max-steps 4000 \
    --output-dir /app/lora_v1 \
    --report-to tensorboard && \
    echo '=== Training complete ===' \
    "