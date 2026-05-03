# FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml ./

RUN uv sync --no-dev
# RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-build-isolation
RUN uv add psutil --no-build-isolation
RUN uv add flash-attn --no-build-isolation

COPY colbert colbert
COPY scripts scripts
COPY configs configs

RUN chmod +x scripts/*.sh

ENTRYPOINT ["uv", "run", "torchrun"]
CMD ["--nproc_per_node=1", "scripts/train.py", "--config", "configs/default.yaml"]
