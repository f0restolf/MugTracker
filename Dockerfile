# MugTracker - GPU-accelerated face tracker for OBS
# Base: ROCm 6.2.4 + Ubuntu 22.04 + Python 3.11
FROM rocm/dev-ubuntu-22.04:6.2.4

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    v4l2loopback-utils \
    v4l-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN pip3 install --upgrade pip

# RDNA2 environment variables (critical for RX 6900 XT)
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
ENV PYTORCH_ROCM_ARCH=gfx1030
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
ENV ROCM_PATH=/opt/rocm

# Install PyTorch with ROCm 6.2.4 support
RUN pip3 install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/rocm6.2.4

# Install project dependencies
RUN pip3 install \
    opencv-python \
    numpy \
    pyyaml \
    pyvirtualcam \
    ultralytics

WORKDIR /app

# Copy project files
COPY src/ ./src/
COPY config.yaml .

# Models directory - weights are mounted at runtime, not baked in
RUN mkdir -p models

# Verify GPU is accessible at build time (optional, comment out if building offline)
# RUN python3 -c "import torch; print('ROCm:', torch.cuda.is_available())"

CMD ["python3", "src/facetracker.py"]
