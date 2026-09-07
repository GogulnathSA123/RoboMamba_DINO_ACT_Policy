# ==============================================================================
# RoboMamba-DINO-ACT & SO-101 Teleoperation + Isaac Sim Simulation Stack
# Ready-to-use Docker container for:
#  1. Leader-Follower Teleoperation & Dataset Collection in Isaac Sim
#  2. Custom Tabletop Scene Generation with PhysX 5.0 Dynamics
#  3. Multi-Seed Simulation Benchmarking (RoboMamba vs. ACT)
#  4. Real-World SO-101 Hardware Deployment & Closed-Loop Evaluation
# ==============================================================================

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl

# 1. System Dependencies (USB serial for Leader/Follower arms, OpenGL, FFmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    pkg-config \
    libusb-1.0-0-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libegl1-mesa-dev \
    udev \
    v4l-utils \
    cmake \
    ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. CUDA Compilation Dependencies for Mamba SSM
RUN pip install --no-cache-dir packaging ninja setuptools wheel

# 3. Install Mamba Selective State-Space Models (compiled against CUDA 12.4)
RUN pip install --no-cache-dir causal-conv1d>=1.4.0 --no-build-isolation
RUN pip install --no-cache-dir mamba-ssm==2.2.0 --no-build-isolation

# 4. Install Vision, Robotics, & ML Libraries
RUN pip install --no-cache-dir \
    timm==1.0.7 \
    transformers==4.43.3 \
    einops==0.8.0 \
    h5py==3.11.0 \
    opencv-python-headless==4.10.0.84 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    wandb==0.17.8 \
    pyserial==3.5 \
    feetech-servo-sdk \
    datasets \
    huggingface_hub

# 5. Set up Project Workspace
WORKDIR /workspace/RoboMamba_DINO_ACT_Policy
COPY . /workspace/RoboMamba_DINO_ACT_Policy

# 6. Install LeIsaac Simulation Package in Editable Mode
WORKDIR /workspace/RoboMamba_DINO_ACT_Policy/leisaac
RUN pip install --no-cache-dir -e "source/leisaac" || true

WORKDIR /workspace/RoboMamba_DINO_ACT_Policy

# Default entrypoint
CMD ["/bin/bash"]
