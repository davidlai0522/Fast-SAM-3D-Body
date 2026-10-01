#!/bin/bash
set -e

# === SAM 3D Body Environment Setup (following official guide) ===
# Requires: uv installed, and system CUDA 12.4 toolkit already installed
# (needed for detectron2 compilation). Set CUDA_HOME if not at default.

# Step 1: Create venv
# uv venv .venv --python 3.11
source .venv/bin/activate

# # Step 2: Point to system CUDA toolkit (for detectron2 compilation)
# export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# # Step 3: Install PyTorch (CUDA 12.4)
# echo "=== Installing PyTorch ==="
# uv pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
#     --extra-index-url https://download.pytorch.org/whl/cu124

# # Step 4: Install Python dependencies
# echo "=== Installing Python dependencies ==="
# uv pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm \
#     dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils \
#     webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope \
#     ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black \
#     pycocotools tensorboard huggingface_hub

# # Step 5: Install Detectron2
# echo "=== Installing Detectron2 ==="
# uv pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
#     --no-build-isolation --no-deps

# # Step 6: Install YOLO (ultralytics, for human detection)
# echo "=== Installing YOLO ==="
# uv pip install ultralytics

# Step 7: Install MoGe
echo "=== Installing MoGe ==="
uv pip install git+https://github.com/microsoft/MoGe.git

# Step 8: Install TensorRT + ONNX (optional, for .engine model conversion)
echo "=== Installing TensorRT & ONNX ==="
uv pip install tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs onnx onnxruntime-gpu nvtx


uv pip install smplx numpy scipy opencv-python tqdm pyzmq pyrealsense2
uv pip install chumpy --no-build-isolation

# Step 9: Install SAM3 (optional, uncomment if needed)
# echo "=== Installing SAM3 ==="
# cd /tmp
# rm -rf sam3
# git clone https://github.com/facebookresearch/sam3.git
# cd sam3
# uv pip install -e .
# uv pip install decord psutil

echo "=== Environment setup complete! ==="
