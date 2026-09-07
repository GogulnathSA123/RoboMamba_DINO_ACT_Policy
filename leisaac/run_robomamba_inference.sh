#!/bin/bash

# Script to run Standard RoboMamba VLA Policy (SigLIP ViT + Mamba LLM + Linear Head) in LeIsaac Sim

echo "=== Running LeIsaac evaluation with Standard RoboMamba VLA Policy ==="

export PYTHONPATH=/home/zozo/lerobot/src:/home/zozo/leisaac/source/leisaac:$PYTHONPATH

# Resolve cuDNN library path to prevent libcudnn_graph missing errors
CUDNN_PATH=$(python3 -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
fi

python scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-CustomTask-v0 \
    --policy_type lerobot-robomamba \
    --policy_checkpoint_path Gogul99/siglip_mamba \
    --policy_host localhost \
    --policy_port 8080 \
    --policy_action_horizon 1 \
    --policy_language_instruction "pick up the object and place it in the target location" \
    --step_hz 30 \
    --policy_timeout_ms 50000 \
    --device cuda \
    --enable_cameras
