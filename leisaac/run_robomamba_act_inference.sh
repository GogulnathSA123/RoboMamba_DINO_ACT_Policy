#!/bin/bash

# Dedicated script to run RoboMamba-ACT Policy (In-Process Execution with Temporal Ensembling) in Isaac Sim

echo "=== Running LeIsaac evaluation with RoboMamba-ACT Policy ==="

export PYTHONPATH=/home/zozo/lerobot/src:/home/zozo/leisaac/source/leisaac:/home/zozo/RoboMamba_ACT_Policy:$PYTHONPATH

PYTHON_BIN="/home/zozo/anaconda3/envs/issac_sim_env/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

# Resolve cuDNN library path
CUDNN_PATH=$($PYTHON_BIN -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null)
if [ -n "$CUDNN_PATH" ]; then
    export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
fi

cd /home/zozo/leisaac

$PYTHON_BIN /home/zozo/leisaac/scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-CustomTask-v0 \
    --policy_type robomamba-act \
    --policy_checkpoint_path Gogul99/Robomamba_act \
    --policy_action_horizon 8 \
    --policy_language_instruction "pick up the red cube and place it into the box." \
    --step_hz 30 \
    --device cuda \
    --enable_cameras "$@"

