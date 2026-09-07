#!/bin/bash

# Run RoboMamba-DINO-ACT Policy in Isaac Sim (LeIsaac)
#
# Episode termination: the environment signals SUCCESS (reset_terminated=True)
# when the red cube is placed inside the grey box, as defined by the
# cube_in_box termination condition in custom_task_env_cfg.py.
#
# Usage:
#   bash run_robomamba_dino_act.sh                  # single episode, no timeout
#   bash run_robomamba_dino_act.sh --eval_rounds 10 # run 10 evaluation rounds
#   bash run_robomamba_dino_act.sh --headless        # headless mode

echo "=== Running LeIsaac evaluation with RoboMamba-DINO-ACT Policy ==="
echo "    Termination: episode ends when the red cube is placed inside the grey box."

export PYTHONPATH=/home/zozo/lerobot/src:/home/zozo/leisaac/source/leisaac:/home/zozo/RoboMamba_DINO_ACT_Policy:$PYTHONPATH

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
    --policy_type robomamba-dino \
    --policy_checkpoint_path Gogul99/RoboMamba_DINO_ACT_Policy \
    --policy_action_horizon 16 \
    --policy_language_instruction "pick up the red cube and place it into the box." \
    --step_hz 30 \
    --device cuda \
    --enable_cameras \
    "$@"
