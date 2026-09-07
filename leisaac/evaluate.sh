#!/bin/bash
export PYTHONPATH=/home/zozo/leisaac/source/leisaac:/home/zozo/lerobot/src:$PYTHONPATH

PYTHON_BIN="/home/zozo/anaconda3/envs/issac_sim_env/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

cd /home/zozo/leisaac

$PYTHON_BIN /home/zozo/leisaac/scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-CustomTask-v0 \
    --policy_type lerobot-robomamba_act \
    --policy_checkpoint_path Gogul99/Robomamba_act \
    --policy_host localhost \
    --policy_port 5555 \
    --policy_action_horizon 8 \
    --step_hz 30 \
    --device cuda \
    --enable_cameras "$@"
