#!/bin/bash
# Real SO-101 Robot Arm Evaluation Script for DINOv2 + RoboMamba + ACT Policy

set -e

# Path to python binary in lerobot-robomamba environment
PYTHON_BIN="/home/cmvmvl/anaconda3/envs/lerobot-robomamba/bin/python"

# Default configuration
CHECKPOINT="Gogul99/RoboMamba_DINO_ACT_Policy_Real"
ROBOT_PORT="/dev/ttyACM0"
CAM_RIGHT=0
CAM_WRIST=2
HZ=30
FLAGS=""

echo "=========================================================="
echo " 🤖 RoboMamba + DINOv2 + ACT Policy: Real SO-101 Evaluator"
echo "=========================================================="

# Check if checkpoint exists locally in checkpoints/best_model.pt
if [ -f "checkpoints/best_model.pt" ]; then
    echo "💡 Local checkpoint detected at 'checkpoints/best_model.pt'."
    CHECKPOINT="checkpoints/best_model.pt"
fi

# Run evaluation script with user passed arguments or defaults
echo "Executing: $PYTHON_BIN eval_real_so101.py --checkpoint '$CHECKPOINT' --robot_port '$ROBOT_PORT' --cam_right $CAM_RIGHT --cam_wrist $CAM_WRIST --hz $HZ $@"
echo "----------------------------------------------------------"

$PYTHON_BIN eval_real_so101.py \
    --checkpoint "$CHECKPOINT" \
    --robot_port "$ROBOT_PORT" \
    --cam_right $CAM_RIGHT \
    --cam_wrist $CAM_WRIST \
    --hz $HZ \
    "$@"
