#!/bin/bash

# Real SO-101 Robot Evaluation Script for ACT Policy
# Automatically downloads Hugging Face weights and starts control loop

echo " Starting SO-101 Policy Evaluation (ACT Realworld)..."

export PATH="/home/zozo/anaconda3/envs/lerobot_libero/bin:$PATH"
PYTHON="/home/zozo/anaconda3/envs/lerobot_libero/bin/python"

$PYTHON -s eval_real_so101.py \
    --checkpoint "Gogul99/act_realworld" \
    --robot_port "/dev/ttyACM0" \
    --cam_right 4 \
    --cam_wrist 6 \
    --hz 30 \
    --max_steps 5000 \
    --viz
