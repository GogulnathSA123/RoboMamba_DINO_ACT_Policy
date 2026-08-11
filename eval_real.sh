#!/bin/bash
# Simple Real SO-101 Robot Arm Evaluation Script

PYTHON="/home/cmvmvl/anaconda3/envs/lerobot-robomamba/bin/python"

# Command 1: Run policy on physical SO-101 robot arm
run_real_robot() {
    echo "🤖 Starting evaluation on real SO-101 robot arm..."
    $PYTHON eval_real_so101.py \
        --checkpoint checkpoints/best_model.pt \
        --robot_port /dev/ttyACM0 \
        --cam_right 0 \
        --cam_wrist 2 \
        --hz 30
}

# Command 2: Download policy from Hugging Face and run on real SO-101 robot arm
run_huggingface_model() {
    echo "🤗 Downloading policy from Hugging Face and starting evaluation..."
    $PYTHON eval_real_so101.py \
        --checkpoint Gogul99/RoboMamba_DINO_ACT_Policy_Real \
        --robot_port /dev/ttyACM0 \
        --cam_right 0 \
        --cam_wrist 2 \
        --hz 30
}

# Command 3: Test policy in simulation / mock mode (no physical robot needed)
run_mock_test() {
    echo "🧪 Starting mock simulation test..."
    $PYTHON eval_real_so101.py \
        --checkpoint checkpoints/best_model.pt \
        --mock \
        --hz 30
}

# Parse command line argument
if [ "$1" == "real" ]; then
    run_real_robot
elif [ "$1" == "hf" ]; then
    run_huggingface_model
elif [ "$1" == "mock" ]; then
    run_mock_test
else
    echo "Usage:"
    echo "  ./eval_real.sh real   -> Test on physical SO-101 robot arm"
    echo "  ./eval_real.sh hf     -> Test using Hugging Face model"
    echo "  ./eval_real.sh mock   -> Test simulation without hardware"
fi
