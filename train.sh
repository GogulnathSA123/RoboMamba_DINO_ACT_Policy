#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup Conda environment
CONDA_SH="/home/zozo/anaconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate lerobot
    echo "Activated conda environment: lerobot"
else
    echo "Warning: Conda setup script not found at $CONDA_SH. Using system python."
fi

# Add current workspace to PYTHONPATH
export PYTHONPATH="/home/zozo/RoboMamba_DINO_ACT_Policy:$PYTHONPATH"

# Run the training script
echo "Starting training..."
python train.py
