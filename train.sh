#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate Conda setup script dynamically
CONDA_SH=""
for path in \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/home/zozo/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$path" ]; then
        CONDA_SH="$path"
        break
    fi
done

if [ -z "$CONDA_SH" ] && command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
        CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
    fi
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate lerobot-robomamba 2>/dev/null || conda activate lerobot 2>/dev/null || echo "Info: Using current python environment."
    echo "Activated conda environment setup from $CONDA_SH"
else
    echo "Info: Conda setup script not found. Using system/current python."
fi

# Add workspace directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the training script
echo "Starting training..."
cd "$SCRIPT_DIR"
python train.py
