#!/bin/bash
# Real SO-101 Robot Arm Evaluation Script for DINOv2 + RoboMamba + ACT Policy

set -e

# Path to python binary in lerobot-robomamba environment
PYTHON_BIN="/home/cmvmvl/anaconda3/envs/lerobot-robomamba/bin/python"
LOCAL_CHECKPOINT="checkpoints/best_model.pt"
HF_CHECKPOINT="Gogul99/RoboMamba_DINO_ACT_Policy_Real"

print_header() {
    echo "=========================================================="
    echo " 🤖 RoboMamba + DINOv2 + ACT Policy: Real SO-101 Evaluator"
    echo "=========================================================="
}

list_serial_ports() {
    echo "🔌 Scanning for connected robot serial ports..."
    ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "⚠️ No /dev/ttyACM* or /dev/ttyUSB* devices found."
}

list_cameras() {
    echo "📷 Scanning for connected camera devices..."
    ls -l /dev/video* 2>/dev/null || echo "⚠️ No /dev/video* devices found."
}

run_real() {
    print_header
    PORT=${1:-"/dev/ttyACM0"}
    RIGHT_CAM=${2:-0}
    WRIST_CAM=${3:-2}
    
    CHECKPOINT="$LOCAL_CHECKPOINT"
    if [ ! -f "$LOCAL_CHECKPOINT" ]; then
        echo "💡 Local checkpoint not found. Fetching from Hugging Face Hub ($HF_CHECKPOINT)..."
        CHECKPOINT="$HF_CHECKPOINT"
    fi

    echo "🚀 Starting Real SO-101 Evaluation on Port: $PORT | Right Cam: $RIGHT_CAM | Wrist Cam: $WRIST_CAM"
    $PYTHON_BIN eval_real_so101.py \
        --checkpoint "$CHECKPOINT" \
        --robot_port "$PORT" \
        --cam_right "$RIGHT_CAM" \
        --cam_wrist "$WRIST_CAM" \
        --hz 30
}

run_hf() {
    print_header
    PORT=${1:-"/dev/ttyACM0"}
    echo "🤗 Loading policy directly from Hugging Face Hub: $HF_CHECKPOINT"
    $PYTHON_BIN eval_real_so101.py \
        --checkpoint "$HF_CHECKPOINT" \
        --robot_port "$PORT" \
        --cam_right 0 \
        --cam_wrist 2 \
        --hz 30
}

run_mock() {
    print_header
    CHECKPOINT="$LOCAL_CHECKPOINT"
    if [ ! -f "$LOCAL_CHECKPOINT" ]; then
        CHECKPOINT="$HF_CHECKPOINT"
    fi
    echo "🧪 Running in Mock / Simulation Mode (No physical hardware required)..."
    $PYTHON_BIN eval_real_so101.py \
        --checkpoint "$CHECKPOINT" \
        --mock \
        --max_steps 300 \
        --hz 30
}

show_menu() {
    print_header
    echo "Select an evaluation command to run:"
    echo "  1) Run on Real SO-101 Hardware (/dev/ttyACM0)"
    echo "  2) Download & Run Hugging Face Model (Gogul99/RoboMamba_DINO_ACT_Policy_Real)"
    echo "  3) Run Mock / Simulation Mode (Test policy pipeline without hardware)"
    echo "  4) Scan Connected USB Cameras (/dev/video*)"
    echo "  5) Scan Connected Serial Ports (/dev/ttyACM* /dev/ttyUSB*)"
    echo "  6) Exit"
    echo "----------------------------------------------------------"
    read -p "Enter choice [1-6]: " choice

    case $choice in
        1) run_real ;;
        2) run_hf ;;
        3) run_mock ;;
        4) list_cameras ;;
        5) list_serial_ports ;;
        6) echo "Exiting."; exit 0 ;;
        *) echo "Invalid option."; exit 1 ;;
    esac
}

# Process command-line argument if passed directly
case "$1" in
    real)
        run_real "$2" "$3" "$4"
        ;;
    hf)
        run_hf "$2"
        ;;
    mock)
        run_mock
        ;;
    ports)
        list_serial_ports
        ;;
    cameras)
        list_cameras
        ;;
    "")
        show_menu
        ;;
    *)
        # Pass all unknown arguments directly to eval_real_so101.py
        $PYTHON_BIN eval_real_so101.py "$@"
        ;;
esac
