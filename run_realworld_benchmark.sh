#!/bin/bash

#  REAL-WORLD SO-101 ROBOT ARM BENCHMARK & INFERENCE PROFILING RUNNER

# Usage:
#   bash run_realworld_benchmark.sh robomamba 10     # Run RoboMamba on real arm (10 trials)
#   bash run_realworld_benchmark.sh act 10           # Run Canonical ACT on real arm (10 trials)
#   bash run_realworld_benchmark.sh robomamba 3 mock # Dry-run mock test without hardware


POLICY=${1:-"robomamba"}
TRIALS=${2:-20}
MODE=${3:-"real"}
ROBOT_PORT="/dev/ttyACM0"
CAM_RIGHT=4
CAM_WRIST=6
HZ=30

PYTHON="/home/zozo/anaconda3/envs/lerobot_libero/bin/python"

if [ ! -f "$PYTHON" ]; then
    PYTHON=$(which python3)
fi


echo "  LAUNCHING REAL-WORLD BENCHMARK: Policy = ${POLICY^^} | Trials = ${TRIALS} | Mode = ${MODE^^}"


EXTRA_ARGS=""
if [ "$MODE" == "mock" ]; then
    EXTRA_ARGS="--mock --no_interactive"
fi

$PYTHON -s /home/zozo/RoboMamba_DINO_ACT_Policy/evaluate_realworld_benchmark.py \
    --policy "$POLICY" \
    --trials "$TRIALS" \
    --robot_port "$ROBOT_PORT" \
    --cam_right "$CAM_RIGHT" \
    --cam_wrist "$CAM_WRIST" \
    --hz "$HZ" \
    $EXTRA_ARGS
