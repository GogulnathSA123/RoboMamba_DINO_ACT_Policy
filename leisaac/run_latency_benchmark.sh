#!/bin/bash


# Latency & Performance Benchmark for RoboMamba-DINO-ACT and ACT

# Usage:
#   bash run_latency_benchmark.sh --policy act --trials 50 --seed 42 --headless
#   bash run_latency_benchmark.sh --policy robomamba-dino --trials 50 --seed 42 --headless
#   bash run_latency_benchmark.sh --policy all --trials 50 --seed 42 --headless


export ISAAC_PATH=/home/zozo/Downloads/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release
export CARB_APP_PATH=$ISAAC_PATH/kit
export EXP_PATH=$ISAAC_PATH/apps
export PYTHONPATH=$ISAAC_PATH/extscache/omni.usd.metrics.assembler-106.5.0+106.5.0.lx64.r.cp310:$ISAAC_PATH/extscache/omni.metrics.core-0.0.1+d02c707b.lx64.r:/home/zozo/IsaacLab/source/isaaclab:/home/zozo/IsaacLab/source/isaaclab_tasks:/home/zozo/IsaacLab/source/isaaclab_assets:/home/zozo/leisaac/source/leisaac:/home/zozo/lerobot/src:/home/zozo/RoboMamba_DINO_ACT_Policy:$PYTHONPATH

PYTHON_BIN="$ISAAC_PATH/python.sh"

cd /home/zozo/leisaac

# Default arguments
POLICY="act"
TRIALS=50
SEED=42
HEADLESS=1
EXTRA_ARGS="--unthrottled"

while [[ $# -gt 0 ]]; do
  case $1 in
    --policy)
      POLICY="$2"
      shift 2
      ;;
    --trials)
      TRIALS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --gui)
      HEADLESS=0
      shift
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      shift
      ;;
  esac
done

HEADLESS_FLAG=""
if [ "$HEADLESS" -eq 1 ]; then
    HEADLESS_FLAG="--headless"
    echo "[Mode] Running in HEADLESS mode"
else
    echo "[Mode] Running in GUI mode"
fi

if [ "$POLICY" == "all" ] || [ "$POLICY" == "both" ]; then
    
    echo "  RUNNING BENCHMARK FOR BOTH ROBOMAMBA-DINO-ACT & ACT (Seed $SEED)"
    
    
    echo -e "\n>>> [1/2] Benchmarking ACT Policy ($TRIALS trials, Seed $SEED) <<<"
    $PYTHON_BIN /home/zozo/leisaac/scripts/evaluation/evaluate_latency_benchmark.py \
        --task LeIsaac-SO101-CustomTask-v0 \
        --policy_type act \
        --policy_checkpoint_path Gogul99/act_isaacsim \
        --eval_rounds $TRIALS \
        --seed $SEED \
        --enable_cameras \
        $HEADLESS_FLAG \
        $EXTRA_ARGS

    echo -e "\n>>> [2/2] Benchmarking RoboMamba-DINO-ACT Policy ($TRIALS trials, Seed $SEED) <<<"
    $PYTHON_BIN /home/zozo/leisaac/scripts/evaluation/evaluate_latency_benchmark.py \
        --task LeIsaac-SO101-CustomTask-v0 \
        --policy_type robomamba-dino \
        --policy_checkpoint_path Gogul99/RoboMamba_DINO_ACT_Policy \
        --eval_rounds $TRIALS \
        --seed $SEED \
        --enable_cameras \
        $HEADLESS_FLAG \
        $EXTRA_ARGS

else
    
    echo "  RUNNING BENCHMARK: Policy = $POLICY | Trials = $TRIALS | Seed = $SEED"
    
    
    if [ "$POLICY" == "act" ] || [ "$POLICY" == "act_isaacsim" ] || [ "$POLICY" == "lerobot-act" ]; then
        CKPT="Gogul99/act_isaacsim"
        PTYPE="act"
    else
        CKPT="Gogul99/RoboMamba_DINO_ACT_Policy"
        PTYPE="robomamba-dino"
    fi

    $PYTHON_BIN /home/zozo/leisaac/scripts/evaluation/evaluate_latency_benchmark.py \
        --task LeIsaac-SO101-CustomTask-v0 \
        --policy_type $PTYPE \
        --policy_checkpoint_path $CKPT \
        --eval_rounds $TRIALS \
        --seed $SEED \
        --enable_cameras \
        $HEADLESS_FLAG \
        $EXTRA_ARGS
fi
