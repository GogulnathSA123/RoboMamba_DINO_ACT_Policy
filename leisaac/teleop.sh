#!/bin/bash
conda run --no-capture-output -n issac_sim_env python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CustomTask-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/dataset_act_dino_real.hdf5 \
    --lerobot_dataset_fps 30 \
    --recalibrate
