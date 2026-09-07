#!/bin/bash
# One-click calibration script for SO-101 Follower Arm

PYTHON="/home/zozo/anaconda3/envs/lerobot_libero/bin/python"
CALIB_SCRIPT="/home/zozo/lerobot/src/lerobot/scripts/lerobot_calibrate.py"


echo "  Starting SO-101 Follower Arm Calibration"


$PYTHON $CALIB_SCRIPT \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm
