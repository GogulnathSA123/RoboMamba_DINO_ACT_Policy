#!/usr/bin/env python3
"""Quick diagnostic: checks joint readings, camera feeds, and model predictions."""
import sys
import os
import time
import numpy as np
import torch
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--robot_port", type=str, default="/dev/ttyACM0")
parser.add_argument("--cam_right", type=int, default=0)
parser.add_argument("--cam_wrist", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="Gogul99/RoboMamba_DINO_ACT_Policy_Real")
args = parser.parse_args()

print("=" * 70)
print("  RoboMamba DINO ACT — Pre-Flight Diagnostic")
print("=" * 70)

# ─── 1. Robot Joint Check ─────────────────────────────────────────────
print("\n[1/4] ROBOT JOINTS")
try:
    try:
        from lerobot.robots.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    except ImportError:
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

    config = SO101FollowerConfig(port=args.robot_port, use_degrees=True)
    robot = SO101Follower(config)
    robot.connect()

    obs = robot.get_observation()
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    raw_state = np.array([obs[f"{j}.pos"] for j in joint_names], dtype=np.float32)

    # Training distribution stats from checkpoint
    train_mean = np.array([-16.9144, -45.6824, 59.8029, -12.9888, 2.9692, 8.9225])
    train_std  = np.array([10.0847, 44.5422, 35.0008, 11.1685, 7.1997, 10.5839])

    print(f"  {'Joint':<16} {'Current':>10} {'Train Mean':>11} {'Train Std':>10} {'Norm (σ)':>9}  Status")
    print(f"  {'─'*16} {'─'*10} {'─'*11} {'─'*10} {'─'*9}  {'─'*10}")
    any_ood = False
    for i, name in enumerate(joint_names):
        norm_val = (raw_state[i] - train_mean[i]) / train_std[i]
        if abs(norm_val) > 3.0:
            status = "OOD!"
            any_ood = True
        elif abs(norm_val) > 2.0:
            status = "Borderline"
        else:
            status = "OK"
        print(f"  {name:<16} {raw_state[i]:>+10.2f}° {train_mean[i]:>+11.2f}° {train_std[i]:>10.2f}° {norm_val:>+9.2f}σ  {status}")

    if any_ood:
        print("\n  WARNING: One or more joints are far outside training distribution!")
        print("  → The model will output frozen/constant predictions (action_mean).")
        print("  → Fix: Recalibrate motors with: python -m lerobot.calibrate --robot.type=so101_follower")
    else:
        print("\n  All joints within training distribution. Joint state looks good.\n")

    robot.disconnect()
except Exception as e:
    print(f" Robot connection failed: {e}")
    raw_state = None

# ─── 2. Camera Check ──────────────────────────────────────────────────
print("\n[2/4] CAMERA STREAMS")
for cam_name, cam_idx in [("right", args.cam_right), ("wrist", args.cam_wrist)]:
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f" Camera '{cam_name}' (index {cam_idx}): FAILED TO OPEN")
        continue

    frames = []
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame.mean())
        time.sleep(0.05)
    cap.release()

    if len(frames) < 2:
        print(f" Camera '{cam_name}' (index {cam_idx}): Only got {len(frames)} frames")
    else:
        frame_var = np.var(frames)
        if frame_var < 0.01:
            print(f" Camera '{cam_name}' (index {cam_idx}): FROZEN — frame variance = {frame_var:.6f}")
        else:
            print(f" Camera '{cam_name}' (index {cam_idx}): LIVE — frame variance = {frame_var:.4f} | mean brightness = {np.mean(frames):.1f}")

# ─── 3. Model Prediction Test ─────────────────────────────────────────
print("\n[3/4] MODEL PREDICTION TEST")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_leisaac import DINOv2PolicyConnector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    connector = DINOv2PolicyConnector(checkpoint_path=args.checkpoint, device=device)

    # Generate two slightly different fake inputs to test if model responds to change
    dummy_img1 = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
    dummy_img2 = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)

    if raw_state is not None:
        test_state = raw_state
    else:
        test_state = train_mean.copy()  # Use training mean as fallback

    connector.reset()
    action1 = connector.get_action(0, [dummy_img1, dummy_img1], test_state)
    action2 = connector.get_action(1, [dummy_img2, dummy_img2], test_state)

    delta = np.abs(action1 - action2).max()
    print(f"\n  Prediction 1: {np.round(action1, 2)}")
    print(f"  Prediction 2: {np.round(action2, 2)}")
    print(f"  Max delta:    {delta:.6f}")

    if delta < 0.01:
        print(f" Model output is FROZEN — predictions identical despite different images")
        print(f"  → Root cause: state input is out-of-distribution or model weights issue")
    else:
        print(f" Model responds to input changes — predictions vary")

    # Test with perfect training-mean state
    print(f"\n  [Test with training-mean state]:")
    connector.reset()
    action_mean_test = connector.get_action(0, [dummy_img1, dummy_img1], train_mean.astype(np.float32))

    action_mean_ref = connector.policy.action_mean.cpu().numpy()
    distance_from_mean = np.abs(action_mean_test - action_mean_ref).max()
    print(f"  Prediction:     {np.round(action_mean_test, 2)}")
    print(f"  Action mean:    {np.round(action_mean_ref, 2)}")
    print(f"  Distance:       {distance_from_mean:.4f}")
    if distance_from_mean < 1.0:
        print(f" Model always outputs ~action_mean regardless of input → possible undertrained model")
    else:
        print(f" Model output differs from action_mean → model has learned")

except Exception as e:
    print(f" Model test failed: {e}")
    import traceback
    traceback.print_exc()

# ─── 4. Observation Key Check ─────────────────────────────────────────
print("\n[4/4] OBSERVATION KEY FORMAT")
try:
    config = SO101FollowerConfig(port=args.robot_port, use_degrees=True)
    robot = SO101Follower(config)
    robot.connect()
    obs = robot.get_observation()
    print(f"  Available keys: {sorted(obs.keys())}")
    for key in sorted(obs.keys()):
        val = obs[key]
        if isinstance(val, (int, float, np.floating)):
            print(f"    {key}: {float(val):.4f}")
    robot.disconnect()
except Exception as e:
    print(f" Could not read observation keys: {e}")

print("\n" + "=" * 70)
print("  Diagnostic Complete")
print("=" * 70)
