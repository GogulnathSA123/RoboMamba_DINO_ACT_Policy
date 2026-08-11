#!/usr/bin/env python3
"""
Real SO-101 Robot Arm Evaluation Script for DINOv2 + RoboMamba + ACT Policy.

Usage:
    # Run in mock/test mode (without physical hardware connected):
    python eval_real_so101.py --checkpoint Gogul99/RoboMamba_DINO_ACT_Policy_Real --mock

    # Run on real SO-101 robot arm with live cameras:
    python eval_real_so101.py --checkpoint checkpoints/best_model.pt --robot_port /dev/ttyACM0 --cam_right 0 --cam_wrist 2
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from PIL import Image

from eval_leisaac import DINOv2PolicyConnector

def capture_camera_frame(cam_source, camera_name="camera"):
    """Captures a single RGB image frame from OpenCV camera index or video device."""
    try:
        import cv2
        cap = cv2.VideoCapture(cam_source)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
    except Exception:
        pass
    
    # Fallback synthetic frame if camera fails or cv2 unavailable
    synthetic = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return synthetic


class MockSO101Robot:
    """Mock interface for SO-101 Robot Arm testing without physical hardware."""

    def __init__(self, state_dim=6):
        self.state_dim = state_dim
        self.current_state = np.zeros(state_dim, dtype=np.float32)

    def read_joint_positions(self):
        # Simulate slight sensor noise around current position
        return self.current_state + np.random.normal(0, 0.001, size=self.state_dim)

    def send_action(self, action_vector):
        self.current_state = np.array(action_vector, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DINOv2 + RoboMamba + ACT Policy on Real SO-101 Robot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to local checkpoint file or Hugging Face repository ID (e.g., 'Gogul99/RoboMamba_DINO_ACT_Policy_Real')",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 robot arm controller (e.g., /dev/ttyACM0 or /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--cam_right",
        type=int,
        default=0,
        help="Camera index for right / side camera view (observation.images.right)",
    )
    parser.add_argument(
        "--cam_wrist",
        type=int,
        default=2,
        help="Camera index for wrist camera view (observation.images.wrist)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Control loop target execution frequency (Hz)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum evaluation timesteps per trial",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock simulation mode without requiring connected physical hardware",
    )
    args = parser.parse_args()

    print("==========================================================")
    print(" 🤖 Starting Real SO-101 DINOv2 + RoboMamba + ACT Evaluation")
    print("==========================================================")
    print(f" -> Checkpoint Source : {args.checkpoint}")
    print(f" -> Control Frequency : {args.hz} Hz")
    print(f" -> Hardware Mode     : {'MOCK / SIMULATED' if args.mock else f'REAL ROBOT ({args.robot_port})'}")
    print(f" -> Right Camera Index: {args.cam_right}")
    print(f" -> Wrist Camera Index: {args.cam_wrist}")
    print("----------------------------------------------------------")

    # 1. Instantiate Policy Connector
    connector = DINOv2PolicyConnector(
        checkpoint_path=args.checkpoint,
        cfg_path=args.cfg_path,
    )
    connector.reset()

    # 2. Instantiate Robot Arm Controller
    if args.mock:
        robot = MockSO101Robot(state_dim=connector.cfg["model"]["state_dim"])
        print("✅ Initialized Mock SO-101 Robot Arm.")
    else:
        try:
            from lerobot.robots.so101 import SO101Robot
            robot = SO101Robot(port=args.robot_port)
            robot.connect()
            print(f"✅ Successfully connected to real SO-101 Robot Arm on '{args.robot_port}'.")
        except Exception as e:
            print(f"⚠️ Could not connect to real SO-101 hardware via lerobot: {e}")
            print(" -> Falling back to Mock Robot mode for safety.")
            robot = MockSO101Robot(state_dim=connector.cfg["model"]["state_dim"])

    step_dt = 1.0 / args.hz
    print("\n🚀 Starting Real-Time Policy Execution Loop... Press Ctrl+C to stop.\n")

    try:
        for t in range(args.max_steps):
            t_start = time.time()

            # A. Capture camera frames
            right_frame = capture_camera_frame(args.cam_right, "right")
            wrist_frame = capture_camera_frame(args.cam_wrist, "wrist")
            raw_images = [right_frame, wrist_frame]

            # B. Read current joint states [6-dim]
            if hasattr(robot, "read_joint_positions"):
                raw_state = robot.read_joint_positions()
            elif hasattr(robot, "get_state"):
                raw_state = robot.get_state()
            else:
                raw_state = np.zeros(6, dtype=np.float32)

            # C. Compute smooth action vector via DINOv2 + Mamba + ACT + Temporal Ensembling
            action = connector.get_action(current_t=t, raw_images=raw_images, raw_state=raw_state)

            # D. Send action to SO-101 robot arm
            if hasattr(robot, "send_action"):
                robot.send_action(action)
            elif hasattr(robot, "step"):
                robot.step(action)

            t_elapsed = time.time() - t_start
            sleep_time = max(0.0, step_dt - t_elapsed)
            time.sleep(sleep_time)

            if t % 10 == 0 or t == args.max_steps - 1:
                fps = 1.0 / max(1e-5, t_elapsed + sleep_time)
                print(
                    f"Step {t:3d}/{args.max_steps} | Loop Rate: {fps:5.1f} Hz | "
                    f"Action: [{', '.join(f'{a:+.3f}' for a in action)}]"
                )

        print("\n✅ Evaluation trial finished successfully.")

    except KeyboardInterrupt:
        print("\n🛑 Stopped real-world evaluation safely via Ctrl+C.")
    finally:
        if not args.mock and hasattr(robot, "disconnect"):
            robot.disconnect()
            print("🔌 Disconnected SO-101 Robot Arm hardware safely.")

if __name__ == "__main__":
    main()
