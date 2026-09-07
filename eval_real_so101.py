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

# Patch importlib.metadata to prevent transformers version check ValueError on numpy
import importlib.metadata
_orig_meta_version = importlib.metadata.version
def _safe_meta_version(pkg):
    try:
        v = _orig_meta_version(pkg)
        if v is not None:
            return v
    except Exception:
        pass
    if pkg == "numpy":
        return getattr(np, "__version__", "2.2.6")
    return "1.0.0"
importlib.metadata.version = _safe_meta_version

import torch
from PIL import Image

from eval_leisaac import DINOv2PolicyConnector

class ThreadedCamera:
    """Robust non-blocking background thread camera reader with MJPEG & dual-thread watchdog."""
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.frame_rgb = None
        self.running = False
        import threading
        self.lock = threading.Lock()
        self.cap = None
        self.last_frame_time = time.time()
        self.reader_thread = None
        self.watchdog_thread = None

    def _open(self):
        import cv2
        cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 300)
            except Exception:
                pass
        return cap

    def start(self):
        self.running = True
        self.last_frame_time = time.time()
        import threading
        self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.reader_thread.start()
        self.watchdog_thread.start()
        return self

    def _read_loop(self):
        import cv2
        self.cap = self._open()
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frame_rgb = rgb
                    self.last_frame_time = time.time()
            time.sleep(0.001)

    def _watchdog_loop(self):
        while self.running:
            time.sleep(0.3)
            # If no frame received for > 0.8s, reboot the VideoCapture object
            if time.time() - self.last_frame_time > 0.8:
                old_cap = self.cap
                self.cap = None
                if old_cap is not None:
                    try:
                        old_cap.release()
                    except Exception:
                        pass
                time.sleep(0.05)
                self.cap = self._open()
                self.last_frame_time = time.time()

    def read_rgb(self):
        with self.lock:
            if self.frame_rgb is not None:
                return self.frame_rgb.copy()
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def stop(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass


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


import queue
import threading

class AsyncInferenceThread:
    """Runs GPU DINOv2 + Mamba + ACT policy inference asynchronously in a background thread."""

    def __init__(self, connector):
        self.connector = connector
        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def predict_async(self, current_t, raw_images, raw_state):
        # Keep queue fresh: drop stale request if worker hasn't picked it up yet
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.queue.put_nowait((current_t, raw_images, raw_state))
        except queue.Full:
            pass

    def _worker(self):
        while self.running:
            try:
                current_t, raw_images, raw_state = self.queue.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                self.connector.get_action(current_t, raw_images, raw_state)
            except Exception as e:
                pass

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


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
        default=4,
        help="Camera index for right / side camera view (observation.images.right)",
    )
    parser.add_argument(
        "--cam_wrist",
        type=int,
        default=6,
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
    parser.add_argument(
        "--no_auto_align",
        action="store_true",
        help="Disable automatic starting pose calibration alignment",
    )
    parser.add_argument(
        "--async_mode",
        action="store_true",
        help="Run GPU inference in background thread",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable live Rerun visualizer",
    )
    args = parser.parse_args()

    
    print(f"  Starting Real SO-101 DINOv2 + RoboMamba + ACT Evaluation ({'ASYNC' if args.async_mode else 'SYNCHRONOUS'} Mode)")
    print("==========================================================")
    print(f" -> Checkpoint Source : {args.checkpoint}")
    print(f" -> Control Frequency : {args.hz} Hz")
    print(f" -> Hardware Mode     : {'MOCK / SIMULATED' if args.mock else f'REAL ROBOT ({args.robot_port})'}")
    print(f" -> Right Camera Index: {args.cam_right}")
    print(f" -> Wrist Camera Index: {args.cam_wrist}")
    print(f" -> Rerun Visualizer  : {'ENABLED' if args.viz else 'DISABLED'}")
    print("----------------------------------------------------------")

    # Initialize Rerun if enabled
    if args.viz:
        try:
            from lerobot.utils.rerun_visualization import init_rerun
            init_rerun("SO-101 Policy Evaluation")
            print(" Rerun Visualizer initialized.")
        except Exception as e:
            print(f" Could not initialize Rerun: {e}")
            args.viz = False

    # 1. Instantiate Policy (Auto-detects native LeRobot ACT vs DINOv2RoboMambaACT)
    is_lerobot_act = False
    try:
        from huggingface_hub import list_repo_files
        if "/" in args.checkpoint and "config.json" in list_repo_files(args.checkpoint):
            is_lerobot_act = True
    except Exception:
        pass

    if is_lerobot_act:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        model_dir = snapshot_download(args.checkpoint)
        print(f" Loading native LeRobot ACTPolicy from '{args.checkpoint}'...")
        policy = ACTPolicy.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
        policy.eval()
        policy.reset()
        connector = None
        state_dim = 6

        # Load normalization buffers for state and action
        norm_file = os.path.join(model_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
        unnorm_file = os.path.join(model_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
        
        act_state_mean = np.array([-16.914392, -45.682438, 59.802876, -12.988785, 2.9692476, 8.922545], dtype=np.float32)
        act_state_std = np.array([10.084669, 44.542244, 35.000786, 11.168477, 7.1997147, 10.583875], dtype=np.float32)
        act_action_mean = np.array([-16.752283, -46.333096, 58.62386, -8.483678, 2.9086468, 7.0913405], dtype=np.float32)
        act_action_std = np.array([10.101624, 44.15465, 35.91024, 22.443579, 7.298939, 11.057688], dtype=np.float32)

        if os.path.exists(norm_file):
            try:
                norm_data = load_file(norm_file)
                if "observation.state.mean" in norm_data:
                    act_state_mean = norm_data["observation.state.mean"].numpy().flatten()
                    act_state_std = norm_data["observation.state.std"].numpy().flatten()
            except Exception as e:
                print(f" Could not load normalizer stats: {e}")

        if os.path.exists(unnorm_file):
            try:
                unnorm_data = load_file(unnorm_file)
                if "action.mean" in unnorm_data:
                    act_action_mean = unnorm_data["action.mean"].numpy().flatten()
                    act_action_std = unnorm_data["action.std"].numpy().flatten()
            except Exception as e:
                print(f" Could not load unnormalizer stats: {e}")
        
        print(" Successfully loaded ACT state/action normalization statistics.")
    else:
        connector = DINOv2PolicyConnector(
            checkpoint_path=args.checkpoint,
            cfg_path=args.cfg_path,
        )
        connector.reset()
        policy = None
        state_dim = connector.cfg["model"]["state_dim"]

    # Optional Async Inference Thread (for DINOv2 connector)
    async_inference = None
    if args.async_mode and connector is not None:
        async_inference = AsyncInferenceThread(connector)
        async_inference.start()

    # 2. Instantiate Robot Arm Controller
    if args.mock:
        robot = MockSO101Robot(state_dim=state_dim)
        print(" Initialized Mock SO-101 Robot Arm.")
    else:
        try:
            try:
                from lerobot.robots.so_follower import SO101Follower
                from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
                config = SO101FollowerConfig(port=args.robot_port, id="my_follower_arm", use_degrees=True, num_read_retries=5)
                robot = SO101Follower(config)
            except ImportError:
                from lerobot.robots.so101 import SO101Robot
                robot = SO101Robot(port=args.robot_port)
            robot.connect(calibrate=False)
            print(f" Successfully connected to real SO-101 Robot Arm on '{args.robot_port}'.")
            
            # Explicitly enable motor torque and set holding limits
            if hasattr(robot, "bus") and robot.bus is not None:
                try:
                    robot.bus.enable_torque()
                except Exception:
                    pass
                for motor_name in robot.bus.motors:
                    try:
                        robot.bus.write("Torque_Enable", motor_name, 1)
                        robot.bus.write("Max_Torque_Limit", motor_name, 1000)
                        robot.bus.write("Lock", motor_name, 1)
                        robot.bus.write("P_Coefficient", motor_name, 24)
                        robot.bus.write("D_Coefficient", motor_name, 32)
                    except Exception as e:
                        print(f" Warning setting motor '{motor_name}' torque: {e}")
        except Exception as e:
            print(f" Could not connect to real SO-101 hardware via lerobot: {e}")
            print(" -> Falling back to Mock Robot mode for safety.")
            robot = MockSO101Robot(state_dim=state_dim)

    # 3. Pre-open cameras in background threads
    caps = {}
    if not args.mock:
        for cam_name, cam_idx in [("right", args.cam_right), ("wrist", args.cam_wrist)]:
            print(f" Starting threaded camera '{cam_name}' (index {cam_idx})...")
            cam = ThreadedCamera(cam_idx).start()
            caps[cam_name] = cam
        print(" Waiting 1.0s for camera streams to stabilize...")
        time.sleep(1.0)

    # Start policy immediately from the user's current physical starting posture (zero pre-movement)
    if not args.mock and hasattr(robot, "get_observation"):
        obs = robot.get_observation()
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        cur_pose = [f"{obs[f'{j}.pos']:+.1f}°" for j in joints if f"{j}.pos" in obs]
        print(f" Starting policy directly from current arm posture: [{', '.join(cur_pose)}]\n")

    step_dt = 1.0 / args.hz
    print("\n Starting Real-Time Policy Execution Loop... Press Ctrl+C to stop.\n")

    try:
        for t in range(args.max_steps):
            t_start = time.time()

            # A. Capture camera frames
            right_frame = caps["right"].read_rgb() if "right" in caps else np.zeros((480, 640, 3), dtype=np.uint8)
            wrist_frame = caps["wrist"].read_rgb() if "wrist" in caps else np.zeros((480, 640, 3), dtype=np.uint8)
            raw_images = [right_frame, wrist_frame]

            # B. Read current joint states [6-dim]
            if hasattr(robot, "read_joint_positions"):
                raw_state = robot.read_joint_positions()
            elif hasattr(robot, "get_state"):
                raw_state = robot.get_state()
            elif hasattr(robot, "get_observation"):
                obs = robot.get_observation()
                joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                raw_state = np.array([obs[f"{j}.pos"] for j in joints if f"{j}.pos" in obs], dtype=np.float32)
            else:
                raw_state = np.zeros(6, dtype=np.float32)
            if is_lerobot_act:
                def prep(img):
                    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                    return t.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
                
                # Normalize proprioceptive state
                norm_state = (raw_state - act_state_mean) / act_state_std
                
                batch = {
                    "observation.images.right": prep(right_frame),
                    "observation.images.wrist": prep(wrist_frame),
                    "observation.state": torch.from_numpy(norm_state).float().unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu"),
                }
                with torch.no_grad():
                    pred_norm_act = policy.select_action(batch).squeeze(0).cpu().numpy()
                
                # Unnormalize action back to physical joint degrees
                action_np = (pred_norm_act * act_action_std) + act_action_mean
            elif args.async_mode and async_inference is not None:
                async_inference.predict_async(t, raw_images, raw_state)
                action = connector.ensemble.get_action(t)
                if action is None or (isinstance(action, torch.Tensor) and torch.all(action == 0)):
                    if 'last_valid_action' in locals() and last_valid_action is not None:
                        action = last_valid_action
                    else:
                        action = raw_state
                else:
                    last_valid_action = action
                action_np = (action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)).copy()
            else:
                # Synchronous execution
                action = connector.get_action(current_t=t, raw_images=raw_images, raw_state=raw_state)
                action_np = (action.cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)).copy()

            # D. Send direct policy action to SO-101 robot arm
            if hasattr(robot, "send_action"):
                if isinstance(robot, MockSO101Robot):
                    robot.send_action(action_np)
                else:
                    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    action_dict = {f"{j}.pos": float(action_np[i]) for i, j in enumerate(joints)}
                    robot.send_action(action_dict)
            elif hasattr(robot, "step"):
                robot.step(action_np)

            # Log to Rerun visualizer if enabled
            if args.viz:
                try:
                    from lerobot.utils.rerun_visualization import log_rerun_data
                    obs_dict = {
                        "images/right": right_frame,
                        "images/wrist": wrist_frame,
                    }
                    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    obs_dict.update({f"joint_positions/{j}": float(raw_state[i]) for i, j in enumerate(joints)})
                    act_dict = {f"joint_targets/{j}": float(action_np[i]) for i, j in enumerate(joints)}
                    log_rerun_data(observation=obs_dict, action=act_dict)
                except Exception:
                    pass

            t_elapsed = time.time() - t_start
            sleep_time = max(0.0, step_dt - t_elapsed)
            time.sleep(sleep_time)

            if t % 10 == 0 or t == args.max_steps - 1:
                fps = 1.0 / max(1e-5, t_elapsed + sleep_time)
                print(
                    f"Step {t:3d}/{args.max_steps} | Loop Rate: {fps:5.1f} Hz | "
                    f"State: [{', '.join(f'{s:+.1f}' for s in raw_state)}] | "
                    f"Action: [{', '.join(f'{a:+.3f}' for a in action_np)}]"
                )

        print("\n Evaluation trial finished successfully.")

    except KeyboardInterrupt:
        print("\n Stopped real-world evaluation safely via Ctrl+C.")
    finally:
        if not args.mock and hasattr(robot, "disconnect"):
            try:
                robot.disconnect()
            except Exception:
                pass
            print(" Disconnected SO-101 Robot Arm hardware safely.")

if __name__ == "__main__":
    main()
