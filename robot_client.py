import os
import sys
import time
import socket
import pickle
import queue
import argparse
import threading
import numpy as np
import torch

# Add parent path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_real_so101 import MockSO101Robot
from eval_leisaac import TemporalEnsembleBuffer

class ThreadedCamera:
    """Robust non-blocking background thread camera reader with MJPEG & dual-thread watchdog."""
    def __init__(self, src, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.frame_rgb = None
        self.running = False
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

    def read(self):
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


def capture_camera_frame(cam_obj):
    """Reads latest frame from a ThreadedCamera object."""
    if isinstance(cam_obj, ThreadedCamera):
        return cam_obj.read()
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

def recv_all(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

class ServerInferenceThread:
    """Manages connection to the GPU policy server and processes predictions in the background."""

    def __init__(self, ensemble_buffer):
        self.ensemble_buffer = ensemble_buffer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.running = False
        self.request_queue = queue.Queue(maxsize=1)
        self.thread = None

    def start(self):
        try:
            self.sock.connect(("127.0.0.1", 8085))
            self.connected = True
            self.running = True
            self.thread = threading.Thread(target=self.worker)
            self.thread.daemon = True
            self.thread.start()
            print(" Successfully connected to Policy Server.")
        except Exception as e:
            print(f" Could not connect to Policy Server: {e}")
            print(" -> Make sure you start 'python policy_server.py' in another terminal first.")
        return self

    def send_request(self, current_t, raw_images, raw_state):
        if not self.connected:
            return
        # Put request in queue (discard older requests if queue is full)
        try:
            self.request_queue.put_nowait((current_t, raw_images, raw_state))
        except queue.Full:
            pass

    def worker(self):
        while self.running:
            try:
                current_t, raw_images, raw_state = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            try:
                # Serialise request
                req = {"current_t": current_t, "raw_images": raw_images, "raw_state": raw_state}
                data = pickle.dumps(req)
                
                # Send request
                self.sock.sendall(len(data).to_bytes(4, byteorder="big") + data)
                
                # Read response header
                header = recv_all(self.sock, 4)
                if not header:
                    break
                resp_size = int.from_bytes(header, byteorder="big")
                
                # Read response data
                resp_data = b""
                while len(resp_data) < resp_size:
                    packet = self.sock.recv(resp_size - len(resp_data))
                    if not packet:
                        break
                    resp_data += packet
                
                if not resp_data:
                    break
                
                pred_chunk = pickle.loads(resp_data)
                
                # Update temporal ensemble buffer
                self.ensemble_buffer.update(current_t, torch.tensor(pred_chunk))
                
            except Exception as e:
                print(f"\n Server connection lost: {e}")
                self.connected = False
                break

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="Async Robot Client for Real SO-101 Robot Arm")
    parser.add_argument(
        "--robot_port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for SO-101 robot arm controller",
    )
    parser.add_argument(
        "--cam_right",
        type=int,
        default=4,
        help="Camera index for right camera",
    )
    parser.add_argument(
        "--cam_wrist",
        type=int,
        default=6,
        help="Camera index for wrist camera",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Control loop target execution frequency",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Maximum evaluation timesteps",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock simulation mode",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable live Rerun visualizer",
    )
    parser.add_argument(
        "--no_auto_align",
        action="store_true",
        help="Disable automatic starting pose calibration alignment",
    )
    args = parser.parse_args()

    
    print("  Starting Real-Time Async Client for SO-101")
    
    print(f" -> Hardware Mode     : {'MOCK / SIMULATED' if args.mock else f'REAL ROBOT ({args.robot_port})'}")
    print(f" -> Control Frequency : {args.hz} Hz")
    print(f" -> Rerun Visualizer  : {'ENABLED' if args.viz else 'DISABLED'}")
    

    # 1. Initialize Rerun if enabled
    if args.viz:
        try:
            from lerobot.utils.rerun_visualization import init_rerun
            init_rerun("SO-101 Async Control Loop")
            print(" Rerun Visualizer initialized.")
        except Exception as e:
            print(f" Could not initialize Rerun: {e}")
            args.viz = False

    # 2. Instantiate Robot Arm Controller
    if args.mock:
        robot = MockSO101Robot(state_dim=6)
        print(" Initialized Mock SO-101 Robot Arm.")
    else:
        try:
            try:
                from lerobot.robots.so_follower import SO101Follower
                from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
            except ImportError:
                from lerobot.robots.so101_follower.so101_follower import SO101Follower
                from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
            
            config = SO101FollowerConfig(
                port=args.robot_port,
                id="my_follower_arm",
                use_degrees=True,
                position_p_coefficient=24,  # Restored to 24 so the arm can lift against gravity
                position_d_coefficient=32   # Restored to 32
            )
            robot = SO101Follower(config)
            robot.connect(calibrate=False)
            
            # Explicitly enable motor torque and set limits
            if hasattr(robot, "bus") and robot.bus is not None:
                for motor_name in robot.bus.motors:
                    try:
                        robot.bus.write("Torque_Enable", motor_name, 1)
                        robot.bus.write("Max_Torque_Limit", motor_name, 1000)
                        robot.bus.write("Lock", motor_name, 1)
                        robot.bus.write("P_Coefficient", motor_name, 24)
                        robot.bus.write("D_Coefficient", motor_name, 32)
                    except Exception:
                        pass
            
            print(f" Connected to real SO-101 Robot Arm on '{args.robot_port}'.")
        except Exception as e:
            print(f" Could not connect to real SO-101: {e}")
            print(" -> Falling back to Mock Robot mode.")
            robot = MockSO101Robot(state_dim=6)

    # 3. Pre-open cameras in background threads
    caps = {}
    if not args.mock:
        for cam_name, cam_idx in [("right", args.cam_right), ("wrist", args.cam_wrist)]:
            print(f" Starting threaded camera '{cam_name}' (index {cam_idx})...")
            cam = ThreadedCamera(cam_idx).start()
            caps[cam_name] = cam
        print(" Waiting 1.0 second for camera threads to initialize...")
        time.sleep(1.0)

    # 4. Set up temporal ensembling & async inference thread
    ensemble_buffer = TemporalEnsembleBuffer(action_dim=6, chunk_size=16)
    inference_thread = ServerInferenceThread(ensemble_buffer).start()

    step_dt = 1.0 / args.hz
    print("\n Starting Control Loop... Press Ctrl+C to stop.\n")

    # Start policy immediately from current posture (zero pre-movement)
    if not args.mock and hasattr(robot, "get_observation"):
        obs = robot.get_observation()
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        cur_pose = [f"{obs[f'{j}.pos']:+.1f}°" for j in joints if f"{j}.pos" in obs]
        print(f" Starting control loop directly from current arm posture: [{', '.join(cur_pose)}]\n")

    print(" Pure Neural Architecture Mode (Client/Server): Actions directly generated by policy.")

    try:
        for t in range(args.max_steps):
            t_start = time.time()

            # A. Capture camera frames
            right_frame = capture_camera_frame(caps.get("right"))
            wrist_frame = capture_camera_frame(caps.get("wrist"))
            raw_images = [right_frame, wrist_frame]

            # B. Read current joint states
            if hasattr(robot, "read_joint_positions"):
                raw_state = robot.read_joint_positions()
            elif hasattr(robot, "get_state"):
                raw_state = robot.get_state()
            elif hasattr(robot, "get_observation"):
                obs = robot.get_observation()
                joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                raw_state = np.array([obs[f"{j}.pos"] for j in joints], dtype=np.float32)
            else:
                raw_state = np.zeros(6, dtype=np.float32)

            # C. Send request to background inference thread (non-blocking)
            inference_thread.send_request(t, raw_images, raw_state)

            # D. Get direct action output from neural network architecture (100% pure policy)
            action_tensor = ensemble_buffer.get_action(t)
            if torch.all(action_tensor == 0):
                if 'last_action' in locals() and last_action is not None:
                    action = last_action.copy()
                else:
                    action = raw_state.copy()
            else:
                action = action_tensor.cpu().numpy().copy()
                last_action = action.copy()

            # E. Send direct policy action to arm
            if hasattr(robot, "send_action"):
                if isinstance(robot, MockSO101Robot):
                    robot.send_action(action)
                else:
                    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    action_dict = {f"{j}.pos": float(action[i]) for i, j in enumerate(joints)}
                    robot.send_action(action_dict)
            elif hasattr(robot, "step"):
                robot.step(action)

            # F. Log to Rerun
            if args.viz:
                try:
                    from lerobot.utils.rerun_visualization import log_rerun_data
                    obs_dict = {
                        "images/right": right_frame,
                        "images/wrist": wrist_frame,
                    }
                    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                    obs_dict.update({f"joint_positions/{j}": float(raw_state[i]) for i, j in enumerate(joints)})
                    act_dict = {f"joint_targets/{j}": float(action[i]) for i, j in enumerate(joints)}
                    log_rerun_data(observation=obs_dict, action=act_dict)
                except Exception:
                    pass

            t_elapsed = time.time() - t_start
            sleep_time = max(0.0, step_dt - t_elapsed)
            time.sleep(sleep_time)

            if t % 15 == 0 or t == args.max_steps - 1:
                fps = 1.0 / max(1e-5, t_elapsed + sleep_time)
                sh_lift = raw_state[1] if len(raw_state) > 1 else 0.0
                print(
                    f"Step {t:4d}/{args.max_steps} | {fps:5.1f} Hz | "
                    f"shoulder_lift: {sh_lift:+.2f}° | "
                    f"Action: [{', '.join(f'{a:+.3f}' for a in action)}]"
                )

        print("\n Trial finished successfully.")

    except KeyboardInterrupt:
        print("\n Stopped safely via Ctrl+C.")
    finally:
        # Stop background threads
        inference_thread.stop()
        for cam_name, cap in caps.items():
            if cap is not None:
                cap.stop()
        
        # Disconnect robot
        if not args.mock and hasattr(robot, "disconnect"):
            try:
                robot.disconnect()
            except Exception:
                pass
            print(" Disconnected SO-101 Robot Arm safely.")

if __name__ == "__main__":
    main()

