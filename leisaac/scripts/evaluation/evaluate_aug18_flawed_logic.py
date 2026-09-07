"""
Comprehensive Latency, Memory Bandwidth, VRAM & Sub-Goal Milestone Benchmark
Aug 18 Unmodified Logic Version (Omits reset_terminated[0] check).
"""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse
import os
import sys

for p in ["/home/zozo/leisaac/source/leisaac", "/home/zozo/lerobot/src", "/home/zozo/RoboMamba_DINO_ACT_Policy"]:
    if p not in sys.path:
        sys.path.insert(0, p)

from isaaclab.app import AppLauncher

# 1. Argument Parser Setup
parser = argparse.ArgumentParser(description="Latency, Hardware & Milestone Benchmarking for SO-101.")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-CustomTask-v0", help="Task name.")
parser.add_argument("--policy_type", type=str, default="robomamba-dino", 
                    choices=["robomamba-dino", "robomamba_dino", "robomamba_dino_act", "act", "act_isaacsim", "lerobot-act"],
                    help="Policy to benchmark.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Custom checkpoint path or HuggingFace repo ID.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon chunk size.")
parser.add_argument("--policy_language_instruction", type=str, 
                    default="pick up the red cube and place it into the box.", 
                    help="Task language instruction.")
parser.add_argument("--eval_rounds", type=int, default=100, help="Number of evaluation trials (default: 100).")
parser.add_argument("--episode_length_s", type=float, default=20.0, help="Maximum episode length in seconds (default: 20.0s).")
parser.add_argument("--step_hz", type=int, default=30, help="Simulation step frequency target in Hz.")
parser.add_argument("--unthrottled", action="store_true", default=False, 
                    help="Run unthrottled (no sleep) to measure maximum throughput/FPS.")
parser.add_argument("--warmup_steps", type=int, default=15, help="Number of warmup steps to discard from latency stats.")
parser.add_argument("--output_dir", type=str, default="/home/zozo/leisaac/benchmark_results", 
                    help="Directory to save benchmark JSON results and reports.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment reproducibility.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse Kit Simulator first (REQUIRED before torch/gym imports)
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
import json
from datetime import datetime
import numpy as np
import torch

# Initialize NVML for hardware & bandwidth profiling
HAS_NVML = False
try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
    print(f"[NVML] Hardware monitoring active on: {pynvml.nvmlDeviceGetName(nvml_handle)}")
except Exception as e:
    print(f"[NVML] Notice: NVML hardware monitoring disabled ({e})")

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
import leisaac  # noqa: F401


# 2. Timing and Rate Limiter Utilities
class RateLimiter:
    def __init__(self, hz):
        self.hz = hz
        self.interval = 1.0 / hz if hz > 0 else 0
        self.last_time = time.time()

    def sleep(self):
        if self.interval <= 0:
            return
        now = time.time()
        elapsed = now - self.last_time
        remaining = self.interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_time = time.time()


def compute_statistics(values):
    if not values or len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def query_gpu_memory_and_bandwidth():
    """Queries current PyTorch VRAM and NVML Memory Controller Bandwidth utilization."""
    res = {
        "torch_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "torch_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "sys_vram_used_mb": 0.0,
        "mem_bandwidth_util_pct": 0.0,
        "gpu_core_util_pct": 0.0,
    }
    if HAS_NVML:
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            res["sys_vram_used_mb"] = float(mem_info.used / (1024**2))
            res["mem_bandwidth_util_pct"] = float(util_info.memory)
            res["gpu_core_util_pct"] = float(util_info.gpu)
        except Exception:
            pass
    return res


# 3. Instrumented Policy Wrappers
class BenchmarkACTAdapter:
    def __init__(self, checkpoint_path="Gogul99/act_isaacsim", device="cuda"):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from huggingface_hub import snapshot_download

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_dir = snapshot_download(checkpoint_path) if "/" in checkpoint_path and not os.path.exists(checkpoint_path) else checkpoint_path
        print(f"[Policy] Loading ACTPolicy from '{model_dir}'...")
        self.policy = ACTPolicy.from_pretrained(model_dir).to(self.device)
        self.policy.eval()
        self.step_t = 0
        self.name = "ACT"
        print("[Policy] ACTPolicy initialized on GPU.")

    def reset(self):
        self.policy.reset()
        self.step_t = 0

    def step(self, obs_dict: dict) -> tuple[torch.Tensor, dict]:
        timing = {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_obs_0 = time.perf_counter()

        if "policy" in obs_dict:
            obs_dict = obs_dict["policy"]

        front_img = obs_dict.get("front", obs_dict.get("observation.images.front", None))
        wrist_img = obs_dict.get("wrist", obs_dict.get("observation.images.wrist", None))
        if front_img is None or wrist_img is None:
            for k, v in obs_dict.items():
                k_str = str(k).lower()
                if front_img is None and "front" in k_str:
                    front_img = v
                elif wrist_img is None and "wrist" in k_str:
                    wrist_img = v

        def prep_cam(img):
            if img is None:
                return torch.zeros((1, 3, 480, 640), dtype=torch.float32, device=self.device)
            if isinstance(img, np.ndarray):
                t = torch.from_numpy(img).float()
            else:
                t = img.float()
            if t.ndim == 4:
                t = t[0]
            if t.ndim == 3 and t.shape[-1] in [3, 4]:
                t = t[..., :3].permute(2, 0, 1)
            if t.max() > 1.0:
                t = t / 255.0
            return t.unsqueeze(0).to(self.device)

        front_t = prep_cam(front_img)
        wrist_t = prep_cam(wrist_img)

        if "joint_pos" in obs_dict:
            raw_state_rad = obs_dict["joint_pos"]
            if isinstance(raw_state_rad, torch.Tensor):
                raw_state_rad = raw_state_rad.cpu().numpy()
            if raw_state_rad.ndim == 1:
                raw_state_rad = raw_state_rad[None, :]
            motor_state = convert_leisaac_action_to_lerobot(raw_state_rad)[0]
        else:
            motor_state = np.zeros(6, dtype=np.float32)

        state_t = torch.from_numpy(motor_state).float().unsqueeze(0).to(self.device)
        batch = {
            "observation.images.front": front_t,
            "observation.images.wrist": wrist_t,
            "observation.state": state_t,
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_obs_1 = time.perf_counter()
        timing["obs_ms"] = (t_obs_1 - t_obs_0) * 1000.0

        t_infer_0 = time.perf_counter()
        with torch.no_grad():
            pred_action = self.policy.select_action(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer_1 = time.perf_counter()
        timing["inference_ms"] = (t_infer_1 - t_infer_0) * 1000.0

        t_post_0 = time.perf_counter()
        pred_motor = pred_action.squeeze(0).cpu().numpy()
        self.step_t += 1
        target_rad = convert_lerobot_action_to_leisaac(pred_motor[None, :])[0]
        action_tensor = torch.tensor(target_rad, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_post_1 = time.perf_counter()
        timing["post_ms"] = (t_post_1 - t_post_0) * 1000.0

        timing["policy_total_ms"] = timing["obs_ms"] + timing["inference_ms"] + timing["post_ms"]
        return action_tensor, timing


class BenchmarkRoboMambaDINOAdapter:
    def __init__(self, checkpoint_path="Gogul99/RoboMamba_DINO_ACT_Policy", device="cuda"):
        if "/home/zozo/RoboMamba_DINO_ACT_Policy" not in sys.path:
            sys.path.insert(0, "/home/zozo/RoboMamba_DINO_ACT_Policy")
        from eval_leisaac import DINOv2PolicyConnector

        cfg_path = "/home/zozo/RoboMamba_DINO_ACT_Policy/config.yaml"
        self.connector = DINOv2PolicyConnector(checkpoint_path=checkpoint_path, cfg_path=cfg_path, device=device)
        self.device = device
        self.step_t = 0
        self.name = "RoboMamba-DINO-ACT"
        print("[Policy] RoboMamba-DINO-ACT Policy initialized on GPU.")

    def reset(self):
        self.connector.reset()
        self.step_t = 0

    def step(self, obs_dict: dict) -> tuple[torch.Tensor, dict]:
        timing = {}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_obs_0 = time.perf_counter()

        if "policy" in obs_dict:
            obs_dict = obs_dict["policy"]

        front_img = obs_dict.get("front", obs_dict.get("observation.images.front", None))
        wrist_img = obs_dict.get("wrist", obs_dict.get("observation.images.wrist", None))
        if front_img is None or wrist_img is None:
            for k, v in obs_dict.items():
                k_str = str(k).lower()
                if front_img is None and "front" in k_str:
                    front_img = v
                elif wrist_img is None and "wrist" in k_str:
                    wrist_img = v

        raw_images = [front_img, wrist_img]
        processed_imgs = []
        for img in raw_images:
            if img is not None:
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                if img.ndim == 4:
                    img = img[0]
                if img.ndim == 3 and img.shape[-1] == 4:
                    img = img[..., :3]
            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            if not isinstance(img, torch.Tensor):
                t_img = torch.tensor(img, dtype=torch.float32, device=self.device)
            else:
                t_img = img.to(self.device).float()

            if t_img.ndim == 3 and t_img.shape[-1] in [3, 4]:
                t_img = t_img[..., :3].permute(2, 0, 1)

            if t_img.shape[-2:] != (224, 224):
                t_img = torch.nn.functional.interpolate(
                    t_img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0)

            if t_img.max() > 1.0:
                t_img = t_img / 255.0

            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
            t_img = (t_img - mean) / std
            processed_imgs.append(t_img)

        batch_imgs = torch.stack(processed_imgs, dim=0).unsqueeze(0)

        if "joint_pos" in obs_dict:
            raw_state_rad = obs_dict["joint_pos"]
            if isinstance(raw_state_rad, torch.Tensor):
                raw_state_rad = raw_state_rad.cpu().numpy()
            if raw_state_rad.ndim == 1:
                raw_state_rad = raw_state_rad[None, :]
            motor_state = convert_leisaac_action_to_lerobot(raw_state_rad)[0]
        else:
            motor_state = np.zeros(6, dtype=np.float32)

        batch_state = torch.tensor(motor_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_obs_1 = time.perf_counter()
        timing["obs_ms"] = (t_obs_1 - t_obs_0) * 1000.0

        t_infer_0 = time.perf_counter()
        with torch.no_grad():
            pred_chunk, _, _ = self.connector.policy(batch_imgs, batch_state)
            pred_chunk = pred_chunk.squeeze(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer_1 = time.perf_counter()
        timing["inference_ms"] = (t_infer_1 - t_infer_0) * 1000.0

        t_post_0 = time.perf_counter()
        self.connector.ensemble.update(self.step_t, pred_chunk)
        raw_action = self.connector.ensemble.get_action(self.step_t)
        pred_motor_action = raw_action.cpu().numpy()
        self.step_t += 1

        raw_gripper = pred_motor_action[5]
        pred_motor_action[5] = np.clip(raw_gripper, 0.0, 60.76)

        target_rad = convert_lerobot_action_to_leisaac(pred_motor_action[None, :])[0]
        action_tensor = torch.tensor(target_rad, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_post_1 = time.perf_counter()
        timing["post_ms"] = (t_post_1 - t_post_0) * 1000.0

        timing["policy_total_ms"] = timing["obs_ms"] + timing["inference_ms"] + timing["post_ms"]
        return action_tensor, timing


# 4. Sub-Goal Milestone Evaluator
class SubGoalMilestoneTracker:
    def __init__(self, env):
        self.env = env
        self.reset()

    def reset(self):
        self.stage1_near_cube = False
        self.stage2_picked_lifted = False
        self.stage3_near_box = False
        self.stage4_placed_in_box = False

        self.min_dist_gripper_cube = float("inf")
        self.max_cube_height = -float("inf")
        self.min_dist_cube_box_xy = float("inf")

        self.stage1_step = None
        self.stage2_step = None
        self.stage3_step = None
        self.stage4_step = None

    def update(self, step_idx):
        try:
            robot = self.env.scene["robot"]
            try:
                gripper_pos = self.env.scene["ee_frame"].data.target_pos_w[0, 0, :3] - self.env.scene.env_origins[0, :3]
            except Exception:
                gripper_pos = robot.data.body_state_w[0, -1, :3] - self.env.scene.env_origins[0, :3]

            cube = self.env.scene["cube"]
            cube_pos = cube.data.root_pos_w[0, :3] - self.env.scene.env_origins[0, :3]

            box = self.env.scene["box"]
            box_pos = box.data.root_pos_w[0, :3] - self.env.scene.env_origins[0, :3]

            dist_gripper_cube = torch.linalg.norm(gripper_pos - cube_pos).item()
            cube_z = cube_pos[2].item()
            dist_cube_box_xy = torch.linalg.norm(cube_pos[:2] - box_pos[:2]).item()

            self.min_dist_gripper_cube = min(self.min_dist_gripper_cube, dist_gripper_cube)
            self.max_cube_height = max(self.max_cube_height, cube_z)
            self.min_dist_cube_box_xy = min(self.min_dist_cube_box_xy, dist_cube_box_xy)

            if not self.stage1_near_cube and dist_gripper_cube <= 0.18:
                self.stage1_near_cube = True
                self.stage1_step = step_idx

            if not self.stage2_picked_lifted and cube_z >= 0.045 and dist_gripper_cube <= 0.20:
                self.stage2_picked_lifted = True
                self.stage1_near_cube = True
                if self.stage2_step is None:
                    self.stage2_step = step_idx

            if not self.stage3_near_box and dist_cube_box_xy <= 0.15 and cube_z >= 0.038:
                self.stage3_near_box = True
                self.stage2_picked_lifted = True
                self.stage1_near_cube = True
                if self.stage3_step is None:
                    self.stage3_step = step_idx

            dx = abs(cube_pos[0].item() - box_pos[0].item())
            dy = abs(cube_pos[1].item() - box_pos[1].item())
            if dx < 0.06 and dy < 0.06 and cube_z < 0.10:
                self.stage4_placed_in_box = True
                self.stage3_near_box = True
                self.stage2_picked_lifted = True
                self.stage1_near_cube = True
                if self.stage4_step is None:
                    self.stage4_step = step_idx
        except Exception as e:
            print(f"[MilestoneTracker Warning] {e}")

    def get_partial_score(self):
        score = 0.0
        if self.stage1_near_cube:
            score += 0.25
        if self.stage2_picked_lifted:
            score += 0.25
        if self.stage3_near_box:
            score += 0.25
        if self.stage4_placed_in_box:
            score += 0.25
        return score

    def get_summary(self):
        return {
            "stage1_near_cube": self.stage1_near_cube,
            "stage1_step": self.stage1_step,
            "stage2_picked_lifted": self.stage2_picked_lifted,
            "stage2_step": self.stage2_step,
            "stage3_near_box": self.stage3_near_box,
            "stage3_step": self.stage3_step,
            "stage4_placed_in_box": self.stage4_placed_in_box,
            "stage4_step": self.stage4_step,
            "partial_goal_score_pct": round(self.get_partial_score() * 100.0, 1),
            "min_dist_gripper_cube_m": round(self.min_dist_gripper_cube, 4),
            "max_cube_height_m": round(self.max_cube_height, 4),
            "min_dist_cube_box_xy_m": round(self.min_dist_cube_box_xy, 4),
        }


# 5. Main Evaluation Runner
def run_evaluation():
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.recorders = None

    print(f"Creating Isaac Sim Environment: {args_cli.task}")
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    dt = env.cfg.sim.dt * env.cfg.decimation
    max_steps_per_episode = int(args_cli.episode_length_s / dt) if dt > 0 else 600

    is_act = args_cli.policy_type.lower() in ["act", "act_isaacsim", "lerobot-act"]
    if is_act:
        ckpt = args_cli.policy_checkpoint_path if args_cli.policy_checkpoint_path else "Gogul99/act_isaacsim"
        policy = BenchmarkACTAdapter(checkpoint_path=ckpt, device=args_cli.device)
        policy_label = "ACT"
    else:
        ckpt = args_cli.policy_checkpoint_path if args_cli.policy_checkpoint_path else "Gogul99/RoboMamba_DINO_ACT_Policy"
        policy = BenchmarkRoboMambaDINOAdapter(checkpoint_path=ckpt, device=args_cli.device)
        policy_label = "RoboMamba-DINO-ACT"

    milestone_tracker = SubGoalMilestoneTracker(env)
    rate_limiter = None if args_cli.unthrottled else RateLimiter(args_cli.step_hz)

    total_trials = args_cli.eval_rounds
    print(f"\n{'='*82}")
    print(f"  STARTING MULTI-METRIC BENCHMARK & EVALUATION ({total_trials} TRIALS)")
    print(f"  Policy:             {policy_label} | Task: {args_cli.task}")
    print(f"  Max Episode Length: {args_cli.episode_length_s}s ({max_steps_per_episode} steps)")
    print(f"  Step Rate Target:   {'Unthrottled (Max FPS)' if args_cli.unthrottled else f'{args_cli.step_hz} Hz'}")
    print(f"  Warmup Discard:     {args_cli.warmup_steps} steps")
    print(f"  Hardware Metrics:   VRAM Footprint + Memory Bandwidth Controller Util (%)")
    print(f"  Task Milestones:    Near Cube (25%) -> Lift (50%) -> Near Box (75%) -> Placed (100%)")
    print(f"{'='*82}\n")

    all_obs_latencies = []
    all_inference_latencies = []
    all_post_latencies = []
    all_policy_latencies = []
    all_sim_latencies = []
    all_step_total_latencies = []

    all_torch_vram_mb = []
    all_sys_vram_mb = []
    all_mem_bw_util_pct = []
    all_gpu_core_util_pct = []

    stage1_success_count = 0
    stage2_success_count = 0
    stage3_success_count = 0
    stage4_success_count = 0
    all_partial_scores = []

    episode_records = []
    total_steps_executed = 0

    with torch.inference_mode():
        obs_dict, _ = env.reset()
        policy.reset()
        milestone_tracker.reset()

    benchmark_start_time = time.time()

    for episode_idx in range(1, total_trials + 1):
        ep_t_start = time.time()
        ep_obs_l, ep_inf_l, ep_post_l, ep_pol_l, ep_sim_l, ep_total_l = [], [], [], [], [], []
        ep_torch_vram_l, ep_sys_vram_l, ep_mem_bw_l, ep_gpu_util_l = [], [], [], []
        step_in_ep = 0
        success = False
        timed_out = False

        milestone_tracker.reset()

        while simulation_app.is_running():
            total_steps_executed += 1
            step_in_ep += 1
            step_start = time.perf_counter()

            with torch.inference_mode():
                actions, timing = policy.step(obs_dict)
                action = actions[0, 0, :]

                if env.cfg.dynamic_reset_gripper_effort_limit:
                    dynamic_reset_gripper_effort_limit_sim(env, task_type)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sim_0 = time.perf_counter()
                obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action.unsqueeze(0))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_sim_1 = time.perf_counter()
                sim_ms = (t_sim_1 - t_sim_0) * 1000.0

                milestone_tracker.update(step_in_ep)
                hw_metrics = query_gpu_memory_and_bandwidth()

                if rate_limiter:
                    rate_limiter.sleep()

            step_end = time.perf_counter()
            step_total_ms = (step_end - step_start) * 1000.0

            if total_steps_executed > args_cli.warmup_steps:
                ep_obs_l.append(timing["obs_ms"])
                ep_inf_l.append(timing["inference_ms"])
                ep_post_l.append(timing["post_ms"])
                ep_pol_l.append(timing["policy_total_ms"])
                ep_sim_l.append(sim_ms)
                ep_total_l.append(step_total_ms)

                ep_torch_vram_l.append(hw_metrics["torch_allocated_mb"])
                ep_sys_vram_l.append(hw_metrics["sys_vram_used_mb"])
                ep_mem_bw_l.append(hw_metrics["mem_bandwidth_util_pct"])
                ep_gpu_util_l.append(hw_metrics["gpu_core_util_pct"])

                all_obs_latencies.append(timing["obs_ms"])
                all_inference_latencies.append(timing["inference_ms"])
                all_post_latencies.append(timing["post_ms"])
                all_policy_latencies.append(timing["policy_total_ms"])
                all_sim_latencies.append(sim_ms)
                all_step_total_latencies.append(step_total_ms)

                all_torch_vram_mb.append(hw_metrics["torch_allocated_mb"])
                all_sys_vram_mb.append(hw_metrics["sys_vram_used_mb"])
                all_mem_bw_util_pct.append(hw_metrics["mem_bandwidth_util_pct"])
                all_gpu_core_util_pct.append(hw_metrics["gpu_core_util_pct"])

            # Check Termination (Aug 18 Unmodified Logic: Omits reset_terminated[0])
            if milestone_tracker.stage4_placed_in_box:
                success = True
                break
            if reset_time_outs[0] or step_in_ep >= max_steps_per_episode:
                timed_out = True
                break

        ep_duration = time.time() - ep_t_start
        milestone_res = milestone_tracker.get_summary()

        if milestone_res["stage1_near_cube"]:
            stage1_success_count += 1
        if milestone_res["stage2_picked_lifted"]:
            stage2_success_count += 1
        if milestone_res["stage3_near_box"]:
            stage3_success_count += 1
        if milestone_res["stage4_placed_in_box"] or success:
            stage4_success_count += 1
            success = True

        all_partial_scores.append(milestone_res["partial_goal_score_pct"])

        status_str = "SUCCESS" if success else ("TIMEOUT" if timed_out else "TERMINATED")
        ep_mean_pol_lat = np.mean(ep_pol_l) if len(ep_pol_l) > 0 else 0.0
        ep_mean_inf_lat = np.mean(ep_inf_l) if len(ep_inf_l) > 0 else 0.0
        ep_mean_sim_lat = np.mean(ep_sim_l) if len(ep_sim_l) > 0 else 0.0
        ep_mean_tot_lat = np.mean(ep_total_l) if len(ep_total_l) > 0 else 0.0
        ep_effective_hz = 1000.0 / ep_mean_tot_lat if ep_mean_tot_lat > 0 else 0.0

        ep_peak_vram = np.max(ep_torch_vram_l) if len(ep_torch_vram_l) > 0 else 0.0
        ep_avg_mem_bw = np.mean(ep_mem_bw_l) if len(ep_mem_bw_l) > 0 else 0.0

        m1 = "yes" if milestone_res["stage1_near_cube"] else "x"
        m2 = "yes" if milestone_res["stage2_picked_lifted"] else "x"
        m3 = "yes" if milestone_res["stage3_near_box"] else "x"
        m4 = "yes" if milestone_res["stage4_placed_in_box"] else "x"
        m_str = f"[{m1} Reach | {m2} Lift | {m3} Box | {m4} Place]"

        print(
            f"[Trial {episode_idx:03d}/{total_trials:03d}] {status_str:<8} | "
            f"Steps: {step_in_ep:4d} | Time: {ep_duration:4.1f}s | "
            f"Infer: {ep_mean_inf_lat:4.1f}ms | Policy: {ep_mean_pol_lat:4.1f}ms | Sim: {ep_mean_sim_lat:4.1f}ms | "
            f"Rate: {ep_effective_hz:4.1f}Hz | VRAM: {ep_peak_vram:4.0f}MB | BW: {ep_avg_mem_bw:3.0f}% | "
            f"{m_str} Score: {milestone_res['partial_goal_score_pct']:3.0f}% | "
            f"SR: {stage4_success_count/episode_idx*100:4.1f}%",
            flush=True,
        )

        episode_records.append({
            "trial_id": episode_idx,
            "status": status_str,
            "success": success,
            "timed_out": timed_out,
            "steps": step_in_ep,
            "duration_s": round(ep_duration, 2),
            "milestones": milestone_res,
        })

        with torch.inference_mode():
            obs_dict, _ = env.reset()
            policy.reset()

    total_benchmark_time = time.time() - benchmark_start_time

    summary_stats = {
        "policy_name": policy_label,
        "task": args_cli.task,
        "checkpoint": str(args_cli.policy_checkpoint_path),
        "seed": args_cli.seed,
        "total_trials": total_trials,
        "total_steps": total_steps_executed,
        "total_time_s": round(total_benchmark_time, 2),
        "task_performance": {
            "full_success_count": stage4_success_count,
            "full_success_rate_pct": round(stage4_success_count / total_trials * 100, 2),
        },
        "episode_records": episode_records
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args_cli.output_dir, f"benchmark_aug18_flawed_logic_{total_trials}trials_seed{args_cli.seed}_{timestamp_str}.json")
    with open(json_path, "w") as f:
        json.dump(summary_stats, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    print(f"[Benchmark] Saved JSON results to: {json_path}", flush=True)

    os._exit(0)


if __name__ == "__main__":
    run_evaluation()
