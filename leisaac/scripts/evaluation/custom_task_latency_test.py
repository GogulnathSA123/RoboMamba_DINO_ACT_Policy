"""
Quick Latency Profiler for SO-101 CustomTask in LeIsaac Simulation.
Profiles step-by-step latency for either RoboMamba-DINO-ACT or ACT policies.
"""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import time
import torch
import argparse
import numpy as np
import gymnasium as gym
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Detailed Latency Test for CustomTask")
parser.add_argument("--task", type=str, default="LeIsaac-SO101-CustomTask-v0", help="Task ID.")
parser.add_argument("--steps", type=int, default=100, help="Number of control steps to test.")
parser.add_argument("--policy_type", type=str, default="robomamba-dino", 
                    choices=["robomamba-dino", "robomamba_dino_act", "act", "act_isaacsim"], 
                    help="Policy architecture.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path or HF repo to model checkpoint.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import leisaac
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type
from leisaac.scripts.evaluation.evaluate_latency_benchmark import BenchmarkACTAdapter, BenchmarkRoboMambaDINOAdapter

def main():
    print(f"[Latency Test] Initializing environment: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if args_cli.policy_type.lower() in ["act", "act_isaacsim"]:
        ckpt = args_cli.checkpoint if args_cli.checkpoint else "Gogul99/act_isaacsim"
        policy = BenchmarkACTAdapter(checkpoint_path=ckpt, device=args_cli.device)
        policy_label = "ACT"
    else:
        ckpt = args_cli.checkpoint if args_cli.checkpoint else "Gogul99/RoboMamba_DINO_ACT_Policy"
        policy = BenchmarkRoboMambaDINOAdapter(checkpoint_path=ckpt, device=args_cli.device)
        policy_label = "RoboMamba-DINO-ACT"

    latencies_obs = []
    latencies_infer = []
    latencies_post = []
    latencies_policy = []
    latencies_sim = []
    latencies_e2e = []

    print(f"\n--- Warming up and running {args_cli.steps} latency test steps for {policy_label} ---")
    obs_dict, _ = env.reset()
    policy.reset()

    # Warmup
    for _ in range(5):
        actions, _ = policy.step(obs_dict)
        obs_dict, _, _, _, _ = env.step(actions[0, 0, :].unsqueeze(0))

    print(f"Starting measurement for {args_cli.steps} steps...")
    for step_idx in range(1, args_cli.steps + 1):
        if not simulation_app.is_running():
            break

        t_e2e_0 = time.perf_counter()
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

        t_e2e_1 = time.perf_counter()
        e2e_ms = (t_e2e_1 - t_e2e_0) * 1000.0

        latencies_obs.append(timing["obs_ms"])
        latencies_infer.append(timing["inference_ms"])
        latencies_post.append(timing["post_ms"])
        latencies_policy.append(timing["policy_total_ms"])
        latencies_sim.append(sim_ms)
        latencies_e2e.append(e2e_ms)

        if step_idx % 20 == 0 or step_idx == args_cli.steps:
            print(f"Step {step_idx:03d}/{args_cli.steps:03d} | Infer: {timing['inference_ms']:5.2f}ms | Policy: {timing['policy_total_ms']:5.2f}ms | Sim: {sim_ms:5.2f}ms | E2E: {e2e_ms:5.2f}ms ({1000.0/e2e_ms:4.1f} Hz)")

        if reset_terminated[0] or reset_time_outs[0]:
            obs_dict, _ = env.reset()
            policy.reset()

    print("\n" + "="*70)
    print(f"LATENCY SUMMARY ({policy_label}, {len(latencies_e2e)} steps):")
    print(f"  Obs Preprocessing:  Mean={np.mean(latencies_obs):5.2f}ms | Median={np.median(latencies_obs):5.2f}ms | P95={np.percentile(latencies_obs, 95):5.2f}ms")
    print(f"  Model Inference:    Mean={np.mean(latencies_infer):5.2f}ms | Median={np.median(latencies_infer):5.2f}ms | P95={np.percentile(latencies_infer, 95):5.2f}ms")
    print(f"  Action Postprocess: Mean={np.mean(latencies_post):5.2f}ms | Median={np.median(latencies_post):5.2f}ms | P95={np.percentile(latencies_post, 95):5.2f}ms")
    print(f"  Total Policy Time:  Mean={np.mean(latencies_policy):5.2f}ms | Median={np.median(latencies_policy):5.2f}ms | P95={np.percentile(latencies_policy, 95):5.2f}ms")
    print(f"  Simulation Step:    Mean={np.mean(latencies_sim):5.2f}ms | Median={np.median(latencies_sim):5.2f}ms | P95={np.percentile(latencies_sim, 95):5.2f}ms")
    print(f"  End-to-End Step:    Mean={np.mean(latencies_e2e):5.2f}ms | Median={np.median(latencies_e2e):5.2f}ms | P95={np.percentile(latencies_e2e, 95):5.2f}ms")
    print(f"  Achieved Loop Hz:   {1000.0/np.mean(latencies_e2e):5.1f} Hz")
    print("="*70)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
