"""Launch Isaac Sim and view LeIsaac-SO101-CustomTask-v0 with zero actions."""

import sys
sys.path.append('/home/zozo/leisaac/source/leisaac')  # Ensure the correct leisaac package is found

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View the custom task environment.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# imports after Isaac Sim is running
import gymnasium as gym
import torch

import leisaac  # registers LeIsaac-SO101-CustomTask-v0
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        "LeIsaac-SO101-CustomTask-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.use_teleop_device('keyboard')  # Configures actions to avoid MISSING errors
    env = gym.make("LeIsaac-SO101-CustomTask-v0", cfg=env_cfg)
    print(f"[INFO]: Obs space : {env.observation_space}")
    print(f"[INFO]: Act space : {env.action_space}")
    env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
