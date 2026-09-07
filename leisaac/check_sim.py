import sys
sys.path.append('/home/zozo/leisaac/source/leisaac')
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless', '--enable_cameras'])
launcher = AppLauncher(args_cli)
simulation_app = launcher.app

import gymnasium as gym
import torch
import leisaac
from leisaac.tasks.custom_task.custom_task_env_cfg import CustomTaskEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
import omni.usd

# Parse configuration
env_cfg = parse_env_cfg('LeIsaac-SO101-CustomTask-v0', device='cuda', num_envs=1)
env_cfg.use_teleop_device('keyboard')

# Create environment
env = gym.make('LeIsaac-SO101-CustomTask-v0', cfg=env_cfg)
env.reset()

env.close()
simulation_app.close()
