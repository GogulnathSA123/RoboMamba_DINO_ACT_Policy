import argparse
import json
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras"])
app = AppLauncher(args).app

import gymnasium as gym
import leisaac
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.utils.env_utils import get_task_type
env_cfg = parse_env_cfg("LeIsaac-SO101-CustomTask-v0", device="cuda", num_envs=1)
task_type = get_task_type("LeIsaac-SO101-CustomTask-v0")
env_cfg.use_teleop_device(task_type)
env = gym.make("LeIsaac-SO101-CustomTask-v0", cfg=env_cfg).unwrapped

env.reset()

robot_pos = env.scene["robot"].data.root_pos_w[0].cpu().numpy().tolist()
cube_pos = env.scene["cube"].data.root_pos_w[0].cpu().numpy().tolist()
box_pos = env.scene["box"].data.root_pos_w[0].cpu().numpy().tolist()

sys.stderr.write(f"ROBOT_POS: {robot_pos}\n")
sys.stderr.write(f"CUBE_POS: {cube_pos}\n")
sys.stderr.write(f"BOX_POS: {box_pos}\n")

rel_cube = [cube_pos[i] - robot_pos[i] for i in range(3)]
rel_box = [box_pos[i] - robot_pos[i] for i in range(3)]

sys.stderr.write(f"RELATIVE_CUBE_FROM_ROBOT: {rel_cube}\n")
sys.stderr.write(f"RELATIVE_BOX_FROM_ROBOT: {rel_box}\n")
sys.stderr.flush()

env.close()
app.close()
