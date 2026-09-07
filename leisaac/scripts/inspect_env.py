import argparse
import sys
import json
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import leisaac
from isaaclab_tasks.utils import parse_env_cfg

env_cfg = parse_env_cfg("LeIsaac-SO101-CustomTask-v0", device="cuda", num_envs=1)
env = gym.make("LeIsaac-SO101-CustomTask-v0", cfg=env_cfg).unwrapped

jnames = list(env.scene["robot"].data.joint_names)
skeys = list(env.scene.sensors.keys())
obs, _ = env.reset()
okeys = {k: list(v.shape) for k, v in obs["policy"].items() if hasattr(v, "shape")}

data = {"joint_names": jnames, "sensor_keys": skeys, "obs_shapes": okeys}
print("INSPECT_OUTPUT_START")
print(json.dumps(data))
print("INSPECT_OUTPUT_END")

env.close()
app.close()
