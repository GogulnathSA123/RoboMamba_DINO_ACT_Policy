"""Script to run a leisaac inference with leisaac in the simulation."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="gr00tn1.5",
    help="Type of policy to use. support gr00tn1.5, gr00tn1.6, lerobot-<model_type>, openpi",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time

import carb
import gymnasium as gym
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)

import leisaac  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "gr00tn1.6", "lerobot", "openpi"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00tServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif args_cli.policy_type == "gr00tn1.6":
        from isaaclab.sensors import Camera
        from leisaac.policy import Gr00t16ServicePolicyClient

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00t16ServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )

    elif "lerobot" in args_cli.policy_type:
        from isaaclab.sensors import Camera, TiledCamera
        from leisaac.policy import LeRobotServicePolicyClient

        model_type = "lerobot"

        policy_type = args_cli.policy_type.split("-")[1]
        camera_keys = ["wrist", "front"]

        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={
                key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, (Camera, TiledCamera)) and key in camera_keys
            },
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type.lower() in ["act", "lerobot-act", "act_isaacsim", "act-direct"]:
        import os
        from lerobot.policies.act.modeling_act import ACTPolicy
        from huggingface_hub import snapshot_download
        from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
        import numpy as np

        model_type = "lerobot"

        class DirectACTPolicyAdapter:
            def __init__(self, checkpoint_path, device="cuda"):
                from safetensors.torch import load_file
                self.device = torch.device(device if torch.cuda.is_available() else "cpu")
                model_dir = snapshot_download(checkpoint_path) if "/" in checkpoint_path and not os.path.exists(checkpoint_path) else checkpoint_path
                print(f"[Policy] Loading ACTPolicy from '{model_dir}'...")
                self.policy = ACTPolicy.from_pretrained(model_dir).to(self.device)
                self.policy.eval()
                self.step_t = 0

                # Load normalization statistics from safetensors
                norm_file = os.path.join(model_dir, "policy_preprocessor_step_3_normalizer_processor.safetensors")
                unnorm_file = os.path.join(model_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")

                if os.path.exists(norm_file) and os.path.exists(unnorm_file):
                    norm = load_file(norm_file)
                    unnorm = load_file(unnorm_file)
                    self.state_mean = norm["observation.state.mean"].to(self.device)
                    self.state_std = norm["observation.state.std"].to(self.device)
                    self.act_mean = unnorm["action.mean"].to(self.device)
                    self.act_std = unnorm["action.std"].to(self.device)
                    self.front_mean = norm["observation.images.front.mean"].to(self.device).view(1, 3, 1, 1)
                    self.front_std = norm["observation.images.front.std"].to(self.device).view(1, 3, 1, 1)
                    self.wrist_mean = norm["observation.images.wrist.mean"].to(self.device).view(1, 3, 1, 1)
                    self.wrist_std = norm["observation.images.wrist.std"].to(self.device).view(1, 3, 1, 1)
                    self.has_stats = True
                    print("[Policy] Successfully loaded normalizer and unnormalizer statistics!")
                else:
                    self.has_stats = False
                    print("[Policy] Warning: Normalization stats not found, running with raw values.")

                print("[Policy] ACTPolicy successfully loaded and initialized on GPU!")
                
            def reset(self):
                self.policy.reset()
                self.step_t = 0
                
            def get_action(self, obs_dict: dict) -> torch.Tensor:
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
                        t = img.clone().float()
                    if t.ndim == 4:
                        t = t[0]
                    if t.ndim == 3 and t.shape[-1] in [3, 4]:
                        t = t[..., :3].permute(2, 0, 1)
                    if t.max() > 1.0:
                        t = t / 255.0
                    t = torch.clamp(t, 0.0, 1.0)
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

                # Normalize inputs according to training statistics
                if self.has_stats:
                    front_t = (front_t - self.front_mean) / (self.front_std + 1e-8)
                    wrist_t = (wrist_t - self.wrist_mean) / (self.wrist_std + 1e-8)
                    state_t = (state_t - self.state_mean) / (self.state_std + 1e-8)

                batch = {
                    "observation.images.front": front_t,
                    "observation.images.wrist": wrist_t,
                    "observation.state": state_t,
                }

                with torch.no_grad():
                    pred_action = self.policy.select_action(batch)
                    # Unnormalize output action according to dataset action distribution
                    if self.has_stats:
                        pred_action = pred_action * self.act_std + self.act_mean
                    pred_motor = pred_action.squeeze(0).cpu().numpy()

                self.step_t += 1
                if self.step_t % 15 == 0:
                    print(f"[ACT Debug] Step {self.step_t} | Pred Motor Deg: {np.round(pred_motor, 2)} | State: {np.round(motor_state, 2)}", flush=True)

                raw_gripper = pred_motor[5]
                pred_motor[5] = np.clip(raw_gripper, 0.0, 60.76)

                target_rad = convert_lerobot_action_to_leisaac(pred_motor[None, :])[0]
                action_tensor = torch.tensor(target_rad, device=self.device, dtype=torch.float32)
                return action_tensor.unsqueeze(0).unsqueeze(0)

        policy = DirectACTPolicyAdapter(
            checkpoint_path=args_cli.policy_checkpoint_path if args_cli.policy_checkpoint_path else "Gogul99/act_isaacsim",
            device=args_cli.device
        )
    elif any(k in args_cli.policy_type.lower() for k in ["robomamba-dino", "robomamba_dino", "robomamba_dino_act"]):
        import sys
        if "/home/zozo/RoboMamba_DINO_ACT_Policy" not in sys.path:
            sys.path.insert(0, "/home/zozo/RoboMamba_DINO_ACT_Policy")
        from eval_leisaac import DINOv2PolicyConnector
        from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
        import numpy as np

        model_type = "lerobot"

        class DirectDINOv2PolicyAdapter:
            def __init__(self, checkpoint_path, device="cuda"):
                cfg_path = "/home/zozo/RoboMamba_DINO_ACT_Policy/config.yaml"
                self.connector = DINOv2PolicyConnector(checkpoint_path=checkpoint_path, cfg_path=cfg_path, device=device)
                self.step_t = 0
                self.device = device
                
            def reset(self):
                self.connector.reset()
                self.step_t = 0
                
            def get_action(self, obs_dict: dict) -> torch.Tensor:
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
                processed_cams = []
                for img in raw_images:
                    if img is not None:
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        if img.ndim == 4:
                            img = img[0]
                        if img.ndim == 3 and img.shape[-1] == 4:
                            img = img[..., :3]
                        processed_cams.append(img)

                if len(processed_cams) < 2:
                    while len(processed_cams) < 2:
                        processed_cams.append(np.zeros((224, 224, 3), dtype=np.uint8))

                if "joint_pos" in obs_dict:
                    raw_state_rad = obs_dict["joint_pos"]
                    if isinstance(raw_state_rad, torch.Tensor):
                        raw_state_rad = raw_state_rad.cpu().numpy()
                    if raw_state_rad.ndim == 1:
                        raw_state_rad = raw_state_rad[None, :]
                    motor_state = convert_leisaac_action_to_lerobot(raw_state_rad)[0]
                else:
                    motor_state = np.zeros(6)

                pred_motor_action = self.connector.get_action(self.step_t, processed_cams, motor_state)
                self.step_t += 1
                
                raw_gripper = pred_motor_action[5]
                print(f"[Policy Debug] Step {self.step_t} | Pred: {np.round(pred_motor_action, 3)} | State: {np.round(motor_state, 3)}", flush=True)

                # Direct gripper command from policy output
                pred_motor_action[5] = np.clip(raw_gripper, 0.0, 60.76)

                target_rad = convert_lerobot_action_to_leisaac(pred_motor_action[None, :])[0]
                action_tensor = torch.tensor(target_rad, device=self.device, dtype=torch.float32)
                return action_tensor.unsqueeze(0).unsqueeze(0)

        policy = DirectDINOv2PolicyAdapter(
            checkpoint_path=args_cli.policy_checkpoint_path if args_cli.policy_checkpoint_path else "Gogul99/RoboMamba_DINO_ACT_Policy",
            device=args_cli.device
        )
    elif args_cli.policy_type == "openpi":
        from isaaclab.sensors import Camera
        from leisaac.policy import OpenPIServicePolicyClient

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    # record the results
    success_count, episode_count = 0, 1

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"[Evaluation] Evaluating episode {episode_count}...")
        success, time_out = False, False
        if hasattr(policy, "reset"):
            policy.reset()
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    if hasattr(policy, "reset"):
                        policy.reset()
                    episode_count += 1
                    break

                obs_dict = preprocess_obs_dict(obs_dict["policy"], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
            if success:
                print(f"[Evaluation] Episode {episode_count} is successful!")
                episode_count += 1
                success_count += 1
                if hasattr(policy, "reset"):
                    policy.reset()
                break
            if time_out:
                print(f"[Evaluation] Episode {episode_count} timed out!")
                episode_count += 1
                if hasattr(policy, "reset"):
                    policy.reset()
                break
        print(
            f"[Evaluation] now success rate: {success_count / (episode_count - 1)} "
            f" [{success_count}/{episode_count - 1}]"
        )
    print(
        f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
        f" [{success_count}/{max_episode_count}]"
    )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
