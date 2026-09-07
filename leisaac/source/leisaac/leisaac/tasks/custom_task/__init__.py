import gymnasium as gym

# Original custom task environment
gym.register(
    id="LeIsaac-SO101-CustomTask-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_task_env_cfg:CustomTaskEnvCfg",
    },
)

# New custom task environment with perpendicular setup cam, swapped assets, and domain randomization

# New custom task mimic environment for dataset generation

