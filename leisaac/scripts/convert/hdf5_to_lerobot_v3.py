import argparse
import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Joint and motor limits for SO-101 Follower
SO101_FOLLOWER_USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}

SO101_FOLLOWER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

def convert_leisaac_action_to_lerobot(action: np.ndarray) -> np.ndarray:
    """Converts joints/actions from physical angles (radians) to LeRobot motor ranges (-100 to 100)."""
    is_1d = (action.ndim == 1)
    if is_1d:
        action = action[np.newaxis, :]

    processed_action = np.zeros_like(action)
    action_deg = action / np.pi * 180.0

    for idx, joint_name in enumerate(SO101_FOLLOWER_USD_JOINT_LIMITS.keys()):
        motor_limit_range = SO101_FOLLOWER_MOTOR_LIMITS[joint_name]
        joint_limit_range = SO101_FOLLOWER_USD_JOINT_LIMITS[joint_name]
        joint_range = joint_limit_range[1] - joint_limit_range[0]
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        
        # Scale to motor limit range
        joint_degree = action_deg[:, idx] - joint_limit_range[0]
        processed_action[:, idx] = (joint_degree / joint_range) * motor_range + motor_limit_range[0]

    if is_1d:
        processed_action = processed_action[0]
    return processed_action

def convert_hdf5_to_lerobot_v3(
    hdf5_path: str,
    repo_id: str,
    fps: int,
    robot_type: str,
    task_description: str,
    vcodec: str,
    push_to_hub: bool,
    only_successful: bool,
    limit_episodes: int = None,
):
    print(f"Opening HDF5 dataset: {hdf5_path}")
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
        
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError("Root 'data' group not found in HDF5 file.")
        
        data_group = f["data"]
        episode_names = list(data_group.keys())
        
        # Sort episode names to keep natural order (e.g. demo_0, demo_1, demo_2, ...)
        try:
            episode_names = sorted(episode_names, key=lambda x: int(x.split("_")[1]))
        except Exception:
            episode_names = sorted(episode_names)
            
        print(f"Found {len(episode_names)} episodes total.")
        
        # Filter episodes if only successful ones are requested
        filtered_episodes = []
        for ep_name in episode_names:
            success = data_group[ep_name].attrs.get("success", False)
            if only_successful and not success:
                continue
            filtered_episodes.append(ep_name)
            
        print(f"Episodes to convert: {len(filtered_episodes)} (only_successful={only_successful})")
        
        if limit_episodes is not None and limit_episodes > 0:
            filtered_episodes = filtered_episodes[:limit_episodes]
            print(f"Limiting to the first {limit_episodes} episodes for testing.")
        
        if len(filtered_episodes) == 0:
            print("No episodes to process. Exiting.")
            return

        # Define dataset features schema
        features = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channels"]
            },
        }
        
        print("Creating LeRobotDataset...")
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=True,
            vcodec=vcodec,
        )
        
        converted_count = 0
        for ep_name in tqdm(filtered_episodes, desc="Converting episodes"):
            ep_group = data_group[ep_name]
            
            # Extract data
            actions = ep_group["actions"][:]
            joint_pos = ep_group["obs"]["joint_pos"][:]
            front_imgs = ep_group["obs"]["front"][:]
            wrist_imgs = ep_group["obs"]["wrist"][:]
            
            num_frames = actions.shape[0]
            if num_frames < 10:
                print(f"Skipping {ep_name} because it has too few frames ({num_frames})")
                continue
                
            # Convert each frame (skipping the first 5 frames to match training script preprocessing)
            for frame_idx in range(5, num_frames):
                scaled_action = convert_leisaac_action_to_lerobot(actions[frame_idx])
                scaled_joint_pos = convert_leisaac_action_to_lerobot(joint_pos[frame_idx])
                
                frame = {
                    "action": scaled_action.astype(np.float32),
                    "observation.state": scaled_joint_pos.astype(np.float32),
                    "observation.images.front": front_imgs[frame_idx],
                    "observation.images.wrist": wrist_imgs[frame_idx],
                    "task": task_description,
                }
                
                dataset.add_frame(frame)
                
            dataset.save_episode()
            converted_count += 1
            
        print(f"Finalizing dataset with {converted_count} episodes...")
        dataset.finalize()
        print(f"Dataset successfully finalized locally at: {dataset.root}")
        
        if push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub (repo: {repo_id})...")
            dataset.push_to_hub()
            print("Successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset308.hdf5 to LeRobot Dataset v3 format.")
    parser.add_argument("--hdf5_path", type=str, default="/home/zozo/leisaac/datasets/dataset308.hdf5", help="Path to the HDF5 file.")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g. username/dataset_name).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second of the dataset.")
    parser.add_argument("--robot_type", type=str, default="so101_follower", help="Robot type name.")
    parser.add_argument("--task_description", type=str, default="pick up the red cube and place it into the box.", help="Task instruction string.")
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"], help="Video codec to use for encoding observations.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the dataset to HF Hub after creation.")
    parser.add_argument("--all_demos", action="store_true", help="If set, converts all demos instead of only successful ones.")
    parser.add_argument("--limit_episodes", type=int, default=None, help="Limit number of episodes to convert (useful for testing).")
    
    args = parser.parse_args()
    
    convert_hdf5_to_lerobot_v3(
        hdf5_path=args.hdf5_path,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        task_description=args.task_description,
        vcodec=args.vcodec,
        push_to_hub=args.push_to_hub,
        only_successful=not args.all_demos,
        limit_episodes=args.limit_episodes,
    )
