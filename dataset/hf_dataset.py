import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class DINOv2LeRobotDataset(Dataset):
    """
    Dataset wrapper for LeRobot datasets formatted for DINOv2 + RoboMamba + ACT training.

    Features:
    - Loads multi-camera images (640x480x3) and resizes to 224x224 for DINOv2
    - ImageNet z-scoring normalization on images
    - Per-channel state/action normalization (z-score)
    - Episode-aware action chunking (pads at episode boundaries instead of crossing)
    """

    def __init__(self, repo_id="Gogul99/300_episode", camera_keys=None, chunk_size=16):
        super().__init__()
        self.repo_id = repo_id
        self.chunk_size = chunk_size

        # Default camera keys if not provided
        if camera_keys is None:
            self.camera_keys = ["observation.images.front", "observation.images.wrist"]
        else:
            self.camera_keys = camera_keys

        print(f"Loading LeRobot dataset '{repo_id}' from HuggingFace...")
        self.dataset = LeRobotDataset(repo_id)

        # Build episode boundary index for safe action chunking
        self.episode_ends = self._build_episode_index()

        # Image transforms: resize 640x480 -> 224x224, then ImageNet z-scoring
        self.image_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.stats = self._compute_stats()

    def _build_episode_index(self):
        """Build a mapping from each sample index to its episode's last valid index."""
        episode_ends = {}
        try:
            # LeRobot v2 datasets have episode_index metadata
            episodes = self.dataset.hf_dataset["episode_index"]
            current_ep = episodes[0]
            ep_start = 0
            for i in range(len(episodes)):
                if i == len(episodes) - 1 or episodes[i + 1] != current_ep:
                    for j in range(ep_start, i + 1):
                        episode_ends[j] = i
                    if i < len(episodes) - 1:
                        current_ep = episodes[i + 1]
                        ep_start = i + 1
        except (AttributeError, KeyError):
            # Fallback: treat entire dataset as single episode
            for i in range(len(self.dataset)):
                episode_ends[i] = len(self.dataset) - 1

        print(f"Built episode index: {len(set(episode_ends.values()))} episodes detected.")
        return episode_ends

    def _compute_stats(self):
        """Compute state and action mean/std statistics for normalization."""
        all_states = []
        all_actions = []

        print("Computing state/action statistics across dataset...")
        for item in self.dataset:
            all_states.append(item["observation.state"])
            all_actions.append(item["action"])

        states_tensor = torch.stack([
            s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32)
            for s in all_states
        ])
        actions_tensor = torch.stack([
            a if isinstance(a, torch.Tensor) else torch.tensor(a, dtype=torch.float32)
            for a in all_actions
        ])

        stats = {
            "state_mean": states_tensor.mean(dim=0),
            "state_std": torch.clamp(states_tensor.std(dim=0), min=1e-5),
            "action_mean": actions_tensor.mean(dim=0),
            "action_std": torch.clamp(actions_tensor.std(dim=0), min=1e-5),
        }
        print(f"State mean: {stats['state_mean']}")
        print(f"Action std: {stats['action_std']}")
        return stats

    def normalize_state(self, state):
        return (state - self.stats["state_mean"]) / self.stats["state_std"]

    def unnormalize_action(self, action):
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # --- Process Camera Images [Num_Cameras, 3, 224, 224] ---
        processed_imgs = []
        for key in self.camera_keys:
            img = item.get(key)
            if img is None:
                # Dynamic candidate key matching
                for k in item.keys():
                    if k.endswith(key.split(".")[-1]):
                        img = item[k]
                        break

            if img is None:
                raise KeyError(f"Camera key '{key}' not found in dataset item. Available keys: {list(item.keys())}")

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # Handle [H, W, C] -> [C, H, W]
            if img.ndim == 3 and img.shape[-1] in [3, 4]:
                img = img[..., :3].permute(2, 0, 1)

            img = self.image_transform(img)
            processed_imgs.append(img)

        imgs_tensor = torch.stack(processed_imgs, dim=0)  # [N_cam, 3, 224, 224]

        # --- Extract and Normalize State ---
        state = item["observation.state"]
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        else:
            state = state.float()
        norm_state = self.normalize_state(state)

        # --- Extract Episode-Safe Action Chunk [chunk_size, action_dim] ---
        episode_end = self.episode_ends.get(idx, len(self.dataset) - 1)
        action_chunk = []

        for offset in range(self.chunk_size):
            # Clamp to episode boundary (pad with last action if beyond end)
            act_idx = min(idx + offset, episode_end)
            act = self.dataset[act_idx]["action"]
            if not isinstance(act, torch.Tensor):
                act = torch.tensor(act, dtype=torch.float32)
            else:
                act = act.float()
            norm_act = (act - self.stats["action_mean"]) / self.stats["action_std"]
            action_chunk.append(norm_act)

        action_chunk_tensor = torch.stack(action_chunk, dim=0)  # [chunk_size, action_dim]

        return {
            "images": imgs_tensor,
            "state": norm_state,
            "action": action_chunk_tensor,
        }
