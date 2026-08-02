import os
import yaml
import torch
import numpy as np

from models.policy import DINOv2RoboMambaACTPolicy

class TemporalEnsembleBuffer:
    """Temporal ensembling buffer for smooth action trajectory execution."""

    def __init__(self, action_dim=6, chunk_size=16, k=0.01):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.k = k
        self.predictions = {}

    def reset(self):
        self.predictions.clear()

    def update(self, current_t, pred_chunk):
        for offset in range(self.chunk_size):
            t = current_t + offset
            if t not in self.predictions:
                self.predictions[t] = []
            self.predictions[t].append((current_t, pred_chunk[offset]))

    def get_action(self, current_t):
        if current_t not in self.predictions or len(self.predictions[current_t]) == 0:
            return torch.zeros(self.action_dim)

        actions_at_t = self.predictions[current_t]
        weights = []
        actions = []

        for start_t, act in actions_at_t:
            age = current_t - start_t
            w = np.exp(-self.k * age)
            weights.append(w)
            actions.append(act)

        weights = np.array(weights)
        weights /= weights.sum()

        actions_tensor = torch.stack(actions)
        weighted_action = (actions_tensor * torch.tensor(weights, device=actions_tensor.device).unsqueeze(1)).sum(dim=0)
        return weighted_action


class DINOv2PolicyConnector:
    """Inference Connector for evaluating DINOv2 + RoboMamba + ACT in Isaac Sim or Hardware."""

    def __init__(self, checkpoint_path=None, cfg_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        base_dir = os.path.dirname(__file__)

        # Set fallback local paths
        if cfg_path is None:
            cfg_path = os.path.join(base_dir, "config.yaml")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(base_dir, "checkpoints", "best_model.pt")

        # Check if checkpoint_path is a Hugging Face repository ID (e.g. "Gogul99/Mamaba_Dino_ACT")
        if checkpoint_path and not os.path.exists(checkpoint_path) and "/" in checkpoint_path:
            hf_repo_id = checkpoint_path
            print(f"'{hf_repo_id}' is not a local path. Attempting to download from Hugging Face Hub...")
            try:
                from huggingface_hub import hf_hub_download
                # Download model weights
                local_checkpoint = hf_hub_download(repo_id=hf_repo_id, filename="best_model.pt")
                checkpoint_path = local_checkpoint
                print(f"Successfully downloaded checkpoint to '{checkpoint_path}'")
                
                # Attempt to download config.yaml from repository
                try:
                    local_config = hf_hub_download(repo_id=hf_repo_id, filename="config.yaml")
                    cfg_path = local_config
                    print(f"Successfully downloaded configuration to '{cfg_path}'")
                except Exception:
                    print("config.yaml not found on Hub. Falling back to local configuration file.")
            except Exception as e:
                print(f"Error fetching from Hugging Face Hub: {e}. Reverting to local search.")

        # Load Configuration
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Build Model
        self.policy = DINOv2RoboMambaACTPolicy(
            dino_model=self.cfg["model"]["dino_backbone"],
            freeze_dino=self.cfg["model"]["freeze_dino"],
            d_model=self.cfg["model"]["d_model"],
            n_mamba_layers=self.cfg["model"]["n_mamba_layers"],
            state_dim=self.cfg["model"]["state_dim"],
            action_dim=self.cfg["model"]["action_dim"],
            chunk_size=self.cfg["model"]["chunk_size"],
        ).to(self.device)

        # Load Weights
        if os.path.exists(checkpoint_path):
            print(f"Loading policy checkpoint from '{checkpoint_path}'...")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(state_dict)
            self.policy.eval()
            print("Successfully loaded model weights and normalization buffers.")
        else:
            print(f"Warning: Checkpoint '{checkpoint_path}' not found! Policy running with random weights.")

        self.ensemble = TemporalEnsembleBuffer(
            action_dim=self.cfg["model"]["action_dim"],
            chunk_size=self.cfg["model"]["chunk_size"],
        )

    def reset(self):
        self.ensemble.reset()

    def get_action(self, current_t, raw_images, raw_state):
        """
        Input:
            current_t: Integer timestep index
            raw_images: List of camera arrays or tensors [front_img, wrist_img]
            raw_state: Array or tensor of joint positions [state_dim]
        Output:
            action: Unnormalized raw action vector [action_dim]
        """
        # Process images
        processed_imgs = []
        for img in raw_images:
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img, dtype=torch.float32, device=self.device)
            else:
                img = img.to(self.device).float()

            if img.ndim == 3 and img.shape[-1] in [3, 4]:
                img = img[..., :3].permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

            if img.shape[-2:] != (224, 224):
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
                ).squeeze(0)

            if img.max() > 1.0:
                img = img / 255.0

            # ImageNet Z-scoring
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
            img = (img - mean) / std
            processed_imgs.append(img)

        batch_imgs = torch.stack(processed_imgs, dim=0).unsqueeze(0)  # [1, N_cams, 3, 224, 224]

        # Process state
        if not isinstance(raw_state, torch.Tensor):
            raw_state = torch.tensor(raw_state, dtype=torch.float32, device=self.device)
        else:
            raw_state = raw_state.to(self.device).float()

        batch_state = raw_state.unsqueeze(0)  # [1, state_dim]

        # Model Inference (Normalizes internally and returns raw unnormalized action chunk)
        with torch.no_grad():
            pred_chunk, _, _ = self.policy(batch_imgs, batch_state)
            pred_chunk = pred_chunk.squeeze(0)  # [chunk_size, action_dim]

        # Temporal Ensembling
        self.ensemble.update(current_t, pred_chunk)
        raw_action = self.ensemble.get_action(current_t)

        return raw_action.cpu().numpy()
