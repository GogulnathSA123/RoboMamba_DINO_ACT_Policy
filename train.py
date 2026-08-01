import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.hf_dataset import DINOv2LeRobotDataset
from models.policy import DINOv2RoboMambaACTPolicy

def compute_kl_divergence(mu, logvar):
    """Compute KL Divergence for CVAE: KL(N(mu, sigma) || N(0, 1))."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()

def train():
    # Load configuration
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"=== Starting DINOv2 + RoboMamba + ACT Training on {device} ===")

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = DINOv2LeRobotDataset(
        repo_id=cfg["dataset"]["repo_id"],
        camera_keys=cfg["dataset"]["cameras"],
        chunk_size=cfg["model"]["chunk_size"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # Save dataset statistics for inference normalization
    stats_path = os.path.join(save_dir, "dataset_stats.pth")
    torch.save(dataset.stats, stats_path)
    print(f"Dataset normalization stats saved to '{stats_path}'.")

    # 2. Model Instantiation
    model = DINOv2RoboMambaACTPolicy(
        dino_model=cfg["model"]["dino_backbone"],
        freeze_dino=cfg["model"]["freeze_dino"],
        d_model=cfg["model"]["d_model"],
        n_mamba_layers=cfg["model"]["n_mamba_layers"],
        state_dim=cfg["model"]["state_dim"],
        action_dim=cfg["model"]["action_dim"],
        chunk_size=cfg["model"]["chunk_size"],
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"]
    )

    l1_loss_fn = nn.L1Loss()
    kl_weight = float(cfg["model"]["kl_weight"])
    best_loss = float("inf")

    # 4. Training Loop
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        model.train()
        total_epoch_loss = 0.0
        total_l1_loss = 0.0
        total_kl_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['training']['num_epochs']}")
        for batch in pbar:
            imgs = batch["images"].to(device)       # [B, N_cams, 3, 224, 224]
            state = batch["state"].to(device)       # [B, state_dim]
            actions = batch["action"].to(device)   # [B, chunk_size, action_dim]

            optimizer.zero_grad()

            pred_actions, mu, logvar = model(imgs, state, actions)

            # Compute L1 Action Reconstruction Loss
            l1_loss = l1_loss_fn(pred_actions, actions)

            # Compute CVAE KL Divergence Loss
            kl_loss = compute_kl_divergence(mu, logvar)

            loss = l1_loss + kl_weight * kl_loss
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_epoch_loss += loss.item()
            total_l1_loss += l1_loss.item()
            total_kl_loss += kl_loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "l1": f"{l1_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
            })

        scheduler.step()

        avg_loss = total_epoch_loss / len(dataloader)
        avg_l1 = total_l1_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)

        print(f"Epoch {epoch} Summary | Total Loss: {avg_loss:.4f} | L1 Loss: {avg_l1:.4f} | KL Loss: {avg_kl:.4f}")

        # Save Best Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model checkpoint to '{best_path}'.")

        # Save Latest Checkpoint
        latest_path = os.path.join(save_dir, "latest_model.pt")
        torch.save(model.state_dict(), latest_path)

    print("=== Training Complete ===")

if __name__ == "__main__":
    train()
