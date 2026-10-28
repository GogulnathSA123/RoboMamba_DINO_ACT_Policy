import os
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.hf_dataset import DINOv2LeRobotDataset
from models.policy import DINOv2RoboMambaACTPolicy


def compute_kl_divergence(mu, logvar):
    """Compute KL Divergence for CVAE: KL(N(mu, sigma) || N(0, 1))."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()


def create_action_trajectory_plot(gt_action, pred_action, max_dims=6):
    """
    Generate a multi-panel trajectory plot comparing Ground Truth vs. Predicted Action Chunks.
    Args:
        gt_action: [chunk_size, action_dim] numpy array
        pred_action: [chunk_size, action_dim] numpy array
        max_dims: Number of action dimensions to display in the plot
    Returns:
        matplotlib Figure object
    """
    chunk_size, action_dim = gt_action.shape
    dims_to_plot = min(max_dims, action_dim)
    rows = (dims_to_plot + 2) // 3
    cols = min(dims_to_plot, 3)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=120)
    axes = np.atleast_1d(axes).flatten()

    timesteps = np.arange(chunk_size)
    for i in range(dims_to_plot):
        ax = axes[i]
        ax.plot(timesteps, gt_action[:, i], label="Ground Truth", color="#1f77b4", linewidth=2.0)
        ax.plot(timesteps, pred_action[:, i], label="Predicted", color="#d62728", linestyle="--", linewidth=2.0)
        ax.set_title(f"Action Dim {i}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Chunk Step", fontsize=8)
        ax.set_ylabel("Normalized Action", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused subplots if any
    for j in range(dims_to_plot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def create_dimension_error_bar_plot(per_dim_l1):
    """
    Generate a bar chart showing L1 reconstruction error across all individual action dimensions.
    Args:
        per_dim_l1: [action_dim] numpy array of average L1 errors
    Returns:
        matplotlib Figure object
    """
    action_dim = len(per_dim_l1)
    fig, ax = plt.subplots(figsize=(max(8, action_dim * 0.4), 4.0), dpi=120)
    indices = np.arange(action_dim)

    bars = ax.bar(indices, per_dim_l1, color="#2ca02c", edgecolor="#1b611b", alpha=0.85)
    ax.set_title("Per-Dimension Action L1 Reconstruction Error", fontsize=11, fontweight="bold")
    ax.set_xlabel("Action Dimension Index", fontsize=9)
    ax.set_ylabel("Mean L1 Error", fontsize=9)
    ax.set_xticks(indices)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate max error bar
    max_idx = np.argmax(per_dim_l1)
    ax.annotate(
        f"Peak: Dim {max_idx} ({per_dim_l1[max_idx]:.3f})",
        xy=(max_idx, per_dim_l1[max_idx]),
        xytext=(max_idx, per_dim_l1[max_idx] * 1.15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=4),
        fontsize=8,
        ha="center",
    )

    plt.tight_layout()
    return fig


def train():
    # Load configuration
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True
    print(f"=== Starting DINOv2 + RoboMamba + ACT Training on {device} (AMP: {use_amp}) ===")

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Initialize Weights & Biases
    wandb_run = wandb.init(
        project=cfg.get("wandb", {}).get("project", "RoboMamba_DINO_ACT_Policy"),
        config=cfg,
        name=f"run_{cfg['model']['dino_backbone']}_mamba_{cfg['model']['n_mamba_layers']}_chunk{cfg['model']['chunk_size']}",
    )

    # Define W&B metric steps for structured dashboard charts
    wandb.define_metric("step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("plots/*", step_metric="epoch")

    # 1. Dataset & DataLoader (yields raw, unnormalized data)
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

    # Determine checkpoint directory
    keep_local = cfg.get("hub", {}).get("keep_local_checkpoints", True)
    if keep_local:
        save_dir = cfg["training"]["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        temp_dir_obj = None
    else:
        import tempfile
        temp_dir_obj = tempfile.TemporaryDirectory()
        save_dir = temp_dir_obj.name
        print(f"Direct-to-Hub Mode Enabled: Saving temporary checkpoints to '{save_dir}'.")

    # Save dataset statistics
    stats_path = os.path.join(save_dir, "dataset_stats.pth")
    torch.save(dataset.stats, stats_path)
    if keep_local:
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
        latent_dim=cfg["model"].get("latent_dim", 32),
    ).to(device)

    # Initialize policy normalization buffers with dataset statistics
    model.state_mean.copy_(dataset.stats["state_mean"])
    model.state_std.copy_(dataset.stats["state_std"])
    model.action_mean.copy_(dataset.stats["action_mean"])
    model.action_std.copy_(dataset.stats["action_std"])
    print("Policy normalization buffers initialized successfully.")

    # Watch model gradients periodically
    wandb.watch(model, log="gradients", log_freq=200)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["num_epochs"], eta_min=1e-6
    )

    l1_loss_fn = nn.L1Loss()
    l1_per_dim_fn = nn.L1Loss(reduction="none")
    kl_weight = float(cfg["model"]["kl_weight"])
    best_loss = float("inf")
    plot_freq = cfg.get("training", {}).get("plot_freq_epochs", 20)

    # AMP GradScaler (bfloat16 does not strictly require scaling on Blackwell, but scaler handles fp16/bf16 gracefully)
    precision_type = cfg.get("training", {}).get("precision", "bfloat16")
    amp_dtype = torch.bfloat16 if (precision_type == "bfloat16" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # 4. Training Loop
    global_step = 0
    try:
        for epoch in range(1, cfg["training"]["num_epochs"] + 1):
            model.train()
            total_epoch_loss = 0.0
            total_l1_loss = 0.0
            total_kl_loss = 0.0
            epoch_dim_l1 = torch.zeros(cfg["model"]["action_dim"], device=device)
            num_batches = len(dataloader)

            # For saving a representative sample to plot
            sample_gt_action = None
            sample_pred_action = None

            pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg['training']['num_epochs']}")
            for batch in pbar:
                step_start_time = time.time()
                imgs = batch["images"].to(device)       # [B, N_cams, 3, 224, 224]
                state = batch["state"].to(device)       # [B, state_dim]
                actions = batch["action"].to(device)   # [B, chunk_size, action_dim]

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    pred_actions_norm, mu, logvar = model(imgs, state, actions)

                    # Normalize target actions manually for computing L1 loss
                    actions_norm = (actions - model.action_mean) / model.action_std

                    # Reconstruction loss
                    l1_loss = l1_loss_fn(pred_actions_norm, actions_norm)
                    kl_loss = compute_kl_divergence(mu, logvar)
                    weighted_kl = kl_weight * kl_loss

                    loss = l1_loss + weighted_kl

                # Backward pass
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                step_duration = max(time.time() - step_start_time, 1e-5)
                throughput_fps = imgs.shape[0] / step_duration

                # Compute per-dimension error on CPU for plotting
                with torch.no_grad():
                    per_dim_batch = l1_per_dim_fn(pred_actions_norm, actions_norm).mean(dim=[0, 1])
                    epoch_dim_l1 += per_dim_batch

                    # Retain last batch sample for visual trajectory plot
                    if sample_gt_action is None:
                        sample_gt_action = actions_norm[0].detach().cpu().numpy()
                        sample_pred_action = pred_actions_norm[0].detach().cpu().numpy()

                total_epoch_loss += loss.item()
                total_l1_loss += l1_loss.item()
                total_kl_loss += kl_loss.item()

                # Step-level W&B Logging
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/step_l1_loss": l1_loss.item(),
                    "train/step_kl_loss": kl_loss.item(),
                    "train/step_weighted_kl": weighted_kl.item(),
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/throughput_fps": throughput_fps,
                    "step": global_step,
                })

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "l1": f"{l1_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}",
                    "fps": f"{throughput_fps:.1f}",
                })
                global_step += 1

            scheduler.step()

            avg_loss = total_epoch_loss / num_batches
            avg_l1 = total_l1_loss / num_batches
            avg_kl = total_kl_loss / num_batches
            avg_dim_l1 = (epoch_dim_l1 / num_batches).detach().cpu().numpy()

            # Epoch-level scalar metrics
            epoch_logs = {
                "epoch/loss": avg_loss,
                "epoch/l1_loss": avg_l1,
                "epoch/kl_loss": avg_kl,
                "epoch/weighted_kl": kl_weight * avg_kl,
                "epoch/learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            # Periodic W&B Plot Generation (every plot_freq epochs and on epoch 1)
            if epoch == 1 or epoch % plot_freq == 0 or epoch == cfg["training"]["num_epochs"]:
                print(f"Generating W&B visual trajectory and per-dimension plots for Epoch {epoch}...")

                # 1. Action Trajectory Plot (Ground Truth vs. Predicted Chunk)
                if sample_gt_action is not None and sample_pred_action is not None:
                    fig_traj = create_action_trajectory_plot(sample_gt_action, sample_pred_action, max_dims=6)
                    epoch_logs["plots/action_trajectories"] = wandb.Image(fig_traj)
                    plt.close(fig_traj)

                # 2. Per-Dimension L1 Error Bar Chart
                fig_bar = create_dimension_error_bar_plot(avg_dim_l1)
                epoch_logs["plots/per_dimension_l1_error"] = wandb.Image(fig_bar)
                plt.close(fig_bar)

            # Log epoch metrics & plots
            wandb.log(epoch_logs)

            print(f"Epoch {epoch} Summary | Total Loss: {avg_loss:.4f} | L1: {avg_l1:.4f} | KL: {avg_kl:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Save Best Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "config": cfg,
                }, best_path)
                if keep_local:
                    print(f"★ Saved new best model checkpoint to '{best_path}' (Loss: {best_loss:.4f})")
                wandb.save(best_path)
                wandb.summary["best_loss"] = best_loss
                wandb.summary["best_epoch"] = epoch

            # Save Latest Checkpoint
            latest_path = os.path.join(save_dir, "latest_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
            }, latest_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Preserving checkpoints...")

    # Set final W&B summary metrics
    wandb.summary["final_loss"] = avg_loss if 'avg_loss' in locals() else None
    wandb.summary["final_l1_loss"] = avg_l1 if 'avg_l1' in locals() else None
    wandb.summary["total_training_steps"] = global_step

    # 5. Push Checkpoints to Hugging Face Hub (if configured)
    push_checkpoints_to_hub(cfg, save_dir)

    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()

    wandb.finish()
    print("=== Training Complete ===")


def push_checkpoints_to_hub(cfg, save_dir):
    hub_cfg = cfg.get("hub", {})
    if not hub_cfg.get("push_to_hub", False):
        return

    repo_id = hub_cfg.get("repo_id")
    if not repo_id:
        print("Warning: hub.push_to_hub is True but no repo_id specified. Skipping upload.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"\nUploading checkpoints to Hugging Face Hub ('{repo_id}')...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=hub_cfg.get("private", False))

        files_to_upload = {
            "best_model.pt": os.path.join(save_dir, "best_model.pt"),
            "latest_model.pt": os.path.join(save_dir, "latest_model.pt"),
            "dataset_stats.pth": os.path.join(save_dir, "dataset_stats.pth"),
            "config.yaml": os.path.join(os.path.dirname(__file__), "config.yaml"),
        }

        for path_in_repo, local_path in files_to_upload.items():
            if os.path.exists(local_path):
                print(f" -> Uploading '{local_path}' as '{path_in_repo}'...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model",
                )
        print(f"Checkpoints successfully uploaded: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Warning: Failed to upload checkpoints: {e}")


if __name__ == "__main__":
    train()
