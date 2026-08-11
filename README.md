# DINOv2 + RoboMamba + ACT Policy Architecture

A state-of-the-art imitation learning policy combining Meta AI's **DINOv2** self-supervised vision backbone, **RoboMamba** State Space Models (SSM) sequence encoder, and **ACT** Action Chunking Transformer decoder for high-precision robotic manipulation.

---

## 🌟 Architecture Highlights

1. **DINOv2 Vision Backbone (`dinov2_vits14`)**:
   - Extract dense spatial patch features ($16 \times 16 = 256$ tokens per camera) pre-trained self-supervised on Meta's LVD-142M dataset.
   - Superior geometric perception and zero-shot spatial localization compared to standard supervised ResNet backbones.

2. **RoboMamba Sequence Encoder**:
   - Linear $\mathcal{O}(N)$ complexity selective state-space model for efficient spatiotemporal sequence encoding.

3. **ACT (Action Chunking Transformer)**:
   - CVAE latent action distribution encoding + Transformer query decoding predicting multi-step action trajectories ($T=16$).

---

## 📁 Repository Structure

```
RoboMamba_DINO_ACT_Policy/
├── config.yaml               # Training & model hyperparameters
├── dataset/
│   └── hf_dataset.py         # LeRobot HuggingFace dataset loader with ImageNet z-scoring & 224x224 resize
├── models/
│   ├── dino_backbone.py      # DINOv2 feature extractor with spatial patch pooling & projection
│   ├── mamba_block.py        # RoboMamba State Space Model encoder layer
│   ├── act_decoder.py        # CVAE Encoder & ACT Transformer Query Decoder
│   └── policy.py             # Full unified DINOv2RoboMambaACTPolicy module
├── train.py                  # Full PyTorch training loop with AdamW, Cosine LR, and checkpointing
├── eval_leisaac.py           # Evaluation connector for Isaac Sim & hardware execution with temporal ensembling
└── README.md
```

---

## 🚀 Quick Start

### 1. Training

Run the training loop on HuggingFace dataset (e.g. `Gogul99/300_episode`):

```bash
python train.py
```

### 2. Configuration (`config.yaml`)

Edit `config.yaml` to customize dataset, backbone, or training parameters:

```yaml
model:
  dino_backbone: "dinov2_vits14"  # Options: dinov2_vits14, dinov2_vitb14
  freeze_dino: true              # Freeze DINOv2 backbone for fast training and low memory
  d_model: 512                   # Mamba and ACT feature dimension
  chunk_size: 16                 # Action prediction horizon
```

---

## 🧪 Evaluation in Isaac Sim

To evaluate the trained checkpoint in Isaac Sim:

```bash
PYTHONPATH=.:$PYTHONPATH \
python scripts/evaluation/policy_inference.py \
    --task LeIsaac-SO101-CustomTask-v0 \
    --policy_type robomamba-act \
    --policy_checkpoint_path ./checkpoints/best_model.pt
```
