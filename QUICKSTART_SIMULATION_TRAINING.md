# 🚀 Quickstart Guide: Training SO-101 Policies in Simulation (RoboMamba-DINO-ACT)

This guide provides a minimal, step-by-step walkthrough to set up, collect data, train, and evaluate **RoboMamba-DINO-ACT** (and baseline ACT) policies on the **SO-101 6-DoF robotic arm** inside **NVIDIA Isaac Sim (LeIsaac)**.

---

## 🛠️ 1. Prerequisites & Environment Setup

### A. Conda Environment
Ensure NVIDIA Isaac Sim (v4.2 / v4.5) is installed on your Linux workstation with an NVIDIA RTX GPU:

```bash
# 1. Clone/Navigate to the repository
cd /path/to/RoboMamba_DINO_ACT_Policy

# 2. Activate Isaac Sim Python environment
conda activate issac_sim_env
# (or the primary LeRobot environment for standalone policy training):
# conda activate lerobot_libero
```

### B. Dependencies
All model and dataset requirements are standard PyTorch and Hugging Face packages:
```bash
pip install torch torchvision timm mamba-ssm einops wandb datasets huggingface_hub
```

---

## 🎮 2. Simulation Environment (LeIsaac SO-101)

The tabletop pick-and-place environment (`LeIsaac-SO101-CustomTask-v0`) runs with PhysX 5.0 contact dynamics and dual RGB cameras (Side camera + Wrist camera).

### Test the Simulation Environment (GUI Mode):
```bash
cd leisaac
# Launch a test episode to verify Isaac Sim asset loading & physics:
bash run_robomamba_dino_act.sh --eval_rounds 1
```

---

## 🏋️ 3. Training the Policy from Scratch

The policy architecture consists of:
1. **Frozen DINOv2 (ViT-B/14)** with $4 \times 4$ spatial adaptive average pooling.
2. **6-Layer Mamba SSM Encoder** ($d_{\text{model}} = 512, d_{\text{state}} = 16$).
3. **CVAE + ACT Transformer Decoder** predicting $K=16$ chunked joint action trajectories.

### Launch Training:
```bash
cd /path/to/RoboMamba_DINO_ACT_Policy

# Method 1: Via shell script
bash train.sh

# Method 2: Via Python directly
python train.py --config config.yaml
```

### Key Training Hyperparameters (`config.yaml`):
* **Dataset / Hugging Face ID:** `Gogul99/RoboMamba_DINO_ACT_Policy` (or local HDF5/LeRobot dataset)
* **Batch Size:** `32` (or `64` depending on GPU VRAM)
* **Learning Rate ($\alpha$):** `1e-4` (AdamW with Cosine Annealing schedule)
* **Weight Decay ($\lambda$):** `1e-4`
* **CVAE KL Loss Weight ($\beta$):** `10.0`
* **Target Steps:** `200,000` steps

Training metrics (L1 action loss, KL divergence, validation curves) automatically log to Weights & Biases (`wandb`). Checkpoints are saved to `checkpoints/`.

---

## 📊 4. Evaluating Policies in Simulation (150-Trial Benchmark)

To benchmark the trained policy across multi-seed randomized initial object positions (seeds `42`, `100`, `2024`):

```bash
cd leisaac

# 1. Run RoboMamba-DINO-ACT (50 trials per seed, Headless GPU mode):
bash run_latency_benchmark.sh --policy robomamba-dino --trials 50 --seed 42 --headless

# 2. Run Baseline ACT for comparison:
bash run_latency_benchmark.sh --policy act --trials 50 --seed 42 --headless

# 3. Run both across all seeds automatically:
bash run_latency_benchmark.sh --policy all --trials 50 --seed 42 --headless
bash run_latency_benchmark.sh --policy all --trials 50 --seed 100 --headless
bash run_latency_benchmark.sh --policy all --trials 50 --seed 2024 --headless
```

Results (success rate, milestone breakdowns, end-to-end compute latency, and trajectory duration) are automatically compiled and exported into `benchmark_results/`.

---

## ⚡ 5. Standalone GPU Latency Profiling

To profile the forward-pass compute latency (DINOv2 + Mamba SSM vs. ResNet + Transformer) using PyTorch CUDA events:
```bash
python scripts/benchmark_latency_reproducible.py --device cuda --num_runs 100
```
