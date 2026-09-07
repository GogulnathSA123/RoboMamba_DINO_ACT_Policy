# RoboMamba-DINO-ACT: Latency-Aware Visuomotor Policies for Robotic Manipulation: A Comparative Simulation and Physical Study on the SO-101 Arm

This repository contains the complete implementation, training pipeline, and evaluation benchmarks for **RoboMamba-DINO-ACT**, evaluated on the 6-DoF SO-101 robotic arm in both physical hardware deployment and NVIDIA Isaac Sim.

---

##  Repository Structure

```text
RoboMamba_DINO_ACT_Policy/
── models/                         # Neural network architecture
   ── dino_backbone.py            # Frozen DINOv2 (ViT-B/14) with 4x4 spatial pooling
   ── mamba_block.py              # 6-layer selective state-space model (SSM) encoder
   ── act_decoder.py              # Transformer action query decoder (K=16 steps)
   ── policy.py                   # Unified RoboMamba-DINO-ACT policy
── dataset/
   ── hf_dataset.py               # Hugging Face / LeRobot dataset loading & caching
── leisaac/                        # Isaac Sim simulation environment & tasks
    ── source/                     # LeIsaac custom task environments & MDP configs
    ─ scripts/evaluation/         # Simulation benchmark & latency evaluation
    ── run_robomamba_dino_act.sh   # Single-episode / test execution launcher
    ── run_latency_benchmark.sh    # Multi-seed 150-trial simulation benchmark runner
── scripts/                        # Analysis, profiling & hyperparameter sweeps
   ── benchmark_latency_reproducible.py # Modular CUDA PyTorch profiling harness
── train.py                        # Model training pipeline
── train.sh                        # Training launcher script
── eval_leisaac.py                 # Isaac Sim policy connector
── eval_real_so101.py              # Real robot SO-101 evaluation script
── evaluate_realworld_benchmark.py # Live physical 10-trial benchmark harness
── run_realworld_benchmark.sh      # Physical evaluation launcher
── policy_server.py                # Asynchronous policy server for serial UART
── robot_client.py                 # SO-101 robot arm control client
── monitor_joints.py               # Hardware joint telemetry monitor
── calibrate.sh                    # Feetech servo calibration script
── config.yaml                     # Model hyperparameters & training configs
```

---

##  1. Real-World Physical Robot Benchmark (SO-101 Arm)

The physical benchmark evaluates closed-loop pick-and-place success and per-step hardware latency on the 6-DoF SO-101 arm using dual USB cameras (`CAM_RIGHT=4`, `CAM_WRIST=6`) at 30 Hz.

### Environment Setup:
```bash
conda activate lerobot_libero
cd /home/zozo/RoboMamba_DINO_ACT_Policy
```

### A. Run RoboMamba-DINO-ACT on the Physical Arm (10 Trials):
```bash
bash run_realworld_benchmark.sh robomamba 10 real
```
 Or invoke Python directly:
```bash
python evaluate_realworld_benchmark.py \
    --policy robomamba \
    --trials 10 \
    --robot_port /dev/ttyACM0 \
    --cam_right 4 \
    --cam_wrist 6 \
    --hz 30
```

### B. Run Canonical ACT Baseline on the Physical Arm (10 Trials):
```bash
bash run_realworld_benchmark.sh act 10 real
```

### C. Mock Dry-Run (Without Physical Hardware Connected):
To verify video capture, policy inference, and pipeline execution without connecting the robot arm:
```bash
bash run_realworld_benchmark.sh robomamba 3 mock
```

---

##  2. Simulation Benchmark (NVIDIA Isaac Sim)

Simulation experiments run on the `LeIsaac-SO101-CustomTask-v0` tabletop environment under PhysX 5.0 dynamics across 3 independent random seeds (`42`, `100`, `2024`) with 50 trials per seed (150 trials total per model).

### A. Full Multi-Seed Benchmark (Both Policies):
Run the automated benchmark runner from within `leisaac/`:
```bash
cd /home/zozo/RoboMamba_DINO_ACT_Policy/leisaac

# Run Seed 42 (50 trials for ACT + 50 trials for RoboMamba-DINO-ACT)
bash run_latency_benchmark.sh --policy all --trials 50 --seed 42 --headless

# Run Seed 100
bash run_latency_benchmark.sh --policy all --trials 50 --seed 100 --headless

# Run Seed 2024
bash run_latency_benchmark.sh --policy all --trials 50 --seed 2024 --headless
```

### B. Run a Specific Policy Individually:
* **RoboMamba-DINO-ACT in Isaac Sim:**
  ```bash
  bash run_latency_benchmark.sh --policy robomamba-dino --trials 50 --seed 42 --headless
  ```
* **ACT Baseline in Isaac Sim:**
  ```bash
  bash run_latency_benchmark.sh --policy act --trials 50 --seed 42 --headless
  ```

### C. Interactive GUI Rollout (Single Episode Visualization):
To view the simulation rollout in the Isaac Sim GUI window:
```bash
cd /home/zozo/RoboMamba_DINO_ACT_Policy/leisaac
bash run_robomamba_dino_act.sh --eval_rounds 1
```

---

## 3. Modular GPU Inference Profiling

To profile the standalone forward-pass latency of the DINOv2 vision backbone, 4x4 spatial pooling, and 6-layer Mamba SSM encoder on an NVIDIA RTX 3080/5090 using PyTorch CUDA events (`torch.cuda.Event`):

```bash
cd /home/zozo/RoboMamba_DINO_ACT_Policy
python scripts/benchmark_latency_reproducible.py --device cuda --num_runs 100
```

---

##  4. Training Pipeline

To train RoboMamba-DINO-ACT from scratch on teleoperated demonstration datasets:

```bash
cd /home/zozo/RoboMamba_DINO_ACT_Policy
bash train.sh
# or:
python train.py --config config.yaml
```
Training telemetry, L1 action reconstruction loss, and CVAE KL divergence regularization will log to Weights & Biases (`wandb`). Checkpoints save automatically to `checkpoints/`.
