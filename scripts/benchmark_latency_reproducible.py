#!/usr/bin/env python3
"""
Reproducible Real-Hardware Latency Benchmark for RoboMamba-DINO-ACT Policy.
Measures both:
  1. Full End-to-End Forward Pass (Dual 224x224 RGB -> DINOv2 -> Pooling -> Mamba -> ACT Decoder -> 16 Actions)
  2. Sequence Model Only (Visual Tokens + State -> Mamba SSM -> ACT Decoder)
across spatial pooling grids (2x2, 4x4, 8x8, 16x16) on the local GPU.
Logs raw timings to CSV and stdout.
"""

import os
import sys
import csv
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.policy import DINOv2RoboMambaACTPolicy

def benchmark_grid(grid_size, device, num_warmup=30, num_iters=100):
    policy = DINOv2RoboMambaACTPolicy(
        dino_model="dinov2_vitb14",
        freeze_dino=True,
        d_model=512,
        n_mamba_layers=6,
        state_dim=6,
        action_dim=6,
        chunk_size=16,
        latent_dim=32,
        pool_grid=grid_size,
    ).to(device)
    policy.eval()

    dummy_imgs = torch.randn(1, 2, 3, 224, 224, device=device)
    dummy_state = torch.randn(1, 6, device=device)

    # 1. Warmup
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(num_warmup):
            _ = policy(dummy_imgs, dummy_state)

    torch.cuda.synchronize()

    # 2. Measure Full End-to-End Latency
    e2e_timings = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            _ = policy(dummy_imgs, dummy_state)
        end.record()
        torch.cuda.synchronize()
        e2e_timings.append(start.elapsed_time(end))

    mean_e2e = float(np.mean(e2e_timings))
    std_e2e = float(np.std(e2e_timings))

    # 3. Measure Sequence Model Only (Tokens -> Mamba -> ACT)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        vis_tokens = policy.vision(dummy_imgs)
        state_norm = (dummy_state - policy.state_mean) / policy.state_std
        state_token = policy.state_encoder(state_norm).unsqueeze(1)
        seq_tokens = torch.cat([vis_tokens, state_token], dim=1)
        z = torch.zeros(1, 32, device=device)

    ssm_timings = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            memory = policy.mamba(seq_tokens)
            _ = policy.decoder(memory, z, state_norm)
        end.record()
        torch.cuda.synchronize()
        ssm_timings.append(start.elapsed_time(end))

    mean_ssm = float(np.mean(ssm_timings))
    std_ssm = float(np.std(ssm_timings))

    return {
        "grid": grid_size,
        "tokens": grid_size * grid_size * 2,
        "e2e_mean_ms": mean_e2e,
        "e2e_std_ms": std_e2e,
        "e2e_hz": 1000.0 / mean_e2e,
        "ssm_mean_ms": mean_ssm,
        "ssm_std_ms": std_ssm,
        "ssm_hz": 1000.0 / mean_ssm,
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    out_csv = "/home/zozo/RoboMamba_DINO_ACT_Policy/experiments/hardware_latency_benchmark.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    
    print(f" REPRODUCIBLE HARDWARE LATENCY BENCHMARK ON: {gpu_name}")
    print("Method: FP16 Autocast, 30 Warmup + 100 Timed CUDA Event Iterations per Grid")
    

    grids = [2, 4, 8, 16]
    records = []

    for g in grids:
        print(f"Benchmarking Grid {g}x{g} ({g*g*2} total visual tokens)...")
        res = benchmark_grid(g, device)
        records.append(res)
        print(f"  -> Full End-to-End: {res['e2e_mean_ms']:6.2f} ± {res['e2e_std_ms']:4.2f} ms ({res['e2e_hz']:5.1f} Hz)")
        print(f"  -> Sequence (Mamba+ACT): {res['ssm_mean_ms']:6.2f} ± {res['ssm_std_ms']:4.2f} ms ({res['ssm_hz']:5.1f} Hz)\n")

    # Save to CSV
    keys = ["grid", "tokens", "e2e_mean_ms", "e2e_std_ms", "e2e_hz", "ssm_mean_ms", "ssm_std_ms", "ssm_hz"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    print(f" Benchmark results saved directly to: {out_csv}")

if __name__ == "__main__":
    main()
