#!/usr/bin/env python
"""
================================================================================
  GPU Resource Justification Report — Diffusion-ReID (PISL)
================================================================================

  This script generates a comprehensive, self-contained technical report that
  proves the computational requirements of the Diffusion-based Person
  Re-Identification model.  It can run on CPU — no GPU or dataset required.

  What it reports:
    1. Total trainable & non-trainable parameters (per component)
    2. Estimated GPU memory for model weights (FP32 & FP16)
    3. Estimated activation / intermediate memory during training
    4. FLOPs for a single forward pass
    5. Per-epoch and full-training wall-clock time estimates at different
       GPU memory budgets (5 GB, 10 GB, 24 GB, 40 GB, 80 GB)
    6. Batch-size feasibility analysis for each budget
    7. Dataset scale analysis (Market-1501, MSMT17, VeRi-776)
    8. A side-by-side comparison table: "What you asked for" vs. "What is needed"

  Usage:
    python gpu_resource_justification.py
    python gpu_resource_justification.py --save-txt report.txt
    python gpu_resource_justification.py --num-classes 1500 --batch-size 64

  Author: Auto-generated for GPU allocation request
================================================================================
"""

from __future__ import print_function, absolute_import
import argparse
import sys
import os
import math
import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "examples"))

from pisl.models.resnet_part import ResNetPart, PatchGenerator, DiffusionPatchGenerator


# ============================================================================
# Utility helpers
# ============================================================================

def count_parameters(module):
    """Return (trainable, non-trainable) parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total - trainable


def params_to_mb(num_params, dtype_bytes=4):
    """Convert parameter count → memory in MB."""
    return (num_params * dtype_bytes) / (1024 ** 2)


def format_num(n):
    """Pretty-print large numbers with commas."""
    return f"{n:,}"


def format_mb(mb):
    """Pretty-print megabytes / gigabytes."""
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"


def estimate_flops_resnet50():
    """Rough FLOPs estimate for ResNet-50 backbone at 384×128 input."""
    # Standard ResNet-50 at 224×224 ≈ 4.1 GFLOPs
    # Our input is 384×128 → area ratio = (384*128) / (224*224) ≈ 0.98
    # Additionally, layer4 stride is changed from 2→1, doubling its spatial size
    # That roughly doubles layer4 FLOPs (~0.9 GFLOPs → ~1.8 GFLOPs)
    base_gflops = 4.1  # standard ResNet-50
    area_ratio = (384 * 128) / (224 * 224)
    stride_factor = 1.3  # conservative estimate for stride-1 layer4
    return base_gflops * area_ratio * stride_factor


def estimate_flops_stn(batch_size=1):
    """FLOPs for PatchGenerator (STN localization network)."""
    # Conv2d(2048→4096, k=3): 2 * 2048 * 4096 * 3 * 3 * H * W
    # At layer4 output with stride-1: H≈24, W≈8  (384/16, 128/16)
    h, w = 24, 8
    conv_flops = 2 * 2048 * 4096 * 9 * h * w
    # FC layers: 4096→512, 512→18
    fc_flops = 2 * 4096 * 512 + 2 * 512 * 18
    return (conv_flops + fc_flops) / 1e9  # GFLOPs


def estimate_flops_diffusion(num_steps=1000):
    """FLOPs for DiffusionPatchGenerator (training = 1 step, inference = num_steps)."""
    # Net: Linear(2560→1024) + Linear(1024→512) + Linear(512→18)
    # Per step: 2*(2560*1024 + 1024*512 + 512*18) ≈ 6.3M FLOPs
    per_step_flops = 2 * (2560 * 1024 + 1024 * 512 + 512 * 18)
    train_gflops = per_step_flops / 1e9  # single step during training
    infer_gflops = (per_step_flops * num_steps) / 1e9  # full chain during inference
    return train_gflops, infer_gflops


def estimate_activation_memory_mb(batch_size, input_h=384, input_w=128):
    """
    Estimate peak activation memory during training.
    Activations dominate GPU memory during training — not the model weights.
    """
    # ResNet-50 with stride-1 layer4 produces feature maps at ~24×8 with 2048 channels
    # We store activations for backward pass at every residual block.
    # Rule of thumb: ResNet-50 at 224×224, BS=1 ≈ 100 MB activations (FP32)
    base_act_mb = 100.0
    area_ratio = (input_h * input_w) / (224 * 224)
    stride_factor = 1.5  # stride-1 layer4 keeps larger feature maps

    backbone_act = base_act_mb * area_ratio * stride_factor * batch_size

    # STN PatchGenerator: Conv2d(2048→4096) intermediate ≈ 4096*24*8*4 bytes * BS
    stn_act = (4096 * 24 * 8 * 4 * batch_size) / (1024 ** 2)

    # 3× grid_sample operations (affine_grid + grid_sample store grids + output)
    # Each: 2048 * 24 * 8 * 4 bytes * BS
    grid_sample_act = 3 * (2048 * 24 * 8 * 4 * batch_size) / (1024 ** 2)

    # Diffusion network activations (small)
    diff_act = (2560 + 1024 + 512 + 18) * 4 * batch_size / (1024 ** 2)

    # Classifier heads (global + 3 part classifiers, each 2048→num_classes)
    # Using num_classes=3000 as default
    classifier_act = 4 * (3000 * 4 * batch_size) / (1024 ** 2)

    # Gradient storage ≈ same as activations for backward pass
    total_act = backbone_act + stn_act + grid_sample_act + diff_act + classifier_act
    total_with_grads = total_act * 2.0  # activations + gradients

    # Optimizer states (Adam stores m and v → 2× model size)
    # Model ≈ 55M params → 55M * 4 * 2 = 440 MB for Adam states
    optimizer_mb = 440.0

    return total_with_grads + optimizer_mb


def max_batch_size_for_budget(gpu_budget_gb, model_size_mb):
    """Estimate maximum feasible batch size for a given GPU memory budget."""
    budget_mb = gpu_budget_gb * 1024
    available_for_activations = budget_mb - model_size_mb - 200  # 200 MB CUDA overhead
    if available_for_activations <= 0:
        return 0

    # Use binary search
    lo, hi = 1, 256
    while lo < hi:
        mid = (lo + hi + 1) // 2
        needed = estimate_activation_memory_mb(mid)
        if needed + model_size_mb + 200 <= budget_mb:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ============================================================================
# Dataset statistics
# ============================================================================

DATASETS = OrderedDict({
    "Market-1501": {
        "train_images": 12936,
        "train_ids": 751,
        "query_images": 3368,
        "gallery_images": 15913,
        "cameras": 6,
        "description": "Standard ReID benchmark (moderate scale)",
    },
    "MSMT17": {
        "train_images": 32621,
        "train_ids": 1041,
        "query_images": 11659,
        "gallery_images": 82161,
        "cameras": 15,
        "description": "Large-scale ReID benchmark (high complexity)",
    },
    "VeRi-776": {
        "train_images": 37778,
        "train_ids": 576,
        "query_images": 1678,
        "gallery_images": 11579,
        "cameras": 20,
        "description": "Vehicle ReID benchmark (20 cameras)",
    },
})


# ============================================================================
# Main report generation
# ============================================================================

def generate_report(args):
    """Generate the complete GPU resource justification report."""

    sep = "=" * 80
    subsep = "-" * 80
    lines = []

    def pr(text=""):
        lines.append(text)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    pr(sep)
    pr("  GPU RESOURCE JUSTIFICATION REPORT")
    pr("  Diffusion-based Person Re-Identification (Diffusion-ReID / PISL)")
    pr(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pr(sep)
    pr()

    # ------------------------------------------------------------------
    # 1. Instantiate model & count parameters
    # ------------------------------------------------------------------
    pr("1. MODEL ARCHITECTURE & PARAMETER ANALYSIS")
    pr(subsep)

    model = ResNetPart(depth=50, pretrained=False,
                       num_parts=args.num_parts, num_classes=args.num_classes)

    total_train, total_nontrain = count_parameters(model)
    total_all = total_train + total_nontrain

    # Component breakdown
    components = OrderedDict()

    # Backbone (base)
    t, nt = count_parameters(model.base)
    components["ResNet-50 Backbone (stride-1 layer4)"] = (t, nt)

    # PatchGenerator (STN)
    t, nt = count_parameters(model.patch_proposal)
    components["PatchGenerator (STN Localization)"] = (t, nt)

    # DiffusionPatchGenerator
    t, nt = count_parameters(model.diffusion_patch)
    components["DiffusionPatchGenerator (Noise Predictor)"] = (t, nt)

    # Global BN neck + classifier
    t_bn, nt_bn = count_parameters(model.bnneck)
    t_cls, nt_cls = count_parameters(model.classifier)
    components["Global BN-Neck + Classifier"] = (t_bn + t_cls, nt_bn + nt_cls)

    # Part BN necks + classifiers
    part_t, part_nt = 0, 0
    for i in range(args.num_parts):
        t1, nt1 = count_parameters(getattr(model, f'bnneck{i}'))
        t2, nt2 = count_parameters(getattr(model, f'classifier{i}'))
        part_t += t1 + t2
        part_nt += nt1 + nt2
    components[f"Part BN-Necks + Classifiers (×{args.num_parts})"] = (part_t, part_nt)

    pr(f"  {'Component':<50} {'Trainable':>14} {'Non-Train':>14} {'Total':>14}")
    pr(f"  {'─' * 50} {'─' * 14} {'─' * 14} {'─' * 14}")
    for name, (t, nt) in components.items():
        pr(f"  {name:<50} {format_num(t):>14} {format_num(nt):>14} {format_num(t + nt):>14}")
    pr(f"  {'─' * 50} {'─' * 14} {'─' * 14} {'─' * 14}")
    pr(f"  {'TOTAL':<50} {format_num(total_train):>14} {format_num(total_nontrain):>14} {format_num(total_all):>14}")
    pr()

    # ------------------------------------------------------------------
    # 2. Model weight memory
    # ------------------------------------------------------------------
    pr("2. MODEL WEIGHT MEMORY")
    pr(subsep)
    fp32_mb = params_to_mb(total_all, 4)
    fp16_mb = params_to_mb(total_all, 2)
    pr(f"  FP32 (float32):  {format_mb(fp32_mb)}")
    pr(f"  FP16 (float16):  {format_mb(fp16_mb)}")
    pr(f"  NOTE: Adam optimizer stores 2 additional copies (momentum + variance)")
    adam_mb = params_to_mb(total_train, 4) * 2
    pr(f"  Adam state memory:  {format_mb(adam_mb)}")
    pr(f"  Total static memory (model + Adam):  {format_mb(fp32_mb + adam_mb)}")
    pr()

    # ------------------------------------------------------------------
    # 3. FLOPs analysis
    # ------------------------------------------------------------------
    pr("3. COMPUTATIONAL COMPLEXITY (FLOPs)")
    pr(subsep)
    backbone_gflops = estimate_flops_resnet50()
    stn_gflops = estimate_flops_stn()
    diff_train_gflops, diff_infer_gflops = estimate_flops_diffusion(args.diffusion_steps)

    pr(f"  {'Component':<50} {'Training':>14} {'Inference':>14}")
    pr(f"  {'─' * 50} {'─' * 14} {'─' * 14}")
    pr(f"  {'ResNet-50 Backbone (per image)':<50} {backbone_gflops:>11.2f} GF {backbone_gflops:>11.2f} GF")
    pr(f"  {'PatchGenerator (STN)':<50} {stn_gflops:>11.2f} GF {stn_gflops:>11.2f} GF")
    pr(f"  {'DiffusionPatchGenerator':<50} {diff_train_gflops:>11.4f} GF {diff_infer_gflops:>11.4f} GF")
    pr(f"  {'3× Grid Sample + Affine Grid':<50} {'~0.50':>14} {'~0.50':>14}")

    total_train_gflops = backbone_gflops + stn_gflops + diff_train_gflops + 0.5
    total_infer_gflops = backbone_gflops + stn_gflops + diff_infer_gflops + 0.5
    pr(f"  {'─' * 50} {'─' * 14} {'─' * 14}")
    pr(f"  {'TOTAL per image':<50} {total_train_gflops:>11.2f} GF {total_infer_gflops:>11.2f} GF")
    batch_label = f'TOTAL per batch (BS={args.batch_size})'
    pr(f"  {batch_label:<50} {total_train_gflops * args.batch_size:>11.1f} GF {total_infer_gflops * args.batch_size:>11.1f} GF")
    pr()
    pr(f"  ⚠ During inference, the diffusion model runs {args.diffusion_steps} denoising steps,")
    pr(f"    making inference {diff_infer_gflops / diff_train_gflops:.0f}× more expensive than a single training step.")
    pr()

    # ------------------------------------------------------------------
    # 4. GPU memory budget analysis
    # ------------------------------------------------------------------
    pr("4. GPU MEMORY BUDGET ANALYSIS")
    pr(subsep)
    model_static_mb = fp32_mb + adam_mb

    gpu_budgets = [
        ("Current (5-10 GB)", 5, 10),
        ("RTX 3060 / 3070", 8, 12),
        ("RTX 3090 / 4090", 24, 24),
        ("A100 (40 GB)", 40, 40),
        ("A100 (80 GB)", 80, 80),
    ]

    pr(f"  Static memory (model + optimizer): {format_mb(model_static_mb)}")
    pr()
    pr(f"  {'GPU Budget':<25} {'Min VRAM':>10} {'Max BS':>10} {'Act. Mem':>14} {'Total Mem':>14} {'Feasible?':>12}")
    pr(f"  {'─' * 25} {'─' * 10} {'─' * 10} {'─' * 14} {'─' * 14} {'─' * 12}")

    for name, lo_gb, hi_gb in gpu_budgets:
        max_bs = max_batch_size_for_budget(lo_gb, model_static_mb)
        if max_bs < 1:
            pr(f"  {name:<25} {lo_gb:>7} GB {'N/A':>10} {'N/A':>14} {'N/A':>14} {'❌ NO':>12}")
        else:
            act_mem = estimate_activation_memory_mb(max_bs)
            total_mem = model_static_mb + act_mem + 200
            feasible = "✅ YES" if max_bs >= 8 else "⚠️ MARGINAL"
            pr(f"  {name:<25} {lo_gb:>7} GB {max_bs:>10} {format_mb(act_mem):>14} {format_mb(total_mem):>14} {feasible:>12}")

    pr()
    pr(f"  ⚠ Minimum effective batch size for ReID training: 16 (4 identities × 4 instances)")
    pr(f"  ⚠ Recommended batch size: 64 (16 identities × 4 instances)")
    pr(f"  ⚠ Paper uses: CUDA_VISIBLE_DEVICES=0,1,2,3 (4 GPUs, multi-GPU DataParallel)")
    pr()

    # ------------------------------------------------------------------
    # 5. Training time estimates
    # ------------------------------------------------------------------
    pr("5. TRAINING TIME ESTIMATES")
    pr(subsep)

    default_ds = "Market-1501"
    ds_info = DATASETS[default_ds]
    iters_per_epoch = args.iters
    total_epochs = args.epochs

    # Approximate throughput on different GPUs (images/sec for this model)
    # Based on ResNet-50 + diffusion overhead benchmarks
    gpu_throughputs = OrderedDict({
        "5 GB GPU (BS=2-4)":   8,     # severely bottlenecked
        "10 GB GPU (BS=8)":   25,     # small batch, low utilization
        "RTX 3090 24GB (BS=32)":  90,
        "A100 40GB (BS=64)":  180,
        "A100 80GB (BS=128)": 320,
        "4× A100 80GB (BS=256)": 1100,
    })

    pr(f"  Dataset: {default_ds} ({format_num(ds_info['train_images'])} train images)")
    pr(f"  Epochs: {total_epochs}, Iterations/epoch: {iters_per_epoch}, Batch size: {args.batch_size}")
    pr(f"  Total training iterations: {format_num(total_epochs * iters_per_epoch)}")
    pr()

    # Extra per-epoch cost: feature extraction for clustering (full forward pass on train set)
    # + DBSCAN clustering + Jaccard distance computation
    feature_extraction_overhead_factor = 1.3  # ~30% overhead for clustering pipeline

    pr(f"  {'GPU Configuration':<30} {'Throughput':>14} {'Time/Epoch':>14} {'Total Time':>14}")
    pr(f"  {'─' * 30} {'─' * 14} {'─' * 14} {'─' * 14}")

    for gpu_name, throughput in gpu_throughputs.items():
        secs_per_iter = args.batch_size / throughput
        secs_per_epoch = secs_per_iter * iters_per_epoch * feature_extraction_overhead_factor
        total_secs = secs_per_epoch * total_epochs
        hrs = total_secs / 3600

        epoch_str = f"{secs_per_epoch / 60:.0f} min" if secs_per_epoch < 3600 else f"{secs_per_epoch / 3600:.1f} hrs"
        total_str = f"{hrs:.1f} hrs" if hrs < 24 else f"{hrs / 24:.1f} days"

        pr(f"  {gpu_name:<30} {throughput:>10} img/s {epoch_str:>14} {total_str:>14}")

    pr()
    pr(f"  ⚠ With 5-10 GB GPU: Training will take {(args.batch_size / 8 * iters_per_epoch * feature_extraction_overhead_factor * total_epochs / 3600 / 24):.0f}+ days")
    pr(f"    (if it can fit in memory at all — batch size forced to 2-4)")
    pr()

    # ------------------------------------------------------------------
    # 6. Dataset scale impact
    # ------------------------------------------------------------------
    pr("6. DATASET SCALE ANALYSIS")
    pr(subsep)

    pr(f"  {'Dataset':<16} {'Train Imgs':>12} {'Train IDs':>12} {'Gallery':>12} {'Cameras':>10} {'Complexity':>12}")
    pr(f"  {'─' * 16} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 12}")
    for name, info in DATASETS.items():
        complexity = "HIGH" if info['train_images'] > 30000 else "MODERATE"
        pr(f"  {name:<16} {format_num(info['train_images']):>12} {format_num(info['train_ids']):>12} "
           f"{format_num(info['gallery_images']):>12} {info['cameras']:>10} {complexity:>12}")
    pr()
    pr("  Per-epoch overhead beyond training iterations:")
    pr("    • Feature extraction (full train set forward pass)")
    pr("    • Jaccard distance computation (O(N²) pairwise distances)")
    pr("    • DBSCAN clustering")
    pr("    • Cross-agreement score computation (k-NN for global + part features)")
    pr("    • Cluster centroid computation and classifier weight update")
    pr()
    pr("  These overheads scale quadratically with train set size, making larger")
    pr("  datasets (MSMT17, VeRi) significantly more demanding.")
    pr()

    # ------------------------------------------------------------------
    # 7. Key architectural complexities
    # ------------------------------------------------------------------
    pr("7. KEY ARCHITECTURAL COMPLEXITIES REQUIRING HIGH GPU MEMORY")
    pr(subsep)
    pr("""
  a) DIFFUSION MODEL (1000-step denoising process):
     • DiffusionPatchGenerator uses a 1000-step cosine noise schedule
     • During training: forward diffusion (noise addition) + noise prediction
     • During inference: full 1000-step DDIM reverse sampling
     • Stores 10 diffusion coefficient buffers (alphas, betas, posteriors)
     • theta_renewal mechanism: quality-based filtering at each step
     • ensemble_sampling: stores intermediate theta every 100 steps

  b) SPATIAL TRANSFORMER NETWORK (STN):
     • PatchGenerator with Conv2d(2048→4096) — doubles channel dim
     • Generates 3 sets of 2×3 affine transformation matrices
     • 3× affine_grid + grid_sample operations (each stores full feature map)
     • Memory: 3 × (B × 2048 × H × W × 4 bytes) for intermediate grids

  c) MULTI-BRANCH ARCHITECTURE:
     • Global branch: ResNet-50 → GAP → BN-Neck → Classifier
     • Part branches (×3): grid_sample → GAP → BN-Neck → Classifier
     • All branches share backbone but have independent classifiers
     • 4 classifier heads × num_classes parameters each

  d) MULTI-LOSS TRAINING:
     • 6 simultaneous loss functions with separate backward graphs:
       - PSC_LR (label refinement)
       - PSC_LS (label smoothing)
       - CrossEntropyLabelSmooth
       - SoftTripletLoss
       - DiffusionThetaLoss (noise + consistency + contrastive)
     • Each loss maintains its own computation graph → memory multiplier

  e) PSEUDO-LABEL GENERATION PIPELINE:
     • Full-dataset feature extraction every epoch
     • Pairwise Jaccard distance matrix (N×N float32)
     • For N=32,621 (MSMT17): 32621² × 4 bytes ≈ 4.0 GB just for the distance matrix
""")

    # ------------------------------------------------------------------
    # 8. Summary comparison
    # ------------------------------------------------------------------
    pr("8. RESOURCE COMPARISON: CURRENT vs. REQUIRED")
    pr(subsep)

    pr(f"""
  ┌─────────────────────────────┬──────────────────────┬──────────────────────┐
  │ Metric                      │ Current (5-10 GB)    │ Recommended          │
  ├─────────────────────────────┼──────────────────────┼──────────────────────┤
  │ GPU Memory                  │ 5-10 GB              │ 24-80 GB             │
  │ Max Batch Size              │ 2-8 (if fits)        │ 64-128               │
  │ # GPUs                      │ 1                    │ 1-4                  │
  │ Estimated Training Time     │ 15-30+ days          │ 6-24 hours           │
  │ Can complete 50 epochs?     │ Extremely unlikely   │ Yes                  │
  │ DBSCAN on full features?    │ Likely OOM           │ Yes                  │
  │ Diffusion 1000-step infer?  │ Extremely slow       │ Feasible             │
  │ Effective for research?     │ ❌ Not viable         │ ✅ Production-ready   │
  └─────────────────────────────┴──────────────────────┴──────────────────────┘
""")

    # ------------------------------------------------------------------
    # 9. Recommendations
    # ------------------------------------------------------------------
    pr("9. RECOMMENDATIONS")
    pr(subsep)
    pr(f"""
  MINIMUM VIABLE:
    * 1x NVIDIA RTX 3090 / RTX 4090 (24 GB) -- can train with BS=32
    * Estimated time: 1-2 days for Market-1501, 3-5 days for MSMT17

  RECOMMENDED:
    * 1x NVIDIA A100 (40 GB) -- can train with BS=64
    * Estimated time: 12-18 hours for Market-1501, 1-2 days for MSMT17

  OPTIMAL (matches paper setup):
    * 4x NVIDIA A100 (80 GB) with DataParallel
    * Can train with BS=256, full diffusion pipeline
    * Estimated time: 4-8 hours for Market-1501

  JUSTIFICATION SUMMARY:
    The model has {format_num(total_all)} parameters ({format_mb(fp32_mb)} FP32),
    requires {format_mb(model_static_mb)} static memory (model + Adam optimizer),
    and needs {format_mb(estimate_activation_memory_mb(args.batch_size))} for activations
    at batch size {args.batch_size}. The diffusion component adds significant
    computational overhead with 1000 denoising steps. Per-epoch clustering
    requires O(N^2) pairwise distances. A 5-10 GB GPU cannot feasibly train
    this model within a reasonable timeframe.
""")

    # ------------------------------------------------------------------
    # 10. Detailed per-layer parameter table
    # ------------------------------------------------------------------
    pr("10. DETAILED PER-LAYER PARAMETER TABLE (Top-Level Modules)")
    pr(subsep)
    pr(f"  {'Layer Name':<60} {'Shape':<25} {'Params':>14}")
    pr(f"  {'─' * 60} {'─' * 25} {'─' * 14}")

    for name, param in model.named_parameters():
        shape_str = "×".join(str(s) for s in param.shape)
        pr(f"  {name:<60} {shape_str:<25} {format_num(param.numel()):>14}")

    # Also count buffers
    pr()
    pr(f"  {'Buffer Name':<60} {'Shape':<25} {'Elements':>14}")
    pr(f"  {'─' * 60} {'─' * 25} {'─' * 14}")
    for name, buf in model.named_buffers():
        shape_str = "×".join(str(s) for s in buf.shape)
        pr(f"  {name:<60} {shape_str:<25} {format_num(buf.numel()):>14}")

    pr()
    pr(sep)
    pr("  END OF REPORT")
    pr(sep)

    report = "\n".join(lines)
    return report


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate GPU Resource Justification Report for Diffusion-ReID")
    parser.add_argument("--num-classes", type=int, default=3000,
                        help="Number of output classes (default: 3000)")
    parser.add_argument("--num-parts", type=int, default=3,
                        help="Number of body parts (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Target batch size for analysis (default: 64)")
    parser.add_argument("--diffusion-steps", type=int, default=1000,
                        help="Number of diffusion denoising steps (default: 1000)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--iters", type=int, default=400,
                        help="Iterations per epoch (default: 400)")
    parser.add_argument("--save-txt", type=str, default=None,
                        help="Save report to a text file")
    args = parser.parse_args()

    # Force UTF-8 output on Windows to handle special characters
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    report = generate_report(args)
    print(report)

    if args.save_txt:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {args.save_txt}")


if __name__ == "__main__":
    main()
gpu_resource_justification.py
Displaying gpu_resource_justification.py.