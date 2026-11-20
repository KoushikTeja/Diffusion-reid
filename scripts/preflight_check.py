#!/usr/bin/env python3
"""
Pre-Flight Verification Script for IIT Kharagpur HPC (PARAM Shakti / DGX).
Run this on the login node BEFORE submitting jobs to Slurm.
It validates all dependencies, paths, weights, and models in ~5 seconds.
"""

import os
import os.path as osp
import sys
import glob

def print_section(title):
    print("\n" + "=" * 70)
    print(f"[*] {title}")
    print("=" * 70)

def main():
    print_section("Pre-Flight Check: Diffusion-reid Environment & Datasets")
    all_ok = True
    home = osp.expanduser("~")
    project_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))
    scratch = os.environ.get("SCRATCH", f"/scratch/scratch26/{os.environ.get('USER', '')}")

    # -------------------------------------------------------------------------
    # 1. Python & Core Libraries Check
    # -------------------------------------------------------------------------
    print("\n--- 1. Checking Dependencies & Version Compatibility ---")
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"       Found {torch.cuda.device_count()} GPU(s): {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
        else:
            print("       (Running in CPU mode on login node - this is normal)")
    except ImportError as e:
        print(f"  [FAIL] PyTorch not installed: {e}")
        all_ok = False

    try:
        import numpy as np
        np_ver = np.__version__
        if int(np_ver.split('.')[0]) >= 2:
            print(f"  [FAIL] NumPy version is {np_ver}. Faiss requires numpy<2.0.0 (e.g. 1.26.4).")
            print("         Fix: pip install 'numpy==1.26.4'")
            all_ok = False
        else:
            print(f"  [OK] NumPy: {np_ver} (Compatible with Faiss)")
    except ImportError as e:
        print(f"  [FAIL] NumPy not installed: {e}")
        all_ok = False

    try:
        import faiss
        print(f"  [OK] Faiss: successfully imported (swigfaiss OK)")
    except Exception as e:
        print(f"  [FAIL] Faiss failed to import: {e}")
        print("         Fix: pip install 'numpy==1.26.4'")
        all_ok = False

    try:
        import torchvision
        print(f"  [OK] Torchvision: {torchvision.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Torchvision missing: {e}")
        all_ok = False

    try:
        import sklearn
        print(f"  [OK] Scikit-Learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"  [FAIL] Scikit-learn missing: {e}")
        all_ok = False

    # -------------------------------------------------------------------------
    # 2. Pretrained Model Weights Check (Offline HPC requirement)
    # -------------------------------------------------------------------------
    print("\n--- 2. Checking Offline ResNet-50 Checkpoint ---")
    ckpt_dir = osp.join(home, ".cache", "torch", "hub", "checkpoints")
    resnet_pth = osp.join(ckpt_dir, "resnet50-0676ba61.pth")
    if osp.isfile(resnet_pth):
        size_mb = osp.getsize(resnet_pth) / (1024 * 1024)
        print(f"  [OK] Found ResNet-50 weights at: {resnet_pth} ({size_mb:.1f} MB)")
    else:
        print(f"  [FAIL] ResNet-50 weights NOT found at: {resnet_pth}")
        print("         Compute nodes cannot connect to internet to download it!")
        print("         Fix: Run this command right now on login node:")
        print(f"         mkdir -p ~/.cache/torch/hub/checkpoints")
        print(f"         wget -c https://download.pytorch.org/models/resnet50-0676ba61.pth -O ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth")
        all_ok = False

    # -------------------------------------------------------------------------
    # 3. Dataset Discovery Check
    # -------------------------------------------------------------------------
    print("\n--- 3. Checking Dataset Directories ---")
    data_roots = [
        osp.join(project_dir, "examples", "data"),
        osp.join(scratch, "data"),
    ]

    market_dir = None
    veri_dir = None
    for root in data_roots:
        m = osp.join(root, "Market1501")
        v = osp.join(root, "VeRi")
        if osp.isdir(m) and market_dir is None:
            market_dir = m
        if osp.isdir(v) and veri_dir is None:
            veri_dir = v

    if market_dir:
        train_dir = osp.join(market_dir, "bounding_box_train")
        test_dir = osp.join(market_dir, "bounding_box_test")
        query_dir = osp.join(market_dir, "query")
        n_train = len(glob.glob(osp.join(train_dir, "*.jpg"))) if osp.isdir(train_dir) else 0
        n_test = len(glob.glob(osp.join(test_dir, "*.jpg"))) if osp.isdir(test_dir) else 0
        n_query = len(glob.glob(osp.join(query_dir, "*.jpg"))) if osp.isdir(query_dir) else 0
        print(f"  [OK] Market-1501 found at: {market_dir}")
        print(f"       Train imgs: {n_train}, Gallery imgs: {n_test}, Query imgs: {n_query}")
        if n_train == 0 or n_test == 0:
            print("       [WARN] Subdirectories in Market1501 appear empty!")
            all_ok = False
    else:
        print(f"  [WARN] Market1501 folder not found in any standard path:")
        for r in data_roots:
            print(f"         - {osp.join(r, 'Market1501')}")

    if veri_dir:
        train_dir = osp.join(veri_dir, "image_train")
        test_dir = osp.join(veri_dir, "image_test")
        query_dir = osp.join(veri_dir, "image_query")
        n_train = len(glob.glob(osp.join(train_dir, "*.jpg"))) if osp.isdir(train_dir) else 0
        n_test = len(glob.glob(osp.join(test_dir, "*.jpg"))) if osp.isdir(test_dir) else 0
        n_query = len(glob.glob(osp.join(query_dir, "*.jpg"))) if osp.isdir(query_dir) else 0
        print(f"  [OK] VeRi-776 found at: {veri_dir}")
        print(f"       Train imgs: {n_train}, Gallery imgs: {n_test}, Query imgs: {n_query}")
    else:
        print(f"  [WARN] VeRi folder not found in any standard path (only needed if training on VeRi)")

    # -------------------------------------------------------------------------
    # 4. Model Architecture & Forward Pass Check
    # -------------------------------------------------------------------------
    print("\n--- 4. Checking ResNetPart Architecture Instantiation ---")
    try:
        sys.path.insert(0, project_dir)
        from pisl.models.resnet_part import resnet50part
        import torch

        # Test instantiation (loads offline weights)
        model = resnet50part(num_parts=3, num_classes=3000)
        dummy_input = torch.randn(2, 3, 384, 128)

        # 4a. Eval mode forward pass
        model.eval()
        with torch.no_grad():
            eval_out = model(dummy_input)
        if isinstance(eval_out, torch.Tensor):
            print(f"  [OK] Eval forward pass successful! Output feature shape: {eval_out.shape}")
        else:
            print(f"  [OK] Eval forward pass successful! Returned {len(eval_out)} elements")

        # 4b. Training mode forward pass
        model.train()
        train_out = model(dummy_input)
        if isinstance(train_out, (tuple, list)):
            print(f"  [OK] Training forward pass successful! Returned {len(train_out)} training outputs (features, logits, diffusion params)")
        else:
            print(f"  [OK] Training forward pass successful! Output shape: {train_out.shape}")
    except Exception as e:
        print(f"  [FAIL] Model instantiation or forward pass failed: {e}")
        all_ok = False

    # -------------------------------------------------------------------------
    # 5. Slurm Script Line-Ending Check
    # -------------------------------------------------------------------------
    print("\n--- 5. Checking Slurm Scripts (Unix LF line endings) ---")
    slurm_files = glob.glob(osp.join(project_dir, "slurm", "*.sh"))
    crlf_found = False
    for sf in slurm_files:
        with open(sf, "rb") as fp:
            if b"\r\n" in fp.read():
                print(f"  [FAIL] Windows CRLF line ending detected in: {osp.basename(sf)}")
                crlf_found = True
                all_ok = False
    if not crlf_found:
        print(f"  [OK] All {len(slurm_files)} Slurm scripts have clean Unix LF line endings.")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print_section("Summary")
    if all_ok:
        print("[SUCCESS] All preflight checks PASSED!")
        print("Your cluster environment is 100% ready. You can safely run:")
        print("  sbatch slurm/submit_plain_market1501.sh")
        print("  sbatch slurm/submit_math_market1501.sh")
    else:
        print("[ACTION REQUIRED] Please address the [FAIL] items above before submitting to Slurm.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
