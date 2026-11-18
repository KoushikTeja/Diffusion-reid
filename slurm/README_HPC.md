# IIT Kharagpur HPC (PARAM Shakti & DGX Cluster) Quick-Start Guide

This guide details how to setup, transfer data, and submit Slurm jobs on IIT Kharagpur's HPC infrastructure (**PARAM Shakti** and the **DGX AI Cluster**).

---

## 1. Remote Login & File Transfer

### Connecting via SSH
* **From Inside Campus:**
  ```bash
  ssh <username>@paramshakti.iitkgp.ac.in
  ```
* **From Outside Campus (Port 4422):**
  ```bash
  ssh <username>@paramshakti.iitkgp.ac.in -p 4422
  ```
*Note: Authenticating requires solving the terminal ASCII challenge, entering your 2FA Google Authenticator code, and entering your password.*

### Transferring Files (`scp`)
* **Upload Repository to Scratch:**
  ```bash
  # Outside campus (Port 4422)
  scp -P 4422 -r ./Diffusion-reid <username>@paramshakti.iitkgp.ac.in:/scratch/<username>/

  # Inside campus
  scp -r ./Diffusion-reid <username>@paramshakti.iitkgp.ac.in:/scratch/<username>/
  ```
* **Upload Datasets to Scratch:**
  ```bash
  scp -P 4422 -r /path/to/data <username>@paramshakti.iitkgp.ac.in:/scratch/<username>/data
  ```
* **Download Best Checkpoints Back to Local Machine:**
  ```bash
  scp -P 4422 <username>@paramshakti.iitkgp.ac.in:/scratch/<username>/Diffusion-reid/logs/math_market1501/best.pth ./
  ```

---

## 2. Storage Architecture & Mandatory Rules

| Filesystem | Path | Quota | Retention Policy | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Home** | `/home/<username>/` | 40–50 GB | Permanent | Source code, config, Conda base |
| **Scratch** | `/scratch/<username>/` | 2 TB (Lustre) | **Purged after 30–60 days** | **Active jobs, datasets, logs, weights** |

> [!IMPORTANT]
> **Cluster Golden Rule:** All Slurm jobs **MUST** run from `/scratch/<username>/`. Never execute heavy training jobs or store large datasets in `/home/<username>/`.

---

## 3. One-Time Environment Setup (On Login Node)

Run these commands once on the login node:

```bash
# 1. Load system CUDA module
module load compilers/cuda/11.8 2>/dev/null || module load cuda/11.8

# 2. Create conda environment
conda create -n diffreid python=3.10 -y
conda activate diffreid

# 3. Install PyTorch matching cluster CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install scikit-learn pillow wandb faiss-gpu==1.7.2
```

---

## 4. Expected Directory Structure on Scratch

Organize your `/scratch/<username>/` directory as follows:

```text
/scratch/<username>/
├── Diffusion-reid/
│   ├── examples/
│   ├── pisl/
│   ├── slurm/
│   │   ├── submit_plain_market1501.sh
│   │   ├── submit_plain_veri776.sh
│   │   ├── submit_math_market1501.sh
│   │   └── submit_math_veri776.sh
│   └── logs_slurm/
└── data/
    ├── market1501/
    │   └── Market-1501-v15.09.15/
    │       ├── bounding_box_train/
    │       ├── bounding_box_test/
    │       └── query/
    └── veri/
        └── VeRi/
            ├── image_train/
            ├── image_test/
            └── image_query/
```

---

## 5. Submitting Jobs via Slurm

All scripts are pre-configured in the `slurm/` directory.

### Step A: Baseline Experiments (`plain-baseline` branch)
```bash
cd /scratch/$USER/Diffusion-reid
git checkout plain-baseline

# Submit Market-1501 Baseline
sbatch slurm/submit_plain_market1501.sh

# Submit VeRi-776 Baseline
sbatch slurm/submit_plain_veri776.sh
```

### Step B: Mathematically Optimized Experiments (`math-optimized` branch)
```bash
cd /scratch/$USER/Diffusion-reid
git checkout math-optimized

# Submit Market-1501 Optimized
sbatch slurm/submit_math_market1501.sh

# Submit VeRi-776 Optimized
sbatch slurm/submit_math_veri776.sh
```

---

## 6. Monitoring and Job Control

* **View Active Jobs:**
  ```bash
  squeue -u $USER
  ```
* **View Node/Partition Status:**
  ```bash
  sinfo -s
  ```
* **Inspect Running Job Details:**
  ```bash
  scontrol show job <job_id>
  ```
* **Follow Training Logs in Real Time:**
  ```bash
  tail -f logs_slurm/<job_id>-pisl_math_market.out
  ```
* **Cancel a Job:**
  ```bash
  scancel <job_id>
  ```
* **Check Disk Quota:**
  ```bash
  myquota
  ```

---

## 7. DGX AI Cluster (H100) Configuration (Optional)

If running on IIT KGP's **DGX AI Cluster** (`gpucluster.iitkgp.ac.in`) instead of PARAM Shakti:
1. In the `#SBATCH` header, change `#SBATCH -p gpu` to:
   ```bash
   #SBATCH -p dgx_all
   #SBATCH --qos=gpu2
   ```
2. The remaining directives (`--nodes=1`, `--gres=gpu:1`, `--cpus-per-task=8`) are identical.
