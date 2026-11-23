#!/bin/bash
# ==============================================================================
# IIT Kharagpur HPC (PARAM Shakti / DGX Cluster) - Slurm Submission Script
# Experiment: Baseline Model (plain-baseline branch) on Market-1501
# ==============================================================================
#SBATCH -J pisl_plain_market           # Job name
#SBATCH -p gpu                         # Partition: 'gpu' for PARAM Shakti (or 'dgx_all' on DGX)
#SBATCH --nodes=1                      # Run on a single compute node
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=16             # 16 CPU cores for DataLoader workers
#SBATCH --gres=gpu:2                   # 2 GPUs (Tesla V100 on PARAM Shakti / H100 on DGX)
#SBATCH --mem=64000                    # 64 GB system memory
#SBATCH -t 24:00:00                    # Walltime HH:MM:SS (Max 72h on 'gpu' partition)
#SBATCH -o logs_slurm/%j-%x.out        # Standard output log (%j = Job ID, %x = Job name)
#SBATCH -e logs_slurm/%j-%x.err        # Standard error log
#SBATCH --mail-type=ALL                # Notify on BEGIN, END, FAIL
# #SBATCH --mail-user=your_email@iitkgp.ac.in   # Uncomment and add your email

# ------------------------------------------------------------------------------
# 1. Mandatory Storage Rule: Execute strictly from $SCRATCH
# ------------------------------------------------------------------------------
SCRATCH_ROOT="${SCRATCH:-/scratch/scratch26/${USER}}"
PROJECT_DIR="${SCRATCH_ROOT}/Diffusion-reid"

# Auto-detect dataset directory
if [ -d "${PROJECT_DIR}/examples/data" ]; then
    DATA_DIR="${PROJECT_DIR}/examples/data"
elif [ -d "${SCRATCH_ROOT}/data" ]; then
    DATA_DIR="${SCRATCH_ROOT}/data"
else
    DATA_DIR="${PROJECT_DIR}/examples/data"
fi

LOGS_DIR="${PROJECT_DIR}/logs/plain_market1501"

cd "${PROJECT_DIR}" || { echo "Directory ${PROJECT_DIR} not found! Exiting."; exit 1; }
mkdir -p logs_slurm "${LOGS_DIR}"

# ------------------------------------------------------------------------------
# 2. Environment Setup (Modules & Conda)
# ------------------------------------------------------------------------------
module purge
module load compiler/cuda/11.7 2>/dev/null || module load compiler/cuda/11.8 2>/dev/null || module load compilers/cuda/11.8 2>/dev/null || module load cuda/11.8 2>/dev/null || true

# Activate conda environment
if [ -f "/home/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate diffreid
elif [ -f "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate diffreid
elif [ -f "/home/${USER}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/home/${USER}/anaconda3/etc/profile.d/conda.sh"
    conda activate diffreid
else
    source ~/.bashrc 2>/dev/null || true
    conda activate diffreid 2>/dev/null || true
fi

# Multi-threading & WandB settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------------------------
# 3. Diagnostics
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start Time:    $(date)"
echo "Working Dir:   $(pwd)"
echo "Data Dir:      ${DATA_DIR}"
echo "Branch:        $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
echo "Commit:        $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "========================================================================"
nvidia-smi
echo "========================================================================"

# ------------------------------------------------------------------------------
# 4. Training Execution
# ------------------------------------------------------------------------------
python examples/train_pisl_cam.py \
    -d market1501 \
    --eps 0.4 \
    --data-dir "${DATA_DIR}" \
    --logs-dir "${LOGS_DIR}" \
    --height 384 \
    --width 128 \
    --batch-size 64 \
    --workers 4 \
    --epochs 50 \
    --eval-step 5

echo "Job finished at: $(date)"
