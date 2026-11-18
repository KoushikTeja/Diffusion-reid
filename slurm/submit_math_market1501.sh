#!/bin/bash
# ==============================================================================
# IIT Kharagpur HPC (PARAM Shakti / DGX Cluster) - Slurm Submission Script
# Experiment: Mathematically Optimized Model (math-optimized branch) on Market-1501
# ==============================================================================
#SBATCH -J pisl_math_market            # Job name
#SBATCH -p gpu                         # Partition: 'gpu' for PARAM Shakti (or 'dgx_all' on DGX)
#SBATCH --nodes=1                      # Single compute node
#SBATCH --ntasks=1                     # Single task
#SBATCH --cpus-per-task=8              # 8 CPU cores for DataLoader workers
#SBATCH --gres=gpu:1                   # 1 GPU (Tesla V100 on PARAM Shakti / H100 on DGX)
#SBATCH --mem=32000                    # 32 GB system memory
#SBATCH -t 24:00:00                    # Walltime HH:MM:SS (Max 72h on 'gpu' partition)
#SBATCH -o logs_slurm/%j-%x.out        # Standard output log
#SBATCH -e logs_slurm/%j-%x.err        # Standard error log
#SBATCH --mail-type=ALL                # Notify on BEGIN, END, FAIL
# #SBATCH --mail-user=your_email@iitkgp.ac.in   # Uncomment and add your email

# ------------------------------------------------------------------------------
# 1. Mandatory Storage Rule: Execute strictly from $SCRATCH
# ------------------------------------------------------------------------------
SCRATCH_ROOT="${SCRATCH:-/scratch/scratch26/${USER}}"
PROJECT_DIR="${SCRATCH_ROOT}/Diffusion-reid"
DATA_DIR="${SCRATCH_ROOT}/data"
LOGS_DIR="${PROJECT_DIR}/logs/math_market1501"

cd "${PROJECT_DIR}" || { echo "Directory ${PROJECT_DIR} not found! Exiting."; exit 1; }
mkdir -p logs_slurm "${LOGS_DIR}"

# ------------------------------------------------------------------------------
# 2. Environment Setup (Modules & Conda)
# ------------------------------------------------------------------------------
module purge
module load compilers/cuda/11.8 2>/dev/null || module load cuda/11.8 2>/dev/null || true

if [ -f "/home/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate diffreid
elif [ -f "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate diffreid
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ------------------------------------------------------------------------------
# 3. Diagnostics
# ------------------------------------------------------------------------------
echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          ${SLURMD_NODENAME}"
echo "Start Time:    $(date)"
echo "Working Dir:   $(pwd)"
echo "Branch:        $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'N/A')"
echo "Commit:        $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "========================================================================"
nvidia-smi
echo "========================================================================"

# ------------------------------------------------------------------------------
# 4. Training Execution (Mathematically Optimized Pipeline)
# ------------------------------------------------------------------------------
python examples/train_pisl_cam.py \
    -d market1501 \
    --data-dir "${DATA_DIR}" \
    --logs-dir "${LOGS_DIR}" \
    --height 384 \
    --width 128 \
    --batch-size 64 \
    --workers 4 \
    --epochs 50 \
    --eval-step 5

echo "Job finished at: $(date)"
