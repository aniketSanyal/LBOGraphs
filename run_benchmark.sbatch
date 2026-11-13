#!/bin/bash
#SBATCH --job-name="no_benchmark"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out

# ---------------------------------------------------------------------------
# Executes a benchmark notebook (default benchmark.ipynb) and writes an
# executed copy with all results, tables and plots embedded.
#
# Submit:   sbatch /storage/user/sany/graphs_benchmark/run_benchmark.sbatch
# Other notebook:
#           sbatch --export=ALL,NOTEBOOK=benchmark_manifolds.ipynb \
#               /storage/user/sany/graphs_benchmark/run_benchmark.sbatch
# Queue:    squeue -u sany
# Log:      tail -f /storage/slurm/logs/slurm-<jobid>.out
# Cancel:   scancel <jobid>
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_DIR="/storage/user/sany/graphs_benchmark"
VENV="/storage/user/sany/venv_graphs"
NOTEBOOK="${NOTEBOOK:-benchmark.ipynb}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_NOTEBOOK="results/${NOTEBOOK%.ipynb}_executed_${SLURM_JOB_ID:-local}_${TIMESTAMP}.ipynb"
CELL_TIMEOUT=100000     # seconds allowed per cell

echo "=============================================="
echo "Job ID   : ${SLURM_JOB_ID:-<none>}"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "Output   : ${PROJECT_DIR}/${OUTPUT_NOTEBOOK}"
echo "=============================================="

module load cuda/12.1.0
source "${VENV}/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p results

# Keep numeric libraries from oversubscribing the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

# Do NOT set MPLBACKEND here: the Jupyter kernel's inline backend already
# works headless, and forcing Agg stops plt.show() from embedding figures in
# the executed notebook (every run before Oct 2026 lost its plots this way).

# --- preflight: fail loudly here, not 30 minutes into the run ---------------
echo "---------- environment ----------"
python - <<'PYCHECK'
import importlib.util
import sys

required = ["torch", "numpy", "scipy", "trimesh", "robust_laplacian",
            "matplotlib", "nbconvert", "nbclient", "ipykernel"]
missing = [m for m in required if importlib.util.find_spec(m) is None]

if missing:
    print("ERROR: missing packages:", ", ".join(missing))
    sys.exit(1)

import torch
print("python        :", sys.version.split()[0])
print("torch         :", torch.__version__)
print("cuda build    :", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu           :", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU visible - running on CPU, this will be slow.")
PYCHECK
echo "---------------------------------"

nvidia-smi || echo "note: nvidia-smi unavailable"

# --- run --------------------------------------------------------------------
srun --unbuffered jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK}" \
    --output "$(basename "${OUTPUT_NOTEBOOK}")" \
    --output-dir results \
    --ExecutePreprocessor.timeout=${CELL_TIMEOUT} \
    --ExecutePreprocessor.kernel_name=python3 \
    --allow-errors

echo "=============================================="
echo "Finished : $(date)"
echo "Result   : ${PROJECT_DIR}/${OUTPUT_NOTEBOOK}"
echo "=============================================="
