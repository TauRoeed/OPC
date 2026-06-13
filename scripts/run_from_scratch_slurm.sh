#!/usr/bin/env bash
#SBATCH -A noamk-users_v2
#SBATCH -p gpu-noamk-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=opc_full
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# TAU PowerSlurm: gpu-noamk-pool (NVIDIA L40S, ~48 GB VRAM each).
# 2 GPUs on Slurm; 8 workers round-robin (4 per GPU via --num-gpus 2).
#
# Full sweep (defaults below):
#   datasets: ml, myket, anime
#   noise mode: kmeans_templates (default; no mode sweep)
#   noise axes: combined, context, action, metadata
#   noise levels: low, medium, high
#   CTR levels: 0.05, 0.1, 0.2
#   train sizes: 5000, 25000, 50000, 100000
#   val sizes: 10000, 50000, 100000
#   seeds: 0-4 (5 seeds)
#   parallel workers: 8 (round-robin over 2 GPUs)
#   slim + require-cuda
#
# Submit:
#   sbatch scripts/run_from_scratch_slurm.sh
#
# Optional env overrides:
#   SKIP_VENV=1  SKIP_BPR=1  SKIP_STUDY=1
#   RUN_TAG=my_tag  MAX_WORKERS=4

set -euo pipefail

mkdir -p logs

echo "[slurm] job_id=${SLURM_JOB_ID:-none}"
echo "[slurm] node=${SLURMD_NODENAME:-$(hostname)}"
echo "[slurm] cpus=${SLURM_CPUS_PER_TASK:-unset}"
echo "[slurm] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

export TMPDIR="${TMPDIR:-${SLURM_TMPDIR:-/tmp}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

nvidia-smi || true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
PYTHON="${VENV}/bin/python"

EMB_DIR="${EMB_DIR:-BPR/embeddings}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
RUN_TAG="${RUN_TAG:-full_ml_myket_anime_30w_s5_slim}"
MAX_WORKERS="${MAX_WORKERS:-30}"
NUM_GPUS="${NUM_GPUS:-3}"

log() { echo "[run_from_scratch_slurm] $*"; }

# ---------------------------------------------------------------------------
# Step 1: Python environment
# ---------------------------------------------------------------------------
if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  log "Step 1/3: Python environment"
  if [[ ! -x "$PYTHON" ]]; then
    if python3 -m venv "$VENV" 2>/dev/null; then
      :
    else
      log "python3-venv missing; bootstrapping pip via get-pip"
      python3 -m venv --without-pip "$VENV"
      curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
      "$PYTHON" /tmp/get-pip.py
    fi
  fi
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r requirements.txt
else
  log "Step 1/3: skipped (SKIP_VENV=1)"
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $PYTHON" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# ---------------------------------------------------------------------------
# Step 2: BPR embeddings (default hyperparams, study datasets only)
# ---------------------------------------------------------------------------
run_bpr() {
  local dataset="$1"
  local root="$2"
  log "BPR: ${dataset} -> ${EMB_DIR}"
  "$PYTHON" -m BPR.generate_artifacts \
    --dataset "$dataset" \
    --root "$root" \
    --emb-dir "$EMB_DIR"
}

if [[ "${SKIP_BPR:-0}" != "1" ]]; then
  log "Step 2/3: BPR artifacts for ml, myket, anime"
  mkdir -p "$EMB_DIR"
  run_bpr ml    datasets/ml-1m
  run_bpr myket datasets/myket
  run_bpr anime datasets/anime
else
  log "Step 2/3: skipped (SKIP_BPR=1)"
fi

# ---------------------------------------------------------------------------
# Step 3: parallel OPC vs no-propensity full sweep
# ---------------------------------------------------------------------------
if [[ "${SKIP_STUDY:-0}" != "1" ]]; then
  log "Step 3/3: parallel study (${MAX_WORKERS} workers / ${NUM_GPUS} GPUs, slim, require-cuda)"
  "$PYTHON" -m training.run_full_study_parallel \
    --datasets ml myket anime \
    --noise-axes combined context action metadata \
    --noise-levels low medium high \
    --ctr-levels 0.05 0.1 0.2 \
    --train-sizes 5000 25000 50000 100000 \
    --val-sizes 10000 50000 100000 \
    --seeds 0 1 2 3 4 \
    --n-trials 20 \
    --num-runs 1 \
    --batch-size 2048 \
    --policy-reward-mode exact \
    --optuna-batch-sizes 256 512 1024 2048 4096 \
    --policy-temperature 1.0 \
    --emb-dir "$EMB_DIR" \
    --out-dir "$OUT_DIR" \
    --run-tag "$RUN_TAG" \
    --max-workers "$MAX_WORKERS" \
    --num-gpus "$NUM_GPUS" \
    --slim \
    --require-cuda
  log "Done. Outputs: ${OUT_DIR}/run_${RUN_TAG}/"
else
  log "Step 3/3: skipped (SKIP_STUDY=1)"
fi

log "All steps finished."
