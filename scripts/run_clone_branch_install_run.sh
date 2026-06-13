#!/usr/bin/env bash
#SBATCH --job-name=opc_full
#SBATCH -A noamk-users_v2
#SBATCH -p gpu-noamk-pool
#SBATCH --qos=owner
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/user/roeedannon/logs/%x-%j.out
#SBATCH --error=/user/roeedannon/logs/%x-%j.err

# Clone/update repo, pip install, BPR artifacts, full OPC study.
#
# Submit (cluster):
#   sbatch scripts/run_clone_branch_install_run.sh
#
# Submit smoke (1 BPR dataset, 1 trial, 1 worker):
#   sbatch --export=ALL,SMOKE=1 scripts/run_clone_branch_install_run.sh
#
# Local smoke (in existing checkout):
#   ./scripts/smoke_clone_install_run.sh
#
# Optional env overrides:
#   REPO_URL=git@github.com:TauRoeed/OPC.git
#   REPO_BRANCH=slim-OPC
#   BASE_DIR=/user/roeedannon
#   REPO_DIR=OPC
#   SKIP_CLONE=1  SKIP_VENV=1  SKIP_BPR=1  SKIP_STUDY=1
#   SMOKE=1
#   OPTUNA_BATCH_SIZES="256 512 1024 2048 4096"

set -euo pipefail

REPO_URL="${REPO_URL:-git@github.com:TauRoeed/OPC.git}"
REPO_BRANCH="${REPO_BRANCH:-slim-OPC}"
REPO_DIR="${REPO_DIR:-OPC}"
BASE_DIR="${BASE_DIR:-/user/roeedannon}"
WORK_DIR="${WORK_DIR:-${BASE_DIR}/${REPO_DIR}}"
VENV_DIR="${VENV_DIR:-${WORK_DIR}/.venv}"
PYTHON="${VENV_DIR}/bin/python"

EMB_DIR="${EMB_DIR:-BPR/embeddings}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
RUN_TAG="${RUN_TAG:-full_ml_myket_anime_8w_s5_slim}"
MAX_WORKERS="${MAX_WORKERS:-30}"
NUM_GPUS="${NUM_GPUS:-3}"
OPTUNA_BATCH_SIZES="${OPTUNA_BATCH_SIZES:-256 512 1024 2048 4096}"

mkdir -p "${BASE_DIR}/logs"
mkdir -p "${BASE_DIR}/.cache/pip"
mkdir -p "${BASE_DIR}/.cache/huggingface"
mkdir -p "${BASE_DIR}/.cache/torch"
mkdir -p "${BASE_DIR}/tmp"

export TMPDIR="${BASE_DIR}/tmp"
export XDG_CACHE_HOME="${BASE_DIR}/.cache"
export PIP_CACHE_DIR="${BASE_DIR}/.cache/pip"
export HF_HOME="${BASE_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${BASE_DIR}/.cache/huggingface"
export TORCH_HOME="${BASE_DIR}/.cache/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

log() { echo "[clone_install_run] $*"; }

log "job_id=${SLURM_JOB_ID:-none} node=${SLURMD_NODENAME:-$(hostname)}"
log "repo=${REPO_URL} branch=${REPO_BRANCH} work_dir=${WORK_DIR}"
log "smoke=${SMOKE:-0} optuna_batch_sizes=${OPTUNA_BATCH_SIZES}"
nvidia-smi || true

# ---------------------------------------------------------------------------
# Step 0: clone / update
# ---------------------------------------------------------------------------
if [[ "${SKIP_CLONE:-0}" != "1" ]]; then
  log "Step 0/4: clone or update repo"
  cd "${BASE_DIR}"
  if [[ ! -d "${WORK_DIR}/.git" ]]; then
    git clone --branch "${REPO_BRANCH}" --single-branch "${REPO_URL}" "${WORK_DIR}"
  else
    cd "${WORK_DIR}"
    git fetch origin "${REPO_BRANCH}"
    git checkout "${REPO_BRANCH}"
    git pull origin "${REPO_BRANCH}"
  fi
else
  log "Step 0/4: skipped (SKIP_CLONE=1)"
fi

cd "${WORK_DIR}"
log "commit: $(git log -1 --oneline 2>/dev/null || echo unknown)"
python3 --version

# ---------------------------------------------------------------------------
# Step 1: venv + pip
# ---------------------------------------------------------------------------
if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  log "Step 1/4: Python environment"
  if [[ ! -x "${PYTHON}" ]]; then
    if python3 -m venv "${VENV_DIR}" 2>/dev/null; then
      :
    else
      log "python3-venv missing; bootstrapping pip via get-pip"
      python3 -m venv --without-pip "${VENV_DIR}"
      curl -sS https://bootstrap.pypa.io/get-pip.py -o "${BASE_DIR}/tmp/get-pip.py"
      "${PYTHON}" "${BASE_DIR}/tmp/get-pip.py"
    fi
  fi
  "${PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${PYTHON}" -m pip install -r requirements.txt
else
  log "Step 1/4: skipped (SKIP_VENV=1)"
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found at ${PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:$PYTHONPATH}"

run_bpr() {
  local dataset="$1"
  local root="$2"
  log "BPR: ${dataset} -> ${EMB_DIR}"
  "${PYTHON}" -m BPR.generate_artifacts \
    --dataset "$dataset" \
    --root "$root" \
    --emb-dir "$EMB_DIR"
}

# ---------------------------------------------------------------------------
# Step 2: BPR artifacts
# ---------------------------------------------------------------------------
if [[ "${SKIP_BPR:-0}" != "1" ]]; then
  log "Step 2/4: BPR artifacts"
  mkdir -p "$EMB_DIR"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    run_bpr ml datasets/ml-1m
  else
    run_bpr ml    datasets/ml-1m
    run_bpr myket datasets/myket
    run_bpr anime datasets/anime
  fi
else
  log "Step 2/4: skipped (SKIP_BPR=1)"
fi

# ---------------------------------------------------------------------------
# Step 3: parallel study (exact reward only; batch size tuned in Optuna)
# ---------------------------------------------------------------------------
if [[ "${SKIP_STUDY:-0}" != "1" ]]; then
  log "Step 3/4: parallel study"
  STUDY_ARGS=(
    --emb-dir "$EMB_DIR"
    --out-dir "$OUT_DIR"
    --policy-reward-mode exact
    --optuna-batch-sizes ${OPTUNA_BATCH_SIZES}
    --slim
    --require-cuda
    --no-skip-completed
  )

  if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_TAG="${RUN_TAG}_smoke"
    STUDY_ARGS+=(
      --datasets ml
      --noise-axes combined
      --noise-levels low
      --ctr-levels 0.05
      --train-sizes 5000
      --val-size 500
      --seeds 0
      --n-trials 1
      --num-runs 1
      --max-workers 1
      --num-gpus 1
      --run-tag "$RUN_TAG"
    )
  else
    STUDY_ARGS+=(
      --datasets ml myket anime
      --noise-axes combined context action metadata
      --noise-levels low medium high
      --ctr-levels 0.05 0.1 0.2
      --train-sizes 5000 25000 50000 100000
      --val-sizes 10000 50000 100000
      --seeds 0 1 2 3 4
      --n-trials 20
      --num-runs 1
      --max-workers "$MAX_WORKERS"
      --num-gpus "$NUM_GPUS"
      --run-tag "$RUN_TAG"
    )
  fi

  "${PYTHON}" -m training.run_full_study_parallel "${STUDY_ARGS[@]}"
  log "Done. Outputs: ${OUT_DIR}/run_${RUN_TAG}/"
else
  log "Step 3/4: skipped (SKIP_STUDY=1)"
fi

log "All steps finished on $(date)"
