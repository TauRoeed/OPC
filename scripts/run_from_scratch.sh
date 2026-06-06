#!/usr/bin/env bash
# Full OPC pipeline from a clean clone:
#   1) Python venv + requirements
#   2) BPR artifacts for all datasets (default hyperparams, auto-download data)
#   3) Parallel OPC vs no-propensity study (run_full_study_parallel defaults)
#
# Usage:
#   ./scripts/run_from_scratch.sh
#
# Optional env overrides:
#   VENV=.venv              venv path
#   SKIP_VENV=1             skip venv create / pip install
#   SKIP_BPR=1              skip BPR artifact generation
#   SKIP_STUDY=1            skip parallel study
#   RUN_TAG=my_run          study output tag (default: timestamp)
#   MAX_WORKERS=4           parallel workers
#   STUDY_DATASETS="ml anime"   datasets for parallel study
#   REQUIRE_CUDA=1          pass --require-cuda to study runner
#   SLIM=1                  pass --slim to study runner
#   SMOKE=1                 tiny fast end-to-end smoke (ml only, 1 trial)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
PYTHON="${VENV}/bin/python"

EMB_DIR="${EMB_DIR:-BPR/embeddings}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MAX_WORKERS="${MAX_WORKERS:-4}"
STUDY_DATASETS="${STUDY_DATASETS:-ml anime}"

log() { echo "[run_from_scratch] $*"; }

if [[ "${SKIP_VENV:-0}" != "1" ]]; then
  log "Step 1/3: Python environment"
  if [[ ! -x "$PYTHON" ]]; then
    if python3 -m venv "$VENV" 2>/dev/null; then
      :
    else
      log "python3-venv missing; creating venv without pip and bootstrapping get-pip"
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

# Repo imports (BPR.*, training.*)
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

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
  log "Step 2/3: BPR artifacts (default params, all datasets)"
  mkdir -p "$EMB_DIR"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    run_bpr ml datasets/ml-1m
  else
    run_bpr ml      datasets/ml-1m
    run_bpr anime   datasets/anime
    run_bpr myket   datasets/myket
    run_bpr lastfm  datasets/lastfm/lastfm_360k.hdf5
    run_bpr msd     datasets/msd/msd_taste_profile.hdf5
  fi
else
  log "Step 2/3: skipped (SKIP_BPR=1)"
fi

if [[ "${SKIP_STUDY:-0}" != "1" ]]; then
  log "Step 3/3: parallel study (run_full_study_parallel defaults)"
  STUDY_ARGS=(
    --datasets ${STUDY_DATASETS}
    --noise-modes kmeans_templates
    --noise-axes combined
    --noise-levels low high
    --ctr-levels 0.05
    --train-sizes 5000 25000 50000 100000
    --seeds 0 1 2
    --n-trials 20
    --num-runs 1
    --batch-size 2048
    --policy-reward-mode exact
    --policy-reward-mc-sim 8
    --policy-temperature 1.0
    --val-frac 0.15
    --val-min 5000
    --emb-dir "$EMB_DIR"
    --out-dir "$OUT_DIR"
    --run-tag "$RUN_TAG"
    --max-workers "$MAX_WORKERS"
  )
  if [[ "${SMOKE:-0}" == "1" ]]; then
    STUDY_DATASETS="ml"
    STUDY_ARGS=(
      --datasets ml
      --noise-modes kmeans_templates
      --noise-axes combined
      --noise-levels low
      --ctr-levels 0.05
      --train-sizes 5000
      --seeds 0
      --n-trials 1
      --num-runs 1
      --batch-size 512
      --policy-reward-mode mc
      --policy-reward-mc-sim 2
      --val-size 500
      --emb-dir "$EMB_DIR"
      --out-dir "$OUT_DIR"
      --run-tag "${RUN_TAG}_smoke"
      --max-workers 1
      --slim
      --no-skip-completed
    )
  fi
  if [[ "${SLIM:-0}" == "1" ]]; then
    STUDY_ARGS+=(--slim)
  fi
  if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
    STUDY_ARGS+=(--require-cuda)
  fi
  "$PYTHON" -m training.run_full_study_parallel "${STUDY_ARGS[@]}"
  log "Done. Outputs: ${OUT_DIR}/run_${RUN_TAG}/"
else
  log "Step 3/3: skipped (SKIP_STUDY=1)"
fi

log "All steps finished."
