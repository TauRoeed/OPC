#!/usr/bin/env bash
# Bias-axis follow-up: fixed validation sizes + more seeds + harder noise.
#
# Default grid:
#   axes: context action metadata combined
#   levels: high extreme brutal catastrophic
#   val sizes: 20k 50k (fixed; NOT val_frac)
#   seeds: 0..9
#   train: 5k 25k 50k 100k
#   ctr: 0.05
#
# Conditions: 4 axes × 4 levels × 10 seeds × 2 vals = 320
# Outputs under: artifacts/full_study/run_<RUN_TAG>/val_20000/ and val_50000/
#
# Native:
#   ./scripts/run_bias_axes_val_sweep.sh
#
# Docker (opc / opc:gpu / opc-docker:latest):
#   IMAGE=opc:gpu ./scripts/run_bias_axes_val_sweep.sh
#   USE_DOCKER=0 ./scripts/run_bias_axes_val_sweep.sh   # force host python
#
# Smoke (tiny):
#   SMOKE=1 ./scripts/run_bias_axes_val_sweep.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-opc:gpu}"
USE_DOCKER="${USE_DOCKER:-}"
if [[ -z "${USE_DOCKER}" ]]; then
  if command -v docker >/dev/null 2>&1 && docker image inspect "$IMAGE" >/dev/null 2>&1; then
    USE_DOCKER=1
  else
    USE_DOCKER=0
  fi
fi

RUN_TAG="${RUN_TAG:-bias_axes_val20_50_s10_hard}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
EMB_DIR="${EMB_DIR:-BPR/embeddings}"
MAX_WORKERS="${MAX_WORKERS:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
SHM="${SHM:-64g}"
N_TRIALS="${N_TRIALS:-20}"

COMMON=(
  -m training.run_full_study_parallel
  --datasets ml
  --noise-axes context action metadata combined
  --noise-modes kmeans_templates
  --ctr-levels 0.05
  --train-sizes 5000 25000 50000 100000
  --val-sizes 20000 50000
  --n-trials "$N_TRIALS"
  --num-runs 1
  --policy-reward-mode exact
  --emb-dir "$EMB_DIR"
  --out-dir "$OUT_DIR"
  --run-tag "$RUN_TAG"
  --max-workers "$MAX_WORKERS"
  --num-gpus "$NUM_GPUS"
  --slim
  --skip-completed
)

if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_TAG="${RUN_TAG}_smoke"
  COMMON=(
    -m training.run_full_study_parallel
    --datasets ml
    --noise-axes context
    --noise-modes kmeans_templates
    --noise-levels brutal catastrophic
    --ctr-levels 0.05
    --train-sizes 5000 25000
    --val-sizes 20000
    --seeds 0 1
    --n-trials 2
    --num-runs 1
    --policy-reward-mode exact
    --emb-dir "$EMB_DIR"
    --out-dir "$OUT_DIR"
    --run-tag "$RUN_TAG"
    --max-workers 1
    --num-gpus 1
    --slim
    --skip-completed
  )
else
  COMMON+=(
    --noise-levels high extreme brutal catastrophic
    --seeds 0 1 2 3 4 5 6 7 8 9
  )
fi

if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
  COMMON+=(--require-cuda)
fi

echo "[bias_axes_val_sweep] run_tag=$RUN_TAG docker=$USE_DOCKER image=$IMAGE"
echo "[bias_axes_val_sweep] workers=$MAX_WORKERS gpus=$NUM_GPUS"

if [[ "$USE_DOCKER" == "1" ]]; then
  GPU_ARGS=()
  if [[ "${REQUIRE_CUDA:-1}" == "1" ]]; then
    GPU_ARGS=(--gpus all)
  fi
  docker run --rm \
    "${GPU_ARGS[@]}" \
    --shm-size="$SHM" \
    -v "$ROOT:/app" -w /app \
    "$IMAGE" \
    "${COMMON[@]}"
else
  python "${COMMON[@]}"
fi

echo "[bias_axes_val_sweep] done -> ${OUT_DIR}/run_${RUN_TAG}/"
echo "[bias_axes_val_sweep] analyze each val folder, e.g.:"
echo "  python -m training.analyze_full_study --run-dir ${OUT_DIR}/run_${RUN_TAG}/val_20000"
echo "  python -m training.analyze_full_study --run-dir ${OUT_DIR}/run_${RUN_TAG}/val_50000"
