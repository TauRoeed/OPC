#!/usr/bin/env bash
# Trial 2: axis × component noise effects at large train, fixed val.
# Run trial 1 first (min val), then set FIXED_VAL to the elbow.
# Usage:
#   FIXED_VAL=100000 ./scripts/run_bias_axis_comp_large_trial.sh
#   FIXED_VAL=100000 IMAGE=opc:gpu ./scripts/run_bias_axis_comp_large_trial.sh
# Overrides:
#   TRAIN_SIZES  FIXED_VAL  RUN_TAG  NOISE_LEVELS
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-bias_axis_comp_tr500k_l5_t20_s5}"
NUM_GPUS="${NUM_GPUS:-2}"
MAX_WORKERS="${MAX_WORKERS:-16}"
SHM="${SHM:-128g}"
IMAGE="${IMAGE:-}"
# Placeholder until trial 1 elbow is known.
FIXED_VAL="${FIXED_VAL:-100000}"
LOG_DIR="${LOG_DIR:-artifacts/full_study/ablation_logs}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/${RUN_TAG}.log"

# shellcheck disable=SC2206
TRAIN_SIZES=(${TRAIN_SIZES:-500000 750000 1000000})
# shellcheck disable=SC2206
NOISE_LEVELS=(${NOISE_LEVELS:-low medium high extreme brutal})

ARGS=(
  -m training.run_full_study_parallel
  --datasets ml
  --noise-axes context action
  --noise-components linear cluster metadata
  --noise-levels "${NOISE_LEVELS[@]}"
  --ctr-levels 0.05
  --train-sizes "${TRAIN_SIZES[@]}"
  --val-size "$FIXED_VAL"
  --seeds 0 1 2 3 4
  --n-trials 20
  --num-gpus "$NUM_GPUS"
  --max-workers "$MAX_WORKERS"
  --require-cuda
  --skip-completed
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --run-tag "$RUN_TAG"
)

N_AXIS=2
N_COMP=3
N_LEVEL=${#NOISE_LEVELS[@]}
N_SEED=5
N_COND=$((N_AXIS * N_COMP * N_LEVEL * N_SEED))

echo "[$(date -Is)] starting trial 2 (large noise) $RUN_TAG"
echo "  train=${TRAIN_SIZES[*]}  fixed_val=$FIXED_VAL"
echo "  levels=${NOISE_LEVELS[*]}"
echo "  grid: ${N_AXIS} axes × ${N_COMP} comps × ${N_LEVEL} levels × ${N_SEED} seeds = ${N_COND} conditions"
echo "  num_gpus=$NUM_GPUS max_workers=$MAX_WORKERS"
echo "  log: $LOGFILE"

if [[ -n "$IMAGE" ]]; then
  echo "  docker image=$IMAGE"
  nohup docker run --rm \
    --gpus all \
    --shm-size="$SHM" \
    -v "$ROOT:/app" -w /app \
    "$IMAGE" \
    "${ARGS[@]}" \
    >"$LOGFILE" 2>&1 &
else
  nohup python "${ARGS[@]}" >"$LOGFILE" 2>&1 &
fi

echo $! >"$LOG_DIR/${RUN_TAG}.pid"
echo "[$(date -Is)] pid=$(cat "$LOG_DIR/${RUN_TAG}.pid")"
echo "  tail -f $LOGFILE"
