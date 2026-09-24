#!/usr/bin/env bash
# Bias trial: each representation-bias type (warp, group, vector) alone and all three
# together, per level, @ large train, fixed val.
# Defaults: sndr train, logging_score q̂, fixed DR score clip M=1 (code default).
# Train curve: 0.5M → 1M → 2M → 5M → 10M.
# Usage:
#   CUDA_VISIBLE_DEVICES=1,2,3 NUM_GPUS=3 MAX_WORKERS=90 FIXED_VAL=100000 \
#     ./scripts/run_bias_axis_comp_large_trial.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-bias_types_sndr_logscore_clip1_tr500k_10m_v100k_t20_s5}"
NUM_GPUS="${NUM_GPUS:-3}"
MAX_WORKERS="${MAX_WORKERS:-90}"
SHM="${SHM:-128g}"
IMAGE="${IMAGE:-}"
FIXED_VAL="${FIXED_VAL:-100000}"
LOG_DIR="${LOG_DIR:-artifacts/full_study/ablation_logs}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/${RUN_TAG}.log"

# shellcheck disable=SC2206
TRAIN_SIZES=(${TRAIN_SIZES:-500000 1000000 2000000 5000000 10000000})
# shellcheck disable=SC2206
BIAS_LEVELS=(${BIAS_LEVELS:-low medium high})
BIAS_CONFIGS=()
for L in "${BIAS_LEVELS[@]}"; do
  BIAS_CONFIGS+=("$L/none/none" "none/$L/none" "none/none/$L" "$L")  # warp, group, vector, all three
done

ARGS=(
  -m training.run_full_study_parallel
  --datasets ml
  --bias-configs "${BIAS_CONFIGS[@]}"
  --ctr-levels 0.05
  --train-sizes "${TRAIN_SIZES[@]}"
  --val-size "$FIXED_VAL"
  --seeds 0 1 2 3 4
  --n-trials 20
  --policy-losses sndr
  --reward-model logging_score
  --slim
  --num-gpus "$NUM_GPUS"
  --max-workers "$MAX_WORKERS"
  --require-cuda
  --skip-completed
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --run-tag "$RUN_TAG"
)

N_BIAS=${#BIAS_CONFIGS[@]}
N_SEED=5
N_COND=$((N_BIAS * N_SEED))

echo "[$(date -Is)] starting bias type trial $RUN_TAG"
echo "  train=${TRAIN_SIZES[*]}  fixed_val=$FIXED_VAL"
echo "  bias configs=${BIAS_CONFIGS[*]}"
echo "  loss=sndr  reward_model=logging_score  DR_score_clip_M=1 (fixed)"
echo "  grid: ${N_BIAS} bias configs × ${N_SEED} seeds = ${N_COND} conditions"
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
