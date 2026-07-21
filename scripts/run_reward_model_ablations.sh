#!/usr/bin/env bash
# Reward-model ablations: regression vs logging_score vs oracle under hurt logging.
# Focus on SNDR (needs q) vs IPW (no q in loss) on the winning hurtlog cell.
#
# Usage:
#   IMAGE=opc:gpu ./scripts/run_reward_model_ablations.sh
#   IMAGE=opc-docker:latest GPU0=0 ./scripts/run_reward_model_ablations.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-opc:gpu}"
SHM="${SHM:-128g}"
WORKERS="${WORKERS:-12}"
GPU0="${GPU0:-2}"
GPU1="${GPU1:-3}"
GPU2="${GPU2:-0}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
LOG_DIR="${LOG_DIR:-artifacts/full_study/ablation_logs}"
mkdir -p "$LOG_DIR"

COMMON=(
  -m training.run_full_study_parallel
  --datasets ml
  --noise-axes combined
  --noise-levels brutal extreme
  --ctr-levels 0.1
  --train-sizes 100000
  --seeds 0 1 2
  --n-trials 20
  --max-workers "$WORKERS"
  --num-gpus 1
  --optuna-selection r_hat
  --logging-uniform-mix 0.3
  --policy-temperature 2
  --policy-reward-mode exact
  --optuna-batch-sizes 4096 8192 16384
  --emb-dir BPR/embeddings
  --out-dir "$OUT_DIR"
  --val-size 25000
  --shared-regression-size 50000
  --slim
  --require-cuda
  --skip-completed
)

run_one() {
  local gpu="$1" tag="$2" loss="$3" q="$4"
  local log="$LOG_DIR/${tag}.log"
  echo "=== GPU $gpu | $tag | loss=$loss reward=$q ===" | tee -a "$log"
  docker run --rm --gpus "device=${gpu}" --shm-size="$SHM" \
    -v "$ROOT:/workspace" -w /workspace "$IMAGE" \
    python "${COMMON[@]}" \
      --policy-losses "$loss" \
      --reward-model "$q" \
      --run-tag "$tag" \
    2>&1 | tee -a "$log"
}

# Priority: SNDR × {oracle, logging_score, regression} — diagnoses whether q hurts DR.
# IPW × regression is a no-q control (should match prior hurtlog IPW).
run_one "$GPU0" "abl_sndr_q_oracle" sndr oracle &
run_one "$GPU1" "abl_sndr_q_logscore" sndr logging_score &
run_one "$GPU2" "abl_sndr_q_regression" sndr regression &
wait

# Optional IPW controls (same grid) — uncomment if GPUs free:
# run_one "$GPU0" "abl_ipw_q_oracle" ipw oracle &
# run_one "$GPU1" "abl_ipw_q_logscore" ipw logging_score &
# wait

echo "Done. Tags under $OUT_DIR/run_abl_*_q_*"
