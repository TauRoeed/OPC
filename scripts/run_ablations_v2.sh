#!/usr/bin/env bash
# Expanded OPC ablations after hurtlog IPW/SNDR pilot.
# Usage (GPU host with nvidia-container-toolkit):
#   IMAGE=opc:gpu ./scripts/run_ablations_v2.sh
#   IMAGE=opc-docker:latest GPU0=0 GPU1=0 ./scripts/run_ablations_v2.sh   # single GPU, sequential
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-opc:gpu}"
SHM="${SHM:-128g}"
WORKERS="${WORKERS:-20}"
GPU0="${GPU0:-2}"
GPU1="${GPU1:-3}"
OUT_DIR="${OUT_DIR:-artifacts/full_study}"
LOG_DIR="${LOG_DIR:-artifacts/full_study/ablation_logs}"
mkdir -p "$LOG_DIR"

COMMON=(
  -m training.run_full_study_parallel
  --datasets ml
  --noise-axes combined
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
  --slim
  --require-cuda
  --skip-completed
)

run_job() {
  local gpu="$1"
  local tag="$2"
  shift 2
  local logfile="$LOG_DIR/${tag}.log"
  echo "[$(date -Is)] starting $tag on GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES="$gpu" docker run --rm \
    --gpus "device=${gpu}" \
    --shm-size="$SHM" \
    -v "$ROOT:/app" -w /app \
    "$IMAGE" \
    "${COMMON[@]}" --run-tag "$tag" "$@" \
    >"$logfile" 2>&1 &
  echo $! >"$LOG_DIR/${tag}.pid"
  echo "[$(date -Is)] pid=$(cat "$LOG_DIR/${tag}.pid") tag=$tag"
}

# 1) IPW expanded: was tiny winner — fill high/extreme/brutal × both CTRs × all trains
run_job "$GPU0" abl_ipw_hurtlog_v2 \
  --policy-losses ipw \
  --noise-levels high extreme brutal \
  --ctr-levels 0.02 0.1 \
  --train-sizes 10000 25000 100000

# 2) SNDR + kl_crm control under same hurt (brutal new + high/extreme for fair kl_crm compare)
#    Split: if GPU0==GPU1, wait for IPW first (caller should set SEQUENTIAL=1)
if [[ "${SEQUENTIAL:-0}" == "1" ]] || [[ "$GPU0" == "$GPU1" ]]; then
  echo "[$(date -Is)] sequential: waiting for IPW job..."
  wait "$(cat "$LOG_DIR/abl_ipw_hurtlog_v2.pid")" || true
fi

run_job "$GPU1" abl_sndr_brutal_hurtlog \
  --policy-losses sndr \
  --noise-levels brutal \
  --ctr-levels 0.02 0.1 \
  --train-sizes 10000 25000 100000

if [[ "${SEQUENTIAL:-0}" == "1" ]] || [[ "$GPU0" == "$GPU1" ]]; then
  wait "$(cat "$LOG_DIR/abl_sndr_brutal_hurtlog.pid")" || true
  run_job "$GPU1" abl_klcrm_hurtlog_ctrl \
    --policy-losses kl_crm \
    --noise-levels high extreme brutal \
    --ctr-levels 0.02 0.1 \
    --train-sizes 10000 25000 100000
else
  # third job shares GPU1 after sndr finishes in background watcher — run klcrm on GPU0 after ipw
  (
    wait "$(cat "$LOG_DIR/abl_ipw_hurtlog_v2.pid")" || true
    run_job "$GPU0" abl_klcrm_hurtlog_ctrl \
      --policy-losses kl_crm \
      --noise-levels high extreme brutal \
      --ctr-levels 0.02 0.1 \
      --train-sizes 10000 25000 100000
    wait
  ) &
fi

echo "[$(date -Is)] launched. Logs in $LOG_DIR"
echo "  tail -f $LOG_DIR/abl_ipw_hurtlog_v2.log"
echo "  tail -f $LOG_DIR/abl_sndr_brutal_hurtlog.log"
