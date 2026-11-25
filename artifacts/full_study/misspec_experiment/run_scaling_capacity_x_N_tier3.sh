#!/bin/bash
# Capacity × N Tier 3 — mlp64 dose-response on R* γ=0.5 (N★ moved ≥1 step; G2 passed).
# Sentinel N = densified ladder {25,50,100,200,400}k. See PROTOCOL_LOCKED_capacity_x_N.md.
set -euo pipefail
cd /home/roee/git-repos/OPC
ROOT=artifacts/full_study/misspec_experiment
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/scaling_capacity_x_N_tier3.log
STATUS=$LOGDIR/scaling_capacity_x_N_tier3.status

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
if [[ "${used:-0}" -gt 500 ]]; then
  echo "ABORT: GPU2 busy (${used}MiB)" | tee -a "$LOG"
  exit 1
fi

GPU_IDS="${GPU_IDS:-2}"
RSTAR_SPEC='s=1,m=4,psi=tanh'
SEEDS=(100 101 102 103 104)
SIZES=(25000 50000 100000 200000 400000)
METHODS=(nll opc)
H=64

{
  echo "===== SCALING CAPACITY×N TIER3 START $(date -Iseconds) ====="
  echo "tag=scale_cap_rstar_g0.5_mlp64 H=${H} sizes=${SIZES[*]}"
} | tee -a "$LOG"
echo "running=scale_cap_rstar_g0.5_mlp64 started=$(date -Iseconds)" > "$STATUS"

set +e
docker run --rm --gpus "device=${GPU_IDS}" --shm-size=64g \
  -e NVIDIA_VISIBLE_DEVICES="$GPU_IDS" -e CUDA_VISIBLE_DEVICES=0 \
  -v /home/roee/git-repos/OPC:/app -w /app \
  opc:gpu -m training.run_full_study_parallel \
  --datasets ml \
  --train-sizes "${SIZES[@]}" \
  --val-size 20000 \
  --shared-regression-size 50000 \
  --nested-train-prefixes \
  --reward-data train --crossfit-folds 5 \
  --n-trials 60 \
  --lr-range 1e-4 2e-3 \
  --epochs-range 5 30 \
  --policy-losses dr \
  --opc-gradient direct \
  --train-weights none \
  --select-weights clip:10 \
  --learn-logit-scale \
  --sampler random \
  --skip-completed \
  --slim \
  --require-cuda \
  --stage development \
  --emb-dir BPR/embeddings \
  --out-dir artifacts/full_study \
  --methods "${METHODS[@]}" \
  --seeds "${SEEDS[@]}" \
  --num-gpus 1 \
  --max-workers 1 \
  --min-workers 1 \
  --policy-transform linear+mlp \
  --mlp-hidden "$H" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 0.5 \
  --run-tag scale_cap_rstar_g0.5_mlp64 \
  2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "===== CAP_XN_T3 END rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
echo "done_all=$(date -Iseconds) rc=$rc" > "$STATUS"
exit "$rc"
