#!/bin/bash
# Capacity × N Tier 2 — densify g0.5+g2 at {50k,200k}; add γ=0 mlp256 sentinel.
# See PROTOCOL_LOCKED_capacity_x_N.md. GPU2 only; methods nll+opc.
set -euo pipefail
cd /home/roee/git-repos/OPC
ROOT=artifacts/full_study/misspec_experiment
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/scaling_capacity_x_N_tier2.log
STATUS=$LOGDIR/scaling_capacity_x_N_tier2.status

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
if [[ "${used:-0}" -gt 500 ]]; then
  echo "ABORT: GPU2 busy (${used}MiB)" | tee -a "$LOG"
  exit 1
fi

GPU_IDS="${GPU_IDS:-2}"
RSTAR_SPEC='s=1,m=4,psi=tanh'
SEEDS=(100 101 102 103 104)
METHODS=(nll opc)
H=256

# Densify: keep Tier1 Ns so --skip-completed reuses them; N_max=400k nested lock.
DENSE_SIZES=(25000 50000 100000 200000 400000)
# γ=0 sentinel (matched DiD to Tier1)
G0_SIZES=(25000 100000 400000)

run_world() {
  local tag="$1"; shift
  local label="$1"; shift
  # remaining args: sizes then residual flags
  local -a sizes=()
  while [[ $# -gt 0 && "$1" != --* ]]; do
    sizes+=("$1"); shift
  done
  {
    echo "===== CAP_XN_T2 ${label} tag=${tag} H=${H} sizes=${sizes[*]} $(date -Iseconds) ====="
    echo "GPU_IDS=$GPU_IDS methods=${METHODS[*]} seeds=${SEEDS[*]}"
  } | tee -a "$LOG"
  echo "running=${tag} started=$(date -Iseconds)" > "$STATUS"

  set +e
  docker run --rm --gpus "device=${GPU_IDS}" --shm-size=64g \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_IDS" -e CUDA_VISIBLE_DEVICES=0 \
    -v /home/roee/git-repos/OPC:/app -w /app \
    opc:gpu -m training.run_full_study_parallel \
    --datasets ml \
    --train-sizes "${sizes[@]}" \
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
    "$@" \
    --run-tag "$tag" \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "===== CAP_XN_T2 ${label} END rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
  echo "done_${tag}=$(date -Iseconds) rc=$rc" >> "$STATUS"
  return "$rc"
}

{
  echo "===== SCALING CAPACITY×N TIER2 START $(date -Iseconds) ====="
  echo "plan=PROTOCOL_LOCKED_capacity_x_N.md densify 50/200k + g0 sentinel"
} | tee -a "$LOG"

run_world scale_cap_rstar_g0.5_mlp256 "R*_g0.5_mlp256_dense" \
  "${DENSE_SIZES[@]}" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 0.5

run_world scale_cap_rstar_g2_mlp256 "R*_g2_mlp256_dense" \
  "${DENSE_SIZES[@]}" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 2

run_world scale_cap_rstar_g0_mlp256 "R*_g0_mlp256" \
  "${G0_SIZES[@]}" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 0

echo "===== SCALING CAPACITY×N TIER2 ALL DONE $(date -Iseconds) =====" | tee -a "$LOG"
echo "done_all=$(date -Iseconds) rc=0" > "$STATUS"
