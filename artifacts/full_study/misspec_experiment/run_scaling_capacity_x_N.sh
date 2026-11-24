#!/bin/bash
# Capacity × N Tier 1 — mlp256 on R* γ∈{0.5,2} at nested N∈{25k,100k,400k}.
# See PROTOCOL_LOCKED_capacity_x_N.md. GPU2 only; methods nll+opc (no hybrid).
set -euo pipefail
cd /home/roee/git-repos/OPC
ROOT=artifacts/full_study/misspec_experiment
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/scaling_capacity_x_N.log
STATUS=$LOGDIR/scaling_capacity_x_N.status

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
if [[ "${used:-0}" -gt 500 ]]; then
  echo "ABORT: GPU2 busy (${used}MiB)" | tee -a "$LOG"
  exit 1
fi

GPU_IDS="${GPU_IDS:-2}"
RSTAR_SPEC='s=1,m=4,psi=tanh'
SEEDS=(100 101 102 103 104)
# Nested prefixes: always include N_max=400k so 25k/100k are true prefixes.
SIZES=(25000 100000 400000)
METHODS=(nll opc)
H=256

COMMON=(
  --datasets ml
  --train-sizes "${SIZES[@]}"
  --val-size 20000
  --shared-regression-size 50000
  --nested-train-prefixes
  --reward-data train --crossfit-folds 5
  --n-trials 60
  --lr-range 1e-4 2e-3
  --epochs-range 5 30
  --policy-losses dr
  --opc-gradient direct
  --train-weights none
  --select-weights clip:10
  --learn-logit-scale
  --sampler random
  --skip-completed
  --slim
  --require-cuda
  --stage development
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --methods "${METHODS[@]}"
  --seeds "${SEEDS[@]}"
  --num-gpus 1
  --max-workers 1
  --min-workers 1
  --policy-transform linear+mlp
  --mlp-hidden "$H"
)

run_world() {
  local tag="$1"; shift
  local label="$1"; shift
  {
    echo "===== CAP_XN ${label} tag=${tag} H=${H} $(date -Iseconds) ====="
    echo "GPU_IDS=$GPU_IDS sizes=${SIZES[*]} methods=${METHODS[*]} seeds=${SEEDS[*]}"
  } | tee -a "$LOG"
  echo "running=${tag} started=$(date -Iseconds)" > "$STATUS"

  set +e
  docker run --rm --gpus "device=${GPU_IDS}" --shm-size=64g \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_IDS" -e CUDA_VISIBLE_DEVICES=0 \
    -v /home/roee/git-repos/OPC:/app -w /app \
    opc:gpu -m training.run_full_study_parallel \
    "${COMMON[@]}" \
    "$@" \
    --run-tag "$tag" \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "===== CAP_XN ${label} END rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
  echo "done_${tag}=$(date -Iseconds) rc=$rc" >> "$STATUS"
  return "$rc"
}

{
  echo "===== SCALING CAPACITY×N TIER1 START $(date -Iseconds) ====="
  echo "plan=PROTOCOL_LOCKED_capacity_x_N.md H=${H} N∈{25,100,400}k methods=nll,opc"
  echo "order: γ=0.5 then γ=2; G1 kill after both"
} | tee -a "$LOG"

run_world scale_cap_rstar_g0.5_mlp256 "R*_g0.5_mlp256" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 0.5

run_world scale_cap_rstar_g2_mlp256 "R*_g2_mlp256" \
  --bias-configs high \
  --residual-kind rstar --residual-spec "$RSTAR_SPEC" --residual-gamma 2

echo "===== SCALING CAPACITY×N TIER1 ALL DONE $(date -Iseconds) =====" | tee -a "$LOG"
echo "done_all=$(date -Iseconds) rc=0" > "$STATUS"
