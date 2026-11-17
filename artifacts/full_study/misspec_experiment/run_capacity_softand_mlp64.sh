#!/bin/bash
# SoftAND* capacity gap under fair reasonable π (CLAIMS_FAIR_PI.md).
# Matched linear+mlp @ hidden 64; γ ∈ {0, 0.5, 100}; ml; seeds 300–309.
# Same HPO/CRN recipe as run_capacity_trial.sh (R*/SoftThresh wave).
#
# Tags:
#   nlres_cap_softand_g0_mlp64
#   nlres_cap_softand_g0.5_mlp64
#   nlres_cap_softand_g100_mlp64
#
# Requires: GPU host with full OPC tree (utils/residual_env.py), docker image opc:gpu,
# and BPR/embeddings. This docs branch checkout alone cannot execute training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Prefer a sibling full checkout if this is the docs-only pdf tree.
if [[ ! -f "$REPO_ROOT/training/run_full_study_parallel.py" ]]; then
  for cand in /home/roee/git-repos/OPC "$PWD" "$REPO_ROOT"; do
    if [[ -f "$cand/training/run_full_study_parallel.py" ]]; then
      REPO_ROOT="$cand"
      break
    fi
  done
fi
cd "$REPO_ROOT"

ROOT=artifacts/full_study/misspec_experiment
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/capacity_softand_mlp64_${WORKER_ID:-main}.log
QUEUE=$LOGDIR/capacity_softand_mlp64_queue.txt
PROGRESS=$LOGDIR/capacity_softand_mlp64_progress_${WORKER_ID:-main}.txt
STATUS=$LOGDIR/capacity_softand_mlp64.status

GPU_IDS="${GPU_IDS:-0}"
N_GPUS=$(awk -F',' '{print NF}' <<<"$GPU_IDS")
GPU_MAX_WORKERS="${GPU_MAX_WORKERS:-2}"
SEEDS=($(seq 300 309))
METHODS=(opc nll hybrid_seq_f0.5)
SOFTAND_SPEC='m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2'
HID=64

COMMON=(
  --datasets ml
  --train-sizes 100000
  --val-size 20000
  --n-trials 20
  --policy-losses dr
  --opc-gradient direct
  --train-weights none
  --select-weights clip:10
  --learn-logit-scale
  --sampler random
  --lr-range 1e-4 2e-3
  --epochs-range 5 30
  --bias-configs high
  --skip-completed
  --slim
  --require-cuda
  --stage confirmatory
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --methods "${METHODS[@]}"
  --seeds "${SEEDS[@]}"
  --num-gpus "$N_GPUS"
  --max-workers "$GPU_MAX_WORKERS"
  --min-workers 1
  --policy-transform linear+mlp
  --mlp-hidden "$HID"
)

build_queue() {
  : > "$QUEUE"
  for g in 0 0.5 100; do
    local tag="nlres_cap_softand_g${g}_mlp${HID}"
    echo "${tag}|softand|${SOFTAND_SPEC}|${g}|${HID}" >> "$QUEUE"
  done
  echo "softand_cap_queue=$(wc -l < "$QUEUE") built=$(date -Iseconds)" | tee -a "$LOG" | tee "$STATUS"
}

run_one() {
  local line="$1"
  IFS='|' read -r tag rk spec g hid <<<"$line"
  local kind="$rk"
  if [[ "$g" == "0" || "$g" == "0.0" ]]; then kind=none; fi
  echo "===== START $tag $(date -Iseconds) =====" | tee -a "$LOG"
  echo "running=$tag started=$(date -Iseconds)" > "$STATUS"
  set +e
  if command -v docker >/dev/null 2>&1 && docker image inspect opc:gpu >/dev/null 2>&1; then
    docker run --rm --gpus all --shm-size=64g \
      -e NVIDIA_VISIBLE_DEVICES="$GPU_IDS" -e CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      -v "$REPO_ROOT:/app" -w /app \
      opc:gpu -m training.run_full_study_parallel \
      "${COMMON[@]}" \
      --residual-kind "$kind" --residual-spec "$spec" --residual-gamma "$g" \
      --run-tag "$tag" 2>&1 | tee -a "$LOG"
  else
    python -m training.run_full_study_parallel \
      "${COMMON[@]}" \
      --residual-kind "$kind" --residual-spec "$spec" --residual-gamma "$g" \
      --run-tag "$tag" 2>&1 | tee -a "$LOG"
  fi
  local rc=${PIPESTATUS[0]}
  set -e
  echo "===== END $tag rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
  return "$rc"
}

claim_and_run() {
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local tag="${line%%|*}"
    local claimed=0
    {
      flock 9
      if grep -Rqx "$tag" "$LOGDIR" --include='capacity_softand_mlp64_progress_*.txt' 2>/dev/null; then
        claimed=1
      else
        echo "$tag" >> "$PROGRESS"
        claimed=0
      fi
    } 9>>"$LOGDIR/capacity_softand_mlp64.lock"
    [[ "$claimed" -eq 1 ]] && continue
    run_one "$line" || true
  done < "$QUEUE"
}

{
  echo "===== SOFTAND CAPACITY mlp64 ${WORKER_ID:-main} $(date -Iseconds) ====="
  echo "REPO_ROOT=$REPO_ROOT GPU_IDS=$GPU_IDS seeds=${SEEDS[*]} methods=${METHODS[*]}"
} | tee -a "$LOG"

if [[ "${SKIP_BUILD_QUEUE:-0}" != "1" || ! -s "$QUEUE" ]]; then
  build_queue
fi

# Always materialize the planned queue for docs / offline hosts.
mkdir -p "$ROOT/capacity_fair_pi"
cp "$QUEUE" "$ROOT/capacity_fair_pi/softand_mlp64_queue.txt"

if [[ "${QUEUE_ONLY:-0}" == "1" ]]; then
  echo "QUEUE_ONLY=1 — wrote queue, skipping training" | tee -a "$LOG"
  echo "status=queued queue_only=1" > "$STATUS"
  exit 0
fi

if [[ ! -f training/run_full_study_parallel.py ]]; then
  echo "ERROR: training/run_full_study_parallel.py not found under $REPO_ROOT" | tee -a "$LOG"
  echo "status=blocked reason=missing_full_tree queue=ready" > "$STATUS"
  echo "Queue is ready at $ROOT/capacity_fair_pi/softand_mlp64_queue.txt — re-run on a full GPU checkout." | tee -a "$LOG"
  exit 2
fi

claim_and_run
echo "===== SOFTAND CAPACITY mlp64 DONE $(date -Iseconds) =====" | tee -a "$LOG"
echo "done=$(date -Iseconds)" > "$STATUS"
