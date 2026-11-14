#!/bin/bash
# Lean capacity × misspec trial (PROTOCOL_DRAFT_capacity.md)
# Caps: linear+mlp @ 64 and 256; families R* + SoftThresh; γ∈{0,0.5,2}; ml; seeds 300-309.
# Linear reference = existing nlres_grid / h1 tags (not re-run).
set -euo pipefail
cd /home/roee/git-repos/OPC
ROOT=artifacts/full_study/misspec_experiment
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/capacity_trial_${WORKER_ID:-main}.log
QUEUE=$LOGDIR/capacity_trial_queue.txt
PROGRESS=$LOGDIR/capacity_trial_progress_${WORKER_ID:-main}.txt
STATUS=$LOGDIR/capacity_trial.status

GPU_IDS="${GPU_IDS:-3}"
N_GPUS=$(awk -F',' '{print NF}' <<<"$GPU_IDS")
GPU_MAX_WORKERS="${GPU_MAX_WORKERS:-2}"
SEEDS=($(seq 300 309))
METHODS=(opc nll hybrid_seq_f0.5)

RSTAR_SPEC='s=1,m=4,psi=tanh'
ST15_SPEC='m=4,tau=1.5,s_th=0.25,proj_tag=default,norm=v2'

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
)

build_queue() {
  : > "$QUEUE"
  for kind in rstar softthresh; do
    local spec=$RSTAR_SPEC rk=rstar short=rstar
    if [[ "$kind" == "softthresh" ]]; then
      spec=$ST15_SPEC; rk=softthresh; short=st15
    fi
    for g in 0 0.5 2; do
      for hid in 64 256; do
        local tag="nlres_cap_${short}_g${g}_mlp${hid}"
        # residual kind none at γ=0
        echo "${tag}|${rk}|${spec}|${g}|${hid}" >> "$QUEUE"
      done
    done
  done
  echo "capacity_queue=$(wc -l < "$QUEUE") built=$(date -Iseconds)" | tee -a "$LOG" | tee "$STATUS"
}

run_one() {
  local line="$1"
  IFS='|' read -r tag rk spec g hid <<<"$line"
  local kind="$rk"
  if [[ "$g" == "0" || "$g" == "0.0" ]]; then kind=none; fi
  echo "===== START $tag $(date -Iseconds) =====" | tee -a "$LOG"
  echo "running=$tag started=$(date -Iseconds)" > "$STATUS"
  set +e
  docker run --rm --gpus all --shm-size=64g \
    -e NVIDIA_VISIBLE_DEVICES="$GPU_IDS" -e CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    -v /home/roee/git-repos/OPC:/app -w /app \
    opc:gpu -m training.run_full_study_parallel \
    "${COMMON[@]}" \
    --policy-transform linear+mlp --mlp-hidden "$hid" \
    --residual-kind "$kind" --residual-spec "$spec" --residual-gamma "$g" \
    --run-tag "$tag" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "===== END $tag rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
}

claim_and_run() {
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local tag="${line%%|*}"
    local claimed=0
    {
      flock 9
      if grep -Rqx "$tag" "$LOGDIR" --include='capacity_trial_progress_*.txt' 2>/dev/null; then
        claimed=1
      else
        echo "$tag" >> "$PROGRESS"
        claimed=0
      fi
    } 9>>"$LOGDIR/capacity_trial.lock"
    [[ "$claimed" -eq 1 ]] && continue
    run_one "$line" || true
  done < "$QUEUE"
}

{
  echo "===== CAPACITY TRIAL ${WORKER_ID:-main} $(date -Iseconds) ====="
  echo "GPU_IDS=$GPU_IDS seeds=${SEEDS[*]} methods=${METHODS[*]}"
} | tee -a "$LOG"

if [[ "${SKIP_BUILD_QUEUE:-0}" != "1" ]]; then
  build_queue
elif [[ ! -s "$QUEUE" ]]; then
  build_queue
fi

claim_and_run
echo "===== CAPACITY TRIAL DONE $(date -Iseconds) =====" | tee -a "$LOG"
echo "done=$(date -Iseconds)" > "$STATUS"
