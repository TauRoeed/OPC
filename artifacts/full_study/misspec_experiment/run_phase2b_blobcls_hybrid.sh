#!/bin/bash
# Phase-2b Tier-1 B1 (+ optional B2): ScoreClass-blob hybrid_seq + opc/nll.
# DO NOT RUN until PROTOCOL_DRAFT_phase2b.md §8 is locked.
# Standalone CausE/BLOB Phase-2 is already complete — this is NOT a re-run of refs.
set -euo pipefail
cd /home/roee/git-repos/OPC
LOG=artifacts/full_study/misspec_experiment/logs/phase2b_blobcls_hybrid.log
mkdir -p artifacts/full_study/misspec_experiment/logs

# Lean confirmatory (B1). Set EXTEND_SEEDS=1 to also finish B2 (305-309) for opc/nll only.
SEEDS=(300 301 302 303 304)
METHODS=(hybrid_seq_f0.5 opc nll)

COMMON=(
  --datasets ml
  --seeds "${SEEDS[@]}"
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
  --methods "${METHODS[@]}"
  --policy-transform blob
  --num-gpus 4
  --max-workers 2
  --min-workers 1
  --skip-completed
  --stage confirmatory
  --slim
  --require-cuda
  --emb-dir BPR/embeddings
  --out-dir artifacts/full_study
  --residual-kind rstar
  --residual-spec s=1,m=4,psi=tanh
)

run_g() {
  local g="$1"
  local tag="nlres_p2b_blobcls_hyb_rstar_g${g}"
  echo "===== START $tag $(date -Iseconds) =====" | tee -a "$LOG"
  docker run --rm --gpus all --shm-size=128g \
    -v /home/roee/git-repos/OPC:/app -w /app \
    opc:gpu -m training.run_full_study_parallel \
    "${COMMON[@]}" --residual-gamma "$g" --run-tag "$tag" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "===== END $tag rc=$rc $(date -Iseconds) =====" | tee -a "$LOG"
  return $rc
}

if [[ "${1:-}" != "--i-locked-section-8" ]]; then
  echo "Refusing to launch. Re-run with: $0 --i-locked-section-8" >&2
  echo "After filling PROTOCOL_DRAFT_phase2b.md §8." >&2
  exit 2
fi

: > "$LOG"
run_g 0
run_g 0.5

if [[ "${EXTEND_SEEDS:-0}" == "1" ]]; then
  # B2: power for opc/nll only on remaining confirmatory seeds (reuse blobcls tags if present).
  SEEDS=(305 306 307 308 309)
  METHODS=(opc nll)
  COMMON=(
    --datasets ml
    --seeds "${SEEDS[@]}"
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
    --methods "${METHODS[@]}"
    --policy-transform blob
    --num-gpus 4
    --max-workers 2
    --min-workers 1
    --skip-completed
    --stage confirmatory
    --slim
    --require-cuda
    --emb-dir BPR/embeddings
    --out-dir artifacts/full_study
    --residual-kind rstar
    --residual-spec s=1,m=4,psi=tanh
  )
  for g in 0 0.5; do
    tag="nlres_blobcls_rstar_g${g}"
    echo "===== START B2 $tag seeds 305-309 $(date -Iseconds) =====" | tee -a "$LOG"
    docker run --rm --gpus all --shm-size=128g \
      -v /home/roee/git-repos/OPC:/app -w /app \
      opc:gpu -m training.run_full_study_parallel \
      "${COMMON[@]}" --residual-gamma "$g" --run-tag "$tag" 2>&1 | tee -a "$LOG"
  done
fi

echo "===== PHASE2B BLOBCLS HYBRID DONE $(date -Iseconds) =====" | tee -a "$LOG"
