#!/usr/bin/env bash
# Compat wrapper → large noise trial (trial 2).
# Prefer:
#   ./scripts/run_bias_min_val_trial.sh          # trial 1: min val @ 1M train
#   FIXED_VAL=... ./scripts/run_bias_axis_comp_large_trial.sh   # trial 2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "[warn] run_bias_axis_comp_trial.sh now launches trial 2 (large train + fixed val)."
echo "       For min-val sweep first: $ROOT/scripts/run_bias_min_val_trial.sh"
exec "$ROOT/scripts/run_bias_axis_comp_large_trial.sh"
