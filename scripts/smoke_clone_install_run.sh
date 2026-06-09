#!/usr/bin/env bash
# Local end-to-end smoke for run_clone_branch_install_run.sh (no Slurm submit).
#
# Usage:
#   ./scripts/smoke_clone_install_run.sh
#
# Uses current checkout (SKIP_CLONE=1), tiny BPR + 1 Optuna trial, exact reward.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export BASE_DIR="$ROOT"
export REPO_DIR="."
export WORK_DIR="$ROOT"
export VENV_DIR="${VENV:-${ROOT}/.venv}"
export SKIP_CLONE=1
export SMOKE=1
export RUN_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
export OPTUNA_BATCH_SIZES="${OPTUNA_BATCH_SIZES:-128 256 512 1024 2048}"

# Skip venv if already built; set SKIP_VENV=0 to force reinstall.
export SKIP_VENV="${SKIP_VENV:-1}"

exec bash "${ROOT}/scripts/run_clone_branch_install_run.sh"
