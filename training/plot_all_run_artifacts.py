"""
Generate all standard plots + per-setup summary for one or more study runs.

Outputs per run_dir:
  figures_initial_reward/{use_log_trick_*,all_trials}/
  figures_actual_hyperparams/{use_log_trick_*,...}/
  figures_actual_reward_comparison/
  summary_by_setup/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from training.plot_actual_reward_opc_vs_noprop import plot_run as plot_comparison
from training.plot_initial_reward_scatters import plot_run as plot_initial_scatters
from training.plot_smoke_actual_hyperparams import main as _hyper_main
from training.summarize_actual_gt_initial_by_setup import summarize_run


def _plot_hyperparams(run_dir: Path) -> None:
    import training.plot_smoke_actual_hyperparams as mod

    argv = [
        "plot_smoke_actual_hyperparams",
        "--run-dir",
        str(run_dir),
        "--split-log-trick",
    ]
    old = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = old


def plot_all_for_run(run_dir: Path) -> bool:
    run_dir = run_dir.resolve()
    if not list(run_dir.rglob("trials_long.csv")) and not list(run_dir.rglob("opc_trials.csv")):
        print(f"skip {run_dir.name}: no trial logs")
        return False
    print(f"\n=== {run_dir.name} ===")
    ok = True
    for step_name, fn in (
        ("initial_reward scatters", lambda: plot_initial_scatters(run_dir)),
        ("hyperparam figures", lambda: _plot_hyperparams(run_dir)),
        ("OPC vs no-prop comparison", lambda: plot_comparison(run_dir)),
        ("summary_by_setup", lambda: summarize_run(run_dir)),
    ):
        try:
            fn()
        except Exception as exc:
            print(f"  FAILED {step_name}: {exc}")
            ok = False
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, action="append", default=None)
    p.add_argument("--study-root", type=Path, default=Path("artifacts/full_study"))
    args = p.parse_args()

    if args.run_dir:
        runs = [Path(d) for d in args.run_dir]
    else:
        root = Path(args.study_root)
        runs = sorted(
            d
            for d in root.glob("run_*")
            if d.is_dir()
            and (list(d.rglob("trials_long.csv")) or list(d.rglob("opc_trials.csv")))
        )

    ok = sum(plot_all_for_run(r) for r in runs)
    print(f"\nfinished: {ok}/{len(runs)} runs")


if __name__ == "__main__":
    main()
