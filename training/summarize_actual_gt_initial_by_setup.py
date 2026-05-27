"""
Per-setup summary: fraction of trials with actual_reward > initial_reward.

Writes ``<run_dir>/summary_by_setup/summary_actual_gt_initial_by_setup.csv``
and a short text report alongside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.load_run_trials import load_run_trials


def _parse_folder(name: str) -> dict:
    out = {}
    for part in name.split("__"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _log_trick_true(v) -> bool:
    return bool(v) if isinstance(v, bool) else int(v) == 1


def _attach_initial_reward(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if "initial_reward" in df.columns and df["initial_reward"].notna().any():
        return df
    summary = run_dir / "all_summary_metrics.csv"
    if not summary.exists():
        return df
    sm = pd.read_csv(summary)
    if "initial_reward" not in sm.columns:
        return df
    keys = [k for k in ("method", "seed") if k in sm.columns and k in df.columns]
    if not keys:
        return df
    ir = sm.groupby(keys, as_index=False)["initial_reward"].first()
    return df.merge(ir, on=keys, how="left")


def _load_with_condition_meta(run_dir: Path) -> pd.DataFrame:
    """trials_long + condition folder fields; fall back to load_run_trials."""
    parts = []
    for p in sorted(run_dir.rglob("trials_long.csv")):
        meta = _parse_folder(p.parent.name)
        df = pd.read_csv(p)
        for k, v in meta.items():
            df[k] = v
        parts.append(df)
    if parts:
        df = pd.concat(parts, ignore_index=True)
        return _attach_initial_reward(df, run_dir)
    return load_run_trials(run_dir)


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    if "actual_reward" not in df.columns:
        return pd.DataFrame()
    for c in ("actual_reward", "initial_reward", "train_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "initial_reward" not in df.columns or df["initial_reward"].isna().all():
        return pd.DataFrame()

    if "param_use_log_trick" in df.columns:
        df["log_trick"] = df["param_use_log_trick"].map(
            lambda v: "true" if _log_trick_true(v) else "false"
        )
    else:
        df["log_trick"] = "unknown"

    setup_keys = [k for k in ("dataset", "noise", "axis", "level", "ctr", "train_size") if k in df.columns]
    if not setup_keys and "train_size" in df.columns:
        setup_keys = ["train_size"]

    rows = []
    group_cols = setup_keys + ["log_trick", "method"]
    if len(setup_keys) == 0:
        group_cols = ["log_trick", "method"]
    for keys, g in df.groupby(group_cols, dropna=False):
        setup = dict(zip(group_cols, keys))
        s = g.dropna(subset=["actual_reward", "initial_reward"])
        n = len(s)
        b = int((s["actual_reward"] > s["initial_reward"]).sum()) if n else 0
        method = setup.pop("method")
        log_trick = setup.pop("log_trick")
        row = {
            **setup,
            "log_trick": log_trick,
            "method": "OPC" if method == "opc" else "no propensity",
            "beat": b,
            "n_trials": n,
            "pct_actual_gt_initial": round(100.0 * b / n, 2) if n else np.nan,
        }
        if "train_size" in row:
            row["train_size"] = int(float(row["train_size"]))
        rows.append(row)

    tab = pd.DataFrame(rows)
    sort_cols = [c for c in ("level", "ctr", "train_size", "log_trick", "method") if c in tab.columns]
    if sort_cols:
        tab = tab.sort_values(sort_cols)
    return tab


def _write_report(tab: pd.DataFrame, path: Path) -> None:
    lines = [
        "Rule: actual_reward > initial_reward",
        f"Rows: {len(tab)}",
        "",
    ]
    if tab.empty:
        lines.append("(no data)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    group_cols = [c for c in ("level", "ctr", "train_size") if c in tab.columns]
    if not group_cols:
        group_cols = [c for c in ("noise", "axis") if c in tab.columns]
    if not group_cols:
        lines.append(tab.to_string(index=False))
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    for setup_vals, sub in tab.groupby(group_cols, sort=True):
        if not isinstance(setup_vals, tuple):
            setup_vals = (setup_vals,)
        hdr = " | ".join(f"{k}={v}" for k, v in zip(group_cols, setup_vals))
        lines.append(f"--- {hdr} ---")
        pivot = sub.pivot_table(
            index="log_trick",
            columns="method",
            values="pct_actual_gt_initial",
            aggfunc="first",
        )
        lines.append(pivot.to_string())
        lines.append(
            sub.pivot_table(index="log_trick", columns="method", values="beat", aggfunc="first").to_string()
            + "  (beat counts)"
        )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_run(run_dir: Path) -> bool:
    run_dir = run_dir.resolve()
    try:
        df = _load_with_condition_meta(run_dir)
    except FileNotFoundError:
        print(f"skip {run_dir.name}: no trial logs")
        return False
    if df.empty:
        print(f"skip {run_dir.name}: empty")
        return False

    tab = build_summary_table(df)
    out_dir = run_dir / "summary_by_setup"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_actual_gt_initial_by_setup.csv"
    tab.to_csv(csv_path, index=False)
    _write_report(tab, out_dir / "summary_actual_gt_initial_by_setup.txt")
    print(f"{run_dir.name}: {len(tab)} rows -> {csv_path}")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="One or more run directories (default: all under --study-root with trials_long)",
    )
    p.add_argument(
        "--study-root",
        type=Path,
        default=Path("artifacts/full_study"),
    )
    args = p.parse_args()

    if args.run_dir:
        runs = [Path(d) for d in args.run_dir]
    else:
        runs = sorted(
            d for d in Path(args.study_root).glob("run_*") if d.is_dir() and list(d.rglob("trials_long.csv"))
        )

    n_ok = sum(summarize_run(r) for r in runs)
    print(f"done: {n_ok}/{len(runs)} runs")


if __name__ == "__main__":
    main()
