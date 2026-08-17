"""Analyze H1 study: where naive beats OPC; empirical threshold surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_summaries(root: Path) -> pd.DataFrame:
    path = root / "all_summary_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    parts = []
    for p in root.glob("dataset=*/summary_metrics.csv"):
        parts.append(pd.read_csv(p))
    if not parts:
        raise FileNotFoundError(f"No summaries under {root}")
    return pd.concat(parts, ignore_index=True)


def _pivot_delta(df: pd.DataFrame, metric: str = "policy_rewards") -> pd.DataFrame:
    if "method" not in df.columns or metric not in df.columns:
        raise ValueError(f"need method and {metric}")
    key = [
        c
        for c in [
            "dataset",
            "noise_level",
            "train_size",
            "val_size",
            "q_error",
            "logging_uniform_mix",
            "target_rand_ctr",
            "measured_rand_ctr",
            "density_regime",
            "seed",
        ]
        if c in df.columns
    ]
    wide = df.pivot_table(
        index=key, columns="method", values=metric, aggfunc="first"
    ).reset_index()
    if "opc" not in wide.columns or "no_propensity" not in wide.columns:
        raise ValueError("need opc and no_propensity columns")
    wide["delta_opc_minus_naive"] = wide["opc"] - wide["no_propensity"]
    wide["naive_wins"] = wide["delta_opc_minus_naive"] < 0
    return wide


def _threshold_grid(wide: pd.DataFrame) -> pd.DataFrame:
    """Coarse empirical boundaries: fraction naive_wins binned by axes."""
    rows = []
    group_cols = [
        c
        for c in [
            "dataset",
            "noise_level",
            "val_size",
            "logging_uniform_mix",
            "q_error",
            "target_rand_ctr",
            "train_size",
        ]
        if c in wide.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    for keys, g4 in wide.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["frac_naive_wins"] = float(g4["naive_wins"].mean()) if len(g4) else np.nan
        row["mean_delta"] = float(g4["delta_opc_minus_naive"].mean())
        row["n_cells"] = len(g4)
        rows.append(row)
    return pd.DataFrame(rows)


def _propose_thresholds(grid: pd.DataFrame, *, min_frac: float = 0.5) -> dict:
    """First-cut thresholds where naive wins ≥ min_frac of seeds."""
    hit = grid[grid["frac_naive_wins"] >= min_frac]
    out = {"min_frac_naive_wins": min_frac, "n_hit_cells": int(len(hit))}
    if hit.empty:
        out["note"] = "no cell met threshold yet"
        return out
    out["q_error_min"] = float(hit["q_error"].min())
    out["train_size_min"] = float(hit["train_size"].min())
    out["logging_mix_min"] = float(hit["logging_uniform_mix"].min())
    out["target_rand_ctr_range"] = [
        float(hit["target_rand_ctr"].min()),
        float(hit["target_rand_ctr"].max()),
    ]
    return out


def main():
    p = argparse.ArgumentParser(description="Analyze H1 OPC vs naive study.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/h1_study/run_h1_v1"),
    )
    p.add_argument("--metric", default="policy_rewards")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    root = Path(args.root)
    out_dir = args.out_dir or root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_summaries(root)
    wide = _pivot_delta(df, metric=args.metric)
    wide.to_csv(out_dir / "h1_delta_pivot.csv", index=False)

    grid = _threshold_grid(wide)
    grid.to_csv(out_dir / "h1_threshold_grid.csv", index=False)

    thresholds = _propose_thresholds(grid, min_frac=0.5)
    (out_dir / "h1_thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # H1 prediction check: naive wins more when q_error high + hurtlog + large n
    if {"q_error", "logging_uniform_mix", "train_size"}.issubset(wide.columns):
        hurt = wide[wide["logging_uniform_mix"] >= 0.3]
        clean = wide[wide["logging_uniform_mix"] < 0.1]
        summary = {
            "naive_win_rate_hurtlog": float(hurt["naive_wins"].mean())
            if len(hurt)
            else None,
            "naive_win_rate_cleanlog": float(clean["naive_wins"].mean())
            if len(clean)
            else None,
            "naive_win_rate_q0": float(wide[wide["q_error"] <= 0]["naive_wins"].mean())
            if (wide["q_error"] <= 0).any()
            else None,
            "naive_win_rate_q1": float(wide[wide["q_error"] >= 1]["naive_wins"].mean())
            if (wide["q_error"] >= 1).any()
            else None,
        }
        (out_dir / "h1_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))

    print(f"Wrote analysis under {out_dir}")


if __name__ == "__main__":
    main()
