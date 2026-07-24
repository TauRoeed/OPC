"""Rebuild study summaries from trial logs (fixes slim actual_reward_selected)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_full_study import (
    _load_trials_long_union,
    _parse_condition_dirname,
)
from training.metrics_utils import enrich_summary_pct_fields


def _merge_keys(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in (
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "ctr",
            "seed",
            "method",
            "train_size",
            "run",
            "val_size",
        )
        if c in df.columns
    ]


def _best_trial_actuals(trials: pd.DataFrame) -> pd.DataFrame:
    if trials.empty or "actual_reward" not in trials.columns:
        return pd.DataFrame()
    t = trials.copy()
    t["train_size"] = pd.to_numeric(t["train_size"], errors="coerce")
    t["run"] = pd.to_numeric(t["run"], errors="coerce")
    if "is_best_in_run" in t.columns and t["is_best_in_run"].any():
        sel = t[t["is_best_in_run"].astype(bool)].copy()
    else:
        keys = _merge_keys(t)
        if len(keys) < 4:
            return pd.DataFrame()
        sel = t.sort_values("value", ascending=False).groupby(keys, as_index=False).first()
    sel["selected_policy_reward"] = pd.to_numeric(sel["actual_reward"], errors="coerce")
    keep = _merge_keys(sel) + ["selected_policy_reward"]
    return sel[keep].drop_duplicates()


def _coerce_merge_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    l = left.copy()
    r = right.copy()
    for k in keys:
        if k not in l.columns or k not in r.columns:
            continue
        if k in ("train_size", "run", "seed", "val_size"):
            l[k] = pd.to_numeric(l[k], errors="coerce")
            r[k] = pd.to_numeric(r[k], errors="coerce")
        elif k == "ctr":
            l[k] = pd.to_numeric(l[k], errors="coerce")
            r[k] = pd.to_numeric(r[k], errors="coerce")
        else:
            l[k] = l[k].astype(str)
            r[k] = r[k].astype(str)
    return l, r


def _apply_selected_reward(runs: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or best.empty:
        return runs
    out = runs.copy()
    keys = [c for c in _merge_keys(out) if c in best.columns]
    if len(keys) < 4:
        return out
    out, best_aligned = _coerce_merge_keys(out, best, keys)
    merged = out.merge(best_aligned, on=keys, how="left")
    sel = merged["selected_policy_reward"]
    for col in ("policy_rewards", "actual_reward_selected"):
        cur = pd.to_numeric(merged.get(col), errors="coerce")
        merged[col] = sel.where(sel.notna(), cur)
    merged = merged.drop(columns=["selected_policy_reward"], errors="ignore")
    return merged


def _enrich_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    extras = []
    for _, row in df.iterrows():
        extras.append(enrich_summary_pct_fields(row.to_dict()))
    extra_df = pd.DataFrame(extras)
    out = df.copy()
    for col in extra_df.columns:
        out[col] = extra_df[col].values
    return out


def rebuild_run_summaries(run_dir: Path, *, write_condition_csv: bool = True) -> pd.DataFrame:
    run_dir = Path(run_dir).resolve()
    trials = _load_trials_long_union(run_dir)
    best = _best_trial_actuals(trials)

    run_frames = []
    cond_dirs = sorted(
        {
            p.parent
            for pat in ("opc_runs_long.csv", "runs_long.csv", "no_prop_runs_long.csv")
            for p in run_dir.rglob(pat)
            if p.parent.name.startswith("dataset=")
        }
    )
    for cond_dir in cond_dirs:
        path = None
        for name in ("opc_runs_long.csv", "runs_long.csv", "no_prop_runs_long.csv"):
            candidate = cond_dir / name
            if candidate.exists():
                path = candidate
                break
        if path is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        tags = _parse_condition_dirname(cond_dir.name)
        for src, dst in (
            ("dataset", "dataset"),
            ("noise", "noise_mode"),
            ("axis", "noise_axis"),
            ("level", "noise_level"),
            ("seed", "seed"),
            ("ctr", "ctr"),
            ("val", "val_size"),
        ):
            if src in tags and dst not in df.columns:
                df[dst] = tags[src]
        df = _apply_selected_reward(df, best)
        df = _enrich_pct_columns(df)
        if write_condition_csv:
            df.to_csv(path, index=False)
        run_frames.append(df)

    if not run_frames:
        return pd.DataFrame()

    all_runs = pd.concat(run_frames, ignore_index=True)
    all_path = run_dir / "all_summary_metrics.csv"
    all_runs.to_csv(all_path, index=False)

    # Per-condition summary_metrics.csv (train_size index rows).
    if write_condition_csv:
        for cond_dir in sorted({p.parent for p in run_dir.rglob("opc_runs_long.csv")}):
            if not cond_dir.name.startswith("dataset="):
                continue
            part = all_runs
            tags = _parse_condition_dirname(cond_dir.name)
            for src, dst in (
                ("dataset", "dataset"),
                ("noise", "noise_mode"),
                ("axis", "noise_axis"),
                ("level", "noise_level"),
                ("seed", "seed"),
                ("ctr", "ctr"),
                ("val", "val_size"),
            ):
                if dst in part.columns and src in tags:
                    part = part[part[dst].astype(str) == str(tags[src])]
            if not part.empty:
                part.to_csv(cond_dir / "summary_metrics.csv", index=False)

    return all_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild all_summary_metrics from trial actual_reward (slim fix)."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--no-write-condition-csv",
        action="store_true",
        help="Only write all_summary_metrics.csv at run root.",
    )
    args = parser.parse_args()
    df = rebuild_run_summaries(
        args.run_dir,
        write_condition_csv=not args.no_write_condition_csv,
    )
    print(f"wrote {len(df)} rows -> {args.run_dir / 'all_summary_metrics.csv'}")


if __name__ == "__main__":
    main()
