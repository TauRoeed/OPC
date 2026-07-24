"""Load per-trial rows from a full-study run (trials_long or opc/no_prop Optuna CSVs)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_condition_dirname(name: str) -> dict:
    out = {}
    for part in name.split("__"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _attach_condition_tags(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.parent.name.startswith("dataset="):
        return df
    tags = _parse_condition_dirname(path.parent.name)
    mapping = {
        "dataset": "dataset",
        "noise": "noise_mode",
        "axis": "noise_axis",
        "level": "noise_level",
        "seed": "seed",
        "ctr": "ctr",
        "val": "val_size",
    }
    out = df.copy()
    for src, dst in mapping.items():
        if src not in tags:
            continue
        tagged = tags[src]
        if dst not in out.columns:
            out[dst] = tagged
        elif dst in ("seed", "ctr", "val_size"):
            current = pd.to_numeric(out[dst], errors="coerce")
            fill = pd.to_numeric(tagged, errors="coerce")
            out[dst] = current.fillna(fill)
    return out


def _seed_from_path(p: Path) -> int:
    m = re.search(r"__seed=(\d+)", str(p))
    return int(m.group(1)) if m else -1


def _parse_scores_dict(s) -> dict:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return {}
    if isinstance(s, dict):
        return s
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return {}


def _normalize_optuna_trials_csv(df: pd.DataFrame, method: str, seed: int) -> pd.DataFrame:
    n = len(df)
    scores = df.get("user_attrs_scores_dict")
    if scores is not None:
        parsed = [_parse_scores_dict(s) for s in scores]
        r_hat_train = [d.get("r_hat_train", np.nan) for d in parsed]
    else:
        r_hat_train = [np.nan] * n
    return pd.DataFrame(
        {
            "method": [method] * n,
            "seed": [int(seed)] * n,
            "trial_number": np.arange(n, dtype=int),
            "value": pd.to_numeric(df.get("value"), errors="coerce"),
            "r_hat": pd.to_numeric(df.get("user_attrs_r_hat"), errors="coerce"),
            "q_error": pd.to_numeric(df.get("user_attrs_q_error"), errors="coerce"),
            "ess": pd.to_numeric(df.get("user_attrs_ess"), errors="coerce"),
            "actual_reward": pd.to_numeric(df.get("user_attrs_actual_reward"), errors="coerce"),
            "r_hat_train": r_hat_train,
        }
    )


def _attach_initial_reward(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    need = "initial_reward" not in df.columns or not df["initial_reward"].notna().any()
    if not need:
        return df
    summary = run_dir / "all_summary_metrics.csv"
    if not summary.exists():
        return df
    sm = pd.read_csv(summary)
    if "initial_reward" not in sm.columns:
        return df
    keys = [k for k in ("method", "seed", "train_size") if k in sm.columns and k in df.columns]
    if not keys:
        keys = [k for k in ("method", "seed") if k in sm.columns and k in df.columns]
    if not keys:
        return df
    ir = sm.groupby(keys, as_index=False)["initial_reward"].first()
    if "initial_reward" in df.columns:
        df = df.drop(columns=["initial_reward"])
    return df.merge(ir, on=keys, how="left")


def load_run_trials(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    for pattern in (
        "trials_long.csv",
        "opc_trials_long.csv",
        "no_prop_trials_long.csv",
    ):
        long_paths = sorted(run_dir.rglob(pattern))
        if long_paths:
            parts = []
            for p in long_paths:
                df = pd.read_csv(p)
                if "seed" not in df.columns:
                    df["seed"] = _seed_from_path(p)
                df = _attach_condition_tags(df, p)
                parts.append(df)
            df = pd.concat(parts, ignore_index=True)
            return _attach_initial_reward(df, run_dir)

    parts = []
    for seed_dir in sorted(run_dir.glob("dataset=*__seed=*")):
        if not seed_dir.is_dir():
            continue
        seed = _seed_from_path(seed_dir)
        for method, fname in (("opc", "opc_trials.csv"), ("no_propensity", "no_prop_trials.csv")):
            p = seed_dir / fname
            if not p.exists():
                continue
            parts.append(_normalize_optuna_trials_csv(pd.read_csv(p), method, seed))

    if not parts:
        raise FileNotFoundError(f"no trials_long or opc/no_prop trials under {run_dir}")

    df = pd.concat(parts, ignore_index=True)
    return _attach_initial_reward(df, run_dir)
