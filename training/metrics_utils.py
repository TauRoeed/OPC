"""Percent-change helpers for study metrics."""

from __future__ import annotations

import numpy as np

_POLICY_VALUE_COL = "policy_rewards"
_ESTIMATOR_COLS = ("ipw", "reg_dm", "conv_dm", "conv_dr", "conv_sndr")


def pct_change(value, baseline, eps: float = 1e-12) -> float:
    """100 * (value - baseline) / |baseline|; nan if baseline ~ 0."""
    v = float(value)
    b = float(baseline)
    if not np.isfinite(v) or not np.isfinite(b) or abs(b) < eps:
        return float("nan")
    return 100.0 * (v - b) / abs(b)


def enrich_trial_pct_fields(
    actual_reward: float,
    value: float,
    initial_reward: float,
) -> dict[str, float]:
    ir = float(initial_reward)
    return {
        "actual_reward_pct_vs_initial": pct_change(actual_reward, ir),
        "value_pct_vs_initial": pct_change(value, ir),
    }


def _scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def enrich_summary_pct_fields(row: dict) -> dict[str, float]:
    """Add % vs initial_reward, ctr, and OPE error % vs policy_rewards."""
    out: dict[str, float] = {}
    pr = row.get(_POLICY_VALUE_COL)
    ir = row.get("initial_reward")
    ctr = row.get("ctr")
    if pr is not None and ir is not None:
        out["policy_rewards_pct_vs_initial"] = pct_change(_scalar(pr), _scalar(ir))
    if pr is not None and ctr is not None:
        out["policy_rewards_pct_vs_ctr"] = pct_change(_scalar(pr), _scalar(ctr))
    if pr is not None:
        truth = _scalar(pr)
        for est in _ESTIMATOR_COLS:
            if est not in row:
                continue
            ev = row[est]
            out[f"{est}_pct_err_vs_truth"] = pct_change(ev, truth)
    return out


def add_paired_method_pct_columns(summary_df):
    """Add opc_vs_noprop_pct on policy_rewards (and actual if present)."""
    import pandas as pd

    df = summary_df.copy()
    idx = [
        c
        for c in (
            "train_size",
            "seed",
            "dataset",
            "noise_mode",
            "noise_axis",
            "noise_level",
            "ctr",
            "val_size",
            "val_size_config",
        )
        if c in df.columns
    ]
    if "method" not in df.columns or len(idx) < 1:
        return df

    piv = df.pivot_table(
        index=idx, columns="method", values="policy_rewards", aggfunc="first"
    )
    if "opc" not in piv.columns or "no_propensity" not in piv.columns:
        return df

    delta = piv["opc"] - piv["no_propensity"]
    denom = piv["no_propensity"].abs().clip(lower=1e-12)
    pct = (100.0 * delta / denom).replace([np.inf, -np.inf], np.nan)
    extra = pct.reset_index(name="opc_vs_noprop_pct")
    return df.merge(extra, on=idx, how="left")
