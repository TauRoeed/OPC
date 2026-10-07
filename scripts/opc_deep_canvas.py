#!/usr/bin/env python3
"""OPC-only deep analysis canvas: improvement, struggle, restored CTR.

All primary tables are separated by train_size. Learning-curve lifts compare
R at smallest vs largest train size with explicit names (not min/max reward).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL_ORDER = ["low", "medium", "high", "extreme", "brutal", "catastrophic"]
SETUP_KEYS = ["dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "train_size"]


def parse_condition_dirname(name: str) -> dict:
    out = {}
    for part in name.split("__"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k] = v
    mapping = {
        "dataset": "dataset",
        "noise": "noise_mode",
        "axis": "noise_axis",
        "level": "noise_level",
        "ctr": "ctr",
        "seed": "seed",
        "val": "val_size",
    }
    return {mapping.get(k, k): v for k, v in out.items()}


def load_opc_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows, trial_rows = [], []
    for cond in sorted(run_dir.glob("dataset=*")):
        tags = parse_condition_dirname(cond.name)
        rp, tp = cond / "runs_long.csv", cond / "trials_long.csv"
        if not rp.exists() or not tp.exists():
            continue
        runs = pd.read_csv(rp)
        trials = pd.read_csv(tp)
        for k, v in tags.items():
            if k not in runs.columns or runs[k].isna().all():
                runs[k] = v
            if k not in trials.columns or trials[k].isna().all():
                trials[k] = v
        run_rows.append(runs)
        trial_rows.append(trials)
    if not run_rows:
        raise SystemExit(f"No conditions under {run_dir}")
    runs = pd.concat(run_rows, ignore_index=True)
    trials = pd.concat(trial_rows, ignore_index=True)
    runs = runs[runs["method"].astype(str) == "opc"].copy()
    trials = trials[trials["method"].astype(str) == "opc"].copy()
    return runs, trials


def coerce(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def restored_ctr_frac(reward: float, initial: float, ctr: float) -> float:
    """(R - R_log) / (CTR - R_log)."""
    r, ir, c = float(reward), float(initial), float(ctr)
    if not (np.isfinite(r) and np.isfinite(ir) and np.isfinite(c)):
        return float("nan")
    denom = c - ir
    if abs(denom) < 1e-12:
        return float("nan")
    return (r - ir) / denom


def reward_over_ctr(reward: float, ctr: float) -> float:
    r, c = float(reward), float(ctr)
    if not (np.isfinite(r) and np.isfinite(c)) or abs(c) < 1e-12:
        return float("nan")
    return r / c


def pct_vs(value: float, baseline: float) -> float:
    v, b = float(value), float(baseline)
    if not (np.isfinite(v) and np.isfinite(b)) or abs(b) < 1e-12:
        return float("nan")
    return 100.0 * (v - b) / abs(b)


def selected_opc(runs: pd.DataFrame) -> pd.DataFrame:
    d = coerce(
        runs,
        [
            "train_size",
            "seed",
            "ctr",
            "policy_rewards",
            "initial_reward",
            "actual_reward_selected",
            "action_delta",
            "context_delta",
            "conv_dr",
            "conv_sndr",
        ],
    )
    if "is_winning_run" in d.columns:
        win = d[d["is_winning_run"].astype(bool)].copy()
        if win.empty:
            win = d.copy()
    else:
        win = d.copy()
    reward = win["policy_rewards"]
    if "actual_reward_selected" in win.columns:
        reward = reward.where(reward.notna(), win["actual_reward_selected"])
    win = win.assign(reward=reward)
    keys = [c for c in SETUP_KEYS + ["seed"] if c in win.columns]
    g = win.groupby(keys, as_index=False).agg(
        reward=("reward", "mean"),
        initial_reward=("initial_reward", "mean"),
        ctr=("ctr", "mean"),
        action_delta=("action_delta", "mean"),
        context_delta=("context_delta", "mean"),
        conv_dr=("conv_dr", "mean"),
        conv_sndr=("conv_sndr", "mean"),
    )
    g["metric"] = "selected"
    return g


def oracle_opc(trials: pd.DataFrame) -> pd.DataFrame:
    d = coerce(trials, ["train_size", "seed", "ctr", "actual_reward", "initial_reward"])
    keys = [c for c in SETUP_KEYS + ["seed"] if c in d.columns]
    g = d.groupby(keys, as_index=False).agg(
        reward=("actual_reward", "max"),
        initial_reward=("initial_reward", "mean"),
        ctr=("ctr", "mean"),
    )
    g["metric"] = "oracle"
    return g


def mean_trials_opc(trials: pd.DataFrame) -> pd.DataFrame:
    d = coerce(trials, ["train_size", "seed", "ctr", "actual_reward", "initial_reward"])
    keys = [c for c in SETUP_KEYS + ["seed"] if c in d.columns]
    g = d.groupby(keys, as_index=False).agg(
        reward=("actual_reward", "mean"),
        initial_reward=("initial_reward", "mean"),
        ctr=("ctr", "mean"),
    )
    g["metric"] = "mean_trials"
    return g


def enrich_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pct_vs_logging"] = [
        pct_vs(r, i) for r, i in zip(out["reward"], out["initial_reward"])
    ]
    out["pct_vs_ctr"] = [pct_vs(r, c) for r, c in zip(out["reward"], out["ctr"])]
    out["restored_ctr"] = [
        restored_ctr_frac(r, i, c)
        for r, i, c in zip(out["reward"], out["initial_reward"], out["ctr"])
    ]
    out["reward_over_ctr"] = [
        reward_over_ctr(r, c) for r, c in zip(out["reward"], out["ctr"])
    ]
    # per-cell struggle at this train size
    out["struggle_worse_than_logging"] = out["reward"] < out["initial_reward"]
    out["struggle_low_restore"] = out["restored_ctr"] < 0.25
    out["struggle"] = out["struggle_worse_than_logging"] | out["struggle_low_restore"]
    return out


def seed_average(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in SETUP_KEYS + ["metric"] if c in df.columns]
    num_cols = [
        "reward",
        "initial_reward",
        "ctr",
        "pct_vs_logging",
        "pct_vs_ctr",
        "restored_ctr",
        "reward_over_ctr",
        "action_delta",
        "context_delta",
        "conv_dr",
        "conv_sndr",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    bool_cols = ["struggle", "struggle_worse_than_logging", "struggle_low_restore"]
    agg = {c: "mean" for c in num_cols}
    for c in bool_cols:
        if c in df.columns:
            agg[c] = "mean"  # fraction of seeds
    agg["seed"] = "count"
    g = df.groupby(keys, as_index=False).agg(agg).rename(columns={"seed": "n_seeds"})
    se = (
        df.groupby(keys)[["reward", "restored_ctr", "pct_vs_logging"]]
        .std()
        .reset_index()
        .rename(
            columns={
                "reward": "reward_std",
                "restored_ctr": "restored_ctr_std",
                "pct_vs_logging": "pct_vs_logging_std",
            }
        )
    )
    g = g.merge(se, on=keys, how="left")
    g["reward_se"] = g["reward_std"] / np.sqrt(g["n_seeds"].clip(lower=1))
    g["restored_ctr_se"] = g["restored_ctr_std"] / np.sqrt(g["n_seeds"].clip(lower=1))
    # seed-averaged struggle if majority of seeds struggle or mean restored low
    g["struggle"] = (g["struggle"] >= 0.5) | (g["restored_ctr"] < 0.25) | (
        g["reward"] < g["initial_reward"]
    )
    return g


def learning_curves(agg: pd.DataFrame) -> pd.DataFrame:
    """Compare R at smallest vs largest train_size (explicit names)."""
    rows = []
    group_cols = [
        c
        for c in ("dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "metric")
        if c in agg.columns
    ]
    for key, part in agg.groupby(group_cols):
        part = part.sort_values("train_size")
        if len(part) < 2:
            continue
        t = part["train_size"].to_numpy(dtype=float)
        r = part["reward"].to_numpy(dtype=float)
        if np.all(np.isfinite(r)) and np.ptp(t) > 0:
            slope_per_10k = float(np.polyfit(t, r, 1)[0] * 10_000.0)
        else:
            slope_per_10k = float("nan")
        n0, n1 = int(t[0]), int(t[-1])
        r0, r1 = float(r[0]), float(r[-1])
        restored = part["restored_ctr"].to_numpy(dtype=float)
        rest0, rest1 = float(restored[0]), float(restored[-1])
        lift_pct = pct_vs(r1, r0)
        # step lifts between consecutive train sizes
        step_rows = []
        for i in range(1, len(part)):
            a, b = part.iloc[i - 1], part.iloc[i]
            step_rows.append(
                {
                    "from_train": int(a["train_size"]),
                    "to_train": int(b["train_size"]),
                    "lift_pct": pct_vs(float(b["reward"]), float(a["reward"])),
                    "reward_delta": float(b["reward"]) - float(a["reward"]),
                }
            )
        flat = bool(np.isfinite(lift_pct) and lift_pct < 0.5)
        regress = bool(np.isfinite(lift_pct) and lift_pct < -0.25)
        low_restore = bool(np.isfinite(rest1) and rest1 < 0.25)
        if isinstance(key, tuple):
            meta = dict(zip(group_cols, key))
        else:
            meta = {group_cols[0]: key}
        rows.append(
            {
                **meta,
                "n_train_small": n0,
                "n_train_large": n1,
                "R_at_n_small": r0,
                "R_at_n_large": r1,
                "reward_delta_large_minus_small": r1 - r0,
                "lift_pct_large_vs_small": lift_pct,
                "slope_per_10k": slope_per_10k,
                "restored_at_n_small": rest0,
                "restored_at_n_large": rest1,
                "restored_delta_large_minus_small": rest1 - rest0
                if np.isfinite(rest1) and np.isfinite(rest0)
                else float("nan"),
                "step_lifts": step_rows,
                "struggle_flat_learning": flat,
                "struggle_regress_learning": regress,
                "struggle_low_restore_at_large_n": low_restore,
                "struggle_learning": flat or regress or low_restore,
            }
        )
    return pd.DataFrame(rows)


def by_train_summary(agg: pd.DataFrame) -> pd.DataFrame:
    """Averages / struggle rates separated by train_size."""
    rows = []
    for metric, mpart in agg.groupby("metric"):
        for train_size, tpart in mpart.groupby("train_size"):
            rows.append(
                {
                    "scope": "all",
                    "metric": metric,
                    "train_size": int(train_size),
                    "n_cells": int(len(tpart)),
                    "mean_reward": float(tpart["reward"].mean()),
                    "mean_logging_reward": float(tpart["initial_reward"].mean()),
                    "mean_restored_ctr": float(tpart["restored_ctr"].mean()),
                    "mean_pct_vs_logging": float(tpart["pct_vs_logging"].mean()),
                    "frac_struggle": float(tpart["struggle"].mean()),
                    "frac_worse_than_logging": float(
                        (tpart["reward"] < tpart["initial_reward"]).mean()
                    ),
                    "frac_low_restore": float((tpart["restored_ctr"] < 0.25).mean()),
                }
            )
            for axis, ap in tpart.groupby("noise_axis"):
                rows.append(
                    {
                        "scope": f"axis={axis}",
                        "metric": metric,
                        "train_size": int(train_size),
                        "n_cells": int(len(ap)),
                        "mean_reward": float(ap["reward"].mean()),
                        "mean_logging_reward": float(ap["initial_reward"].mean()),
                        "mean_restored_ctr": float(ap["restored_ctr"].mean()),
                        "mean_pct_vs_logging": float(ap["pct_vs_logging"].mean()),
                        "frac_struggle": float(ap["struggle"].mean()),
                        "frac_worse_than_logging": float(
                            (ap["reward"] < ap["initial_reward"]).mean()
                        ),
                        "frac_low_restore": float((ap["restored_ctr"] < 0.25).mean()),
                    }
                )
            for level, lp in tpart.groupby("noise_level"):
                rows.append(
                    {
                        "scope": f"level={level}",
                        "metric": metric,
                        "train_size": int(train_size),
                        "n_cells": int(len(lp)),
                        "mean_reward": float(lp["reward"].mean()),
                        "mean_logging_reward": float(lp["initial_reward"].mean()),
                        "mean_restored_ctr": float(lp["restored_ctr"].mean()),
                        "mean_pct_vs_logging": float(lp["pct_vs_logging"].mean()),
                        "frac_struggle": float(lp["struggle"].mean()),
                        "frac_worse_than_logging": float(
                            (lp["reward"] < lp["initial_reward"]).mean()
                        ),
                        "frac_low_restore": float((lp["restored_ctr"] < 0.25).mean()),
                    }
                )
    return pd.DataFrame(rows)


def level_key(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return 999


def _clean(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def build_payload(agg: pd.DataFrame, curves: pd.DataFrame, by_train: pd.DataFrame) -> dict:
    axes = sorted(agg["noise_axis"].dropna().unique().tolist())
    levels = sorted(agg["noise_level"].dropna().unique().tolist(), key=level_key)
    trains = sorted(pd.to_numeric(agg["train_size"], errors="coerce").dropna().unique().tolist())
    metrics = ["selected", "oracle", "mean_trials"]

    rows = []
    for _, r in agg.iterrows():
        rows.append(
            {
                k: _clean(r.get(k))
                for k in [
                    "dataset",
                    "noise_mode",
                    "noise_axis",
                    "noise_level",
                    "ctr",
                    "train_size",
                    "metric",
                    "reward",
                    "initial_reward",
                    "pct_vs_logging",
                    "pct_vs_ctr",
                    "restored_ctr",
                    "reward_over_ctr",
                    "reward_se",
                    "restored_ctr_se",
                    "n_seeds",
                    "struggle",
                ]
            }
        )

    curve_rows = []
    for _, r in curves.iterrows():
        curve_rows.append(
            {
                k: _clean(r.get(k))
                for k in [
                    "dataset",
                    "noise_mode",
                    "noise_axis",
                    "noise_level",
                    "ctr",
                    "metric",
                    "n_train_small",
                    "n_train_large",
                    "R_at_n_small",
                    "R_at_n_large",
                    "reward_delta_large_minus_small",
                    "lift_pct_large_vs_small",
                    "slope_per_10k",
                    "restored_at_n_small",
                    "restored_at_n_large",
                    "restored_delta_large_minus_small",
                    "struggle_learning",
                    "struggle_flat_learning",
                    "struggle_regress_learning",
                    "struggle_low_restore_at_large_n",
                ]
            }
        )

    by_train_rows = [{k: _clean(v) for k, v in r.items()} for r in by_train.to_dict(orient="records")]

    return {
        "run_tag": "run_bias_axes_l5_t20_s5",
        "axes": axes,
        "levels": levels,
        "train_sizes": [int(x) for x in trains],
        "metrics": metrics,
        "rows": rows,
        "curves": curve_rows,
        "by_train": by_train_rows,
        "definitions": {
            "selected": "policy_rewards of HPO-selected OPC run (is_winning_run)",
            "oracle": "max actual_reward over Optuna trials at that train_size",
            "mean_trials": "mean actual_reward over Optuna trials at that train_size",
            "R": "true policy reward (seed-averaged)",
            "R_log": "initial_reward = logging policy true reward",
            "CTR": "logging CTR target (0.05)",
            "restored_ctr": "(R - R_log) / (CTR - R_log)",
            "pct_vs_logging": "100 * (R - R_log) / |R_log|",
            "R_at_n_small": "R at smallest train_size (usually 5000), NOT min reward",
            "R_at_n_large": "R at largest train_size (usually 100000), NOT max reward",
            "lift_pct_large_vs_small": "100 * (R_at_n_large - R_at_n_small) / |R_at_n_small|",
            "struggle_at_train": "R < R_log OR restored_ctr < 0.25 (at that train_size)",
        },
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OPC deep analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;550;650&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg0:#10151c; --bg1:#1a222d; --bg2:#243041; --ink:#e8eef4; --muted:#9aabbc;
    --opc:#3dbb8f; --warn:#e07a5f; --line:#334155; --gold:#f0c14a;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--ink); font-family:"IBM Plex Sans",sans-serif;
    background:
      radial-gradient(1000px 500px at 0% -10%, #1a3a40 0%, transparent 55%),
      radial-gradient(800px 480px at 100% 0%, #2a2f48 0%, transparent 50%),
      var(--bg0);
    min-height:100vh;
  }
  header{max-width:1460px;margin:0 auto;padding:28px 32px 8px}
  header h1{margin:0 0 6px;font-family:"IBM Plex Serif",Georgia,serif;font-size:1.85rem;font-weight:600}
  header p{margin:0;color:var(--muted);max-width:90ch;line-height:1.45}
  .defs{margin-top:10px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:var(--bg1);font-size:.82rem;color:var(--muted);line-height:1.45}
  .defs b{color:var(--ink)}
  .controls{display:flex;flex-wrap:wrap;gap:12px;max-width:1460px;margin:16px auto 0;padding:0 32px}
  label{display:flex;flex-direction:column;gap:4px;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  select,button{background:var(--bg1);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:.95rem;min-width:130px}
  button{cursor:pointer;background:var(--bg2)}
  button.active{border-color:var(--opc);color:var(--opc)}
  .metric-toggle{display:flex;gap:8px;align-items:end}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;max-width:1460px;margin:16px auto 0;padding:0 32px}
  @media(max-width:1000px){.kpis{grid-template-columns:repeat(2,1fr)}}
  .kpi{background:var(--bg1);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .kpi .lbl{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}
  .kpi .val{font-size:1.35rem;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
  .kpi .sub{color:var(--muted);font-size:.78rem;margin-top:2px}
  .grid{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;max-width:1460px;margin:18px auto 40px;padding:0 32px}
  @media(max-width:980px){.grid{grid-template-columns:1fr}}
  .card{background:color-mix(in srgb,var(--bg1) 90%,transparent);border:1px solid var(--line);border-radius:14px;padding:16px 18px}
  .card h2{margin:0 0 10px;font-size:1rem}
  .full{grid-column:1/-1}
  canvas.chart{width:100%;height:360px;display:block}
  .legend{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:.85rem;margin-bottom:8px}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
  .note{color:var(--muted);font-size:.8rem;margin-top:8px}
  table{width:100%;border-collapse:collapse;font-size:.84rem;font-variant-numeric:tabular-nums}
  th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
  th{color:var(--muted);font-weight:550;font-size:.68rem;text-transform:uppercase;letter-spacing:.04em}
  tr:hover td{background:color-mix(in srgb,var(--bg2) 55%,transparent)}
  .pos{color:var(--opc)} .neg{color:var(--warn)} .flag{color:var(--gold);font-weight:650}
  .scroll{max-height:460px;overflow:auto}
</style>
</head>
<body>
<header>
  <h1>OPC deep dive</h1>
  <p>OPC only (<code>run_bias_axes_l5_t20_s5</code>). Primary view is <b>per train size</b>. Learning lift compares R at small n vs large n with explicit names.</p>
  <div class="defs">
    <b>R</b> = OPC true reward &nbsp;|&nbsp;
    <b>R_log</b> = logging reward (<code>initial_reward</code>) &nbsp;|&nbsp;
    <b>CTR</b> = 0.05 &nbsp;|&nbsp;
    <b>restored CTR</b> = (R − R_log) / (CTR − R_log) &nbsp;|&nbsp;
    <b>Selected</b> = HPO-picked policy &nbsp;|&nbsp;
    <b>Oracle</b> = best trial actual_reward &nbsp;|&nbsp;
    <b>Mean trials</b> = mean trial actual_reward
  </div>
</header>

<div class="controls">
  <div class="metric-toggle" id="metricToggle"></div>
  <label>Train size<select id="trainSel"></select></label>
  <label>Axis<select id="axisSel"></select></label>
  <label>Level<select id="levelSel"></select></label>
  <label>View
    <select id="viewSel">
      <option value="restore_heat">Restored CTR heatmap (this train size)</option>
      <option value="reward_heat">Reward heatmap (this train size)</option>
      <option value="restored">Restored CTR vs all train sizes</option>
      <option value="reward">Reward vs all train sizes</option>
      <option value="lift">Lift % : R@large_n vs R@small_n</option>
    </select>
  </label>
</div>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="card">
    <h2 id="chartTitle">Chart</h2>
    <div class="legend" id="legend"></div>
    <canvas class="chart" id="mainChart" width="920" height="360"></canvas>
    <p class="note" id="chartNote"></p>
  </div>
  <div class="card">
    <h2>Averages at selected train size</h2>
    <div class="scroll">
      <table id="byTrainTable">
        <thead>
          <tr>
            <th>Scope</th><th>Train n</th><th>R</th><th>R_log</th><th>Restored CTR</th><th>% vs log</th><th>Struggle%</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card full">
    <h2>Per train size × axis × level</h2>
    <div class="scroll">
      <table id="cellTable">
        <thead>
          <tr>
            <th>Axis</th><th>Level</th><th>Train n</th><th>R</th><th>R_log</th>
            <th>% vs logging</th><th>Restored CTR</th><th>R/CTR</th><th>Struggle?</th><th>Seeds</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="card full">
    <h2>Learning across train sizes (same noise setup)</h2>
    <p class="note" style="margin-top:0">
      <b>R @ n_small</b> / <b>R @ n_large</b> = reward at smallest / largest train size (not min/max of rewards).
      Negative lift means R at large n is slightly below R at small n.
    </p>
    <div class="scroll">
      <table id="curveTable">
        <thead>
          <tr>
            <th>Axis</th><th>Level</th>
            <th>n_small</th><th>R @ n_small</th>
            <th>n_large</th><th>R @ n_large</th>
            <th>ΔR (large−small)</th><th>Lift %</th><th>Slope/10k</th>
            <th>Restored @ n_small</th><th>Restored @ n_large</th>
            <th>Learning struggle?</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const LEVEL_COLORS = {
  low:'#7dd3a7', medium:'#6ea8fe', high:'#f0c14a', extreme:'#e07a5f', brutal:'#c084fc', catastrophic:'#fb7185'
};
let state = {
  metric:'selected',
  train: String(DATA.train_sizes[0]),
  axis:'all',
  level:'all',
  view:'restore_heat'
};

function fmt(x,d=5){ if(x==null||Number.isNaN(x)) return '—'; return Number(x).toFixed(d); }
function fmtPct(x,d=2){ if(x==null||Number.isNaN(x)) return '—'; return (x>=0?'+':'')+Number(x).toFixed(d)+'%'; }
function cls(x){ return x>0?'pos':(x<0?'neg':''); }
function avg(xs){ const a=xs.filter(x=>x!=null&&!Number.isNaN(x)); return a.length?a.reduce((p,c)=>p+c,0)/a.length:NaN; }

function init(){
  const mt=document.getElementById('metricToggle');
  const labels={selected:'Selected', oracle:'Oracle', mean_trials:'Mean trials'};
  DATA.metrics.forEach(m=>{
    const b=document.createElement('button');
    b.textContent=labels[m]||m; b.dataset.metric=m;
    if(m===state.metric) b.classList.add('active');
    b.onclick=()=>{state.metric=m; sync();};
    mt.appendChild(b);
  });
  const trainSel=document.getElementById('trainSel');
  trainSel.innerHTML = DATA.train_sizes.map(t=>`<option value="${t}">n = ${t}</option>`).join('');
  trainSel.value = state.train;
  trainSel.onchange=e=>{state.train=e.target.value; sync();};
  const axisSel=document.getElementById('axisSel');
  axisSel.innerHTML='<option value="all">All axes</option>'+DATA.axes.map(a=>`<option value="${a}">${a}</option>`).join('');
  axisSel.onchange=e=>{state.axis=e.target.value; sync();};
  const levelSel=document.getElementById('levelSel');
  levelSel.innerHTML='<option value="all">All levels</option>'+DATA.levels.map(a=>`<option value="${a}">${a}</option>`).join('');
  levelSel.onchange=e=>{state.level=e.target.value; sync();};
  document.getElementById('viewSel').onchange=e=>{state.view=e.target.value; sync();};
}

function filterRows(forTrainOnly){
  return DATA.rows.filter(r=>{
    if(r.metric!==state.metric) return false;
    if(forTrainOnly && String(r.train_size)!==String(state.train)) return false;
    if(state.axis!=='all' && r.noise_axis!==state.axis) return false;
    if(state.level!=='all' && r.noise_level!==state.level) return false;
    return true;
  });
}
function filterCurves(){
  return DATA.curves.filter(r=>{
    if(r.metric!==state.metric) return false;
    if(state.axis!=='all' && r.noise_axis!==state.axis) return false;
    if(state.level!=='all' && r.noise_level!==state.level) return false;
    return true;
  });
}

function renderKpis(rowsAtTrain, curves){
  const restored=avg(rowsAtTrain.map(r=>r.restored_ctr));
  const reward=avg(rowsAtTrain.map(r=>r.reward));
  const pct=avg(rowsAtTrain.map(r=>r.pct_vs_logging));
  const struggleN=rowsAtTrain.filter(r=>r.struggle).length;
  const lift=avg(curves.map(r=>r.lift_pct_large_vs_small));
  document.getElementById('kpis').innerHTML=`
    <div class="kpi"><div class="lbl">R at n=${state.train}</div><div class="val">${fmt(reward)}</div><div class="sub">${rowsAtTrain.length} cells</div></div>
    <div class="kpi"><div class="lbl">Restored CTR @ n=${state.train}</div><div class="val ${cls(restored)}">${fmt(restored,3)}</div><div class="sub">(R−R_log)/(CTR−R_log)</div></div>
    <div class="kpi"><div class="lbl">% vs logging @ n</div><div class="val ${cls(pct)}">${fmtPct(pct)}</div><div class="sub">100*(R−R_log)/|R_log|</div></div>
    <div class="kpi"><div class="lbl">Struggle @ n</div><div class="val ${struggleN?'flag':''}">${struggleN}/${rowsAtTrain.length}</div><div class="sub">R&lt;R_log or restored&lt;0.25</div></div>
    <div class="kpi"><div class="lbl">Lift % n_large vs n_small</div><div class="val ${cls(lift)}">${fmtPct(lift)}</div><div class="sub">across filtered setups</div></div>
  `;
}

function renderByTrain(){
  const tb=document.querySelector('#byTrainTable tbody');
  const rows=DATA.by_train.filter(r=>r.metric===state.metric && String(r.train_size)===String(state.train));
  const order=s=>s==='all'?0:(s.startsWith('axis=')?1:2);
  rows.sort((a,b)=>order(a.scope)-order(b.scope)||a.scope.localeCompare(b.scope));
  tb.innerHTML=rows.map(r=>`
    <tr>
      <td>${r.scope}</td>
      <td>${r.train_size}</td>
      <td>${fmt(r.mean_reward)}</td>
      <td>${fmt(r.mean_logging_reward)}</td>
      <td class="${cls(r.mean_restored_ctr)}">${fmt(r.mean_restored_ctr,3)}</td>
      <td class="${cls(r.mean_pct_vs_logging)}">${fmtPct(r.mean_pct_vs_logging)}</td>
      <td class="${r.frac_struggle>0.3?'flag':''}">${(100*r.frac_struggle).toFixed(0)}%</td>
    </tr>`).join('');
}

function renderCells(rows){
  const tb=document.querySelector('#cellTable tbody');
  const sorted=[...rows].sort((a,b)=>
    a.noise_axis.localeCompare(b.noise_axis)||
    DATA.levels.indexOf(a.noise_level)-DATA.levels.indexOf(b.noise_level)||
    a.train_size-b.train_size
  );
  tb.innerHTML=sorted.map(r=>`
    <tr>
      <td>${r.noise_axis}</td><td>${r.noise_level}</td><td>${r.train_size}</td>
      <td>${fmt(r.reward)}</td>
      <td>${fmt(r.initial_reward)}</td>
      <td class="${cls(r.pct_vs_logging)}">${fmtPct(r.pct_vs_logging)}</td>
      <td class="${cls(r.restored_ctr)}">${fmt(r.restored_ctr,3)}</td>
      <td>${fmt(r.reward_over_ctr,3)}</td>
      <td class="${r.struggle?'flag':''}">${r.struggle?'YES':'no'}</td>
      <td>${r.n_seeds}</td>
    </tr>`).join('');
}

function renderCurves(curves){
  const tb=document.querySelector('#curveTable tbody');
  const sorted=[...curves].sort((a,b)=>
    a.noise_axis.localeCompare(b.noise_axis)||
    DATA.levels.indexOf(a.noise_level)-DATA.levels.indexOf(b.noise_level)
  );
  tb.innerHTML=sorted.map(r=>`
    <tr>
      <td>${r.noise_axis}</td><td>${r.noise_level}</td>
      <td>${r.n_train_small}</td><td>${fmt(r.R_at_n_small)}</td>
      <td>${r.n_train_large}</td><td>${fmt(r.R_at_n_large)}</td>
      <td class="${cls(r.reward_delta_large_minus_small)}">${fmt(r.reward_delta_large_minus_small,6)}</td>
      <td class="${cls(r.lift_pct_large_vs_small)}">${fmtPct(r.lift_pct_large_vs_small)}</td>
      <td class="${cls(r.slope_per_10k)}">${fmt(r.slope_per_10k,6)}</td>
      <td>${fmt(r.restored_at_n_small,3)}</td>
      <td class="${cls(r.restored_at_n_large)}">${fmt(r.restored_at_n_large,3)}</td>
      <td class="${r.struggle_learning?'flag':''}">${r.struggle_learning?'YES':'no'}</td>
    </tr>`).join('');
}

function clearCanvas(ctx,c){ ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#15202b'; ctx.fillRect(0,0,c.width,c.height); }
function drawFrame(ctx,c,pad,ymin,ymax,xlabel){
  ctx.strokeStyle='#334155'; ctx.fillStyle='#9aabbc'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,c.height-pad.b); ctx.lineTo(c.width-pad.r,c.height-pad.b); ctx.stroke();
  ctx.font='12px IBM Plex Sans,sans-serif';
  for(let i=0;i<=4;i++){
    const yv=ymin+(ymax-ymin)*(i/4);
    const y=c.height-pad.b-(i/4)*(c.height-pad.t-pad.b);
    ctx.fillText(yv.toFixed(3), 6, y+4);
    ctx.strokeStyle='#1f2a36'; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(c.width-pad.r,y); ctx.stroke();
  }
  ctx.fillStyle='#9aabbc'; ctx.fillText(xlabel, c.width/2-40, c.height-8);
}
function XYf(pad,c,xmin,xmax,ymin,ymax,x,y){
  const X=pad.l+((x-xmin)/(xmax-xmin||1))*(c.width-pad.l-pad.r);
  const Y=c.height-pad.b-((y-ymin)/(ymax-ymin||1))*(c.height-pad.t-pad.b);
  return [X,Y];
}

function drawLines(ctx,c,rows,ykey){
  clearCanvas(ctx,c);
  const pad={l:64,r:16,t:16,b:36};
  const trains=DATA.train_sizes;
  const levels=state.level==='all'?DATA.levels:[state.level];
  const out={};
  levels.forEach(lv=>{
    out[lv]=trains.map(t=>{
      const cell=rows.filter(r=>r.noise_level===lv && r.train_size===t);
      return avg(cell.map(r=>r[ykey]));
    });
  });
  const vals=[].concat(...Object.values(out)).filter(v=>v!=null&&!Number.isNaN(v));
  if(!vals.length) return;
  const ymin=Math.min(...vals)-0.02*Math.abs(Math.min(...vals)||1);
  const ymax=Math.max(...vals)+0.02*Math.abs(Math.max(...vals)||1);
  drawFrame(ctx,c,pad,ymin,ymax,'train_size');
  trains.forEach((t,i)=>{
    const [X]=XYf(pad,c,0,trains.length-1,ymin,ymax,i,ymin);
    ctx.fillStyle='#9aabbc'; ctx.fillText(String(t), X-16, c.height-18);
  });
  if(ykey==='restored_ctr'){
    [0,1].forEach(ref=>{
      const [,Y]=XYf(pad,c,0,trains.length-1,ymin,ymax,0,ref);
      ctx.strokeStyle=ref===1?'#f0c14a':'#64748b'; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(pad.l,Y); ctx.lineTo(c.width-pad.r,Y); ctx.stroke();
      ctx.setLineDash([]);
    });
  }
  levels.forEach(lv=>{
    const series=out[lv];
    ctx.strokeStyle=LEVEL_COLORS[lv]||'#fff'; ctx.fillStyle=ctx.strokeStyle; ctx.lineWidth=2.4;
    ctx.beginPath(); let started=false;
    series.forEach((v,i)=>{
      if(v==null||Number.isNaN(v)) return;
      const [X,Y]=XYf(pad,c,0,trains.length-1,ymin,ymax,i,v);
      if(!started){ctx.moveTo(X,Y); started=true;} else ctx.lineTo(X,Y);
    });
    ctx.stroke();
    series.forEach((v,i)=>{
      if(v==null||Number.isNaN(v)) return;
      const [X,Y]=XYf(pad,c,0,trains.length-1,ymin,ymax,i,v);
      ctx.beginPath(); ctx.arc(X,Y,3.5,0,Math.PI*2); ctx.fill();
    });
  });
}

function drawHeatFromRows(ctx,c,rows,key){
  clearCanvas(ctx,c);
  const axes=state.axis==='all'?DATA.axes:[state.axis];
  const levels=state.level==='all'?DATA.levels:[state.level];
  const pad={l:90,r:16,t:24,b:40};
  const cw=(c.width-pad.l-pad.r)/Math.max(levels.length,1);
  const ch=(c.height-pad.t-pad.b)/Math.max(axes.length,1);
  const cell={};
  rows.forEach(r=>{ cell[r.noise_axis+'||'+r.noise_level]=r; });
  let maxAbs=1e-9;
  Object.values(cell).forEach(r=>{ if(r&&r[key]!=null) maxAbs=Math.max(maxAbs, Math.abs(r[key])); });
  axes.forEach((ax,i)=>{
    levels.forEach((lv,j)=>{
      const r=cell[ax+'||'+lv];
      const m=r?r[key]:null;
      const x=pad.l+j*cw, y=pad.t+i*ch;
      let color='#1f2a36';
      if(m!=null && !Number.isNaN(m)){
        const t=Math.min(1, Math.abs(m)/maxAbs);
        color = m>=0 ? `rgba(61,187,143,${0.25+0.75*t})` : `rgba(224,122,95,${0.25+0.75*t})`;
      }
      ctx.fillStyle=color; ctx.fillRect(x+2,y+2,cw-4,ch-4);
      ctx.fillStyle='#e8eef4'; ctx.font='12px IBM Plex Sans,sans-serif';
      const txt = (m==null||Number.isNaN(m)) ? '—' : (key.includes('pct')?((m>=0?'+':'')+m.toFixed(2)+'%'):m.toFixed(3));
      ctx.fillText(txt, x+8, y+ch/2+4);
      if(r && r.struggle){
        ctx.strokeStyle='#f0c14a'; ctx.lineWidth=2; ctx.strokeRect(x+3,y+3,cw-6,ch-6);
      }
      if(i===axes.length-1){ ctx.fillStyle='#9aabbc'; ctx.fillText(lv, x+6, c.height-16); }
    });
    ctx.fillStyle='#9aabbc'; ctx.fillText(ax, 10, pad.t+i*ch+ch/2+4);
  });
}

function drawHeatFromCurves(ctx,c,curves,key){
  clearCanvas(ctx,c);
  const axes=state.axis==='all'?DATA.axes:[state.axis];
  const levels=state.level==='all'?DATA.levels:[state.level];
  const pad={l:90,r:16,t:24,b:40};
  const cw=(c.width-pad.l-pad.r)/Math.max(levels.length,1);
  const ch=(c.height-pad.t-pad.b)/Math.max(axes.length,1);
  const cell={};
  curves.forEach(r=>{ cell[r.noise_axis+'||'+r.noise_level]=r; });
  let maxAbs=1e-9;
  Object.values(cell).forEach(r=>{ if(r&&r[key]!=null) maxAbs=Math.max(maxAbs, Math.abs(r[key])); });
  axes.forEach((ax,i)=>{
    levels.forEach((lv,j)=>{
      const r=cell[ax+'||'+lv];
      const m=r?r[key]:null;
      const x=pad.l+j*cw, y=pad.t+i*ch;
      let color='#1f2a36';
      if(m!=null && !Number.isNaN(m)){
        const t=Math.min(1, Math.abs(m)/maxAbs);
        color = m>=0 ? `rgba(61,187,143,${0.25+0.75*t})` : `rgba(224,122,95,${0.25+0.75*t})`;
      }
      ctx.fillStyle=color; ctx.fillRect(x+2,y+2,cw-4,ch-4);
      ctx.fillStyle='#e8eef4'; ctx.font='12px IBM Plex Sans,sans-serif';
      const txt = (m==null||Number.isNaN(m)) ? '—' : (key.includes('pct')?((m>=0?'+':'')+m.toFixed(2)+'%'):m.toFixed(3));
      ctx.fillText(txt, x+8, y+ch/2+4);
      if(r && r.struggle_learning){
        ctx.strokeStyle='#f0c14a'; ctx.lineWidth=2; ctx.strokeRect(x+3,y+3,cw-6,ch-6);
      }
      if(i===axes.length-1){ ctx.fillStyle='#9aabbc'; ctx.fillText(lv, x+6, c.height-16); }
    });
    ctx.fillStyle='#9aabbc'; ctx.fillText(ax, 10, pad.t+i*ch+ch/2+4);
  });
}

function renderChart(rowsAll, rowsAtTrain, curves){
  const canvas=document.getElementById('mainChart');
  const ctx=canvas.getContext('2d');
  const title=document.getElementById('chartTitle');
  const note=document.getElementById('chartNote');
  const legend=document.getElementById('legend');
  const levels=state.level==='all'?DATA.levels:[state.level];
  legend.innerHTML=levels.map(lv=>`<span><i class="swatch" style="background:${LEVEL_COLORS[lv]||'#fff'}"></i>${lv}</span>`).join('')
    + '<span><i class="swatch" style="background:#f0c14a"></i>gold = struggle</span>';

  if(state.view==='restore_heat'){
    title.textContent=`Restored CTR heatmap at train n=${state.train}`;
    note.textContent=DATA.definitions.restored_ctr;
    drawHeatFromRows(ctx,canvas,rowsAtTrain,'restored_ctr');
  } else if(state.view==='reward_heat'){
    title.textContent=`Reward heatmap at train n=${state.train}`;
    note.textContent=DATA.definitions.R;
    drawHeatFromRows(ctx,canvas,rowsAtTrain,'reward');
  } else if(state.view==='restored'){
    title.textContent='Restored CTR vs train size';
    note.textContent='Dashed lines: 0 = logging, 1 = CTR.';
    drawLines(ctx,canvas,rowsAll,'restored_ctr');
  } else if(state.view==='reward'){
    title.textContent='Reward vs train size';
    note.textContent='Seed-averaged R by noise level.';
    drawLines(ctx,canvas,rowsAll,'reward');
  } else {
    title.textContent='Lift % : R @ n_large vs R @ n_small';
    note.textContent=DATA.definitions.lift_pct_large_vs_small;
    drawHeatFromCurves(ctx,canvas,curves,'lift_pct_large_vs_small');
  }
}

function sync(){
  document.querySelectorAll('#metricToggle button').forEach(b=>b.classList.toggle('active', b.dataset.metric===state.metric));
  const rowsAll=filterRows(false);
  const rowsAtTrain=filterRows(true);
  const curves=filterCurves();
  renderKpis(rowsAtTrain, curves);
  renderByTrain();
  renderCells(rowsAtTrain);
  renderCurves(curves);
  renderChart(rowsAll, rowsAtTrain, curves);
}
init(); sync();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/full_study/run_bias_axes_l5_t20_s5"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/analysis/opc_deep"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs, trials = load_opc_frames(args.run_dir)
    long = pd.concat(
        [
            enrich_row_metrics(selected_opc(runs)),
            enrich_row_metrics(oracle_opc(trials)),
            enrich_row_metrics(mean_trials_opc(trials)),
        ],
        ignore_index=True,
    )
    agg = seed_average(long)
    curves = learning_curves(agg)
    by_train = by_train_summary(agg)

    long.to_csv(args.out_dir / "opc_long_by_seed.csv", index=False)
    agg.to_csv(args.out_dir / "opc_seed_averaged.csv", index=False)
    curves.drop(columns=["step_lifts"], errors="ignore").to_csv(
        args.out_dir / "opc_learning_curves.csv", index=False
    )
    by_train.to_csv(args.out_dir / "opc_by_train_summary.csv", index=False)

    payload = build_payload(agg, curves, by_train)
    (args.out_dir / "canvas_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = HTML.replace("__DATA_JSON__", json.dumps(payload))
    out_html = args.out_dir / "opc_deep_canvas.html"
    out_html.write_text(html, encoding="utf-8")

    art = Path("/opt/cursor/artifacts")
    if art.exists():
        (art / "opc_deep_canvas.html").write_text(html, encoding="utf-8")
        by_train.to_csv(art / "opc_deep_by_train.csv", index=False)

    print(f"wrote {out_html}")
    print(by_train[(by_train.metric == "selected") & (by_train.scope == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
