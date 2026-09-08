#!/usr/bin/env python3
"""OPC-only deep analysis canvas: improvement, struggle, restored CTR."""

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
    """Fraction of gap from logging reward toward CTR closed by learned policy.

    1.0 = fully restored to CTR; 0 = stuck at logging; <0 = worse than logging;
    >1 = beat CTR. NaN if CTR ≈ initial.
    """
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
            "selection_val_score",
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
    out["pct_vs_initial"] = [
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
    return out


def seed_average(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in SETUP_KEYS + ["metric"] if c in df.columns]
    num_cols = [
        "reward",
        "initial_reward",
        "ctr",
        "pct_vs_initial",
        "pct_vs_ctr",
        "restored_ctr",
        "reward_over_ctr",
        "action_delta",
        "context_delta",
        "conv_dr",
        "conv_sndr",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    agg = {c: "mean" for c in num_cols}
    agg["seed"] = "count"
    g = df.groupby(keys, as_index=False).agg(agg).rename(columns={"seed": "n_seeds"})
    # SE for restored + reward
    se = (
        df.groupby(keys)[["reward", "restored_ctr", "pct_vs_initial"]]
        .std()
        .reset_index()
        .rename(
            columns={
                "reward": "reward_std",
                "restored_ctr": "restored_ctr_std",
                "pct_vs_initial": "pct_vs_initial_std",
            }
        )
    )
    g = g.merge(se, on=keys, how="left")
    g["reward_se"] = g["reward_std"] / np.sqrt(g["n_seeds"].clip(lower=1))
    g["restored_ctr_se"] = g["restored_ctr_std"] / np.sqrt(g["n_seeds"].clip(lower=1))
    return g


def learning_curves(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (axis, level, metric): improvement rate across train sizes."""
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
        # linear slope per 10k train samples
        if np.all(np.isfinite(r)) and np.ptp(t) > 0:
            slope = np.polyfit(t, r, 1)[0]
            slope_per_10k = slope * 10_000.0
        else:
            slope_per_10k = float("nan")
        r0, r1 = float(r[0]), float(r[-1])
        t0, t1 = float(t[0]), float(t[-1])
        total_lift_pct = pct_vs(r1, r0)
        # consecutive step improvements
        step_lifts = []
        for i in range(1, len(r)):
            step_lifts.append(pct_vs(float(r[i]), float(r[i - 1])))
        mean_step = float(np.nanmean(step_lifts)) if step_lifts else float("nan")
        worst_step = float(np.nanmin(step_lifts)) if step_lifts else float("nan")
        restored = part["restored_ctr"].to_numpy(dtype=float)
        restored_start = float(restored[0]) if len(restored) else float("nan")
        restored_end = float(restored[-1]) if len(restored) else float("nan")
        # struggle flags
        flat = bool(np.isfinite(total_lift_pct) and total_lift_pct < 0.5)
        regress = bool(np.isfinite(total_lift_pct) and total_lift_pct < -0.25)
        low_restore = bool(np.isfinite(restored_end) and restored_end < 0.25)
        struggle = flat or regress or low_restore
        if isinstance(key, tuple):
            meta = dict(zip(group_cols, key))
        else:
            meta = {group_cols[0]: key}
        rows.append(
            {
                **meta,
                "train_min": t0,
                "train_max": t1,
                "reward_min_train": r0,
                "reward_max_train": r1,
                "total_lift_pct": total_lift_pct,
                "slope_per_10k": slope_per_10k,
                "mean_step_lift_pct": mean_step,
                "worst_step_lift_pct": worst_step,
                "restored_ctr_start": restored_start,
                "restored_ctr_end": restored_end,
                "restored_ctr_gain": restored_end - restored_start
                if np.isfinite(restored_end) and np.isfinite(restored_start)
                else float("nan"),
                "struggle": struggle,
                "struggle_flat": flat,
                "struggle_regress": regress,
                "struggle_low_restore": low_restore,
            }
        )
    return pd.DataFrame(rows)


def struggle_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, part in curves.groupby("metric"):
        n = len(part)
        rows.append(
            {
                "scope": "all",
                "metric": metric,
                "n_setups": n,
                "frac_struggle": float(part["struggle"].mean()) if n else 0.0,
                "frac_flat": float(part["struggle_flat"].mean()) if n else 0.0,
                "frac_regress": float(part["struggle_regress"].mean()) if n else 0.0,
                "frac_low_restore": float(part["struggle_low_restore"].mean()) if n else 0.0,
                "mean_total_lift_pct": float(part["total_lift_pct"].mean()),
                "mean_restored_end": float(part["restored_ctr_end"].mean()),
                "mean_slope_per_10k": float(part["slope_per_10k"].mean()),
            }
        )
        for axis, ap in part.groupby("noise_axis"):
            rows.append(
                {
                    "scope": f"axis={axis}",
                    "metric": metric,
                    "n_setups": len(ap),
                    "frac_struggle": float(ap["struggle"].mean()),
                    "frac_flat": float(ap["struggle_flat"].mean()),
                    "frac_regress": float(ap["struggle_regress"].mean()),
                    "frac_low_restore": float(ap["struggle_low_restore"].mean()),
                    "mean_total_lift_pct": float(ap["total_lift_pct"].mean()),
                    "mean_restored_end": float(ap["restored_ctr_end"].mean()),
                    "mean_slope_per_10k": float(ap["slope_per_10k"].mean()),
                }
            )
        for level, lp in part.groupby("noise_level"):
            rows.append(
                {
                    "scope": f"level={level}",
                    "metric": metric,
                    "n_setups": len(lp),
                    "frac_struggle": float(lp["struggle"].mean()),
                    "frac_flat": float(lp["struggle_flat"].mean()),
                    "frac_regress": float(lp["struggle_regress"].mean()),
                    "frac_low_restore": float(lp["struggle_low_restore"].mean()),
                    "mean_total_lift_pct": float(lp["total_lift_pct"].mean()),
                    "mean_restored_end": float(lp["restored_ctr_end"].mean()),
                    "mean_slope_per_10k": float(lp["slope_per_10k"].mean()),
                }
            )
    return pd.DataFrame(rows)


def level_key(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return 999


def build_payload(agg: pd.DataFrame, curves: pd.DataFrame, struggle: pd.DataFrame) -> dict:
    axes = sorted(agg["noise_axis"].dropna().unique().tolist())
    levels = sorted(agg["noise_level"].dropna().unique().tolist(), key=level_key)
    trains = sorted(pd.to_numeric(agg["train_size"], errors="coerce").dropna().unique().tolist())
    metrics = ["selected", "oracle", "mean_trials"]

    def clean(x):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, (np.integer, int)):
            return int(x)
        if isinstance(x, (np.bool_, bool)):
            return bool(x)
        return x

    rows = []
    for _, r in agg.iterrows():
        rows.append({k: clean(r.get(k)) for k in [
            "dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "train_size",
            "metric", "reward", "initial_reward", "pct_vs_initial", "pct_vs_ctr",
            "restored_ctr", "reward_over_ctr", "reward_se", "restored_ctr_se",
            "n_seeds", "action_delta", "context_delta",
        ]})

    curve_rows = []
    for _, r in curves.iterrows():
        curve_rows.append({k: clean(r.get(k)) for k in [
            "dataset", "noise_mode", "noise_axis", "noise_level", "ctr", "metric",
            "train_min", "train_max", "reward_min_train", "reward_max_train",
            "total_lift_pct", "slope_per_10k", "mean_step_lift_pct", "worst_step_lift_pct",
            "restored_ctr_start", "restored_ctr_end", "restored_ctr_gain",
            "struggle", "struggle_flat", "struggle_regress", "struggle_low_restore",
        ]})

    struggle_rows = []
    for _, r in struggle.iterrows():
        struggle_rows.append({k: clean(r.get(k)) for k in r.index})

    return {
        "run_tag": "run_bias_axes_l5_t20_s5",
        "axes": axes,
        "levels": levels,
        "train_sizes": [int(x) for x in trains],
        "metrics": metrics,
        "rows": rows,
        "curves": curve_rows,
        "struggle": struggle_rows,
        "definitions": {
            "restored_ctr": "(R_pi - R_log) / (CTR - R_log): fraction of gap from logging reward toward CTR closed by OPC",
            "reward_over_ctr": "R_pi / CTR",
            "total_lift_pct": "100*(R_at_max_train - R_at_min_train)/|R_at_min_train|",
            "slope_per_10k": "linear fit slope of reward vs train_size, scaled per 10k samples",
            "struggle": "flat lift (<0.5%), regress (< -0.25%), or restored_ctr_end < 0.25",
        },
    }


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OPC deep analysis — noise & restored CTR</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;550;650&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg0:#10151c; --bg1:#1a222d; --bg2:#243041; --ink:#e8eef4; --muted:#9aabbc;
    --opc:#3dbb8f; --warn:#e07a5f; --accent:#6ea8fe; --line:#334155; --gold:#f0c14a;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--ink);
    font-family:"IBM Plex Sans",sans-serif;
    background:
      radial-gradient(1000px 500px at 0% -10%, #1a3a40 0%, transparent 55%),
      radial-gradient(800px 480px at 100% 0%, #2a2f48 0%, transparent 50%),
      var(--bg0);
    min-height:100vh;
  }
  header{max-width:1400px;margin:0 auto;padding:28px 32px 8px}
  header h1{margin:0 0 6px;font-family:"IBM Plex Serif",Georgia,serif;font-size:1.85rem;font-weight:600}
  header p{margin:0;color:var(--muted);max-width:78ch;line-height:1.45}
  .controls{display:flex;flex-wrap:wrap;gap:12px;max-width:1400px;margin:16px auto 0;padding:0 32px}
  label{display:flex;flex-direction:column;gap:4px;font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
  select,button{background:var(--bg1);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:.95rem;min-width:130px}
  button{cursor:pointer;background:var(--bg2)}
  button.active{border-color:var(--opc);color:var(--opc)}
  .metric-toggle{display:flex;gap:8px;align-items:end}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;max-width:1400px;margin:16px auto 0;padding:0 32px}
  @media(max-width:1000px){.kpis{grid-template-columns:repeat(2,1fr)}}
  .kpi{background:var(--bg1);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .kpi .lbl{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.05em}
  .kpi .val{font-size:1.35rem;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
  .kpi .sub{color:var(--muted);font-size:.78rem;margin-top:2px}
  .grid{display:grid;grid-template-columns:1.15fr 1fr;gap:16px;max-width:1400px;margin:18px auto 40px;padding:0 32px}
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
  th{color:var(--muted);font-weight:550;font-size:.7rem;text-transform:uppercase;letter-spacing:.04em}
  tr:hover td{background:color-mix(in srgb,var(--bg2) 55%,transparent)}
  .pos{color:var(--opc)} .neg{color:var(--warn)} .flag{color:var(--gold);font-weight:650}
  .scroll{max-height:440px;overflow:auto}
  code{font-size:.85em}
</style>
</head>
<body>
<header>
  <h1>OPC deep dive</h1>
  <p>
    OPC only on <code>run_bias_axes_l5_t20_s5</code>.
    Tracks learning <b>improvement rate</b>, <b>struggle</b> cells (flat / regress / low restore),
    and <b>restored CTR</b> = <code>(Rπ − Rlog) / (CTR − Rlog)</code> across noise axis, level, and train size.
  </p>
</header>

<div class="controls">
  <div class="metric-toggle" id="metricToggle"></div>
  <label>Axis<select id="axisSel"></select></label>
  <label>Level<select id="levelSel"></select></label>
  <label>View
    <select id="viewSel">
      <option value="restored">Restored CTR vs train</option>
      <option value="reward">Reward vs train</option>
      <option value="lift">Improvement rate heatmap</option>
      <option value="restore_heat">Restored CTR heatmap</option>
    </select>
  </label>
</div>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="card">
    <h2 id="chartTitle">Restored CTR vs train</h2>
    <div class="legend" id="legend"></div>
    <canvas class="chart" id="mainChart" width="920" height="360"></canvas>
    <p class="note" id="chartNote"></p>
  </div>
  <div class="card">
    <h2>Struggle / averages by group</h2>
    <div class="scroll">
      <table id="struggleTable">
        <thead>
          <tr>
            <th>Scope</th><th>Lift%</th><th>Restored@max</th><th>Slope/10k</th><th>Struggle</th><th>Low restore</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="card full">
    <h2>Per noise setup — learning curve summary</h2>
    <div class="scroll">
      <table id="curveTable">
        <thead>
          <tr>
            <th>Axis</th><th>Level</th><th>R@min</th><th>R@max</th><th>Lift%</th><th>Slope/10k</th>
            <th>Restored start</th><th>Restored end</th><th>Δ restored</th><th>Struggle?</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="card full">
    <h2>Seed-averaged cells (axis × level × train)</h2>
    <div class="scroll">
      <table id="cellTable">
        <thead>
          <tr>
            <th>Axis</th><th>Level</th><th>Train</th><th>Reward</th><th>% vs log</th>
            <th>Restored CTR</th><th>R/CTR</th><th>Seeds</th>
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
let state = {metric:'selected', axis:'all', level:'all', view:'restored'};

function fmt(x,d=5){ if(x==null||Number.isNaN(x)) return '—'; return Number(x).toFixed(d); }
function fmtPct(x,d=2){ if(x==null||Number.isNaN(x)) return '—'; return (x>=0?'+':'')+Number(x).toFixed(d)+'%'; }
function cls(x){ return x>0?'pos':(x<0?'neg':''); }

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
  const axisSel=document.getElementById('axisSel');
  axisSel.innerHTML='<option value="all">All axes</option>'+DATA.axes.map(a=>`<option value="${a}">${a}</option>`).join('');
  axisSel.onchange=e=>{state.axis=e.target.value; sync();};
  const levelSel=document.getElementById('levelSel');
  levelSel.innerHTML='<option value="all">All levels</option>'+DATA.levels.map(a=>`<option value="${a}">${a}</option>`).join('');
  levelSel.onchange=e=>{state.level=e.target.value; sync();};
  document.getElementById('viewSel').onchange=e=>{state.view=e.target.value; sync();};
}

function filterRows(){
  return DATA.rows.filter(r=>{
    if(r.metric!==state.metric) return false;
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

function avg(xs){ const a=xs.filter(x=>x!=null && !Number.isNaN(x)); return a.length?a.reduce((p,c)=>p+c,0)/a.length:NaN; }

function renderKpis(rows, curves){
  const restored=avg(rows.map(r=>r.restored_ctr));
  const lift=avg(curves.map(r=>r.total_lift_pct));
  const slope=avg(curves.map(r=>r.slope_per_10k));
  const struggleN=curves.filter(c=>c.struggle).length;
  const reward=avg(rows.map(r=>r.reward));
  document.getElementById('kpis').innerHTML=`
    <div class="kpi"><div class="lbl">Mean reward</div><div class="val">${fmt(reward)}</div><div class="sub">${rows.length} cells</div></div>
    <div class="kpi"><div class="lbl">Mean restored CTR</div><div class="val ${cls(restored)}">${fmt(restored,3)}</div><div class="sub">1 = gap to CTR closed</div></div>
    <div class="kpi"><div class="lbl">Mean train lift</div><div class="val ${cls(lift)}">${fmtPct(lift)}</div><div class="sub">min→max train</div></div>
    <div class="kpi"><div class="lbl">Mean slope / 10k</div><div class="val ${cls(slope)}">${fmt(slope,6)}</div><div class="sub">linear learning rate</div></div>
    <div class="kpi"><div class="lbl">Struggle setups</div><div class="val ${struggleN?'flag':''}">${struggleN}/${curves.length}</div><div class="sub">flat / regress / low restore</div></div>
  `;
}

function renderStruggle(){
  const tb=document.querySelector('#struggleTable tbody');
  const rows=DATA.struggle.filter(r=>r.metric===state.metric);
  const order=s=>s==='all'?0:(s.startsWith('axis=')?1:2);
  rows.sort((a,b)=>order(a.scope)-order(b.scope)||a.scope.localeCompare(b.scope));
  tb.innerHTML=rows.map(r=>`
    <tr>
      <td>${r.scope}</td>
      <td class="${cls(r.mean_total_lift_pct)}">${fmtPct(r.mean_total_lift_pct)}</td>
      <td class="${cls(r.mean_restored_end)}">${fmt(r.mean_restored_end,3)}</td>
      <td class="${cls(r.mean_slope_per_10k)}">${fmt(r.mean_slope_per_10k,6)}</td>
      <td class="${r.frac_struggle>0.3?'flag':''}">${(100*r.frac_struggle).toFixed(0)}%</td>
      <td>${(100*r.frac_low_restore).toFixed(0)}%</td>
    </tr>`).join('');
}

function renderCurves(curves){
  const tb=document.querySelector('#curveTable tbody');
  const sorted=[...curves].sort((a,b)=>a.noise_axis.localeCompare(b.noise_axis)||DATA.levels.indexOf(a.noise_level)-DATA.levels.indexOf(b.noise_level));
  tb.innerHTML=sorted.map(r=>`
    <tr>
      <td>${r.noise_axis}</td><td>${r.noise_level}</td>
      <td>${fmt(r.reward_min_train)}</td><td>${fmt(r.reward_max_train)}</td>
      <td class="${cls(r.total_lift_pct)}">${fmtPct(r.total_lift_pct)}</td>
      <td class="${cls(r.slope_per_10k)}">${fmt(r.slope_per_10k,6)}</td>
      <td>${fmt(r.restored_ctr_start,3)}</td>
      <td class="${cls(r.restored_ctr_end)}">${fmt(r.restored_ctr_end,3)}</td>
      <td class="${cls(r.restored_ctr_gain)}">${fmt(r.restored_ctr_gain,3)}</td>
      <td class="${r.struggle?'flag':''}">${r.struggle?'YES':'no'}</td>
    </tr>`).join('');
}

function renderCells(rows){
  const tb=document.querySelector('#cellTable tbody');
  const sorted=[...rows].sort((a,b)=>a.noise_axis.localeCompare(b.noise_axis)||DATA.levels.indexOf(a.noise_level)-DATA.levels.indexOf(b.noise_level)||a.train_size-b.train_size);
  tb.innerHTML=sorted.map(r=>`
    <tr>
      <td>${r.noise_axis}</td><td>${r.noise_level}</td><td>${r.train_size}</td>
      <td>${fmt(r.reward)}</td>
      <td class="${cls(r.pct_vs_initial)}">${fmtPct(r.pct_vs_initial)}</td>
      <td class="${cls(r.restored_ctr)}">${fmt(r.restored_ctr,3)}</td>
      <td>${fmt(r.reward_over_ctr,3)}</td>
      <td>${r.n_seeds}</td>
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

function seriesByLevel(rows, ykey){
  const trains=DATA.train_sizes;
  const levels=state.level==='all'?DATA.levels:[state.level];
  const out={};
  levels.forEach(lv=>{
    out[lv]=trains.map(t=>{
      const cell=rows.filter(r=>r.noise_level===lv && r.train_size===t);
      return avg(cell.map(r=>r[ykey]));
    });
  });
  return {trains, levels, out};
}

function drawLines(ctx,c,rows,ykey,ylabel){
  clearCanvas(ctx,c);
  const pad={l:64,r:16,t:16,b:36};
  const {trains, levels, out}=seriesByLevel(rows,ykey);
  const vals=[].concat(...Object.values(out)).filter(v=>v!=null&&!Number.isNaN(v));
  if(!vals.length) return;
  const ymin=Math.min(...vals)-0.02*Math.abs(Math.min(...vals)||1);
  const ymax=Math.max(...vals)+0.02*Math.abs(Math.max(...vals)||1);
  drawFrame(ctx,c,pad,ymin,ymax,'train_size');
  trains.forEach((t,i)=>{
    const [X]=XYf(pad,c,0,trains.length-1,ymin,ymax,i,ymin);
    ctx.fillStyle='#9aabbc'; ctx.fillText(String(t), X-16, c.height-18);
  });
  // reference lines for restored
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
    ctx.beginPath();
    let started=false;
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

function drawHeat(ctx,c,curves,key){
  clearCanvas(ctx,c);
  const axes=state.axis==='all'?DATA.axes:[state.axis];
  const levels=state.level==='all'?DATA.levels:[state.level];
  const pad={l:90,r:16,t:24,b:40};
  const cw=(c.width-pad.l-pad.r)/levels.length;
  const ch=(c.height-pad.t-pad.b)/axes.length;
  const cell={};
  curves.forEach(r=>{ (cell[r.noise_axis+'||'+r.noise_level]=r); });
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
      const txt = m==null||Number.isNaN(m) ? '—' : (key.includes('pct')?((m>=0?'+':'')+m.toFixed(2)+'%'):m.toFixed(3));
      ctx.fillText(txt, x+8, y+ch/2+4);
      if(r && r.struggle){
        ctx.strokeStyle='#f0c14a'; ctx.lineWidth=2;
        ctx.strokeRect(x+3,y+3,cw-6,ch-6);
      }
      if(i===axes.length-1){ ctx.fillStyle='#9aabbc'; ctx.fillText(lv, x+6, c.height-16); }
    });
    ctx.fillStyle='#9aabbc'; ctx.fillText(ax, 10, pad.t+i*ch+ch/2+4);
  });
}

function renderChart(rows, curves){
  const canvas=document.getElementById('mainChart');
  const ctx=canvas.getContext('2d');
  const title=document.getElementById('chartTitle');
  const note=document.getElementById('chartNote');
  const legend=document.getElementById('legend');
  const levels=state.level==='all'?DATA.levels:[state.level];
  legend.innerHTML=levels.map(lv=>`<span><i class="swatch" style="background:${LEVEL_COLORS[lv]||'#fff'}"></i>${lv}</span>`).join('')
    + (state.view.includes('heat')?'<span><i class="swatch" style="background:#f0c14a"></i>gold border = struggle</span>':'');

  if(state.view==='restored'){
    title.textContent='Restored CTR vs train size';
    note.textContent=DATA.definitions.restored_ctr+'  Dashed: 0 = logging, 1 = CTR.';
    drawLines(ctx,canvas,rows,'restored_ctr','restored');
  } else if(state.view==='reward'){
    title.textContent='OPC reward vs train size';
    note.textContent='Seed-averaged selected/oracle/mean-trials reward by noise level.';
    drawLines(ctx,canvas,rows,'reward','reward');
  } else if(state.view==='lift'){
    title.textContent='Improvement rate (total lift % min→max train)';
    note.textContent=DATA.definitions.total_lift_pct+'  Gold box = struggle setup.';
    drawHeat(ctx,canvas,curves,'total_lift_pct');
  } else {
    title.textContent='Restored CTR at max train (axis × level)';
    note.textContent='End-of-curve restored CTR. Gold box = struggle.';
    drawHeat(ctx,canvas,curves,'restored_ctr_end');
  }
}

function sync(){
  document.querySelectorAll('#metricToggle button').forEach(b=>b.classList.toggle('active', b.dataset.metric===state.metric));
  const rows=filterRows();
  const curves=filterCurves();
  renderKpis(rows, curves);
  renderStruggle();
  renderCurves(curves);
  renderCells(rows);
  renderChart(rows, curves);
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
    struggle = struggle_summary(curves)

    long.to_csv(args.out_dir / "opc_long_by_seed.csv", index=False)
    agg.to_csv(args.out_dir / "opc_seed_averaged.csv", index=False)
    curves.to_csv(args.out_dir / "opc_learning_curves.csv", index=False)
    struggle.to_csv(args.out_dir / "opc_struggle_summary.csv", index=False)

    payload = build_payload(agg, curves, struggle)
    (args.out_dir / "canvas_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = HTML.replace("__DATA_JSON__", json.dumps(payload))
    out_html = args.out_dir / "opc_deep_canvas.html"
    out_html.write_text(html, encoding="utf-8")

    art = Path("/opt/cursor/artifacts")
    if art.exists():
        (art / "opc_deep_canvas.html").write_text(html, encoding="utf-8")
        struggle.to_csv(art / "opc_deep_struggle.csv", index=False)
        curves.to_csv(art / "opc_deep_curves.csv", index=False)

    print(f"wrote {out_html}")
    sel = struggle[struggle.metric == "selected"]
    print(sel[sel.scope == "all"].to_string(index=False))
    hard = curves[(curves.metric == "selected") & curves.struggle].sort_values("total_lift_pct")
    print("struggle setups:", len(hard))
    if len(hard):
        print(hard[["noise_axis", "noise_level", "total_lift_pct", "restored_ctr_end"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
