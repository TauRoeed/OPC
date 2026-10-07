#!/usr/bin/env python3
"""Build OPC vs naive (no_propensity) canvas: selected / oracle / mean trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVEL_ORDER = ["low", "medium", "high", "extreme", "brutal"]
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


def load_condition_frames(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    trial_rows = []
    for cond in sorted(run_dir.glob("dataset=*")):
        tags = parse_condition_dirname(cond.name)
        runs_path = cond / "runs_long.csv"
        trials_path = cond / "trials_long.csv"
        if not runs_path.exists() or not trials_path.exists():
            continue
        runs = pd.read_csv(runs_path)
        trials = pd.read_csv(trials_path)
        for k, v in tags.items():
            if k not in runs.columns or runs[k].isna().all():
                runs[k] = v
            if k not in trials.columns or trials[k].isna().all():
                trials[k] = v
        run_rows.append(runs)
        trial_rows.append(trials)
    if not run_rows:
        raise SystemExit(f"No condition folders under {run_dir}")
    return pd.concat(run_rows, ignore_index=True), pd.concat(trial_rows, ignore_index=True)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def selected_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Winning-run policy reward per setup×seed×method (selected)."""
    d = coerce_numeric(
        runs,
        ["train_size", "seed", "ctr", "policy_rewards", "actual_reward_selected", "initial_reward"],
    )
    if "is_winning_run" in d.columns:
        win = d[d["is_winning_run"].astype(bool)].copy()
        if win.empty:
            win = d.copy()
    else:
        win = d.copy()
    reward = win["policy_rewards"]
    if "actual_reward_selected" in win.columns:
        ar = win["actual_reward_selected"]
        reward = reward.where(reward.notna(), ar)
    win = win.assign(reward=reward)
    keys = [c for c in SETUP_KEYS + ["seed", "method"] if c in win.columns]
    g = (
        win.groupby(keys, as_index=False)
        .agg(reward=("reward", "mean"), initial_reward=("initial_reward", "mean"))
    )
    g["metric"] = "selected"
    return g


def oracle_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Max actual_reward over Optuna trials per setup×seed×method."""
    d = coerce_numeric(
        trials,
        ["train_size", "seed", "ctr", "actual_reward", "initial_reward", "trial_number"],
    )
    keys = [c for c in SETUP_KEYS + ["seed", "method"] if c in d.columns]
    g = (
        d.groupby(keys, as_index=False)
        .agg(reward=("actual_reward", "max"), initial_reward=("initial_reward", "mean"))
    )
    g["metric"] = "oracle"
    return g


def mean_trials_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Mean actual_reward over all Optuna trials per setup×seed×method."""
    d = coerce_numeric(
        trials,
        ["train_size", "seed", "ctr", "actual_reward", "initial_reward"],
    )
    keys = [c for c in SETUP_KEYS + ["seed", "method"] if c in d.columns]
    g = (
        d.groupby(keys, as_index=False)
        .agg(reward=("actual_reward", "mean"), initial_reward=("initial_reward", "mean"))
    )
    g["metric"] = "mean_trials"
    return g


def pivot_delta(long: pd.DataFrame) -> pd.DataFrame:
    idx = [c for c in SETUP_KEYS + ["seed", "metric"] if c in long.columns]
    piv = long.pivot_table(index=idx, columns="method", values="reward", aggfunc="mean")
    need = {"opc", "no_propensity"}
    if piv.empty or not need.issubset(set(piv.columns)):
        return pd.DataFrame()
    out = piv.reset_index()
    out["delta"] = out["opc"] - out["no_propensity"]
    out["lift_pct"] = 100.0 * out["delta"] / out["no_propensity"].abs().clip(lower=1e-12)
    ir = long.groupby(idx, as_index=False)["initial_reward"].mean()
    out = out.merge(ir, on=idx, how="left")
    return out


def aggregate_seeds(delta: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in SETUP_KEYS + ["metric"] if c in delta.columns]
    agg = (
        delta.groupby(keys, as_index=False)
        .agg(
            opc=("opc", "mean"),
            no_propensity=("no_propensity", "mean"),
            delta=("delta", "mean"),
            delta_std=("delta", "std"),
            lift_pct=("lift_pct", "mean"),
            n_seeds=("delta", "count"),
            initial_reward=("initial_reward", "mean"),
        )
    )
    agg["delta_se"] = agg["delta_std"] / np.sqrt(agg["n_seeds"].clip(lower=1))
    return agg


def overall_avg(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, part in agg.groupby("metric"):
        rows.append(
            {
                "scope": "all_setups",
                "metric": metric,
                "opc": part["opc"].mean(),
                "no_propensity": part["no_propensity"].mean(),
                "delta": part["delta"].mean(),
                "lift_pct": part["lift_pct"].mean(),
                "frac_opc_wins": float((part["delta"] > 0).mean()),
                "n": int(len(part)),
            }
        )
        for axis, ap in part.groupby("noise_axis"):
            rows.append(
                {
                    "scope": f"axis={axis}",
                    "metric": metric,
                    "opc": ap["opc"].mean(),
                    "no_propensity": ap["no_propensity"].mean(),
                    "delta": ap["delta"].mean(),
                    "lift_pct": ap["lift_pct"].mean(),
                    "frac_opc_wins": float((ap["delta"] > 0).mean()),
                    "n": int(len(ap)),
                }
            )
        for level, lp in part.groupby("noise_level"):
            rows.append(
                {
                    "scope": f"level={level}",
                    "metric": metric,
                    "opc": lp["opc"].mean(),
                    "no_propensity": lp["no_propensity"].mean(),
                    "delta": lp["delta"].mean(),
                    "lift_pct": lp["lift_pct"].mean(),
                    "frac_opc_wins": float((lp["delta"] > 0).mean()),
                    "n": int(len(lp)),
                }
            )
    return pd.DataFrame(rows)


def level_sort_key(level: str) -> int:
    try:
        return LEVEL_ORDER.index(level)
    except ValueError:
        return 999


def build_payload(agg: pd.DataFrame, overall: pd.DataFrame, seed_delta: pd.DataFrame) -> dict:
    axes = sorted(agg["noise_axis"].dropna().unique().tolist())
    levels = sorted(agg["noise_level"].dropna().unique().tolist(), key=level_sort_key)
    trains = sorted(pd.to_numeric(agg["train_size"], errors="coerce").dropna().unique().tolist())
    metrics = ["selected", "oracle", "mean_trials"]
    records = []
    for _, r in agg.iterrows():
        records.append(
            {
                "dataset": r.get("dataset"),
                "noise_mode": r.get("noise_mode"),
                "noise_axis": r["noise_axis"],
                "noise_level": r["noise_level"],
                "ctr": float(r["ctr"]) if pd.notna(r.get("ctr")) else None,
                "train_size": int(r["train_size"]),
                "metric": r["metric"],
                "opc": float(r["opc"]),
                "naive": float(r["no_propensity"]),
                "delta": float(r["delta"]),
                "delta_se": float(r["delta_se"]) if pd.notna(r["delta_se"]) else 0.0,
                "lift_pct": float(r["lift_pct"]),
                "n_seeds": int(r["n_seeds"]),
                "initial_reward": float(r["initial_reward"]) if pd.notna(r.get("initial_reward")) else None,
            }
        )
    overall_recs = []
    for _, r in overall.iterrows():
        overall_recs.append(
            {
                "scope": r["scope"],
                "metric": r["metric"],
                "opc": float(r["opc"]),
                "naive": float(r["no_propensity"]),
                "delta": float(r["delta"]),
                "lift_pct": float(r["lift_pct"]),
                "frac_opc_wins": float(r["frac_opc_wins"]),
                "n": int(r["n"]),
            }
        )
    return {
        "run_tag": "run_bias_axes_l5_t20_s5",
        "axes": axes,
        "levels": levels,
        "train_sizes": [int(x) for x in trains],
        "metrics": metrics,
        "rows": records,
        "overall": overall_recs,
        "n_seed_rows": int(len(seed_delta)),
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OPC vs Naive — bias axes</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;550;650&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet"/>
<style>
  :root {
    --bg0: #0f1419;
    --bg1: #1a222c;
    --bg2: #243040;
    --ink: #e8eef4;
    --muted: #9aabbc;
    --opc: #3dbb8f;
    --naive: #e07a5f;
    --delta: #6ea8fe;
    --line: #334155;
    --warn: #f0c14a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background:
      radial-gradient(1200px 600px at 10% -10%, #1d3a33 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #2a3348 0%, transparent 50%),
      var(--bg0);
    color: var(--ink);
    min-height: 100vh;
  }
  header {
    padding: 28px 32px 8px;
    max-width: 1400px;
    margin: 0 auto;
  }
  header h1 {
    margin: 0 0 6px;
    font-family: "IBM Plex Serif", Georgia, serif;
    font-weight: 600;
    font-size: 1.85rem;
    letter-spacing: -0.02em;
  }
  header p { margin: 0; color: var(--muted); max-width: 70ch; line-height: 1.45; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 12px;
    max-width: 1400px; margin: 18px auto 0; padding: 0 32px;
  }
  label {
    display: flex; flex-direction: column; gap: 4px;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted);
  }
  select, button {
    background: var(--bg1); color: var(--ink);
    border: 1px solid var(--line); border-radius: 8px;
    padding: 8px 12px; font-size: 0.95rem; min-width: 140px;
  }
  button { cursor: pointer; background: var(--bg2); }
  button.active { border-color: var(--opc); color: var(--opc); }
  .metric-toggle { display: flex; gap: 8px; align-items: end; }
  .grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 16px;
    max-width: 1400px;
    margin: 18px auto 40px;
    padding: 0 32px;
  }
  @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: color-mix(in srgb, var(--bg1) 88%, transparent);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px 18px;
    backdrop-filter: blur(6px);
  }
  .card h2 {
    margin: 0 0 10px;
    font-size: 1rem;
    font-weight: 600;
  }
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    max-width: 1400px;
    margin: 16px auto 0;
    padding: 0 32px;
  }
  @media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
  .kpi {
    background: var(--bg1);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 16px;
  }
  .kpi .lbl { color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .kpi .val { font-size: 1.45rem; font-weight: 650; margin-top: 4px; font-variant-numeric: tabular-nums; }
  .kpi .sub { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  canvas.chart { width: 100%; height: 340px; display: block; }
  table {
    width: 100%; border-collapse: collapse; font-size: 0.86rem;
    font-variant-numeric: tabular-nums;
  }
  th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  th { color: var(--muted); font-weight: 550; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; }
  tr:hover td { background: color-mix(in srgb, var(--bg2) 55%, transparent); }
  .pos { color: var(--opc); }
  .neg { color: var(--naive); }
  .legend { display: flex; gap: 14px; color: var(--muted); font-size: 0.85rem; margin-bottom: 8px; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
  .note { color: var(--muted); font-size: 0.8rem; margin-top: 8px; }
  .full { grid-column: 1 / -1; }
  .scroll { max-height: 420px; overflow: auto; }
</style>
</head>
<body>
<header>
  <h1>OPC vs Naive</h1>
  <p>
    Bias-axis study <code>run_bias_axes_l5_t20_s5</code>.
    Compare OPC against naive (no propensity) for <b>selected</b> policy,
    <b>oracle</b> (best trial actual reward), and <b>mean of trials</b>.
    Per setup, pooled, and averages.
  </p>
</header>

<div class="controls">
  <div class="metric-toggle" id="metricToggle"></div>
  <label>Axis
    <select id="axisSel"></select>
  </label>
  <label>Level
    <select id="levelSel"></select>
  </label>
  <label>View
    <select id="viewSel">
      <option value="curves">Reward vs train size</option>
      <option value="delta">Delta (OPC − naive)</option>
      <option value="heatmap">Heatmap by axis × level</option>
    </select>
  </label>
</div>

<div class="kpis" id="kpis"></div>

<div class="grid">
  <div class="card">
    <h2 id="chartTitle">Reward vs train size</h2>
    <div class="legend">
      <span><i class="swatch" style="background:var(--opc)"></i>OPC</span>
      <span><i class="swatch" style="background:var(--naive)"></i>Naive</span>
      <span><i class="swatch" style="background:var(--delta)"></i>Delta</span>
    </div>
    <canvas class="chart" id="mainChart" width="900" height="340"></canvas>
    <p class="note" id="chartNote"></p>
  </div>
  <div class="card">
    <h2>Overall &amp; group averages</h2>
    <div class="scroll">
      <table id="overallTable">
        <thead>
          <tr><th>Scope</th><th>Metric</th><th>OPC</th><th>Naive</th><th>Δ</th><th>Lift%</th><th>OPC wins</th></tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
  <div class="card full">
    <h2>Per-setup table (seed-averaged)</h2>
    <div class="scroll">
      <table id="setupTable">
        <thead>
          <tr>
            <th>Axis</th><th>Level</th><th>Train</th><th>OPC</th><th>Naive</th><th>Δ</th><th>SE</th><th>Lift%</th><th>Seeds</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

let state = {
  metric: 'selected',
  axis: 'all',
  level: 'all',
  view: 'curves',
};

function fmt(x, d=5) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  return Number(x).toFixed(d);
}
function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—';
  const s = (x >= 0 ? '+' : '') + Number(x).toFixed(2) + '%';
  return s;
}
function clsDelta(x) { return x > 0 ? 'pos' : (x < 0 ? 'neg' : ''); }

function initControls() {
  const mt = document.getElementById('metricToggle');
  const labels = {selected: 'Selected', oracle: 'Oracle', mean_trials: 'Mean trials'};
  DATA.metrics.forEach(m => {
    const b = document.createElement('button');
    b.textContent = labels[m] || m;
    b.dataset.metric = m;
    if (m === state.metric) b.classList.add('active');
    b.onclick = () => { state.metric = m; sync(); };
    mt.appendChild(b);
  });
  const axisSel = document.getElementById('axisSel');
  axisSel.innerHTML = '<option value="all">All axes</option>' +
    DATA.axes.map(a => `<option value="${a}">${a}</option>`).join('');
  axisSel.onchange = e => { state.axis = e.target.value; sync(); };
  const levelSel = document.getElementById('levelSel');
  levelSel.innerHTML = '<option value="all">All levels</option>' +
    DATA.levels.map(a => `<option value="${a}">${a}</option>`).join('');
  levelSel.onchange = e => { state.level = e.target.value; sync(); };
  document.getElementById('viewSel').onchange = e => { state.view = e.target.value; sync(); };
}

function filteredRows() {
  return DATA.rows.filter(r => {
    if (r.metric !== state.metric) return false;
    if (state.axis !== 'all' && r.noise_axis !== state.axis) return false;
    if (state.level !== 'all' && r.noise_level !== state.level) return false;
    return true;
  });
}

function renderKpis(rows) {
  const avg = (xs) => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : NaN;
  const deltas = rows.map(r => r.delta);
  const lifts = rows.map(r => r.lift_pct);
  const wins = rows.filter(r => r.delta > 0).length;
  const el = document.getElementById('kpis');
  const meanDelta = avg(deltas);
  el.innerHTML = `
    <div class="kpi"><div class="lbl">Mean OPC</div><div class="val">${fmt(avg(rows.map(r=>r.opc)))}</div><div class="sub">${rows.length} setups</div></div>
    <div class="kpi"><div class="lbl">Mean Naive</div><div class="val">${fmt(avg(rows.map(r=>r.naive)))}</div><div class="sub">${state.metric}</div></div>
    <div class="kpi"><div class="lbl">Mean Δ (OPC−naive)</div><div class="val ${clsDelta(meanDelta)}">${meanDelta>=0?'+':''}${fmt(meanDelta)}</div><div class="sub">lift ${fmtPct(avg(lifts))}</div></div>
    <div class="kpi"><div class="lbl">OPC wins</div><div class="val">${wins}/${rows.length}</div><div class="sub">${rows.length?((100*wins/rows.length).toFixed(0)):'0'}% of setups</div></div>
  `;
}

function renderOverall() {
  const tb = document.querySelector('#overallTable tbody');
  const rows = DATA.overall.filter(r => r.metric === state.metric);
  const order = (s) => s === 'all_setups' ? 0 : (s.startsWith('axis=') ? 1 : 2);
  rows.sort((a,b) => order(a.scope) - order(b.scope) || a.scope.localeCompare(b.scope));
  tb.innerHTML = rows.map(r => `
    <tr>
      <td>${r.scope}</td>
      <td>${r.metric}</td>
      <td>${fmt(r.opc)}</td>
      <td>${fmt(r.naive)}</td>
      <td class="${clsDelta(r.delta)}">${r.delta>=0?'+':''}${fmt(r.delta)}</td>
      <td class="${clsDelta(r.lift_pct)}">${fmtPct(r.lift_pct)}</td>
      <td>${(100*r.frac_opc_wins).toFixed(0)}% (n=${r.n})</td>
    </tr>`).join('');
}

function renderSetupTable(rows) {
  const tb = document.querySelector('#setupTable tbody');
  const sorted = [...rows].sort((a,b) =>
    a.noise_axis.localeCompare(b.noise_axis) ||
    DATA.levels.indexOf(a.noise_level) - DATA.levels.indexOf(b.noise_level) ||
    a.train_size - b.train_size
  );
  tb.innerHTML = sorted.map(r => `
    <tr>
      <td>${r.noise_axis}</td>
      <td>${r.noise_level}</td>
      <td>${r.train_size}</td>
      <td>${fmt(r.opc)}</td>
      <td>${fmt(r.naive)}</td>
      <td class="${clsDelta(r.delta)}">${r.delta>=0?'+':''}${fmt(r.delta)}</td>
      <td>${fmt(r.delta_se, 6)}</td>
      <td class="${clsDelta(r.lift_pct)}">${fmtPct(r.lift_pct)}</td>
      <td>${r.n_seeds}</td>
    </tr>`).join('');
}

function clearCanvas(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#15202b';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawAxes(ctx, pad, w, h, xmin, xmax, ymin, ymax, xlabel) {
  ctx.strokeStyle = '#334155';
  ctx.fillStyle = '#9aabbc';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
  ctx.font = '12px IBM Plex Sans, sans-serif';
  for (let i = 0; i <= 4; i++) {
    const yv = ymin + (ymax - ymin) * (i / 4);
    const y = h - pad.b - (i / 4) * (h - pad.t - pad.b);
    ctx.fillText(yv.toFixed(4), 8, y + 4);
    ctx.strokeStyle = '#1f2a36';
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
  }
  ctx.fillStyle = '#9aabbc';
  ctx.fillText(xlabel, w / 2 - 30, h - 8);
}

function xy(pad, w, h, xmin, xmax, ymin, ymax, x, y) {
  const X = pad.l + ((x - xmin) / (xmax - xmin || 1)) * (w - pad.l - pad.r);
  const Y = h - pad.b - ((y - ymin) / (ymax - ymin || 1)) * (h - pad.t - pad.b);
  return [X, Y];
}

function drawCurves(ctx, canvas, rows) {
  clearCanvas(ctx, canvas);
  const pad = {l: 64, r: 16, t: 16, b: 36};
  const trains = DATA.train_sizes;
  // average across filtered setups at each train size
  const byT = {};
  trains.forEach(t => { byT[t] = {opc: [], naive: []}; });
  rows.forEach(r => {
    if (!byT[r.train_size]) byT[r.train_size] = {opc: [], naive: []};
    byT[r.train_size].opc.push(r.opc);
    byT[r.train_size].naive.push(r.naive);
  });
  const seriesO = trains.map(t => {
    const a = byT[t].opc; return a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
  });
  const seriesN = trains.map(t => {
    const a = byT[t].naive; return a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
  });
  const vals = [...seriesO, ...seriesN].filter(v => v !== null);
  if (!vals.length) return;
  const ymin = Math.min(...vals) - 0.0005;
  const ymax = Math.max(...vals) + 0.0005;
  const xmin = 0, xmax = trains.length - 1;
  drawAxes(ctx, pad, canvas.width, canvas.height, xmin, xmax, ymin, ymax, 'train_size');
  trains.forEach((t, i) => {
    const [X] = xy(pad, canvas.width, canvas.height, xmin, xmax, ymin, ymax, i, ymin);
    ctx.fillStyle = '#9aabbc';
    ctx.fillText(String(t), X - 16, canvas.height - 18);
  });
  function strokeSeries(series, color) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
    ctx.beginPath();
    let started = false;
    series.forEach((v, i) => {
      if (v === null) return;
      const [X, Y] = xy(pad, canvas.width, canvas.height, xmin, xmax, ymin, ymax, i, v);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      ctx.beginPath(); ctx.arc(X, Y, 4, 0, Math.PI*2); ctx.fill();
      if (i === 0) ctx.moveTo(X, Y); else {
        // redraw line segments
      }
    });
    ctx.beginPath();
    started = false;
    series.forEach((v, i) => {
      if (v === null) return;
      const [X, Y] = xy(pad, canvas.width, canvas.height, xmin, xmax, ymin, ymax, i, v);
      if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
    });
    ctx.stroke();
    series.forEach((v, i) => {
      if (v === null) return;
      const [X, Y] = xy(pad, canvas.width, canvas.height, xmin, xmax, ymin, ymax, i, v);
      ctx.beginPath(); ctx.arc(X, Y, 4, 0, Math.PI*2); ctx.fill();
    });
  }
  strokeSeries(seriesO, '#3dbb8f');
  strokeSeries(seriesN, '#e07a5f');
}

function drawDelta(ctx, canvas, rows) {
  clearCanvas(ctx, canvas);
  const pad = {l: 64, r: 16, t: 16, b: 36};
  const trains = DATA.train_sizes;
  const byT = {};
  trains.forEach(t => { byT[t] = []; });
  rows.forEach(r => { (byT[r.train_size] ||= []).push(r.delta); });
  const means = trains.map(t => {
    const a = byT[t] || []; return a.length ? a.reduce((x,y)=>x+y,0)/a.length : null;
  });
  const vals = means.filter(v => v !== null);
  if (!vals.length) return;
  const maxAbs = Math.max(...vals.map(Math.abs), 1e-6);
  const ymin = -maxAbs * 1.2, ymax = maxAbs * 1.2;
  drawAxes(ctx, pad, canvas.width, canvas.height, 0, trains.length-1, ymin, ymax, 'train_size');
  // zero line
  const [, y0] = xy(pad, canvas.width, canvas.height, 0, trains.length-1, ymin, ymax, 0, 0);
  ctx.strokeStyle = '#f0c14a'; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(canvas.width-pad.r, y0); ctx.stroke();
  ctx.setLineDash([]);
  const barW = (canvas.width - pad.l - pad.r) / trains.length * 0.55;
  means.forEach((v, i) => {
    if (v === null) return;
    const [X] = xy(pad, canvas.width, canvas.height, 0, trains.length-1, ymin, ymax, i, 0);
    const [, Y] = xy(pad, canvas.width, canvas.height, 0, trains.length-1, ymin, ymax, i, v);
    ctx.fillStyle = v >= 0 ? '#3dbb8f' : '#e07a5f';
    const top = Math.min(Y, y0), bot = Math.max(Y, y0);
    ctx.fillRect(X - barW/2, top, barW, Math.max(bot - top, 1));
    ctx.fillStyle = '#9aabbc';
    ctx.fillText(String(trains[i]), X - 16, canvas.height - 18);
  });
}

function drawHeatmap(ctx, canvas, rows) {
  clearCanvas(ctx, canvas);
  const axes = state.axis === 'all' ? DATA.axes : [state.axis];
  const levels = state.level === 'all' ? DATA.levels : [state.level];
  // mean delta per axis x level
  const cell = {};
  rows.forEach(r => {
    const k = r.noise_axis + '||' + r.noise_level;
    (cell[k] ||= []).push(r.delta);
  });
  const pad = {l: 90, r: 16, t: 24, b: 40};
  const cw = (canvas.width - pad.l - pad.r) / levels.length;
  const ch = (canvas.height - pad.t - pad.b) / axes.length;
  let maxAbs = 1e-9;
  Object.values(cell).forEach(arr => {
    const m = arr.reduce((a,b)=>a+b,0)/arr.length;
    maxAbs = Math.max(maxAbs, Math.abs(m));
  });
  axes.forEach((ax, i) => {
    levels.forEach((lv, j) => {
      const arr = cell[ax + '||' + lv] || [];
      const m = arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : null;
      const x = pad.l + j * cw, y = pad.t + i * ch;
      let color = '#1f2a36';
      if (m !== null) {
        const t = Math.min(1, Math.abs(m) / maxAbs);
        if (m >= 0) color = `rgba(61,187,143,${0.25 + 0.75*t})`;
        else color = `rgba(224,122,95,${0.25 + 0.75*t})`;
      }
      ctx.fillStyle = color;
      ctx.fillRect(x+2, y+2, cw-4, ch-4);
      ctx.fillStyle = '#e8eef4';
      ctx.font = '12px IBM Plex Sans, sans-serif';
      ctx.fillText(m === null ? '—' : ((m>=0?'+':'') + m.toFixed(5)), x + 10, y + ch/2 + 4);
      if (i === axes.length - 1) {
        ctx.fillStyle = '#9aabbc';
        ctx.fillText(lv, x + 8, canvas.height - 16);
      }
    });
    ctx.fillStyle = '#9aabbc';
    ctx.fillText(ax, 12, pad.t + i * ch + ch/2 + 4);
  });
}

function renderChart(rows) {
  const canvas = document.getElementById('mainChart');
  const ctx = canvas.getContext('2d');
  const title = document.getElementById('chartTitle');
  const note = document.getElementById('chartNote');
  if (state.view === 'curves') {
    title.textContent = 'Mean reward vs train size (filtered setups)';
    note.textContent = 'Averages OPC and naive across currently filtered setups at each train size.';
    drawCurves(ctx, canvas, rows);
  } else if (state.view === 'delta') {
    title.textContent = 'Mean Δ (OPC − naive) vs train size';
    note.textContent = 'Positive = OPC better. Yellow dashed = zero.';
    drawDelta(ctx, canvas, rows);
  } else {
    title.textContent = 'Mean Δ heatmap (axis × level)';
    note.textContent = 'Cell = mean delta over train sizes (and seeds already averaged in rows).';
    drawHeatmap(ctx, canvas, rows);
  }
}

function sync() {
  document.querySelectorAll('#metricToggle button').forEach(b => {
    b.classList.toggle('active', b.dataset.metric === state.metric);
  });
  const rows = filteredRows();
  renderKpis(rows);
  renderOverall();
  renderSetupTable(rows);
  renderChart(rows);
}

initControls();
sync();
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
        default=Path("artifacts/analysis/opc_vs_naive"),
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs, trials = load_condition_frames(args.run_dir)
    long = pd.concat(
        [selected_table(runs), oracle_table(trials), mean_trials_table(trials)],
        ignore_index=True,
    )
    seed_delta = pivot_delta(long)
    agg = aggregate_seeds(seed_delta)
    overall = overall_avg(agg)

    long.to_csv(args.out_dir / "long_by_seed_method.csv", index=False)
    seed_delta.to_csv(args.out_dir / "delta_by_seed.csv", index=False)
    agg.to_csv(args.out_dir / "delta_seed_averaged.csv", index=False)
    overall.to_csv(args.out_dir / "overall_averages.csv", index=False)

    payload = build_payload(agg, overall, seed_delta)
    (args.out_dir / "canvas_data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload))
    out_html = args.out_dir / "opc_vs_naive_canvas.html"
    out_html.write_text(html, encoding="utf-8")

    # also copy to cursor artifacts for easy open
    art = Path("/opt/cursor/artifacts")
    if art.exists():
        (art / "opc_vs_naive_canvas.html").write_text(html, encoding="utf-8")
        overall.to_csv(art / "opc_vs_naive_overall.csv", index=False)
        agg.to_csv(art / "opc_vs_naive_setups.csv", index=False)

    print(f"wrote {out_html}")
    print(f"setups (seed-avg rows): {len(agg)}")
    print(overall[overall.scope == "all_setups"].to_string(index=False))


if __name__ == "__main__":
    main()
