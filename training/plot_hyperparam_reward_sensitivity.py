"""
Per experimental condition: grid of reward vs each logged hyperparameter.

Rows = hyperparameters (columns named param_* in trials_long).
Columns = train_size (within that condition).

Uses actual_reward when present; otherwise value (Optuna objective).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.analyze_full_study import _parse_condition_dirname


def _setup_key_without_seed(condition_dirname: str) -> str:
    """Folder tag for grouping all seeds (drops __seed=... from run_key)."""
    t = _parse_condition_dirname(condition_dirname)
    parts = []
    if "dataset" in t:
        parts.append(f"dataset={t['dataset']}")
    if "noise" in t:
        parts.append(f"noise={t['noise']}")
    if "axis" in t:
        parts.append(f"axis={t['axis']}")
    if "level" in t:
        parts.append(f"level={t['level']}")
    if "ctr" in t:
        parts.append(f"ctr={t['ctr']}")
    return "__".join(parts) if parts else condition_dirname.replace("/", "_")


def _load_condition_trials(condition_dir: Path) -> pd.DataFrame:
    unified = condition_dir / "trials_long.csv"
    if unified.exists():
        df = pd.read_csv(unified)
    else:
        parts = []
        for name in ("opc_trials_long.csv", "no_prop_trials_long.csv"):
            p = condition_dir / name
            if p.exists():
                parts.append(pd.read_csv(p))
        if not parts:
            return pd.DataFrame()
        df = pd.concat(parts, ignore_index=True)

    tags = _parse_condition_dirname(condition_dir.name)
    mapping = {
        "dataset": "dataset",
        "noise": "noise_mode",
        "axis": "noise_axis",
        "level": "noise_level",
        "seed": "seed",
        "ctr": "ctr",
    }
    for src, dst in mapping.items():
        if src in tags and dst not in df.columns:
            df[dst] = tags[src]
    for c in ("train_size", "run", "seed", "trial_number", "value", "actual_reward", "ctr", "initial_reward"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if c.startswith("param_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _reward_column(df: pd.DataFrame) -> str:
    if "actual_reward" in df.columns and df["actual_reward"].notna().any():
        return "actual_reward"
    return "value"


def _filter_beat_initial_reward(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where actual_reward > initial_reward (both finite). Ignores Optuna value."""
    if df.empty:
        return df
    if "initial_reward" not in df.columns or "actual_reward" not in df.columns:
        return pd.DataFrame()
    r = pd.to_numeric(df["actual_reward"], errors="coerce")
    ir = pd.to_numeric(df["initial_reward"], errors="coerce")
    ok = r.notna() & ir.notna() & (r > ir)
    return df.loc[ok].copy()


def _param_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("param_")]
    out = []
    for c in cols:
        if df[c].notna().sum() < 2:
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return sorted(out)


def _fmt_param_level(u) -> str:
    try:
        fu = float(u)
        if abs(fu - round(fu)) < 1e-9:
            return str(int(round(fu)))
    except (TypeError, ValueError):
        pass
    return f"{u:.4g}"


def _is_integer_like(x: np.ndarray) -> bool:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False
    return np.all(np.abs(x - np.round(x)) < 1e-6)


def _plot_cell(ax, sub: pd.DataFrame, pcol: str, reward_col: str):
    pv = sub[pcol].to_numpy()
    rv = sub[reward_col].to_numpy()
    ok = np.isfinite(pv) & np.isfinite(rv)
    pv, rv = pv[ok], rv[ok]
    if len(rv) < 2:
        ax.text(0.5, 0.5, "n<2", ha="center", va="center", transform=ax.transAxes)
        return

    # Integer-like hyperparameters: jitter scatter of reward vs value.
    if _is_integer_like(pv):
        vals = np.sort(np.unique(pv))
        positions = np.arange(len(vals), dtype=float)
        for pos, v in zip(positions, vals):
            m = pv == v
            if not m.any():
                continue
            jitter = np.random.uniform(-0.12, 0.12, size=m.sum())
            ax.scatter(
                np.full(m.sum(), pos) + jitter,
                rv[m],
                alpha=0.6,
                s=18,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([_fmt_param_level(v) for v in vals], rotation=35, ha="right", fontsize=7)
        ax.set_xlabel(pcol.replace("param_", ""), fontsize=8)
        ax.set_ylabel(reward_col, fontsize=8)
        return

    # Continuous params: bin into quantiles, then plot mean reward with CI over bins.
    try:
        pv_f = pv.astype(float)
    except Exception:
        ax.text(0.5, 0.5, "non-numeric", ha="center", va="center", transform=ax.transAxes)
        return
    if pv_f.size < 3:
        ax.text(0.5, 0.5, "n<3", ha="center", va="center", transform=ax.transAxes)
        return

    n_bins = min(7, max(3, int(np.sqrt(len(pv_f)))))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(pv_f, qs)
    edges = np.unique(edges)
    if edges.size < 3:
        edges = np.linspace(pv_f.min(), pv_f.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    xs, means, ses = [], [], []
    for lo, hi, c in zip(edges[:-1], edges[1:], centers):
        m = (pv_f >= lo) & (pv_f <= hi)
        rr = rv[m]
        if rr.size < 2:
            continue
        xs.append(c)
        means.append(rr.mean())
        ses.append(rr.std(ddof=1) / np.sqrt(len(rr)))

    if not xs:
        ax.text(0.5, 0.5, "no bins", ha="center", va="center", transform=ax.transAxes)
        return

    xs = np.asarray(xs, dtype=float)
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    ax.errorbar(xs, means, yerr=1.96 * ses, marker="o", linestyle="-", linewidth=1.1, markersize=4)
    if np.nanmin(xs) > 0 and any(k in pcol.lower() for k in ("lr", "decay")):
        ax.set_xscale("log")
    ax.set_xlabel(pcol.replace("param_", ""), fontsize=8)
    ax.set_ylabel(reward_col, fontsize=8)


def _plot_sensitivity_df(
    df: pd.DataFrame,
    setup_label: str,
    out_dir: Path,
    *,
    title_suffix: str = "",
) -> int:
    """One setup label (folder name) -> per-method, per-param PNGs."""
    if df.empty or "method" not in df.columns:
        return 0

    reward_col = _reward_column(df)
    param_cols = _param_columns(df)
    if not param_cols:
        return 0

    train_sizes = sorted(df["train_size"].dropna().unique().tolist())
    if not train_sizes:
        return 0

    safe_name = setup_label.replace("/", "_")
    n_made = 0
    for method, g_all in df.groupby("method"):
        g_all = g_all.copy()
        method_dir = out_dir / safe_name / f"method={method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        for pcol in param_cols:
            fig, axes = plt.subplots(
                1,
                len(train_sizes),
                figsize=(3.2 * len(train_sizes), 3.0),
                squeeze=False,
            )
            for j, ts in enumerate(train_sizes):
                ax = axes[0][j]
                sub = g_all[g_all["train_size"] == ts]
                _plot_cell(ax, sub, pcol, reward_col)
                ax.set_title(f"train={int(ts)}", fontsize=8)
            suf = f" | {title_suffix}" if title_suffix else ""
            fig.suptitle(
                f"{safe_name}\nmethod={method} | {pcol.replace('param_', '')} | reward={reward_col}{suf}",
                fontsize=9,
            )
            fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.9])
            out_path = method_dir / f"{pcol}.png"
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            n_made += 1
    return n_made


def _plot_condition(condition_dir: Path, out_dir: Path) -> int:
    df = _load_condition_trials(condition_dir)
    return _plot_sensitivity_df(df, condition_dir.name, out_dir)


def _plot_combined_seeds(
    cond_dirs: list[Path],
    out_dir: Path,
    *,
    beat_initial_reward: bool = False,
) -> int:
    """Pool trials across all seeds that share the same setup (no seed in key)."""
    from collections import defaultdict

    groups: dict[str, list[Path]] = defaultdict(list)
    for d in cond_dirs:
        groups[_setup_key_without_seed(d.name)].append(d)

    total = 0
    for key, dirs in sorted(groups.items()):
        frames = []
        for d in sorted(dirs):
            df = _load_condition_trials(d)
            if not df.empty:
                frames.append(df)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        if beat_initial_reward:
            merged = _filter_beat_initial_reward(merged)
            if merged.empty:
                print(f"- combined [{key}] ({len(frames)} seeds): no trials with actual_reward > initial_reward")
                continue
        suffix = "actual_reward > initial_reward" if beat_initial_reward else ""
        n = _plot_sensitivity_df(merged, key, out_dir, title_suffix=suffix)
        total += n
        if n:
            tag = "beat-initial " if beat_initial_reward else ""
            print(f"+ combined {tag}[{key}] ({len(frames)} seeds): {n} figure(s)")
    return total


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter vs reward sensitivity plots per condition.")
    parser.add_argument("--run-dir", type=Path, required=True, help="run_* directory with dataset=*/")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/figures_hp_reward_sensitivity)",
    )
    parser.add_argument(
        "--mode",
        choices=("per_seed", "combined", "both"),
        default="per_seed",
        help="per_seed: one folder per dataset=...__seed=...; combined: pool all seeds per setup; both: run both.",
    )
    parser.add_argument(
        "--beat-initial-reward",
        action="store_true",
        default=False,
        help="Combined mode only: keep trials where actual_reward > initial_reward (needs columns in trials_long).",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "figures_hp_reward_sensitivity")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cond_dirs: set[Path] = set()
    for suffix in ("trials_long.csv", "opc_trials_long.csv", "no_prop_trials_long.csv"):
        for p in run_dir.rglob(suffix):
            if p.parent.name.startswith("dataset="):
                cond_dirs.add(p.parent)
    cond_dirs = sorted(cond_dirs)

    total = 0
    if args.mode in ("per_seed", "both"):
        for d in cond_dirs:
            n = _plot_condition(d, out_dir)
            total += n
            if n:
                print(f"+ {d.name}: {n} figure(s)")
    if args.mode in ("combined", "both"):
        sub = "combined_seeds_beat_initial" if args.beat_initial_reward else "combined_seeds"
        combined_root = out_dir / sub
        combined_root.mkdir(parents=True, exist_ok=True)
        n2 = _plot_combined_seeds(
            cond_dirs,
            combined_root,
            beat_initial_reward=args.beat_initial_reward,
        )
        total += n2
    print(f"Done. {total} figures -> {out_dir}")


if __name__ == "__main__":
    main()
