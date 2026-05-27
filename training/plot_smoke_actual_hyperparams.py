"""
Build hyperparameter scatter figures for a smoke run: color points by
actual_reward > initial_reward (green vs blue).

Writes PNGs under ``<run_dir>/figures_actual_hyperparams/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3d projection

from training.load_run_trials import load_run_trials

LOG_PARAMS = frozenset({"param_lr", "param_lr_decay"})

SCATTER3D_PAIRS: list[tuple[str, str]] = [
    ("param_lr", "param_lr_decay"),
    ("param_num_epochs", "param_batch_size"),
    ("param_lr", "param_num_epochs"),
    ("param_batch_size", "param_lr_decay"),
    ("param_batch_size", "param_kl_gamma"),
    ("param_lr", "param_kl_gamma"),
    ("param_lr", "param_batch_size"),
    ("param_lr_decay", "param_kl_gamma"),
    ("param_num_epochs", "param_lr_decay"),
    ("param_num_epochs", "param_kl_gamma"),
]


def _pad_linear(lo: float, hi: float, frac: float = 0.03) -> tuple[float, float]:
    p = max((hi - lo) * frac, 1e-12)
    return lo - p, hi + p


def _pad_log_positive(lo: float, hi: float, factor: float = 1.06) -> tuple[float, float]:
    """Multiplicative padding; keeps limits strictly positive for log axes."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= 0:
        return lo, hi
    if hi < lo:
        lo, hi = hi, lo
    return lo / factor, hi * factor


def _log_trick_is_true(v) -> bool:
    return bool(v) if isinstance(v, (bool, np.bool_)) else int(v) == 1


def _beat_mask(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df["actual_reward"], errors="coerce")
    i0 = pd.to_numeric(df["initial_reward"], errors="coerce")
    return (a > i0) & a.notna() & i0.notna()


def plot_actual_vs_each_hyperparam(df: pd.DataFrame, out: Path) -> None:
    param_cols = [c for c in df.columns if c.startswith("param_")]
    param_cols.sort()
    if not param_cols:
        raise ValueError("no param_* columns in trials_long")

    need = param_cols + ["actual_reward", "initial_reward"]
    sub = df.dropna(subset=["actual_reward", "initial_reward"]).copy()
    for c in param_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.dropna(subset=param_cols, how="any")

    beat = _beat_mask(sub)
    y = pd.to_numeric(sub["actual_reward"], errors="coerce").to_numpy()

    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, pc in zip(axes_flat, param_cols):
        x = sub[pc].to_numpy()
        b = beat.to_numpy()
        ax.scatter(x[~b], y[~b], alpha=0.75, s=22, c="C0", edgecolors="none", label="actual ≤ initial")
        ax.scatter(x[b], y[b], alpha=0.75, s=22, c="tab:green", edgecolors="none", label="actual > initial")
        if pc in LOG_PARAMS:
            ax.set_xscale("log")
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        if pc in LOG_PARAMS:
            ax.set_xlim(*_pad_log_positive(xmin, xmax))
        else:
            ax.set_xlim(*_pad_linear(xmin, xmax))
        ax.set_ylim(*_pad_linear(ymin, ymax))
        ax.set_xlabel(pc)
        ax.set_ylabel("actual_reward")
        ax.set_title(f"actual_reward vs {pc}")
        ax.legend(loc="best", fontsize=7)

    for ax in axes_flat[len(param_cols) :]:
        ax.set_visible(False)

    fig.suptitle("actual_reward vs hyperparameters (all seeds, trials_long)", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter3d_pairs(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df.dropna(subset=["actual_reward", "initial_reward"]).copy()
    for c in ["actual_reward", "initial_reward"]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    beat = _beat_mask(sub)
    z = pd.to_numeric(sub["actual_reward"], errors="coerce").to_numpy()
    b = beat.to_numpy()

    pairs = [(px, py) for px, py in SCATTER3D_PAIRS if px in sub.columns and py in sub.columns]
    for px, py in pairs:
        for c in (px, py):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        m = sub[[px, py, "actual_reward"]].notna().all(axis=1)
        d = sub.loc[m]
        if d.empty:
            continue
        x = d[px].to_numpy(dtype=float)
        y = d[py].to_numpy(dtype=float)
        zz = d["actual_reward"].to_numpy(dtype=float)
        bb = _beat_mask(d).to_numpy()

        fig = plt.figure(figsize=(7, 5.5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x[~bb], y[~bb], zz[~bb], alpha=0.75, s=22, c="C0", depthshade=True, label="actual ≤ initial")
        ax.scatter(x[bb], y[bb], zz[bb], alpha=0.75, s=22, c="tab:green", depthshade=True, label="actual > initial")
        if px in LOG_PARAMS:
            ax.set_xscale("log")
        if py in LOG_PARAMS:
            ax.set_yscale("log")
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        zmin, zmax = float(np.nanmin(zz)), float(np.nanmax(zz))
        ax.set_xlim(
            *_pad_log_positive(xmin, xmax) if px in LOG_PARAMS else _pad_linear(xmin, xmax)
        )
        ax.set_ylim(
            *_pad_log_positive(ymin, ymax) if py in LOG_PARAMS else _pad_linear(ymin, ymax)
        )
        ax.set_zlim(*_pad_linear(zmin, zmax))
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.set_zlabel("actual_reward")
        ax.set_title(f"actual_reward vs {px} vs {py}")
        ax.legend(loc="upper left", fontsize=7)
        fname = f"scatter3d_actual_{px}__{py}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _split_by_log_trick(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "param_use_log_trick" not in df.columns:
        return [("", df)]
    col = df["param_use_log_trick"]
    uniq = sorted({bool(_log_trick_is_true(v)) for v in col.dropna().unique()}, key=lambda b: (not b, b))
    out: list[tuple[str, pd.DataFrame]] = []
    for flag in uniq:
        mask = col.map(_log_trick_is_true) == flag
        label = "use_log_trick_true" if flag else "use_log_trick_false"
        out.append((label, df.loc[mask].copy()))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/smoke_train_val_true/run_smoke_n500_s0123456789"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <run-dir>/figures_actual_hyperparams",
    )
    p.add_argument(
        "--split-log-trick",
        action="store_true",
        help="write figures under use_log_trick_true/ and use_log_trick_false/ subdirs",
    )
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    out_root = (args.out_dir or (run_dir / "figures_actual_hyperparams")).resolve()

    df = load_run_trials(run_dir)
    groups = _split_by_log_trick(df) if args.split_log_trick else [("", df)]

    for label, sub in groups:
        out_dir = out_root / label if label else out_root
        out_dir.mkdir(parents=True, exist_ok=True)
        param_cols = [c for c in sub.columns if c.startswith("param_")]
        if not param_cols:
            print(
                f"skip hyperparam figures for {out_dir}: no param_* columns "
                "(need trials_long or fuller Optuna export)"
            )
            continue
        plot_actual_vs_each_hyperparam(sub, out_dir / "scatter_actual_reward_vs_each_hyperparam.png")
        plot_scatter3d_pairs(sub, out_dir)
        print(f"wrote figures under {out_dir} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
