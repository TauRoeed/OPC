"""
initial_reward vs r_hat (val), r_hat_train, actual_reward — OPC | no_prop panels.
Points: green if actual_reward > initial_reward, else blue.
actual vs initial panel includes y=x over visible axis box.
Outputs one subfolder per param_use_log_trick value.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.load_run_trials import load_run_trials

PANELS = [
    ("opc", "OPC"),
    ("no_propensity", "no propensity"),
]


def _pad_linear(lo: float, hi: float, frac: float = 0.03) -> tuple[float, float]:
    p = max((hi - lo) * frac, 1e-12)
    return lo - p, hi + p


def _y_equals_x_segment(xlo: float, xhi: float, ylo: float, yhi: float) -> tuple[float, float] | None:
    t0 = max(xlo, ylo)
    t1 = min(xhi, yhi)
    if t0 < t1:
        return t0, t1
    return None


def _plot_panel(
    ax,
    sub: pd.DataFrame,
    y_col: str,
    y_label: str,
    title: str,
    *,
    draw_line: bool,
) -> None:
    need = [c for c in (y_col, "initial_reward", "actual_reward") if c in sub.columns]
    sub = sub.dropna(subset=need)
    if sub.empty or "initial_reward" not in sub.columns:
        ax.set_title(f"{title} (no data)")
        return
    x = sub["initial_reward"].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    if "actual_reward" in sub.columns and sub["actual_reward"].notna().any():
        beat = sub["actual_reward"].to_numpy(dtype=float) > sub["initial_reward"].to_numpy(dtype=float)
        ax.scatter(x[~beat], y[~beat], alpha=0.75, s=30, c="C0", edgecolors="none", label="actual ≤ initial")
        ax.scatter(x[beat], y[beat], alpha=0.75, s=30, c="tab:green", edgecolors="none", label="actual > initial")
    else:
        ax.scatter(x, y, alpha=0.75, s=30, c="C0", edgecolors="none")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xlo, xhi = _pad_linear(xmin, xmax)
    ylo, yhi = _pad_linear(ymin, ymax)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("auto")
    if draw_line:
        seg = _y_equals_x_segment(xlo, xhi, ylo, yhi)
        if seg:
            t0, t1 = seg
            ax.plot([t0, t1], [t0, t1], "k--", lw=1.2, label="y = x")
    ax.set_xlabel("initial_reward")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def _save_pair(
    df: pd.DataFrame,
    out_dir: Path,
    y_col: str,
    y_label: str,
    fname: str,
    suptitle: str,
    *,
    draw_line: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (meth, panel_title) in zip(axes, PANELS):
        mask = df["method"] == meth
        _plot_panel(ax, df.loc[mask], y_col, y_label, panel_title, draw_line=draw_line)
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_trick_is_true(v) -> bool:
    return bool(v) if isinstance(v, (bool, np.bool_)) else int(v) == 1


def _log_trick_subdirs(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "param_use_log_trick" not in df.columns:
        return [("all_trials", df)]
    col = df["param_use_log_trick"]
    uniq = sorted({bool(_log_trick_is_true(v)) for v in col.dropna().unique()}, key=lambda b: (not b, b))
    out: list[tuple[str, pd.DataFrame]] = []
    for flag in uniq:
        mask = col.map(_log_trick_is_true) == flag
        label = "use_log_trick_true" if flag else "use_log_trick_false"
        out.append((label, df.loc[mask].copy()))
    return out


def plot_run(run_dir: Path, out_root: Path | None = None) -> None:
    run_dir = run_dir.resolve()
    out_root = (out_root or (run_dir / "figures_initial_reward")).resolve()
    df = load_run_trials(run_dir)
    for c in ("initial_reward", "r_hat", "r_hat_train", "actual_reward"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "initial_reward" not in df.columns or not df["initial_reward"].notna().any():
        print(f"skip {out_root}: no initial_reward in trial logs")
        return

    specs = [
        ("r_hat", "r_hat (val)", "initial_reward_vs_r_hat_val.png", False),
        ("r_hat_train", "r_hat_train", "initial_reward_vs_r_hat_train.png", False),
        ("actual_reward", "actual_reward", "initial_reward_vs_actual_reward.png", True),
    ]
    for label, sub in _log_trick_subdirs(df):
        out_dir = out_root / label
        n = len(sub)
        tag = (
            label.replace("use_log_trick_", "log trick: ")
            if label.startswith("use_log_trick_")
            else label
        )
        for y_col, y_label, fname, draw_line in specs:
            if y_col not in sub.columns:
                continue
            _save_pair(
                sub,
                out_dir,
                y_col,
                y_label,
                fname,
                f"initial_reward vs {y_label} ({tag}, n={n})",
                draw_line=draw_line,
            )
        print(f"{out_dir}: {n} rows")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/full_study/run_n100_t1000_s0"),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    plot_run(args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()
