"""
Compare actual_reward between OPC and no propensity for a full-study run.

Outputs under ``<run_dir>/figures_actual_reward_comparison/``:
  - distribution comparison (box + strip) by log trick
  - paired scatter (matched hyperparameters per seed): OPC vs no propensity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.load_run_trials import load_run_trials

PARAM_KEYS = [
    "param_lr",
    "param_num_epochs",
    "param_batch_size",
    "param_lr_decay",
    "param_kl_gamma",
    "param_use_log_trick",
]
MERGE_KEYS = ["seed", "train_size", "run"] + PARAM_KEYS


def _log_trick_true(v) -> bool:
    return bool(v) if isinstance(v, bool) else int(v) == 1


def _log_trick_label(v) -> str:
    return "log trick ON" if _log_trick_true(v) else "log trick OFF"


def _log_trick_slices(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "param_use_log_trick" not in df.columns:
        return [("all", df)]
    return [
        ("false", df[df["param_use_log_trick"].map(_log_trick_true) == False]),
        ("true", df[df["param_use_log_trick"].map(_log_trick_true) == True]),
    ]


def _ctr_slug(ctr: float) -> str:
    return f"{ctr:g}".replace(".", "p")


def _level_slug(level: str) -> str:
    return str(level).replace(" ", "_")


def _prepare_reward_distribution_df(
    df: pd.DataFrame,
    *,
    train_only: bool = True,
) -> pd.DataFrame:
    sub = df.copy()
    sub["ctr"] = pd.to_numeric(sub.get("ctr"), errors="coerce")
    sub["actual_reward"] = pd.to_numeric(sub.get("actual_reward"), errors="coerce")
    if "initial_reward" in sub.columns:
        sub["initial_reward"] = pd.to_numeric(sub["initial_reward"], errors="coerce")
    if train_only and "train_size" in sub.columns:
        sub = sub[pd.to_numeric(sub["train_size"], errors="coerce") > 0]
    return sub.dropna(subset=["ctr", "actual_reward"])


def _draw_reward_distribution_panel(ax, part: pd.DataFrame, title: str) -> None:
    method_specs = [("opc", "OPC", "C0"), ("no_propensity", "no propensity", "C1")]
    labels, data = [], []
    for meth, label, _ in method_specs:
        vals = part.loc[part["method"] == meth, "actual_reward"].dropna().to_numpy()
        if vals.size:
            labels.append(label)
            data.append(vals)
    if not data:
        ax.set_title(f"{title}\n(no data)")
        return
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
    )
    colors = ("C0", "C1")
    for patch, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    rng = np.random.default_rng(0)
    for meth, label, c in method_specs:
        block = part.loc[part["method"] == meth]
        vals = block["actual_reward"].dropna().to_numpy()
        if vals.size == 0:
            continue
        x = labels.index(label) + 1
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        if "initial_reward" in block.columns and block["initial_reward"].notna().any():
            beat = vals > block["initial_reward"].to_numpy(dtype=float)
            ax.scatter(
                x + jitter[~beat],
                vals[~beat],
                s=14,
                alpha=0.45,
                c=c,
                edgecolors="none",
            )
            ax.scatter(
                x + jitter[beat],
                vals[beat],
                s=14,
                alpha=0.55,
                c="tab:green",
                edgecolors="none",
            )
        else:
            ax.scatter(x + jitter, vals, s=14, alpha=0.45, c=c, edgecolors="none")
    if "initial_reward" in part.columns:
        ir = part["initial_reward"].dropna()
        if not ir.empty:
            ax.axhline(
                float(ir.median()),
                color="k",
                ls="--",
                lw=1,
                alpha=0.6,
                label="median initial_reward",
            )
    ax.set_title(title)
    ax.set_ylabel("actual_reward")


def plot_actual_reward_distribution_by_ctr(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    train_only: bool = True,
) -> int:
    """Distribution panels by CTR and noise level (OPC vs no propensity)."""
    if "ctr" not in df.columns or "actual_reward" not in df.columns:
        return 0

    sub = _prepare_reward_distribution_df(df, train_only=train_only)
    if sub.empty:
        return 0

    ctrs = sorted(sub["ctr"].unique())
    out_dir = Path(out_dir)
    by_ctr_dir = out_dir / "by_ctr"
    by_ctr_dir.mkdir(parents=True, exist_ok=True)

    level_col = "noise_level" if "noise_level" in sub.columns else None
    if level_col:
        level_order = ["low", "medium", "high"]
        levels = [lv for lv in level_order if lv in set(sub[level_col].astype(str))]
        levels += sorted(
            lv for lv in sub[level_col].astype(str).unique() if lv not in levels
        )
    else:
        levels = [None]

    by_ctr_level_dir = out_dir / "by_ctr_and_level"
    by_ctr_level_dir.mkdir(parents=True, exist_ok=True)
    n_saved = 0

    # Combined grid: rows = noise level, cols = CTR.
    if level_col:
        fig, axes = plt.subplots(
            len(levels),
            len(ctrs),
            figsize=(4.6 * len(ctrs), 4.2 * len(levels)),
            sharey=True,
            squeeze=False,
        )
        for i, level in enumerate(levels):
            for j, ctr in enumerate(ctrs):
                part = sub[(sub["ctr"] == ctr) & (sub[level_col].astype(str) == str(level))]
                _draw_reward_distribution_panel(
                    axes[i, j],
                    part,
                    f"level={level}, ctr={ctr:g} (n={len(part)})",
                )
        for ax in axes[:, 0]:
            ax.legend(loc="lower left", fontsize=7)
        fig.suptitle(
            "actual_reward by noise level x CTR (green = beat initial)",
            y=1.01,
        )
        fig.tight_layout()
        combined = out_dir / "actual_reward_distribution_by_ctr_and_level.png"
        fig.savefig(combined, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    # One file per CTR (all levels on one row).
    fig, axes = plt.subplots(1, len(ctrs), figsize=(4.8 * len(ctrs), 4.8), sharey=True)
    if len(ctrs) == 1:
        axes = [axes]
    for ax, ctr in zip(axes, ctrs):
        part = sub[sub["ctr"] == ctr]
        _draw_reward_distribution_panel(ax, part, f"ctr={ctr:g} (n={len(part)})")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("actual_reward distribution by CTR (all noise levels)", y=1.02)
    fig.tight_layout()
    combined_ctr = out_dir / "actual_reward_distribution_by_ctr.png"
    fig.savefig(combined_ctr, dpi=150, bbox_inches="tight")
    plt.close(fig)
    n_saved += 1

    for ctr in ctrs:
        part_ctr = sub[sub["ctr"] == ctr]
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        _draw_reward_distribution_panel(ax, part_ctr, f"ctr={ctr:g} (n={len(part_ctr)})")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        path = by_ctr_dir / f"actual_reward_distribution_ctr_{_ctr_slug(ctr)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

        if level_col:
            for level in levels:
                part = part_ctr[part_ctr[level_col].astype(str) == str(level)]
                fig, ax = plt.subplots(figsize=(5.2, 4.8))
                _draw_reward_distribution_panel(
                    ax,
                    part,
                    f"level={level}, ctr={ctr:g} (n={len(part)})",
                )
                ax.legend(loc="lower left", fontsize=8)
                fig.tight_layout()
                path = by_ctr_level_dir / (
                    f"actual_reward_distribution_ctr_{_ctr_slug(ctr)}_level_{_level_slug(level)}.png"
                )
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                n_saved += 1

    return n_saved


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    slices = _log_trick_slices(df)
    fig, axes = plt.subplots(1, len(slices), figsize=(5 * len(slices), 4.5), sharey=True)
    if len(slices) == 1:
        axes = [axes]
    for ax, (slug, sub) in zip(axes, slices):
        flag_label = _log_trick_label(slug == "true") if slug in ("true", "false") else "all trials"
        labels, data = [], []
        for meth, label in [("opc", "OPC"), ("no_propensity", "no propensity")]:
            vals = sub.loc[sub["method"] == meth, "actual_reward"].dropna().to_numpy()
            if vals.size:
                labels.append(label)
                data.append(vals)
        if not data:
            ax.set_title(flag_label + " (no data)")
            continue
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )
        colors = ("C0", "C1")
        for patch, c in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        for meth, label, c in zip(
            ("opc", "no_propensity"),
            labels,
            colors,
        ):
            vals = sub.loc[sub["method"] == meth, "actual_reward"].dropna().to_numpy()
            if vals.size:
                x = labels.index(label) + 1
                jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=vals.size)
                ax.scatter(x + jitter, vals, s=14, alpha=0.45, c=c, edgecolors="none")
        if "initial_reward" in sub.columns:
            ir = sub["initial_reward"].dropna()
            if not ir.empty:
                ax.axhline(float(ir.iloc[0]), color="k", ls="--", lw=1, alpha=0.6, label="initial_reward")
        ax.set_title(flag_label)
        ax.set_ylabel("actual_reward")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("actual_reward: OPC vs no propensity (all seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_scatter(df: pd.DataFrame, out_path: Path) -> None:
    opc = df[df["method"] == "opc"].copy()
    nop = df[df["method"] == "no_propensity"].copy()
    keys = [k for k in MERGE_KEYS if k in opc.columns and k in nop.columns]
    merged = opc.merge(
        nop,
        on=keys,
        suffixes=("_opc", "_nop"),
        how="inner",
    )
    if "param_use_log_trick" in merged.columns:
        pair_slices = [("false", False), ("true", True)]
    else:
        pair_slices = [("all", None)]

    fig, axes = plt.subplots(1, len(pair_slices), figsize=(5 * len(pair_slices), 4.5), sharex=True, sharey=True)
    if len(pair_slices) == 1:
        axes = [axes]
    if merged.empty:
        for ax in axes:
            ax.set_visible(False)
        fig.text(0.5, 0.5, "no paired trials (hyperparameter match)", ha="center")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    for ax, (slug, flag) in zip(axes, pair_slices):
        if flag is None:
            sub = merged
            title_suffix = "all"
        else:
            sub = merged[merged["param_use_log_trick_opc"].map(_log_trick_true) == flag]
            title_suffix = _log_trick_label(flag)
        if sub.empty:
            ax.set_title(title_suffix + " (no pairs)")
            continue
        x = sub["actual_reward_nop"].to_numpy(dtype=float)
        y = sub["actual_reward_opc"].to_numpy(dtype=float)
        if "initial_reward_opc" in sub.columns and "initial_reward_nop" in sub.columns:
            beat = (y > sub["initial_reward_opc"].to_numpy(dtype=float)) & (
                x > sub["initial_reward_nop"].to_numpy(dtype=float)
            )
            ax.scatter(x[~beat], y[~beat], s=28, alpha=0.7, c="C0", edgecolors="none", label="either ≤ initial")
            ax.scatter(x[beat], y[beat], s=28, alpha=0.7, c="tab:green", edgecolors="none", label="both > initial")
        else:
            ax.scatter(x, y, s=28, alpha=0.7, c="C0", edgecolors="none")
        lo = float(np.min([x.min(), y.min()]))
        hi = float(np.max([x.max(), y.max()]))
        pad = max((hi - lo) * 0.03, 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.1, label="y = x")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("no propensity actual_reward")
        ax.set_ylabel("OPC actual_reward")
        ax.set_title(f"{title_suffix} (n={len(sub)} pairs)")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("Paired trials (same seed + hyperparameters)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_vs_actual(df: pd.DataFrame, out_path: Path) -> None:
    if "value" not in df.columns:
        return
    for c in ("value", "actual_reward", "initial_reward"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (meth, title) in zip(axes, [("opc", "OPC"), ("no_propensity", "no propensity")]):
        need = [c for c in ("value", "actual_reward") if c in df.columns]
        sub = df[df["method"] == meth].dropna(subset=need)
        if sub.empty:
            ax.set_title(f"{title} (no data)")
            continue
        x = sub["value"].to_numpy()
        y = sub["actual_reward"].to_numpy()
        if "initial_reward" in sub.columns and sub["initial_reward"].notna().any():
            beat = y > sub["initial_reward"].to_numpy()
            ax.scatter(x[~beat], y[~beat], s=22, alpha=0.55, c="C0", edgecolors="none")
            ax.scatter(x[beat], y[beat], s=22, alpha=0.55, c="tab:green", edgecolors="none")
        else:
            ax.scatter(x, y, s=22, alpha=0.55, c="C0", edgecolors="none")
        ax.set_xlabel("score (Optuna value)")
        ax.set_ylabel("actual_reward")
        ax.set_title(title)
    fig.suptitle("score vs actual_reward", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run(run_dir: Path, out_dir: Path | None = None) -> None:
    run_dir = run_dir.resolve()
    out_dir = (out_dir or (run_dir / "figures_actual_reward_comparison")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_run_trials(run_dir)
    for c in ("actual_reward", "initial_reward", "value") + tuple(PARAM_KEYS):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") if c != "param_use_log_trick" else df[c]

    plot_distributions(df, out_dir / "actual_reward_distribution_opc_vs_noprop.png")
    n_ctr = plot_actual_reward_distribution_by_ctr(df, out_dir)
    plot_paired_scatter(df, out_dir / "actual_reward_paired_opc_vs_noprop.png")
    plot_score_vs_actual(df, out_dir / "score_vs_actual_reward_opc_vs_noprop.png")
    print(f"wrote figures under {out_dir}" + (f" ({n_ctr} CTR panels)" if n_ctr else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/full_study/run_serious_high_t20_tr400k_v100k_full"),
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    plot_run(args.run_dir, args.out_dir)


if __name__ == "__main__":
    main()
