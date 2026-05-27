"""CTR and axis ablation plots for full-study runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.analyze_full_study import _ensure_dir, _load_summary, _pivot_delta

LEVELS = ["low", "medium", "high"]
AXES = ["combined", "context", "action"]
METHOD_LABELS = {
    "opc": "OPC",
    "no_propensity": "no propensity",
}
COMBINED_EPS = {
    "low": (0.05, 0.05, 0.0),
    "medium": (0.10, 0.15, 0.05),
    "high": (0.20, 0.25, 0.10),
}
PER_AXIS_EPS = {
    "context": {"low": 0.05, "medium": 0.10, "high": 0.20},
    "action": {"low": 0.05, "medium": 0.15, "high": 0.25},
    "metadata": {"low": 0.0, "medium": 0.05, "high": 0.10},
}


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in (
        "train_size",
        "policy_rewards",
        "conv_dr",
        "seed",
        "ctr",
        "initial_reward",
        "action_delta",
        "context_delta",
        "action_diff_to_real",
        "context_diff_to_real",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fmt_float(value: float) -> str:
    return f"{float(value):g}"


def _agg_with_ci(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False)[value_col].agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def _plot_method_curves(part: pd.DataFrame, out_path: Path, title: str) -> None:
    grp = _agg_with_ci(
        part,
        ["dataset", "noise_mode", "noise_axis", "noise_level", "method", "train_size"],
        "policy_rewards",
    )
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
    for ax, level in zip(axes, LEVELS):
        pp = grp[grp["noise_level"] == level]
        for method, mm in pp.groupby("method"):
            mm = mm.sort_values("train_size")
            x = mm["train_size"].to_numpy(dtype=float)
            y = mm["mean"].to_numpy(dtype=float)
            se = mm["se"].to_numpy(dtype=float)
            if len(x) == 0:
                continue
            ax.plot(x, y, marker="o", label=METHOD_LABELS.get(method, method))
            ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.18)
        ax.set_xscale("log")
        ax.set_title(level)
        ax.set_xlabel("train_size")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("policy_rewards")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_delta_curves(part: pd.DataFrame, out_path: Path, title: str) -> None:
    delta = _pivot_delta(part, "policy_rewards")
    if delta.empty:
        return
    grp = _agg_with_ci(delta, ["dataset", "noise_mode", "noise_axis", "noise_level", "train_size"], "delta")
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
    for ax, level in zip(axes, LEVELS):
        pp = grp[grp["noise_level"] == level].sort_values("train_size")
        if not pp.empty:
            ax.errorbar(
                pp["train_size"],
                pp["mean"],
                yerr=1.96 * pp["se"],
                marker="o",
                capsize=3,
            )
        ax.axhline(0.0, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(level)
        ax.set_xlabel("train_size")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("OPC - no propensity")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_ctr_slices(df: pd.DataFrame, out_dir: Path) -> None:
    for ctr, ctr_df in df.dropna(subset=["ctr"]).groupby("ctr"):
        ctr_label = _fmt_float(ctr)
        ctr_dir = out_dir / f"ctr_{ctr_label}"
        _ensure_dir(ctr_dir)
        for (dataset, noise_mode, axis), part in ctr_df.groupby(["dataset", "noise_mode", "noise_axis"]):
            _plot_method_curves(
                part,
                ctr_dir / f"curves_{dataset}_{noise_mode}_{axis}_ctr_{ctr_label}.png",
                f"Policy curves ({dataset}, {noise_mode}, axis={axis}, ctr={ctr_label})",
            )
            _plot_delta_curves(
                part,
                ctr_dir / f"delta_{dataset}_{noise_mode}_{axis}_ctr_{ctr_label}.png",
                f"Delta with CI ({dataset}, {noise_mode}, axis={axis}, ctr={ctr_label})",
            )


def plot_ctr_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "ctr_sensitivity"
    _ensure_dir(out)
    train_size = float(pd.to_numeric(df["train_size"], errors="coerce").max())
    max_train = df[pd.to_numeric(df["train_size"], errors="coerce") == train_size]

    delta = _pivot_delta(max_train, "policy_rewards")
    if not delta.empty:
        grp = _agg_with_ci(delta, ["dataset", "noise_mode", "noise_axis", "noise_level", "ctr"], "delta")
        for (dataset, noise_mode), part in grp.groupby(["dataset", "noise_mode"]):
            fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
            for ax, level in zip(axes, LEVELS):
                pp = part[part["noise_level"] == level]
                for axis, aa in pp.groupby("noise_axis"):
                    aa = aa.sort_values("ctr")
                    ax.errorbar(aa["ctr"], aa["mean"], yerr=1.96 * aa["se"], marker="o", capsize=3, label=axis)
                ax.axhline(0.0, ls="--", lw=1)
                ax.set_title(level)
                ax.set_xlabel("ctr")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel("OPC - no propensity")
            handles, labels = axes[-1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
            fig.suptitle(f"Delta vs CTR ({dataset}, {noise_mode}, train_size={int(train_size)})")
            fig.tight_layout()
            fig.savefig(out / f"delta_vs_ctr_{dataset}_{noise_mode}_train_{int(train_size)}.png", dpi=180)
            plt.close(fig)

    grp = _agg_with_ci(
        max_train,
        ["dataset", "noise_mode", "noise_axis", "noise_level", "method", "ctr"],
        "policy_rewards",
    )
    for (dataset, noise_mode), part in grp.groupby(["dataset", "noise_mode"]):
        fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
        for ax, level in zip(axes, LEVELS):
            pp = part[part["noise_level"] == level]
            for (axis, method), aa in pp.groupby(["noise_axis", "method"]):
                aa = aa.sort_values("ctr")
                label = f"{axis} / {METHOD_LABELS.get(method, method)}"
                ax.errorbar(aa["ctr"], aa["mean"], yerr=1.96 * aa["se"], marker="o", capsize=3, label=label)
            ax.set_title(level)
            ax.set_xlabel("ctr")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("policy_rewards")
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
        fig.suptitle(f"Policy reward vs CTR ({dataset}, {noise_mode}, train_size={int(train_size)})")
        fig.tight_layout()
        fig.savefig(out / f"policy_vs_ctr_{dataset}_{noise_mode}_train_{int(train_size)}.png", dpi=180)
        plt.close(fig)


def plot_axis_side_by_side(df: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "axis_comparison"
    _ensure_dir(out)
    grp = _agg_with_ci(
        df,
        ["dataset", "noise_mode", "ctr", "noise_axis", "noise_level", "method", "train_size"],
        "policy_rewards",
    )
    for (dataset, noise_mode, ctr), part in grp.dropna(subset=["ctr"]).groupby(["dataset", "noise_mode", "ctr"]):
        ctr_label = _fmt_float(ctr)
        fig, axes = plt.subplots(len(LEVELS), len(AXES), figsize=(5 * len(AXES), 3.6 * len(LEVELS)), sharex=True, sharey=True)
        for i, level in enumerate(LEVELS):
            for j, axis in enumerate(AXES):
                ax = axes[i][j]
                pp = part[(part["noise_level"] == level) & (part["noise_axis"] == axis)]
                if pp.empty:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                else:
                    counts = pp.groupby("method")["count"].max().astype(int)
                    for method, mm in pp.groupby("method"):
                        mm = mm.sort_values("train_size")
                        x = mm["train_size"].to_numpy(dtype=float)
                        y = mm["mean"].to_numpy(dtype=float)
                        se = mm["se"].to_numpy(dtype=float)
                        ax.plot(x, y, marker="o", label=METHOD_LABELS.get(method, method))
                        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, alpha=0.18)
                    seed_text = ", ".join(f"{METHOD_LABELS.get(k, k)} n={v}" for k, v in counts.items())
                    ax.text(0.02, 0.04, seed_text, transform=ax.transAxes, fontsize=7, va="bottom")
                ax.set_xscale("log")
                ax.set_title(f"{axis} / {level}")
                if i == len(LEVELS) - 1:
                    ax.set_xlabel("train_size")
                if j == 0:
                    ax.set_ylabel("policy_rewards")
                ax.grid(True, alpha=0.25)
        handles, labels = axes[0][-1].get_legend_handles_labels()
        if not handles:
            handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
        fig.suptitle(f"Axis comparison ({dataset}, {noise_mode}, ctr={ctr_label})")
        fig.tight_layout()
        fig.savefig(out / f"axis_side_by_side_{dataset}_{noise_mode}_ctr_{ctr_label}.png", dpi=180)
        plt.close(fig)


def plot_axis_delta_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    out = out_dir / "axis_comparison"
    _ensure_dir(out)
    delta = _pivot_delta(df, "policy_rewards")
    if delta.empty:
        return
    agg = (
        delta.groupby(["dataset", "noise_mode", "ctr", "train_size", "noise_level", "noise_axis"], dropna=False)["delta"]
        .agg(["mean", "count"])
        .reset_index()
    )
    for (dataset, noise_mode, ctr, train_size), part in agg.dropna(subset=["ctr"]).groupby(
        ["dataset", "noise_mode", "ctr", "train_size"]
    ):
        matrix = np.full((len(LEVELS), len(AXES)), np.nan)
        counts = np.zeros((len(LEVELS), len(AXES)), dtype=int)
        for i, level in enumerate(LEVELS):
            for j, axis in enumerate(AXES):
                pp = part[(part["noise_level"] == level) & (part["noise_axis"] == axis)]
                if pp.empty:
                    continue
                matrix[i, j] = float(pp["mean"].iloc[0])
                counts[i, j] = int(pp["count"].iloc[0])
        vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
        vmax = max(float(vmax), 1e-6)
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(AXES)), labels=AXES)
        ax.set_yticks(np.arange(len(LEVELS)), labels=LEVELS)
        ax.set_xlabel("noise_axis")
        ax.set_ylabel("noise_level")
        for i in range(len(LEVELS)):
            for j in range(len(AXES)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.4f}\nn={counts[i, j]}", ha="center", va="center", fontsize=8)
                else:
                    ax.text(j, i, "no data", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean OPC - no propensity")
        ctr_label = _fmt_float(ctr)
        ax.set_title(f"Mean delta heatmap ({dataset}, {noise_mode}, ctr={ctr_label}, train={int(train_size)})")
        fig.tight_layout()
        fig.savefig(out / f"axis_delta_heatmap_{dataset}_{noise_mode}_ctr_{ctr_label}_train_{int(train_size)}.png", dpi=180)
        plt.close(fig)


def write_matched_eps(out_dir: Path) -> None:
    out = out_dir / "axis_comparison"
    _ensure_dir(out)
    rows = []
    for level in LEVELS:
        context_eps, action_eps, metadata_eps = COMBINED_EPS[level]
        rows.append(
            {
                "noise_level": level,
                "combined_context_eps": context_eps,
                "combined_action_eps": action_eps,
                "combined_metadata_eps": metadata_eps,
                "context_axis_eps": PER_AXIS_EPS["context"][level],
                "action_axis_eps": PER_AXIS_EPS["action"][level],
                "metadata_axis_eps": PER_AXIS_EPS["metadata"][level],
            }
        )
    pd.DataFrame(rows).to_csv(out / "matched_eps_readout.csv", index=False)


def plot_drift_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in ("action_delta", "context_delta", "action_diff_to_real", "context_diff_to_real") if c in df.columns]
    if not cols:
        return
    out = out_dir / "drift_diagnostics"
    _ensure_dir(out)
    for col in cols:
        grp = _agg_with_ci(df.dropna(subset=[col]), ["dataset", "noise_mode", "noise_axis", "noise_level", "train_size"], col)
        for (dataset, noise_mode), part in grp.groupby(["dataset", "noise_mode"]):
            fig, axes = plt.subplots(1, len(LEVELS), figsize=(5 * len(LEVELS), 4), sharey=True)
            for ax, level in zip(axes, LEVELS):
                pp = part[part["noise_level"] == level]
                for axis, aa in pp.groupby("noise_axis"):
                    aa = aa.sort_values("train_size")
                    ax.errorbar(aa["train_size"], aa["mean"], yerr=1.96 * aa["se"], marker="o", capsize=3, label=axis)
                ax.set_xscale("log")
                ax.set_title(level)
                ax.set_xlabel("train_size")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel(col)
            handles, labels = axes[-1].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
            fig.suptitle(f"{col} by axis ({dataset}, {noise_mode})")
            fig.tight_layout()
            fig.savefig(out / f"{col}_by_axis_{dataset}_{noise_mode}.png", dpi=180)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CTR and axis ablations for a full-study run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary_csv = (args.summary_csv or (run_dir / "all_summary_metrics.csv")).resolve()
    out_dir = (args.out_dir or (run_dir / "figures")).resolve()
    _ensure_dir(out_dir)

    df = _numeric(_load_summary(summary_csv))
    plot_ctr_slices(df, out_dir)
    plot_ctr_sensitivity(df, out_dir)
    plot_axis_side_by_side(df, out_dir)
    plot_axis_delta_heatmaps(df, out_dir)
    write_matched_eps(out_dir)
    plot_drift_diagnostics(df, out_dir)
    print(f"wrote CTR and axis ablation figures under {out_dir}")


if __name__ == "__main__":
    main()
