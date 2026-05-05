import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _load_summary(path: Path):
    df = pd.read_csv(path)
    if "train_size" in df.columns:
        df["train_size"] = df["train_size"].astype(int)
    return df


def _pivot_delta(df: pd.DataFrame, metric: str):
    tmp = (
        df.pivot_table(
            index=["dataset", "noise_mode", "noise_level", "seed", "train_size"],
            columns="method",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
        .dropna(subset=["opc", "no_propensity"])
    )
    tmp["delta"] = tmp["opc"] - tmp["no_propensity"]
    return tmp


def _plot_performance_curves(df: pd.DataFrame, out_dir: Path):
    grouped = (
        df.groupby(["dataset", "noise_mode", "method", "train_size"])["policy_rewards"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    for (dataset, noise_mode), part in grouped.groupby(["dataset", "noise_mode"]):
        plt.figure(figsize=(8, 5))
        for method, p in part.groupby("method"):
            p = p.sort_values("train_size")
            plt.plot(p["train_size"], p["mean"], marker="o", label=method)
            plt.fill_between(
                p["train_size"],
                p["mean"] - 1.96 * p["se"],
                p["mean"] + 1.96 * p["se"],
                alpha=0.2,
            )
        plt.xscale("log")
        plt.title(f"Performance vs train size ({dataset}, {noise_mode})")
        plt.xlabel("train_size")
        plt.ylabel("policy_rewards")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{dataset}_{noise_mode}.png", dpi=180)
        plt.close()


def _plot_delta_heatmaps(df: pd.DataFrame, out_dir: Path):
    delta = _pivot_delta(df, "policy_rewards")
    for (dataset, noise_mode), part in delta.groupby(["dataset", "noise_mode"]):
        heat = (
            part.groupby(["noise_level", "train_size"])["delta"]
            .mean()
            .unstack("train_size")
            .reindex(["low", "medium", "high"])
        )
        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm")
        ax.set_xticks(np.arange(len(heat.columns)))
        ax.set_xticklabels(heat.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(heat.index)))
        ax.set_yticklabels(heat.index)
        ax.set_title(f"OPC - NoProp delta heatmap ({dataset}, {noise_mode})")
        plt.colorbar(im, ax=ax, label="policy_rewards delta")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_{dataset}_{noise_mode}.png", dpi=180)
        plt.close()


def _plot_box_violin(df: pd.DataFrame, out_dir: Path):
    delta = _pivot_delta(df, "policy_rewards")
    delta["cond"] = (
        delta["dataset"]
        + "|"
        + delta["noise_mode"]
        + "|"
        + delta["noise_level"]
        + "|n="
        + delta["train_size"].astype(str)
    )
    top = (
        delta.groupby("cond")["delta"].count().sort_values(ascending=False).head(18).index
    )
    part = delta[delta["cond"].isin(top)]
    plt.figure(figsize=(14, 5))
    data = [part.loc[part["cond"] == c, "delta"].values for c in top]
    plt.violinplot(data, showmeans=True, showmedians=False)
    plt.xticks(np.arange(1, len(top) + 1), top, rotation=60, ha="right")
    plt.title("Seed distribution of OPC delta (selected conditions)")
    plt.ylabel("policy_rewards delta")
    plt.tight_layout()
    plt.savefig(out_dir / "delta_violin_selected_conditions.png", dpi=180)
    plt.close()


def _plot_pareto(df: pd.DataFrame, out_dir: Path):
    agg = (
        df.groupby(["dataset", "method"])[["policy_rewards", "conv_dr_var"]]
        .mean()
        .reset_index()
    )
    for dataset, part in agg.groupby("dataset"):
        plt.figure(figsize=(6, 4))
        for _, row in part.iterrows():
            plt.scatter(row["conv_dr_var"], row["policy_rewards"], s=120)
            plt.text(row["conv_dr_var"], row["policy_rewards"], row["method"])
        plt.xlabel("variance (conv_dr_var)")
        plt.ylabel("reward (policy_rewards)")
        plt.title(f"Pareto: reward vs variance ({dataset})")
        plt.tight_layout()
        plt.savefig(out_dir / f"pareto_{dataset}.png", dpi=180)
        plt.close()


def _plot_calibration(df: pd.DataFrame, out_dir: Path):
    for dataset, part in df.groupby("dataset"):
        plt.figure(figsize=(6, 5))
        for method, p in part.groupby("method"):
            x = p["conv_dr"].to_numpy()
            y = p["policy_rewards"].to_numpy()
            if len(x) == 0:
                continue
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            bins = np.array_split(np.arange(len(x)), min(20, len(x)))
            xb = np.array([x[b].mean() for b in bins])
            yb = np.array([y[b].mean() for b in bins])
            plt.plot(xb, yb, marker="o", label=method)
        lo, hi = plt.xlim()
        plt.plot([lo, hi], [lo, hi], "k--", alpha=0.6)
        plt.xlabel("estimated (conv_dr)")
        plt.ylabel("true (policy_rewards)")
        plt.title(f"Calibration plot ({dataset})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"calibration_{dataset}.png", dpi=180)
        plt.close()


def _robustness_ranking(df: pd.DataFrame):
    delta = _pivot_delta(df, "policy_rewards")
    summary = (
        delta.groupby(["dataset", "noise_mode", "noise_level"])["delta"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values(["dataset", "mean"], ascending=[True, False])
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize full study outputs.")
    parser.add_argument("--summary-csv", default="artifacts/full_study/all_summary_metrics.csv")
    parser.add_argument("--out-dir", default="artifacts/full_study/figures")
    args = parser.parse_args()

    summary_csv = Path(args.summary_csv)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    df = _load_summary(summary_csv)

    _plot_performance_curves(df, out_dir)
    _plot_delta_heatmaps(df, out_dir)
    _plot_box_violin(df, out_dir)
    _plot_pareto(df, out_dir)
    _plot_calibration(df, out_dir)

    robust = _robustness_ranking(df)
    robust.to_csv(out_dir / "robustness_ranking.csv", index=False)

    delta = _pivot_delta(df, "policy_rewards")
    sig = (
        delta.groupby(["dataset", "noise_mode", "noise_level", "train_size"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    sig["se"] = sig["std"] / np.sqrt(sig["count"].clip(lower=1))
    sig["ci_low"] = sig["mean"] - 1.96 * sig["se"]
    sig["ci_high"] = sig["mean"] + 1.96 * sig["se"]
    sig["significant_positive"] = sig["ci_low"] > 0
    sig.to_csv(out_dir / "delta_significance.csv", index=False)


if __name__ == "__main__":
    main()
