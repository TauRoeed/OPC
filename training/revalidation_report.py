"""Figures and tables of the revalidation report (docs/simulator_fix_opc_revalidation_20261004.md, Phase 4).

Inputs (summaries written by ``training.analyze_recoverability``):
  --old  the buggy-simulator summaries (artifacts/full_study/summaries_20260927): stage2_learned_rows.csv,
         stage3_lgs_*/stage2_learned_rows.csv
  --new  the fixed-simulator summaries (artifacts/full_study/opc_revalidation_20261004/summaries): the same files,
         plus budget_* and misspec_* learned rows (the reward-model tests) and the old counterparts of those under
         --old-extra (rebuilt from the local old run folders by ``old_extra_rows``)
  --tuning  the Phase 2 weighting tables (tuning/weights_*.csv)

Outputs (--out): fig1..fig7 as PNG and PDF, each with the plotted values in a CSV of the same name, and the old-vs-new
delta tables (``old_new_*.csv``) with each finding's classification (training.revalidation_compare).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.representation_report import (BIASES, COLORS, DATASETS, LABEL, METHOD_NAMES, SIZES, _plt, _save)
from training.revalidation_compare import KEYS, mean_ci, old_new_table

SHARES = (0.6, 0.8, 0.95)
ALL_BIASES = ("none",) + BIASES
VERDICT_COLORS = {"unchanged": "#4D4D4D", "same direction, different magnitude": "#0072B2", "weakened": "#E69F00",
                  "unsupported": "#D55E00", "reversed": "#CC0000", "new": "#009E73"}


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["train_size"] > 0].copy()


def cell_means(rows: pd.DataFrame, value: str, scale: float = 100.0, methods=None) -> pd.DataFrame:
    """Mean and 95% interval over the worlds of each bias × train size × arm."""
    out = []
    rows = rows if methods is None else rows[rows["method"].isin(methods)]
    for (b, n, m), g in rows.groupby(["bias", "train_size", "method"]):
        mm, lo, hi, k = mean_ci(scale * g[value])
        out.append(dict(bias=b, train_size=n, method=m, mean=mm, ci_low=lo, ci_high=hi, n=k))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------- figures


def fig1_recovery(old: pd.DataFrame, new: pd.DataFrame, out: Path) -> None:
    """OPC and DM-only greedy fraction of the oracle repair against data size, old (dashed) vs new (solid)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(BIASES), figsize=(12.5, 3.0), sharey=True)
    data = []
    for tag, rows, ls, alpha in (("old (buggy logs)", old, "--", 0.55), ("new (fixed logs)", new, "-", 1.0)):
        c = cell_means(rows, "fraction_of_oracle_repair_greedy", scale=1.0, methods=["opc", "dm"])
        data.append(c.assign(simulator=tag))
        for ax, b in zip(axes, BIASES):
            for i, m in enumerate(("opc", "dm")):
                g = c[(c["bias"] == b) & (c["method"] == m)].sort_values("train_size")
                xs = np.log10(g["train_size"]) + (i - 0.5) * 0.03 + (0.012 if ls == "-" else -0.012)
                ax.errorbar(xs, g["mean"], yerr=[g["mean"] - g["ci_low"], g["ci_high"] - g["mean"]], color=COLORS[m],
                            linestyle=ls, alpha=alpha, marker="o" if ls == "-" else "s", markersize=3.3, linewidth=1.3,
                            capsize=2, elinewidth=0.8, label=f"{METHOD_NAMES[m]}, {tag}")
    for ax, b in zip(axes, BIASES):
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        ax.set_title(LABEL[b])
        ax.set_xlabel("logged training rows")
    axes[0].set_ylabel("fraction of the oracle ranking repair\n(greedy)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Stage 2 recovery, old vs corrected logs (mean and 95% CI over 3 datasets × 2 seeds; the oracle bound is "
                 "unchanged)", fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig1_recovery_old_vs_new", pd.concat(data, ignore_index=True))


def fig2_arms(new: pd.DataFrame, out: Path, arms=("opc", "dm", "no_propensity", "tempered_logger")) -> None:
    """Corrected Stage 2: every arm's true stochastic gain over the logger."""
    plt = _plt()
    extra = [m for m in new["method"].unique() if m.startswith("opc_")]
    arms = tuple(arms) + tuple(sorted(extra))
    fig, axes = plt.subplots(1, len(ALL_BIASES), figsize=(14, 3.0), sharey=True)
    c = cell_means(new, "learned_gain", methods=list(arms))
    for ax, b in zip(axes, ALL_BIASES):
        for i, m in enumerate(arms):
            g = c[(c["bias"] == b) & (c["method"] == m)].sort_values("train_size")
            xs = np.log10(g["train_size"]) + (i - (len(arms) - 1) / 2) * 0.025
            ax.errorbar(xs, g["mean"], yerr=[g["mean"] - g["ci_low"], g["ci_high"] - g["mean"]],
                        color=COLORS.get(m, "#56B4E9"), linestyle=":" if m.startswith("opc_") else "-", marker="o",
                        markersize=3.2, linewidth=1.3, capsize=2, elinewidth=0.8, label=METHOD_NAMES.get(m, m))
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        ax.set_title(LABEL.get(b, b))
        ax.set_xlabel("logged training rows")
    axes[0].set_ylabel("true gain over the logger\n(stochastic, CTR points)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("Corrected Stage 2: OPC, DM-only, no-propensity and the tempered logger (mean and 95% CI over 6 worlds)",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig2_corrected_arms", c)


def fig3_contrasts(table: pd.DataFrame, out: Path, contrast: str = "OPC - DM-only", name: str = "fig3_opc_minus_dm_old_vs_new") -> None:
    """A contrast per bias × size: old and new mean with 95% CI, side by side, coloured by the verdict."""
    plt = _plt()
    t = table[table["contrast"] == contrast]
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    order = [b for b in ALL_BIASES if b in set(t["bias"])]
    for i, b in enumerate(order):
        for j, n in enumerate(SIZES):
            r = t[(t["bias"] == b) & (t["train_size"] == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            x0 = i + (j - 1) * 0.27
            ax.errorbar(x0 - 0.05, r["old"], yerr=[[r["old"] - r["old_lo"]], [r["old_hi"] - r["old"]]], fmt="s",
                        color="#999999", markersize=3.6, capsize=2, elinewidth=0.9, label="old" if (i, j) == (0, 0) else None)
            ax.errorbar(x0 + 0.05, r["new"], yerr=[[r["new"] - r["new_lo"]], [r["new_hi"] - r["new"]]], fmt="o",
                        color=VERDICT_COLORS[r["verdict"]], markersize=4.2, capsize=2, elinewidth=1.0)
            ax.text(x0, 0.01, f"{n // 1000}k", ha="center", va="bottom", fontsize=6.5, color="#666666",
                    transform=ax.get_xaxis_transform())  # x in data, y in axes fraction: fixed to the frame
    for v, c in VERDICT_COLORS.items():
        if v in set(t["verdict"]):
            ax.plot([], [], "o", color=c, label=f"new: {v}")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(range(len(order)), [LABEL.get(b, b) for b in order])
    ax.set_ylabel(f"{contrast}, true CTR points")
    ax.set_title(f"{contrast}: old (grey) vs corrected logs (coloured by the classification); paired over 6 worlds "
                 f"(5k / 25k / 100k left to right)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, name, t)


def fig_verdicts(findings: pd.DataFrame, out: Path) -> None:
    """Fig 7: every key finding, old vs new effect (points or fraction), coloured by its classification."""
    plt = _plt()
    f = findings.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * len(f) + 1.2))
    for i, r in f.iterrows():
        y = len(f) - 1 - i
        ax.errorbar(r["old"], y + 0.15, xerr=[[r["old"] - r["old_lo"]], [r["old_hi"] - r["old"]]], fmt="s", color="#999999",
                    markersize=3.5, capsize=2, elinewidth=0.8)
        ax.errorbar(r["new"], y - 0.15, xerr=[[r["new"] - r["new_lo"]], [r["new_hi"] - r["new"]]], fmt="o",
                    color=VERDICT_COLORS[r["verdict"]], markersize=4, capsize=2, elinewidth=1.0)
    ax.set_yticks(range(len(f)), [f"{r['finding']}  [{r['verdict']}]" for _, r in f.iloc[::-1].iterrows()], fontsize=7.5)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("effect (CTR points; fractions for the recovery rows); grey = old, coloured = corrected")
    ax.set_title("Old vs corrected: the key findings and their classification")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig7_conclusions", f)


# --------------------------------------------------------------------------------------------- tables


STAGE2_CONTRASTS = {
    "OPC fraction (greedy)": ("opc", None, "fraction_of_oracle_repair_greedy"),
    "OPC gain (stochastic)": ("opc", None, "learned_gain"),
    "DM-only fraction (greedy)": ("dm", None, "fraction_of_oracle_repair_greedy"),
    "OPC - DM-only": ("opc", "dm", "V_method"),
    "OPC - no-propensity": ("opc", "no_propensity", "V_method"),
    "OPC - tempered logger": ("opc", "tempered_logger", "V_method"),
}


def stage2_old_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    return old_new_table(old, new, STAGE2_CONTRASTS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = load_rows(Path(a.old) / "stage2_learned_rows.csv")
    new = load_rows(Path(a.new) / "stage2_learned_rows.csv")
    t2 = stage2_old_new(old, new)
    t2.to_csv(out / "old_new_stage2.csv", index=False)
    fig1_recovery(old, new, out)
    fig2_arms(new, out)
    fig3_contrasts(t2, out)
    print(t2.groupby("contrast")["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
