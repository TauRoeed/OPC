"""Figures and tables of the revalidation report (docs/simulator_fix_opc_revalidation_20261004.md, Phase 4).

Inputs (summaries written by ``training.analyze_recoverability``):
  --old  the buggy-simulator summaries (artifacts/full_study/summaries_20260927): stage2_learned_rows.csv,
         stage3_lgs_*/stage2_learned_rows.csv
  --new  the fixed-simulator summaries (artifacts/full_study/opc_revalidation_20261004/summaries,
         training.revalidation_phase3): the same files, the reward-model reruns' learned rows (reward_model/), the old
         configuration on the corrected logs (decomposition/), and the old reward-model and Su-slice rows rebuilt
         from the old run folders (old/), so that the report needs no run folder
  --tuning  the Phase 2 weighting tables (tuning/weights_*.csv)

Outputs (--out): fig1..fig8 as PNG and PDF, each with the plotted values in a CSV of the same name, the old-vs-new
delta tables (``old_new_*.csv``) with each finding's classification (training.revalidation_compare), and the
simulator-vs-retuning decomposition (``decomposition_simulator_vs_retuning.csv``) when the old configuration's rerun on
the corrected logs is present (--new/decomposition).
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
VERDICT_COLORS = {"unchanged": "#4D4D4D", "unchanged size, less precise": "#8C8C8C",
                  "same direction, different magnitude": "#0072B2", "weakened": "#E69F00",
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


def fig4_weighting(screen: pd.DataFrame, sel_diag: pd.DataFrame, out: Path) -> None:
    """Fig 4: (a–c) the corrected weighting study: the selected policy's true gain against each family's parameter,
    one panel per train size, raw weights as a line; (d) the selected OPC policies' raw-weight ESS and (e) share of
    weights above 10, by arm and train size (corrected Stage 2)."""
    from training.analyze_revalidation import weight_family

    plt = _plt()
    fams = {"clip": ("#0072B2", "o", "clip:M (M)"), "shrink": ("#E69F00", "s", "Su shrink:λ (λ)"),
            "harmonic": ("#009E73", "^", "Metelli harmonic:λ (1/λ)")}
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
    sc = screen.copy()
    fam_param = [weight_family(r) for r in sc["run"]]
    sc["family"] = [f for f, _ in fam_param]
    sc["param"] = np.array([p for _, p in fam_param], dtype=float)
    # one common x-axis: the weight cap (clip M, √λ for shrink's peak position, 1/λ for harmonic's cap)
    sc["cap"] = np.where(sc["family"] == "shrink", np.sqrt(sc["param"]),
                         np.where(sc["family"] == "harmonic", 1.0 / sc["param"], sc["param"]))
    for ax, n in zip(axes[:3], SIZES):
        g = sc[sc["train_size"] == n]
        ax.axhline(0, color="#555555", linestyle="--", linewidth=1, label="raw weights (reference)")
        for fam, (c, mk, lab) in fams.items():
            f = g[g["family"] == fam].sort_values("cap")
            ax.errorbar(f["cap"], f["selected_vs_ref"], yerr=[f["selected_vs_ref"] - f["selected_vs_ref_lo"],
                                                           f["selected_vs_ref_hi"] - f["selected_vs_ref"]],
                        color=c, marker=mk, markersize=4, linewidth=1.3, capsize=2, elinewidth=0.7, label=lab)
        ax.set_xscale("log")
        ax.set_title(f"{n // 1000}k: selected policy vs raw weights")
        ax.set_xlabel("weight cap (clip M; √λ; 1/λ)")
    axes[0].set_ylabel("true CTR difference to raw (points;\npaired over conditions)")
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")
    d = sel_diag[sel_diag["method"].isin(["opc", "dm"])] if "method" in sel_diag else sel_diag
    for ax, col, lab in ((axes[3], "ess_raw", "raw-weight ESS of the selected policy\n(median over worlds)"),
                         (axes[4], "w_share_gt10", "share of weights > 10 (%)\n(median over worlds)")):
        for m in ("opc", "dm"):
            g = d[d["method"] == m].groupby("train_size")[col].median()
            if g.empty:
                continue
            ax.plot(np.log10(g.index), g.values * (100 if col == "w_share_gt10" else 1), marker="o",
                    color=COLORS[m], label=METHOD_NAMES[m])
        ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        ax.set_title(lab)
    axes[3].set_yscale("log")
    axes[3].legend(frameon=False, fontsize=7)
    fig.suptitle("Corrected weighting study (OPC, tuning seed 201) and weight diagnostics of the corrected Stage 2",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig4_weights_ess", sc)


def _per_world(rows: pd.DataFrame, a: str, b: str | None, value: str = "V_method", scale: float = 100.0) -> pd.Series:
    pa = rows[rows["method"] == a].set_index(KEYS)[value]
    if b is None:
        return scale * pa
    return scale * (pa - rows[rows["method"] == b].set_index(KEYS)[value]).dropna()


def reward_model_table(settings: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """For each reward-model test {label: (rows under the reference q̂, rows under the altered q̂)}: per bias × size,
    OPC − DM-only under each, and the change of each arm (altered − reference), mean and 95% interval over worlds."""
    out = []
    for label, (ref, alt) in settings.items():
        for (b, n), _ in ref.groupby(["bias", "train_size"]):
            r_ref, r_alt = ref[(ref["bias"] == b) & (ref["train_size"] == n)], alt[(alt["bias"] == b) & (alt["train_size"] == n)]
            if r_alt.empty:
                continue
            row = dict(setting=label, bias=b, train_size=n)
            for tag, rows in (("reference", r_ref), ("altered", r_alt)):
                m, lo, hi, k = mean_ci(_per_world(rows, "opc", "dm"))
                row.update({f"OPC-DM {tag}": m, f"OPC-DM {tag} lo": lo, f"OPC-DM {tag} hi": hi})
            for arm in ("opc", "dm"):
                d = (_per_world(r_alt, arm, None) - _per_world(r_ref, arm, None)).dropna()
                m, lo, hi, k = mean_ci(d)
                row.update({f"change {arm}": m, f"change {arm} lo": lo, f"change {arm} hi": hi, "worlds": k})
                # the reward model's error on the arm's selected policy, and that policy's raw-weight support
                for tag, rows in (("reference", r_ref), ("altered", r_alt)):
                    a = rows[rows["method"] == arm]
                    if "qhat_error" in a:
                        m, lo, hi, _ = mean_ci(100 * a["qhat_error"])
                        row.update({f"qhat error {arm} {tag}": m, f"qhat error {arm} {tag} lo": lo,
                                    f"qhat error {arm} {tag} hi": hi})
                    row[f"ess {arm} {tag}"] = a["ess_raw"].mean() if len(a) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def fig5_reward_model(old_t: pd.DataFrame, new_t: pd.DataFrame, out: Path) -> None:
    """Fig 5: the reward-model tests, old (older pipeline, buggy logs) vs corrected: OPC − DM-only under the
    reference and the altered q̂, and each arm's change, for the misspecified (concat) and the external-data q̂."""
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.0))
    for r, setting in enumerate(("misspecified (concat) q̂", "external 50k-row q̂")):
        for c, (col, title) in enumerate((("OPC-DM altered", "OPC − DM-only under the altered q̂"),
                                          ("change dm", "DM-only: altered − reference q̂"),
                                          ("change opc", "OPC: altered − reference q̂"))):
            ax = axes[r, c]
            for k, (tag, t, mk) in enumerate((("old", old_t, "s"), ("new", new_t, "o"))):
                g = t[t["setting"] == setting]
                for j, b in enumerate(("medium", "high")):
                    gb = g[g["bias"] == b].sort_values("train_size")
                    if gb.empty:
                        continue
                    xs = np.log10(gb["train_size"]) + (k - 0.5) * 0.04 + (j - 0.5) * 0.015
                    ax.errorbar(xs, gb[col], yerr=[gb[col] - gb[f"{col} lo"], gb[f"{col} hi"] - gb[col]], marker=mk,
                                color=("#999999" if tag == "old" else ("#0072B2" if b == "medium" else "#D55E00")),
                                linestyle="--" if tag == "old" else "-", markersize=3.5, capsize=2, elinewidth=0.8,
                                label=f"{tag}, combined {b}")
            ax.axhline(0, color="#444444", linewidth=0.8)
            ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
            ax.set_title(f"{setting}: {title}", fontsize=8.5)
        axes[r, 0].set_ylabel("true CTR points")
    axes[0, 2].legend(frameon=False, fontsize=7)
    fig.suptitle("Reward-model tests: old (grey, older pipeline on the buggy logs) vs corrected (colour); mean and 95% CI "
                 "over 3 datasets × 2 seeds", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out, "fig5_reward_model_tests", pd.concat([old_t.assign(sim="old"), new_t.assign(sim="new")], ignore_index=True))


SINGLES = ("w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high")


def support_table(rows_by_share: dict[float, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for share, rows in rows_by_share.items():
        for b in SINGLES:
            r = rows[rows["bias"] == b]
            if r.empty:
                continue
            frac = r[r["method"] == "opc"]["fraction_of_oracle_repair_greedy"]
            m, lo, hi, _ = mean_ci(frac)
            d, dlo, dhi, k = mean_ci(_per_world(r, "opc", "dm"))
            o, dm = r[r["method"] == "opc"], r[r["method"] == "dm"]
            out.append(dict(share=share, bias=b, worlds=k, opc_fraction=m, opc_fraction_lo=lo, opc_fraction_hi=hi,
                            dm_fraction=dm["fraction_of_oracle_repair_greedy"].mean(),
                            opc_minus_dm=d, opc_minus_dm_lo=dlo, opc_minus_dm_hi=dhi, opc_ess=o["ess_raw"].mean(),
                            opc_w_gt10_pct=100 * o["w_share_gt10"].mean(), opc_regret_pct=100 * o["regret"].mean(),
                            opc_sel_error_pct=100 * o["sel_error_point"].mean(), dm_regret_pct=100 * dm["regret"].mean(),
                            logger_value_pct=100 * o["V_logger"].mean(),
                            oracle_gain_greedy_pct=100 * (o["oracle_repair_greedy"] - o["V_logger_greedy"]).mean()))
    return pd.DataFrame(out)


def fig6_support(old_t: pd.DataFrame, new_t: pd.DataFrame, out: Path) -> None:
    """Fig 6: the logging-support sweep at 25k, old vs corrected: the oracle bound (unchanged), OPC's fraction of it,
    OPC − DM-only and OPC's raw-weight ESS against the logger's greedy share."""
    from training.representation_report import BIAS_COLORS, BIAS_MARKERS

    plt = _plt()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
    for tag, t, ls in (("old", old_t, "--"), ("new", new_t, "-")):
        for i, b in enumerate(SINGLES):
            g = t[t["bias"] == b].sort_values("share")
            xs = g["share"] + (i - 1) * 0.006 + (0.002 if tag == "new" else -0.002)
            kw = dict(color=BIAS_COLORS[b], marker=BIAS_MARKERS[b], markersize=4, linestyle=ls,
                      alpha=1.0 if tag == "new" else 0.5)
            if tag == "new":
                axes[0].plot(xs, g["oracle_gain_greedy_pct"], label=LABEL[b], **kw)
            for ax, col in ((axes[1], "opc_fraction"), (axes[2], "opc_minus_dm")):
                ax.errorbar(xs, g[col], yerr=[g[col] - g[f"{col}_lo"], g[f"{col}_hi"] - g[col]], capsize=2,
                            elinewidth=0.8, label=f"{LABEL[b]} ({tag})" if ax is axes[1] else None, **kw)
            axes[3].plot(xs, g["opc_ess"], **kw)
    for ax, title in zip(axes, ("(a) oracle ranking gain (points)", "(b) OPC fraction of the oracle repair",
                                "(c) OPC − DM-only (points)", "(d) OPC raw-weight ESS (of 20,000)")):
        ax.set_title(title)
        ax.set_xticks(SHARES, ["0.6", "0.8", "0.95"])
        ax.set_xlabel("logger greedy share")
    axes[0].set_ylim(bottom=0)
    axes[2].axhline(0, color="#444444", linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=6.5, ncol=1)
    fig.suptitle("Logging-support sweep at 25k: old (dashed, buggy logs) vs corrected (solid); ml, kuairand × 2 seeds",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig6_support_old_vs_new", pd.concat([old_t.assign(sim="old"), new_t.assign(sim="new")], ignore_index=True))


def fig_verdicts(findings: pd.DataFrame, out: Path) -> None:
    """Fig 7: every key finding, old (grey) vs new effect (coloured by its classification), one panel per group of
    findings with the group's own unit."""
    plt = _plt()
    f = findings[findings["verdict"] != "no data"].reset_index(drop=True)
    if "group" not in f:
        f = f.assign(group="findings")
    groups = list(dict.fromkeys(f["group"]))
    sizes = [int((f["group"] == g).sum()) for g in groups]
    # two columns of panels: groups in order, the left column until it holds half the rows
    columns, acc = ([], []), 0
    for g, s in zip(groups, sizes):
        columns[0 if acc < sum(sizes) / 2 else 1].append((g, s))
        acc += s
    rows = max(sum(s for _, s in c) for c in columns)
    panels = max(len(c) for c in columns)
    fig = plt.figure(figsize=(17, 0.25 * rows + 0.85 * panels + 0.8), layout="constrained")
    outer = fig.add_gridspec(1, 2)
    first = None
    for k, col in enumerate(columns):
        if not col:
            continue
        sub = outer[k].subgridspec(len(col), 1, height_ratios=[s + 1.8 for _, s in col])
        for j, (g, _) in enumerate(col):
            ax = fig.add_subplot(sub[j])
            first = first or ax
            fg = f[f["group"] == g].reset_index(drop=True)
            for i, r in fg.iterrows():
                y = len(fg) - 1 - i
                ax.errorbar(r["old"], y + 0.17, xerr=[[r["old"] - r["old_lo"]], [r["old_hi"] - r["old"]]], fmt="s",
                            color="#999999", markersize=3.3, capsize=2, elinewidth=0.8)
                ax.errorbar(r["new"], y - 0.17, xerr=[[r["new"] - r["new_lo"]], [r["new_hi"] - r["new"]]], fmt="o",
                            color=VERDICT_COLORS[r["verdict"]], markersize=3.8, capsize=2, elinewidth=1.0)
            ax.set_yticks(range(len(fg)), [f"{r['finding']}  [{r['verdict']}]" for _, r in fg.iloc[::-1].iterrows()],
                          fontsize=7)
            ax.set_ylim(-0.7, len(fg) - 0.3)
            lo, hi = ax.get_xlim()
            if lo <= 0 <= hi:  # a zero reference only where the data reach it (not for ESS or tail shares)
                ax.axvline(0, color="#444444", linewidth=0.8)
            ax.set_title(g, loc="left", fontsize=8.5)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="x", labelsize=7)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color="#999999", label="old (buggy logs)")]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=c, label=f"new: {v}") for v, c in VERDICT_COLORS.items()
                if v in set(f["verdict"])]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=len(handles), loc="outside lower center")
    fig.suptitle("Old vs corrected: the key findings (mean and 95% CI over worlds; the new value coloured by its "
                 "classification)", fontsize=9.5)
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


def stage2_old_new(old: pd.DataFrame, new: pd.DataFrame, by=("bias", "train_size")) -> pd.DataFrame:
    return old_new_table(old, new, STAGE2_CONTRASTS, by=by)


def finding(label: str, old: pd.Series, new: pd.Series, material_abs: float | None = None) -> dict:
    """One paired finding from per-world values (old and new indexed by world): both means with 95% intervals, the
    paired change on the common worlds, and the classification (``material_abs``: see ``classify``)."""
    from training.revalidation_compare import classify

    common = old.index.intersection(new.index)
    o, n = old.loc[common], new.loc[common]
    om, olo, ohi, k = mean_ci(o)
    nm, nlo, nhi, _ = mean_ci(n)
    cm, clo, chi, _ = mean_ci(n - o)
    return dict(finding=label, worlds=k, old=om, old_lo=olo, old_hi=ohi, new=nm, new_lo=nlo, new_hi=nhi,
                change=cm, change_lo=clo, change_hi=chi, change_rel=cm / abs(om) if om else np.nan,
                verdict=classify((om, olo, ohi), (nm, nlo, nhi), (cm, clo, chi), material_abs))


def _sel(rows, size=None, biased=None, bias=None):
    r = rows
    if size is not None:
        r = r[r["train_size"] == size]
    if biased is not None:
        r = r[(r["bias"] != "none") == biased]
    if bias is not None:
        r = r[r["bias"] == bias]
    return r


FRACTION = "fraction_of_oracle_repair_greedy"
G_RECOVERY = "Stage 2 recovery (fraction of the oracle ranking repair, greedy; biased worlds unless named)"
G_BASELINES = "Stage 2: OPC minus each baseline (true CTR points)"
G_NOBIAS = "No-bias control (true CTR points)"
G_REWARD = "Reward-model tests (true CTR points; combined medium and high)"
G_SUPPORT = "Logging support at 25k (difference in OPC's fraction of the oracle repair)"
G_WEIGHTS = "Importance weighting (true CTR points)"
G_SELECTION = "Selection (true CTR points; biased worlds)"
G_ESS = "Selected OPC policy's raw-weight ESS (log10; biased worlds)"
G_TAIL = "Selected OPC policy's share of weights above 10 (%; biased worlds)"


def build_findings(old2, new2, rm_old, rm_new, sup_old_rows, sup_new_rows, su_old=None, su_new=None) -> pd.DataFrame:
    """The key findings of the old report (docs/representation_repair_experimental_report_20260928.md, sections D–J),
    each as a paired old-vs-new comparison (see ``finding``), grouped by topic; each group has one unit."""
    F = []

    def add(group, label, old, new, material_abs=None):
        F.append(dict(group=group, **finding(label, old.dropna(), new.dropna(), material_abs)))

    def both(rows_old, rows_new, a, b=None, value="V_method", scale=100.0, **sel):
        return (_per_world(_sel(rows_old, **sel), a, b, value, scale), _per_world(_sel(rows_new, **sel), a, b, value, scale))

    def singles(rows, sizes=None):
        r = rows[rows["bias"].isin(SINGLES)]
        return r if sizes is None else r[r["train_size"].isin(sizes)]

    k = lambda n: f"{n // 1000}k"
    # Stage 2 recovery
    for arm, name in (("opc", "OPC"), ("dm", "DM-only")):
        for n in SIZES:
            add(G_RECOVERY, f"{name}, {k(n)}", *both(old2, new2, arm, None, FRACTION, 1.0, size=n, biased=True))
    for n in (5000, 100000):
        add(G_RECOVERY, f"no-propensity, {k(n)}", *both(old2, new2, "no_propensity", None, FRACTION, 1.0, size=n, biased=True))
    for n in (5000, 25000):
        add(G_RECOVERY, f"OPC, Anime vector high, {k(n)}",
            *both(old2[old2["dataset"] == "anime"], new2[new2["dataset"] == "anime"], "opc", None, FRACTION, 1.0, size=n,
                  bias="w-none.g-none.v-high"))
    # OPC vs the baselines
    for n in SIZES:
        add(G_BASELINES, f"OPC − DM-only, biased, {k(n)}", *both(old2, new2, "opc", "dm", size=n, biased=True))
    for n in (25000, 100000):
        add(G_BASELINES, f"OPC − DM-only, single-type high, {k(n)}",
            _per_world(singles(old2, [n]), "opc", "dm"), _per_world(singles(new2, [n]), "opc", "dm"))
    add(G_BASELINES, "OPC − DM-only, combined high, 5k", *both(old2, new2, "opc", "dm", size=5000, bias="high"))
    for d in ("ml", "kuairand", "anime"):
        add(G_BASELINES, f"OPC − DM-only, single-type high, 25k + 100k, {d}",
            _per_world(singles(old2[old2["dataset"] == d], [25000, 100000]), "opc", "dm"),
            _per_world(singles(new2[new2["dataset"] == d], [25000, 100000]), "opc", "dm"))
    for arm, name in (("no_propensity", "no-propensity"), ("tempered_logger", "tempered logger")):
        for n in SIZES:
            add(G_BASELINES, f"OPC − {name}, biased, {k(n)}", *both(old2, new2, "opc", arm, size=n, biased=True))
    # the no-bias control
    for n in (5000, 100000):
        add(G_NOBIAS, f"OPC − tempered logger, no bias, {k(n)}", *both(old2, new2, "opc", "tempered_logger", size=n, biased=False))
    add(G_NOBIAS, "OPC − no-propensity, no bias, 5k", *both(old2, new2, "opc", "no_propensity", size=5000, biased=False))
    add(G_NOBIAS, "OPC − DM-only, no bias, 100k", *both(old2, new2, "opc", "dm", size=100000, biased=False))
    # reward-model tests: rm_*[test] = (reference-setting rows, altered-setting rows)
    ext_old, ext_new = rm_old["budget"][0], rm_new["budget"][0]
    add(G_REWARD, "external 50k-row q̂: OPC − DM-only, 5k", *both(ext_old, ext_new, "opc", "dm", size=5000))
    for setting, (ra_old, ra_new), sizes in (("budget-fair − external q̂", (rm_old["budget"], rm_new["budget"]), (5000, 25000)),
                                             ("concat − interaction q̂", (rm_old["concat"], rm_new["concat"]), SIZES)):
        for arm, name in (("dm", "DM-only"), ("opc", "OPC")):
            for n in sizes:
                ch = lambda ra: _per_world(_sel(ra[1], n), arm, None) - _per_world(_sel(ra[0], n), arm, None)
                add(G_REWARD, f"{setting}: change in {name}, {k(n)}", ch(ra_old), ch(ra_new))
    # logging support (single-type highs at 25k)
    def share_gap(rows_by_share, lo, hi, biases):
        f_lo = _per_world(rows_by_share[lo][rows_by_share[lo]["bias"].isin(biases)], "opc", None, FRACTION, 1.0)
        f_hi = _per_world(rows_by_share[hi][rows_by_share[hi]["bias"].isin(biases)], "opc", None, FRACTION, 1.0)
        return f_lo.droplevel("train_size") - f_hi.droplevel("train_size")
    for b in SINGLES:
        add(G_SUPPORT, f"share 0.6 − 0.8, {LABEL[b]}", share_gap(sup_old_rows, 0.6, 0.8, [b]), share_gap(sup_new_rows, 0.6, 0.8, [b]))
    add(G_SUPPORT, "share 0.95 − 0.8, single-type highs", share_gap(sup_old_rows, 0.95, 0.8, SINGLES),
        share_gap(sup_new_rows, 0.95, 0.8, SINGLES))
    # weighting: the Su robustness slice
    if su_old is not None and su_new is not None:
        add(G_WEIGHTS, "harmonic:0.1 − shrink:100 training weights, single-type high, 25k", su_old, su_new)
    # selection and the weights of the selected policies
    for arm, name, what in (("opc", "OPC", "DR point estimate − truth"), ("dm", "DM-only", "q̂ estimate − truth")):
        for n in SIZES:
            add(G_SELECTION, f"{name}: {what}, {k(n)}", *both(old2, new2, arm, None, "sel_error_point", size=n, biased=True))
        for n in SIZES:
            add(G_SELECTION, f"{name}: true selection regret, {k(n)}", *both(old2, new2, arm, None, "regret", size=n, biased=True))
    lo2 = lambda rows: rows.assign(log10_ess=np.log10(rows["ess_raw"]))
    for n in SIZES:
        add(G_ESS, f"OPC, {k(n)}", *both(lo2(old2), lo2(new2), "opc", None, "log10_ess", 1.0, size=n, biased=True),
            material_abs=np.log10(1.2))  # material: a 20% change of the ESS itself
    for n in SIZES:
        add(G_TAIL, f"OPC, {k(n)}", *both(old2, new2, "opc", None, "w_share_gt10", 100.0, size=n, biased=True))
    return pd.DataFrame(F)


DECOMPOSITION = {"OPC": ("opc", None), "DM-only": ("dm", None), "OPC - DM-only": ("opc", "dm")}
EFFECTS = {"simulator": ("mid", "old"), "configuration": ("new", "mid"), "total": ("new", "old")}


def decomposition_table(old: pd.DataFrame, mid: pd.DataFrame, new: pd.DataFrame, by=("train_size",)) -> pd.DataFrame:
    """Separates the simulator fix from the retuning, world by world: ``old`` = buggy logs with the old configuration,
    ``mid`` = corrected logs with the old configuration, ``new`` = corrected logs with the new configuration. Per cell of
    ``by``: each quantity's mean in the three runs and the paired simulator (mid − old), configuration (new − mid) and
    total (new − old) effects, with 95% intervals over the worlds present in all three (true CTR points)."""
    out = []
    for label, (a, b) in DECOMPOSITION.items():
        v = {k: _per_world(r, a, b) for k, r in (("old", old), ("mid", mid), ("new", new))}
        common = v["old"].index.intersection(v["mid"].index).intersection(v["new"].index)
        df = pd.DataFrame({k: s.loc[common] for k, s in v.items()}).reset_index()
        for cell, g in df.groupby(list(by)):
            rec = dict(quantity=label, **dict(zip(by, cell if isinstance(cell, tuple) else (cell,))), worlds=len(g))
            for k in ("old", "mid", "new"):
                m, lo, hi, _ = mean_ci(g[k])
                rec.update({k: m, f"{k}_lo": lo, f"{k}_hi": hi})
            for k, (x, y) in EFFECTS.items():
                m, lo, hi, _ = mean_ci(g[x] - g[y])
                rec.update({k: m, f"{k}_lo": lo, f"{k}_hi": hi})
            out.append(rec)
    return pd.DataFrame(out)


def fig8_decomposition(table: pd.DataFrame, out: Path) -> None:
    """Fig 8: the old-to-new change of OPC, DM-only and OPC − DM-only on the biased worlds, split into the simulator
    effect (old configuration, buggy → corrected logs) and the configuration effect (corrected logs, old → new
    configuration), per train size."""
    plt = _plt()
    colors = {"simulator": "#0072B2", "configuration": "#E69F00", "total": "#333333"}
    fig, axes = plt.subplots(1, len(DECOMPOSITION), figsize=(12, 3.2), sharey=True)
    for ax, q in zip(axes, DECOMPOSITION):
        t = table[table["quantity"] == q].sort_values("train_size")
        for i, (k, c) in enumerate(colors.items()):
            xs = np.arange(len(t)) + (i - 1) * 0.22
            ax.errorbar(xs, t[k], yerr=[t[k] - t[f"{k}_lo"], t[f"{k}_hi"] - t[k]], fmt="o", color=c, markersize=4.5,
                        capsize=2.5, elinewidth=1.0, label={"simulator": "simulator fix (old configuration)",
                                                            "configuration": "retuning (corrected logs)",
                                                            "total": "total (old report → corrected)"}[k])
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(range(len(t)), [f"{n // 1000}k" for n in t["train_size"]])
        ax.set_title(q.replace(" - ", " − "))
        ax.set_xlabel("logged training rows")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("change, true CTR points\n(paired over worlds)")
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    worlds = int(table["worlds"].min()) if len(table) else 0
    fig.suptitle(f"What changed OPC and DM-only: the simulator fix vs the retuning (biased worlds of ml and kuairand; "
                 f"{worlds} worlds per size)", fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig8_simulator_vs_retuning", table)


def m5_config_table(m5: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Phase 5: the OPC side of the CausE M5 comparison (fixed-simulator logs; the pre-revalidation configuration: old
    search space, fixed logit scale) against the corrected Stage 2 at 25k on the same worlds (the same logs and
    splits: a one-size run reproduces that size's trials of a multi-size run). Per arm and bias: both means and the
    paired difference (revalidated − M5) in true CTR points. The tempered logger has no training configuration, so its
    difference must be exactly 0 if the logs are identical."""
    out = []
    n25 = new[new["train_size"] == 25000]
    for arm in ("opc", "dm", "tempered_logger"):
        a = m5[m5["method"] == arm].set_index(KEYS)["V_method"]
        b = n25[n25["method"] == arm].set_index(KEYS)["V_method"]
        common = a.index.intersection(b.index)
        both = pd.DataFrame({"m5": 100 * a.loc[common], "new": 100 * b.loc[common]}).reset_index()
        for bias, g in list(both.groupby("bias")) + [("all", both), ("biased", both[both["bias"] != "none"])]:
            m, lo, hi, k = mean_ci(g["new"] - g["m5"])
            out.append(dict(method=arm, bias=bias, worlds=k, m5=g["m5"].mean(), revalidated=g["new"].mean(), diff=m,
                            diff_lo=lo, diff_hi=hi, max_abs_diff=float((g["new"] - g["m5"]).abs().max())))
    return pd.DataFrame(out)


def _md_table(df: pd.DataFrame, cols, fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "" for c in cols) + " |\n"
                   for _, r in df.iterrows())
    return head + body


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="artifacts/full_study/summaries_20260927")
    ap.add_argument("--new", default="artifacts/full_study/opc_revalidation_20261004/summaries")
    ap.add_argument("--tuning", default="artifacts/full_study/opc_revalidation_20261004/tuning")
    ap.add_argument("--out", default="artifacts/full_study/opc_revalidation_20261004/report")
    ap.add_argument("--phase0", default="artifacts/full_study/opc_revalidation_20261004/phase0")
    a = ap.parse_args(argv)
    out, old_dir, new_dir = Path(a.out), Path(a.old), Path(a.new)
    out.mkdir(parents=True, exist_ok=True)
    old2 = load_rows(old_dir / "stage2_learned_rows.csv")
    new2 = load_rows(new_dir / "stage2" / "stage2_learned_rows.csv")
    new2_main = new2[~new2["method"].str.startswith("opc_")]
    t2 = stage2_old_new(old2, new2_main)
    t2.to_csv(out / "old_new_stage2.csv", index=False)
    pooled = stage2_old_new(old2[old2["bias"] != "none"], new2_main[new2_main["bias"] != "none"], by=("train_size",))
    pooled.to_csv(out / "old_new_stage2_biased_pooled.csv", index=False)
    by_ds = stage2_old_new(old2, new2_main, by=("dataset", "train_size"))
    by_ds.to_csv(out / "old_new_stage2_by_dataset.csv", index=False)
    fig1_recovery(old2, new2_main, out)
    fig2_arms(new2, out)
    fig3_contrasts(t2, out)
    # weighting (Phase 2) and the corrected Stage 2's weight diagnostics
    screen = pd.read_csv(Path(a.tuning) / "weights_screen_s201.csv")
    fig4_weighting(screen, new2_main.rename(columns={}), out)
    # reward-model tests: old = the older pipeline's runs on the buggy logs (rebuilt into --new/old by
    # training.revalidation_phase3), new = the reruns and the Stage 2 rerun
    mh = lambda d: d[d["bias"].isin(["medium", "high"])]
    old_rm = lambda name: load_rows(new_dir / "old" / f"reward_model_{name}.csv")
    rm_old = {"budget": (old_rm("external"), old_rm("budget_fair")), "concat": (old_rm("interaction"), old_rm("concat"))}
    rm_new = {"budget": (load_rows(new_dir / "reward_model" / "learned_rows_external.csv"), mh(new2_main)),
              "concat": (mh(new2_main), load_rows(new_dir / "reward_model" / "learned_rows_concat.csv"))}
    labels = {"budget": "external 50k-row q̂", "concat": "misspecified (concat) q̂"}
    # the figure shows OPC − DM under the altered q̂ and each arm's change altered − reference; for the budget test the
    # altered setting is the external q̂ (the old report's 'external' rows), the reference the budget-fair one
    rmt_old = reward_model_table({labels["concat"]: rm_old["concat"], labels["budget"]: (rm_old["budget"][1], rm_old["budget"][0])})
    rmt_new = reward_model_table({labels["concat"]: rm_new["concat"], labels["budget"]: (rm_new["budget"][1], rm_new["budget"][0])})
    rmt_old.to_csv(out / "reward_model_tests_old.csv", index=False)
    rmt_new.to_csv(out / "reward_model_tests_new.csv", index=False)
    fig5_reward_model(rmt_old, rmt_new, out)
    # logging support
    sup_old = {sh: load_rows(old_dir / f"stage3_lgs_{str(sh).replace('.', '_')}" / "stage2_learned_rows.csv") for sh in SHARES}
    sup_new = {sh: load_rows(new_dir / f"stage3_lgs_{str(sh).replace('.', '_')}" / "stage2_learned_rows.csv") for sh in SHARES}
    st_old, st_new = support_table(sup_old), support_table(sup_new)
    st_old.to_csv(out / "support_old.csv", index=False)
    st_new.to_csv(out / "support_new.csv", index=False)
    fig6_support(st_old, st_new, out)
    # Su robustness slice: harmonic:0.1 − shrink:100 on the single-type highs at 25k (ml, kuairand), old vs new
    su_old = su_new = None
    su_path = new_dir / "old" / "su_shrink100.csv"
    rob = new2[new2["method"] == "opc_shrink100"]
    if su_path.exists() and not rob.empty:
        su_rows = load_rows(su_path)
        keys = ["dataset", "bias", "seed", "train_size"]
        o_h = old2[old2["method"] == "opc"].set_index(keys)["V_method"]
        o_s = su_rows[su_rows["method"] == "opc"].set_index(keys)["V_method"]
        su_old = (100 * (o_h - o_s)).dropna()
        n_h = new2[new2["method"] == "opc"].set_index(keys)["V_method"]
        n_s = rob.set_index(keys)["V_method"]
        su_new = (100 * (n_h - n_s)).dropna()
        su_new = su_new[su_new.index.isin(su_old.index)]
    # simulator effect vs retuning: the old configuration rerun on the corrected logs (ml, kuairand)
    mid_path = new_dir / "decomposition" / "learned_rows_oldspace.csv"
    if mid_path.exists():
        mid2 = load_rows(mid_path)
        dec = pd.concat([decomposition_table(_sel(old2, biased=True), _sel(mid2, biased=True), _sel(new2_main, biased=True))
                         .assign(worlds_kind="biased"),
                         decomposition_table(_sel(old2, biased=False), _sel(mid2, biased=False), _sel(new2_main, biased=False))
                         .assign(worlds_kind="no bias"),
                         decomposition_table(old2, mid2, new2_main, by=("bias", "train_size")).assign(worlds_kind="per bias")],
                        ignore_index=True)
        dec.to_csv(out / "decomposition_simulator_vs_retuning.csv", index=False)
        fig8_decomposition(dec[dec["worlds_kind"] == "biased"], out)
    # Phase 5: the M5 OPC side vs the revalidated configuration on the same logs
    m5_path = new_dir / "m5" / "learned_rows_m5_opc_side.csv"
    if m5_path.exists():
        m5_config_table(load_rows(m5_path), new2_main).to_csv(out / "phase5_m5_opc_side_vs_revalidated.csv", index=False)
    findings = build_findings(old2, new2_main, rm_old, rm_new, sup_old, sup_new, su_old, su_new)
    findings.to_csv(out / "findings_old_vs_new.csv", index=False)
    fig_verdicts(findings, out)
    cols = ["finding", "worlds", "old", "new", "change", "change_lo", "change_hi", "verdict"]
    fmt = {c: "{:+.2f}" for c in ("old", "new", "change", "change_lo", "change_hi")}
    fmt["worlds"] = "{:.0f}"
    (out / "findings_old_vs_new.md").write_text(_md_table(findings, cols, fmt))
    print(findings[cols].round(3).to_string(index=False))
    from training.revalidation_tables import main as tables

    tables(["--old", str(old_dir), "--new", str(new_dir), "--report", str(out), "--phase0", a.phase0])


if __name__ == "__main__":
    main()
