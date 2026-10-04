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


def learned_rows_matching(run_dir, oracle_root, pattern: str) -> pd.DataFrame:
    """``learned_recovery`` rows of the condition folders of ``run_dir`` whose name matches the regular expression
    ``pattern`` (the old reward-model runs mix several settings in one run folder)."""
    import re
    import shutil
    import tempfile

    from training.analyze_recoverability import learned_recovery, load_learned, load_oracle

    run_dir = Path(run_dir)
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / run_dir.name
        sub.mkdir()
        for cond in run_dir.glob("dataset=*"):
            if re.search(pattern, cond.name):
                (sub / cond.name).symlink_to(cond.resolve())
        rows = load_learned(sub)
    if rows.empty:
        raise FileNotFoundError(f"no condition of {run_dir} matches {pattern!r}")
    return learned_recovery(rows, load_oracle(oracle_root))


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
    for ax, col, lab in ((axes[3], "ess_raw", "raw-weight ESS of the selected policy"),
                         (axes[4], "w_share_gt10", "share of weights > 10 (%)")):
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
            o = r[r["method"] == "opc"]
            out.append(dict(share=share, bias=b, worlds=k, opc_fraction=m, opc_fraction_lo=lo, opc_fraction_hi=hi,
                            opc_minus_dm=d, opc_minus_dm_lo=dlo, opc_minus_dm_hi=dhi, opc_ess=o["ess_raw"].median(),
                            opc_w_gt10_pct=100 * o["w_share_gt10"].mean(), logger_value_pct=100 * o["V_logger"].mean(),
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


def stage2_old_new(old: pd.DataFrame, new: pd.DataFrame, by=("bias", "train_size")) -> pd.DataFrame:
    return old_new_table(old, new, STAGE2_CONTRASTS, by=by)


def finding(label: str, old: pd.Series, new: pd.Series) -> dict:
    """One paired finding from per-world values (old and new indexed by world): both means with 95% intervals, the
    paired change on the common worlds, and the classification."""
    from training.revalidation_compare import classify

    common = old.index.intersection(new.index)
    o, n = old.loc[common], new.loc[common]
    om, olo, ohi, k = mean_ci(o)
    nm, nlo, nhi, _ = mean_ci(n)
    cm, clo, chi, _ = mean_ci(n - o)
    return dict(finding=label, worlds=k, old=om, old_lo=olo, old_hi=ohi, new=nm, new_lo=nlo, new_hi=nhi,
                change=cm, change_lo=clo, change_hi=chi, verdict=classify((om, olo, ohi), (nm, nlo, nhi), (cm, clo, chi)))


def _sel(rows, size=None, biased=None, bias=None):
    r = rows
    if size is not None:
        r = r[r["train_size"] == size]
    if biased is not None:
        r = r[(r["bias"] != "none") == biased]
    if bias is not None:
        r = r[r["bias"] == bias]
    return r


def build_findings(old2, new2, rm_old, rm_new, sup_old_rows, sup_new_rows, su_old=None, su_new=None) -> pd.DataFrame:
    """The key findings of the old report, each as a paired old-vs-new comparison (see ``finding``)."""
    F = []
    for n in SIZES:
        k = f"{n // 1000}k"
        F.append(finding(f"OPC fraction of the oracle repair, biased, {k}",
                         _per_world(_sel(old2, n, True), "opc", None, "fraction_of_oracle_repair_greedy", 1.0),
                         _per_world(_sel(new2, n, True), "opc", None, "fraction_of_oracle_repair_greedy", 1.0)))
    for arm, name in (("dm", "DM-only"), ("no_propensity", "no-propensity"), ("tempered_logger", "tempered logger")):
        for n in SIZES:
            F.append(finding(f"OPC − {name}, biased, {n // 1000}k",
                             _per_world(_sel(old2, n, True), "opc", arm), _per_world(_sel(new2, n, True), "opc", arm)))
    for n in (5000, 100000):
        F.append(finding(f"no-bias cost: OPC − tempered logger, {n // 1000}k",
                         _per_world(_sel(old2, n, False), "opc", "tempered_logger"),
                         _per_world(_sel(new2, n, False), "opc", "tempered_logger")))
    for setting, (ref_alt_old, ref_alt_new) in {"budget-fair vs external q̂": (rm_old["budget"], rm_new["budget"]),
                                                "concat vs interaction q̂": (rm_old["concat"], rm_new["concat"])}.items():
        for arm, name in (("dm", "DM-only"), ("opc", "OPC")):
            for n in (5000, 25000):
                ch = lambda ra: _per_world(_sel(ra[1], n), arm, None) - _per_world(_sel(ra[0], n), arm, None)
                F.append(finding(f"{setting}: change in {name}, {n // 1000}k", ch(ref_alt_old).dropna(), ch(ref_alt_new).dropna()))
    for b in SINGLES:
        def gap(rows_by_share):
            f6 = _per_world(_sel(rows_by_share[0.6], bias=b), "opc", None, "fraction_of_oracle_repair_greedy", 1.0)
            f8 = _per_world(_sel(rows_by_share[0.8], bias=b), "opc", None, "fraction_of_oracle_repair_greedy", 1.0)
            return (f6.droplevel("train_size") - f8.droplevel("train_size")).dropna()
        F.append(finding(f"logger share 0.6 − 0.8: OPC fraction, {LABEL[b]}", gap(sup_old_rows), gap(sup_new_rows)))
    if su_old is not None and su_new is not None:
        F.append(finding("harmonic:0.1 − shrink:100, single-type high, 25k (V)", su_old, su_new))
    return pd.DataFrame(F)


def _md_table(df: pd.DataFrame, cols, fmt=None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "" for c in cols) + " |\n"
                   for _, r in df.iterrows())
    return head + body


def main(argv=None) -> None:
    from training.analyze_recoverability import learned_recovery, load_learned, load_oracle

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default="artifacts/full_study/summaries_20260927")
    ap.add_argument("--new", default="artifacts/full_study/opc_revalidation_20261004/summaries")
    ap.add_argument("--tuning", default="artifacts/full_study/opc_revalidation_20261004/tuning")
    ap.add_argument("--runs", default="artifacts/full_study", help="run folders (old reward-model tests, Su slices)")
    ap.add_argument("--out", default="artifacts/full_study/opc_revalidation_20261004/report")
    a = ap.parse_args(argv)
    out, old_dir, new_dir, runs = Path(a.out), Path(a.old), Path(a.new), Path(a.runs)
    out.mkdir(parents=True, exist_ok=True)
    oracle_root = runs / "run_oracle_repair_20260927"
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
    # reward-model tests: old from the old run folders (older pipeline), new from the reruns + the Stage 2 rerun
    mh = lambda d: d[d["bias"].isin(["medium", "high"])]
    rm_old = {"budget": (learned_rows_matching(runs / "run_logger_explore", oracle_root, r"__lgs=0\.8$"),
                         learned_rows_matching(runs / "run_logger_explore", oracle_root, r"__lgs=0\.8__qhat=train__cf=5$")),
              "concat": (learned_rows_matching(runs / "run_logger_explore_budget", oracle_root, r"__lgs=0\.8__qhat=train__cf=5__val=20000$"),
                         learned_rows_matching(runs / "run_qhat_concat", oracle_root, r"__lgs=0\.8__qhat=train__cf=5__val=20000$"))}
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
    su_old_dir = runs / "run_stage2_su_shrink_100"
    rob = new2[new2["method"] == "opc_shrink100"]
    if su_old_dir.exists() and not rob.empty:
        oracle = load_oracle(oracle_root)
        su_rows = learned_recovery(load_learned(su_old_dir), oracle)
        keys = ["dataset", "bias", "seed", "train_size"]
        o_h = old2[old2["method"] == "opc"].set_index(keys)["V_method"]
        o_s = su_rows[su_rows["method"] == "opc"].set_index(keys)["V_method"]
        su_old = (100 * (o_h - o_s)).dropna()
        n_h = new2[new2["method"] == "opc"].set_index(keys)["V_method"]
        n_s = rob.set_index(keys)["V_method"]
        su_new = (100 * (n_h - n_s)).dropna()
        su_new = su_new[su_new.index.isin(su_old.index)]
    findings = build_findings(old2, new2_main, rm_old, rm_new, sup_old, sup_new, su_old, su_new)
    findings.to_csv(out / "findings_old_vs_new.csv", index=False)
    fig_verdicts(findings, out)
    cols = ["finding", "worlds", "old", "new", "change", "change_lo", "change_hi", "verdict"]
    fmt = {c: "{:+.2f}" for c in ("old", "new", "change", "change_lo", "change_hi")}
    fmt["worlds"] = "{:.0f}"
    (out / "findings_old_vs_new.md").write_text(_md_table(findings, cols, fmt))
    print(findings[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
