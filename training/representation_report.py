"""Figures and source tables for the representation-repair experimental report (development results, 2026-09-28).

Inputs:
  - the committed summaries (``artifacts/full_study/summaries_20260927``): the Stage 1 oracle rows, the Stage 2
    learned rows and paired differences, the shrink:100 slice, the Stage 3 per-share tables and the follow-up
    tables (oracle validation, per-dataset views, gap decomposition);
  - the local run folders (``artifacts/full_study/run_*``, listed in ``run_registry.csv``) of the earlier
    development runs only: the reward-model budget and misspecification tests and the objective / weighting study.

Their compact tables are written to the output folder. When the run folders are absent, the tables already there
are used, so every figure rebuilds from committed tables alone. Each figure is written as PNG and PDF, with the
plotted values in a CSV of the same name. ``tables.md`` holds the report's tables.

Usage: python -m training.representation_report --out artifacts/full_study/report_20260928
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_recoverability import _tags, mean_ci

SUMMARIES = Path("artifacts/full_study/summaries_20260927")
RUNS = Path("artifacts/full_study")
DATASETS = ("ml", "kuairand", "anime")
DATASET_NAMES = {"ml": "MovieLens", "kuairand": "KuaiRand", "anime": "Anime"}
BIASES = ("w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high", "medium", "high")
LABEL = {"none": "no bias", "w-high.g-none.v-none": "warp", "w-none.g-high.v-none": "group",
         "w-none.g-none.v-high": "vector", "medium": "combined medium", "high": "combined high"}
METHODS = ("opc", "dm", "no_propensity", "tempered_logger")
METHOD_NAMES = {"opc": "OPC", "dm": "DM-only", "no_propensity": "no-propensity", "tempered_logger": "tempered logger"}
# Okabe-Ito (colour-blind safe): one colour per entity across all figures
COLORS = {"opc": "#0072B2", "dm": "#E69F00", "no_propensity": "#CC79A7", "tempered_logger": "#009E73",
          "ml": "#56B4E9", "kuairand": "#D55E00", "anime": "#009E73"}
GAP_COLORS = {"learned": "#0072B2", "learning": "#9ECAE1", "structural": "#BDBDBD"}
# bias types (Paul Tol muted), distinct from the method colours
BIAS_COLORS = {"w-high.g-none.v-none": "#332288", "w-none.g-high.v-none": "#117733", "w-none.g-none.v-high": "#882255"}
BIAS_MARKERS = {"w-high.g-none.v-none": "o", "w-none.g-high.v-none": "s", "w-none.g-none.v-high": "^"}
SIZES = (5000, 25000, 100000)
SHARES = (0.6, 0.8, 0.95)


# ---------------------------------------------------------------------------------------------------- tables


def _ci_row(x) -> dict:
    mean, lo, hi, n = mean_ci(x)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n}


def stage1_table(s: Path = SUMMARIES) -> pd.DataFrame:
    """Per dataset and overall (``dataset`` = "all"), per bias: the logger's ranking loss, the oracle repair's ranking
    gain (CTR points), structural recoverability (greedy; Stage 1 bound) with its range over the worlds, and the
    validated recoverability (the follow-up's widened search, same selection rule)."""
    rows = pd.read_csv(s / "stage1_oracle_rows.csv")
    val = pd.read_csv(s / "followup" / "oracle_validation_by_world.csv")
    x = rows[rows["bias"] != "none"].merge(val[["dataset", "bias", "seed", "new recoverability greedy"]],
                                           on=["dataset", "bias", "seed"], validate="one_to_one")
    out = []
    groups = [((d, b), g) for (d, b), g in x.groupby(["dataset", "bias"])] + [(("all", b), g) for b, g in x.groupby("bias")]
    for (d, b), g in groups:
        out.append({"dataset": d, "bias": b, "n": len(g),
                    "logger ranking loss %": 100 * g["representation_loss_greedy"].mean(),
                    "oracle repair gain %": 100 * g["gain_repair_greedy"].mean(),
                    "structural recoverability": g["recoverability_repair_greedy"].mean(),
                    "recoverability min": g["recoverability_repair_greedy"].min(),
                    "recoverability max": g["recoverability_repair_greedy"].max(),
                    "validated recoverability": g["new recoverability greedy"].mean()})
    return _sort(pd.DataFrame(out))


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    order_b = {b: i for i, b in enumerate(("none",) + BIASES)}
    order_d = {d: i for i, d in enumerate(DATASETS + ("all",))}
    order_m = {m: i for i, m in enumerate(METHODS)}
    keys = [k for k in ("dataset", "bias", "train_size", "method", "share") if k in df]
    df = df.assign(**{f"_{k}": df[k].map({"dataset": order_d, "bias": order_b, "method": order_m}.get(k, {})).fillna(df[k])
                      if k in ("dataset", "bias", "method") else df[k] for k in keys})
    df = df.sort_values([f"_{k}" for k in ("bias", "dataset", "train_size", "method", "share") if k in keys], kind="stable")
    df = df.drop(columns=[f"_{k}" for k in keys]).reset_index(drop=True)
    df.insert(df.columns.get_loc("bias") + 1, "bias type", df["bias"].map(LABEL))
    return df


def stage2_table(s: Path = SUMMARIES) -> pd.DataFrame:
    """Per bias × train size × arm, overall and per dataset: the true gain over the logger (greedy and stochastic,
    CTR points) and the greedy fraction of the oracle repair, as mean and 95% t-interval over the conditions."""
    m = pd.read_csv(s / "stage2_learned_rows.csv")
    out = []
    for scope, frame in [("all", m)] + [(d, m[m["dataset"] == d]) for d in DATASETS]:
        for (b, n, meth), g in frame.groupby(["bias", "train_size", "method"]):
            row = {"dataset": scope, "bias": b, "train_size": n, "method": meth}
            for col, name, scale in (("learned_gain_greedy", "gain greedy %", 100), ("learned_gain", "gain %", 100),
                                     ("fraction_of_oracle_repair_greedy", "fraction greedy", 1)):
                c = _ci_row(scale * g[col])
                row.update({name: c["mean"], f"{name} ci_low": c["ci_low"], f"{name} ci_high": c["ci_high"]})
            row["n"] = len(g)
            out.append(row)
    return _sort(pd.DataFrame(out))


def paired_table(s: Path = SUMMARIES) -> pd.DataFrame:
    """OPC minus DM-only / no-propensity / tempered logger in true CTR (stochastic and greedy), per bias × train size,
    overall (the committed paired table) and per dataset (mean and the two seeds' range)."""
    p = pd.read_csv(s / "stage2_paired.csv").assign(dataset="all")
    m = pd.read_csv(s / "stage2_learned_rows.csv")
    rows = []
    for measure, col in (("stochastic", "V_method"), ("greedy", "V_method_greedy")):
        w = m.pivot_table(index=["dataset", "bias", "seed", "train_size"], columns="method", values=col)
        for other in ("dm", "no_propensity", "tempered_logger"):
            d = 100 * (w["opc"] - w[other])
            for (ds, b, n), g in d.groupby(level=["dataset", "bias", "train_size"]):
                rows.append({"dataset": ds, "bias": b, "train_size": n, "contrast": f"opc - {other}", "measure": measure,
                             "mean": g.mean(), "ci_low": g.min(), "ci_high": g.max(), "n": len(g)})
    per_ds = pd.DataFrame(rows).assign(interval="seed range")
    p = p.drop(columns=["bias type"]).assign(interval="95% t")
    return _sort(pd.concat([p, per_ds], ignore_index=True))


def gap_table(s: Path = SUMMARIES) -> pd.DataFrame:
    g = pd.read_csv(s / "followup" / "gap_decomposition.csv").assign(dataset="all")
    gd = pd.read_csv(s / "followup" / "gap_decomposition_by_dataset.csv")
    both = pd.concat([g, gd], ignore_index=True).drop(columns=["bias type"])
    return _sort(both)


def stage3_table(s: Path = SUMMARIES) -> pd.DataFrame:
    """Per logger share × bias (ml, kuairand × 2 seeds, 25k): the oracle's ranking gain over the logger, the greedy
    fraction of the oracle repair of OPC and DM-only, OPC minus DM-only (stochastic, CTR points), and OPC's
    selected-policy raw-weight ESS and share of weights above 10."""
    out = []
    for share in SHARES:
        m = pd.read_csv(s / f"stage3_lgs_{str(share).replace('.', '_')}" / "stage2_learned_rows.csv")
        for b, g in m.groupby("bias"):
            opc, dm = g[g["method"] == "opc"].set_index(["dataset", "seed"]), g[g["method"] == "dm"].set_index(["dataset", "seed"])
            row = {"share": share, "bias": b, "n": len(opc),
                   "oracle gain greedy %": 100 * (opc["oracle_repair_greedy"] - opc["V_logger_greedy"]).mean(),
                   "logger value %": 100 * opc["V_logger"].mean(), "logger greedy %": 100 * opc["V_logger_greedy"].mean()}
            for name, series in (("OPC fraction", opc["fraction_of_oracle_repair_greedy"]),
                                 ("DM fraction", dm["fraction_of_oracle_repair_greedy"]),
                                 ("OPC-DM %", 100 * (opc["V_method"] - dm["V_method"]))):
                c = _ci_row(series)
                row.update({name: c["mean"], f"{name} ci_low": c["ci_low"], f"{name} ci_high": c["ci_high"]})
            row["OPC ESS"] = opc["ess_raw"].mean()
            row["OPC w>10 %"] = 100 * opc["w_share_gt10"].mean()
            row["DM ESS"] = dm["ess_raw"].mean()
            out.append(row)
    return _sort(pd.DataFrame(out))


def _condition_values(run: Path, **tags) -> pd.DataFrame:
    """Selected-policy true CTR (stochastic) per condition × train size × arm of one run, for the conditions whose folder
    tags match ``tags`` (a tag set to None must be absent)."""
    rows = []
    for cond in sorted(run.glob("dataset=*")):
        t = _tags(cond.name)
        if not (cond / "summary_metrics.csv").exists() or any((t.get(k) != v) for k, v in tags.items()):
            continue
        s = pd.read_csv(cond / "summary_metrics.csv")
        for r in s[s["train_size"] > 0].itertuples():
            rows.append({"dataset": t["dataset"], "bias": t["bias"], "seed": int(t["seed"]), "train_size": int(r.train_size),
                         "method": r.method, "V": float(r.policy_rewards)})
    return pd.DataFrame(rows)


def propensity_earlier_table(runs: Path = RUNS) -> pd.DataFrame:
    """The earlier development tests of the reward model (previous defaults: legacy minibatch SNDR, log trick,
    shrink:100, TPE; pre-fix code), per q_hat setting × bias × train size over ml, kuairand, anime × 2 seeds:
    OPC, DM-only and OPC minus DM-only (true CTR points, mean and 95% t-interval), and for each pair of settings the
    change in DM-only and in OPC from the first to the second (paired by condition)."""
    settings = {
        "external 50k q_hat": (runs / "run_logger_explore", dict(lgs="0.8", qhat=None, cf=None)),
        "budget-fair q_hat (cf5)": (runs / "run_logger_explore", dict(lgs="0.8", qhat="train", cf="5")),
        "budget-fair q_hat, interaction features (cf5, val 20k)": (runs / "run_logger_explore_budget", dict(lgs="0.8", qhat="train", cf="5", val="20000")),
        "budget-fair q_hat, misspecified concat features (cf5, val 20k)": (runs / "run_qhat_concat", dict(lgs="0.8", qhat="train", cf="5", val="20000")),
    }
    vals = {}
    for name, (run, tags) in settings.items():
        v = _condition_values(run, **tags)
        if v.empty:
            return pd.DataFrame()
        vals[name] = v.pivot_table(index=["dataset", "bias", "seed", "train_size"], columns="method", values="V")
    rows = []
    for name, w in vals.items():
        for (b, n), g in (100 * w).groupby(level=["bias", "train_size"]):
            row = {"setting": name, "bias": b, "train_size": n, "OPC %": g["opc"].mean(), "DM %": g["dm"].mean()}
            c = _ci_row(g["opc"] - g["dm"])
            row.update({"OPC-DM": c["mean"], "OPC-DM ci_low": c["ci_low"], "OPC-DM ci_high": c["ci_high"], "n": c["n"]})
            rows.append(row)
    for a, b_ in (("external 50k q_hat", "budget-fair q_hat (cf5)"),
                  ("budget-fair q_hat, interaction features (cf5, val 20k)", "budget-fair q_hat, misspecified concat features (cf5, val 20k)")):
        both = vals[a].join(vals[b_], lsuffix="_a", rsuffix="_b", how="inner")
        for (b, n), g in (100 * both).groupby(level=["bias", "train_size"]):
            row = {"setting": f"change: {a} -> {b_}", "bias": b, "train_size": n}
            for arm in ("opc", "dm"):
                c = _ci_row(g[f"{arm}_b"] - g[f"{arm}_a"])
                row.update({f"change {METHOD_NAMES[arm]}": c["mean"], f"change {METHOD_NAMES[arm]} ci_low": c["ci_low"],
                            f"change {METHOD_NAMES[arm]} ci_high": c["ci_high"], "n": c["n"]})
            rows.append(row)
    return _sort(pd.DataFrame(rows))


def _opc_trials(run: Path) -> pd.DataFrame:
    rows = []
    for cond in sorted(run.glob("dataset=*")):
        if not (cond / "trials_long.csv").exists():
            continue
        t = pd.read_csv(cond / "trials_long.csv", usecols=lambda c: c in ("method", "train_size", "trial_number", "actual_reward",
                                                                            "is_best_in_run"))
        t = t[t["method"] == "opc"].assign(condition=cond.name)
        rows.append(t)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def weighting_study_table(runs: Path = RUNS) -> pd.DataFrame:
    """The paired objective / gradient / weighting comparisons (docs/training_losses.md section 9; random sampler, OPC
    only, ml, kuairand, anime × medium, high × 2 seeds × 5k, 25k, 100k, 20 trials): per comparison and cell, the per-trial
    difference in true CTR (same configuration and seed; CTR points; mean over the trials of each condition, then mean
    and 95% t-interval over the 6 conditions), and the difference between the two runs' selected policies."""
    comparisons = {
        "legacy SNDR - dr (log trick, shrink:100)": ("run_replay_sndr_batch_shrink_100", "run_gradcmp_dr_shrink_100_log_trick"),
        "global SNDR - dr (log trick, shrink:100)": ("run_replay_sndr_global_shrink_100", "run_gradcmp_dr_shrink_100_log_trick"),
        "direct - log trick (dr, shrink:100)": ("run_gradcmp_dr_shrink_100_direct", "run_gradcmp_dr_shrink_100_log_trick"),
        "direct - log trick (dr, none: control)": ("run_gradcmp_dr_none_direct", "run_gradcmp_dr_none_log_trick"),
        "shrink:100 - raw (dr, direct)": ("run_gradcmp_dr_shrink_100_direct", "run_gradcmp_dr_none_direct"),
        "harmonic:0.1 - raw (dr, direct)": ("run_final_dr_harmonic_0_1_direct", "run_gradcmp_dr_none_direct"),
        "harmonic:0.1 - shrink:100 (dr, direct)": ("run_final_dr_harmonic_0_1_direct", "run_gradcmp_dr_shrink_100_direct"),
    }
    cache, rows = {}, []
    for name, (a, b) in comparisons.items():
        for r in (a, b):
            if r not in cache:
                cache[r] = _opc_trials(runs / r)
        ta, tb = cache[a], cache[b]
        if ta.empty or tb.empty:
            return pd.DataFrame()
        key = ["condition", "train_size", "trial_number"]
        both = ta.merge(tb, on=key, suffixes=("_a", "_b"), validate="one_to_one")
        both["d"] = 100 * (both["actual_reward_a"] - both["actual_reward_b"])
        per_cond = both.groupby(["condition", "train_size"])["d"].mean().reset_index()
        sel = lambda t: t[t["is_best_in_run"].astype(bool)].set_index(["condition", "train_size"])["actual_reward"]
        sd = (100 * (sel(ta) - sel(tb))).rename("sel").reset_index()
        per_cond = per_cond.merge(sd, on=["condition", "train_size"])
        per_cond["bias"] = [_tags(c)["bias"] for c in per_cond["condition"]]
        for (bias, n), g in per_cond.groupby(["bias", "train_size"]):
            c, cs = _ci_row(g["d"]), _ci_row(g["sel"])
            rows.append({"comparison": name, "bias": bias, "train_size": n, "per-trial": c["mean"], "per-trial ci_low": c["ci_low"],
                         "per-trial ci_high": c["ci_high"], "selected": cs["mean"], "selected ci_low": cs["ci_low"],
                         "selected ci_high": cs["ci_high"], "n": c["n"], "trials": int(both[both["train_size"] == n]["condition"].isin(g["condition"]).sum())})
    return pd.DataFrame(rows)


def su_table(s: Path = SUMMARIES) -> pd.DataFrame:
    return pd.read_csv(s / "stage2_su_harmonic_minus_shrink100.csv")


# ---------------------------------------------------------------------------------------------------- figures


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
    return plt


def _save(fig, out: Path, name: str, data: pd.DataFrame) -> None:
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}.pdf")
    data.to_csv(out / f"{name}.csv", index=False)
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_structural(t1: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [2.4, 1]})
    x = np.arange(len(BIASES))
    width = 0.24
    for i, d in enumerate(DATASETS):
        g = t1[t1["dataset"] == d].set_index("bias").loc[list(BIASES)]
        ax.bar(x + (i - 1) * width, g["structural recoverability"], width, color=COLORS[d], label=DATASET_NAMES[d],
               yerr=[g["structural recoverability"] - g["recoverability min"], g["recoverability max"] - g["structural recoverability"]],
               error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#444444"})
    agg = t1[t1["dataset"] == "all"].set_index("bias").loc[list(BIASES)]
    ax.scatter(x, agg["structural recoverability"], marker="_", s=900, color="black", linewidths=1.6, zorder=3, label="all datasets")
    ax.set_xticks(x, [LABEL[b] for b in BIASES])
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("structural recoverability (greedy)\n= oracle repair gain / logger ranking loss")
    ax.set_title("(a) What the linear repair class can recover with the true rewards")
    ax.legend(ncol=4, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    w = t1[t1["dataset"] != "all"]
    for d in DATASETS:
        g = w[w["dataset"] == d]
        ax2.scatter(g["structural recoverability"], g["validated recoverability"], color=COLORS[d], s=22, label=DATASET_NAMES[d], zorder=3)
    ax2.plot([0.25, 1.02], [0.25, 1.02], color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax2.set_xlim(0.25, 1.02)
    ax2.set_ylim(0.25, 1.02)
    ax2.set_xlabel("Stage 1 bound")
    ax2.set_ylabel("validated bound (widened search)")
    ax2.set_title("(b) Oracle validation: per dataset × bias")
    fig.tight_layout()
    _save(fig, out, "fig1_structural_recoverability", t1)


def fig_fraction(t2: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(BIASES), figsize=(12, 2.9), sharey=True)
    data = t2[(t2["dataset"] == "all") & t2["bias"].isin(BIASES) & t2["method"].isin(["opc", "dm", "no_propensity"])]
    for ax, b in zip(axes, BIASES):
        for i, meth in enumerate(("opc", "dm", "no_propensity")):
            g = data[(data["bias"] == b) & (data["method"] == meth)].sort_values("train_size")
            xs = np.log10(g["train_size"]) + (i - 1) * 0.03
            ax.errorbar(xs, g["fraction greedy"], yerr=[g["fraction greedy"] - g["fraction greedy ci_low"],
                                                        g["fraction greedy ci_high"] - g["fraction greedy"]],
                        color=COLORS[meth], marker="o", markersize=3.5, linewidth=1.4, capsize=2, elinewidth=0.8,
                        label=METHOD_NAMES[meth])
        ax.axhline(0, color="#666666", linewidth=0.7)
        ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        ax.set_title(LABEL[b])
        ax.set_xlabel("logged training rows")
    axes[0].set_ylabel("fraction of the oracle ranking repair")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Learned repair as a fraction of what the class can do (greedy; mean and 95% CI over 3 datasets × 2 seeds; "
                 "tempered logger = 0 by construction)", fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out, "fig2_fraction_of_oracle_repair", data)


def fig_opc_minus_dm(tp: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    order = ("none",) + BIASES
    data = tp[(tp["dataset"] == "all") & (tp["contrast"] == "opc - dm") & (tp["measure"] == "stochastic") & tp["bias"].isin(order)]
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    shades = {5000: "#9ECAE1", 25000: "#4292C6", 100000: "#08519C"}
    for i, b in enumerate(order):
        for j, n in enumerate(SIZES):
            r = data[(data["bias"] == b) & (data["train_size"] == n)].iloc[0]
            ax.errorbar(i + (j - 1) * 0.22, r["mean"], yerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]], fmt="o",
                        color=shades[n], markersize=4.5, capsize=2.5, elinewidth=1.0, label=f"{n // 1000}k" if i == 0 else None)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(range(len(order)), [LABEL[b] for b in order])
    ax.set_ylabel("OPC − DM-only, true CTR points")
    ax.set_title("Value of the propensity correction over the reward model alone (paired; mean and 95% CI over 3 datasets × 2 seeds)")
    ax.legend(title="training rows", frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig3_opc_minus_dm", data)


def fig_gap(tg: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    g = tg[(tg["dataset"] == "all") & tg["bias"].isin(BIASES)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.35]})
    at = g[g["train_size"] == 100000].set_index("bias").loc[list(BIASES)[::-1]]
    ylab = [LABEL[b] for b in at.index]
    left = np.zeros(len(at))
    for col, key, lab in (("learned_repair_gain %", "learned", "learned by OPC"), ("learning_gap %", "learning", "expressible, not learned"),
                          ("structural_gap %", "structural", "not expressible (structural)")):
        ax.barh(ylab, at[col], left=left, color=GAP_COLORS[key], label=lab, edgecolor="white", linewidth=0.8)
        left += at[col].to_numpy()
    ax.set_xlabel("greedy CTR points below the target best (representation loss)")
    ax.set_title("(a) The representation loss at 100k, in CTR points")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", visible=False)
    # shares at each size
    ypos, labels = [], []
    for i, b in enumerate(list(BIASES)[::-1]):
        for j, n in enumerate(SIZES[::-1]):
            r = g[(g["bias"] == b) & (g["train_size"] == n)].iloc[0]
            y = i * 4 + j
            ypos.append(y)
            labels.append(f"{LABEL[b]}  {n // 1000}k")
            lft = 0.0
            for col, key in (("learned_repair_gain share", "learned"), ("learning_gap share", "learning"), ("structural_gap share", "structural")):
                ax2.barh(y, r[col], left=lft, color=GAP_COLORS[key], edgecolor="white", linewidth=0.6, height=0.85)
                lft += r[col]
    ax2.set_yticks(ypos, labels, fontsize=7.5)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("share of the representation loss")
    ax2.set_title("(b) Shares at 5k / 25k / 100k")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, out, "fig4_gap_decomposition", g)


def fig_logging_support(t3: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    singles = BIASES[:3]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
    for i, b in enumerate(singles):
        g = t3[t3["bias"] == b].sort_values("share")
        xs = g["share"] + (i - 1) * 0.008
        kw = dict(color=BIAS_COLORS[b], marker=BIAS_MARKERS[b], markersize=4)
        axes[0].plot(xs, g["oracle gain greedy %"], label=LABEL[b], **kw)
        for ax, col in ((axes[1], "OPC fraction"), (axes[2], "OPC-DM %")):
            ax.errorbar(xs, g[col], yerr=[g[col] - g[f"{col} ci_low"], g[f"{col} ci_high"] - g[col]], capsize=2, elinewidth=0.8, **kw)
        axes[3].plot(xs, g["OPC ESS"], **kw)
    titles = ("(a) oracle ranking gain (points)", "(b) OPC fraction of oracle repair", "(c) OPC − DM-only (points)",
              "(d) OPC raw-weight ESS (of 20,000)")
    for ax, t in zip(axes, titles):
        ax.set_title(t)
        ax.set_xticks(SHARES, ["0.6", "0.8", "0.95"])
        ax.set_xlabel("logger greedy share")
    axes[0].set_ylim(bottom=0)
    for ax in axes[1:3]:
        ax.axhline(0, color="#444444", linewidth=0.8)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Logging-support sweep (ml, kuairand × 2 seeds, 25k; mean and 95% CI)", fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig6_logging_support", t3)


def fig_per_dataset(t2: pd.DataFrame, tp: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, len(BIASES), figsize=(12, 5.0), sharey="row")
    frac = t2[(t2["method"] == "opc") & t2["dataset"].isin(DATASETS) & t2["bias"].isin(BIASES)]
    diff = tp[(tp["contrast"] == "opc - dm") & (tp["measure"] == "stochastic") & tp["dataset"].isin(DATASETS) & tp["bias"].isin(BIASES)]
    for k, b in enumerate(BIASES):
        for i, d in enumerate(DATASETS):
            f = frac[(frac["bias"] == b) & (frac["dataset"] == d)].sort_values("train_size")
            axes[0, k].plot(np.log10(f["train_size"]), f["fraction greedy"], marker="o", markersize=3.5, color=COLORS[d], label=DATASET_NAMES[d])
            q = diff[(diff["bias"] == b) & (diff["dataset"] == d)].sort_values("train_size")
            xs = np.log10(q["train_size"]) + (i - 1) * 0.03
            axes[1, k].errorbar(xs, q["mean"], yerr=[q["mean"] - q["ci_low"], q["ci_high"] - q["mean"]], marker="o", markersize=3.5,
                                color=COLORS[d], capsize=2, elinewidth=0.8)
        axes[0, k].set_title(LABEL[b])
        for r in (0, 1):
            axes[r, k].axhline(0, color="#666666", linewidth=0.7)
            axes[r, k].set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        axes[1, k].set_xlabel("logged training rows")
    axes[0, 0].set_ylabel("OPC fraction of oracle repair")
    axes[1, 0].set_ylabel("OPC − DM-only (points;\nbars: the two seeds)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Per dataset (mean of 2 seeds)", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out, "fig7_per_dataset", pd.concat([frac.assign(panel="OPC fraction"), diff.assign(panel="OPC - DM")], ignore_index=True))


def fig_propensity_value(te: pd.DataFrame, tp: pd.DataFrame, out: Path) -> None:
    if te.empty:
        return
    plt = _plt()
    settings = [s for s in te["setting"].unique() if not s.startswith("change")]
    styles = dict(zip(settings, (("#999999", "o", "--"), ("#0072B2", "o", "-"), ("#56B4E9", "s", "-"), ("#D55E00", "s", "-"))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.1), sharey=True)
    data = te[te["setting"].isin(settings)]
    for ax, b in zip(axes, ("medium", "high")):
        for i, st in enumerate(settings):
            g = data[(data["setting"] == st) & (data["bias"] == b)].sort_values("train_size")
            c, mk, ls = styles[st]
            xs = np.log10(g["train_size"]) + (i - 1.5) * 0.025
            ax.errorbar(xs, g["OPC-DM"], yerr=[g["OPC-DM"] - g["OPC-DM ci_low"], g["OPC-DM ci_high"] - g["OPC-DM"]], color=c,
                        marker=mk, linestyle=ls, markersize=3.5, capsize=2, elinewidth=0.8, label=st)
        cur = tp[(tp["dataset"] == "all") & (tp["contrast"] == "opc - dm") & (tp["measure"] == "stochastic") & (tp["bias"] == b)].sort_values("train_size")
        ax.errorbar(np.log10(cur["train_size"]) + 0.06, cur["mean"], yerr=[cur["mean"] - cur["ci_low"], cur["ci_high"] - cur["mean"]],
                    color="black", marker="D", markersize=3.5, capsize=2, elinewidth=0.8, linestyle=":", label="current pipeline (Stage 2)")
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_xticks(np.log10(SIZES), ["5k", "25k", "100k"])
        ax.set_title(LABEL[b])
        ax.set_xlabel("logged training rows")
    axes[0].set_ylabel("OPC − DM-only, true CTR points")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("OPC − DM-only by reward-model setting (earlier runs: previous defaults, pre-fix code; mean and 95% CI over "
                 "3 datasets × 2 seeds)", fontsize=9, y=1.03)
    fig.tight_layout()
    _save(fig, out, "fig5_propensity_value", data)


# ---------------------------------------------------------------------------------------------------- markdown


def _md(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, p=2, sign=False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"


def _ci(r, col, p=2, sign=True) -> str:
    return f"{_f(r[col], p, sign)} [{_f(r[f'{col} ci_low'], p, sign)}, {_f(r[f'{col} ci_high'], p, sign)}]"


def tables_markdown(t1, t2, tp, tg, t3, te, tw, tsu) -> str:
    parts = []
    # T1
    rows = []
    for b in BIASES:
        cells = [LABEL[b]]
        for d in DATASETS + ("all",):
            r = t1[(t1["dataset"] == d) & (t1["bias"] == b)].iloc[0]
            cells.append(f"{r['logger ranking loss %']:.2f} / {r['oracle repair gain %']:.2f} / {r['structural recoverability']:.3f} "
                         f"({r['validated recoverability']:.3f})")
        rows.append(cells)
    parts.append("### T1. Structural recoverability (Stage 1, greedy)\n\nEach cell: logger ranking loss (points) / oracle repair gain "
                 "(points) / structural recoverability (validated bound in brackets). Datasets: mean of 2 seeds; all: mean of 6.\n\n"
                 + _md(["bias"] + [DATASET_NAMES[d] for d in DATASETS] + ["all"], rows))
    # T2 aggregate
    rows = []
    agg = t2[t2["dataset"] == "all"]
    for b in ("none",) + BIASES:
        for n in SIZES:
            g = agg[(agg["bias"] == b) & (agg["train_size"] == n)].set_index("method")
            cells = [LABEL[b], f"{n // 1000}k"]
            cells += [" / ".join(_f(g.loc[m, "gain greedy %"], 2, True) for m in ("opc", "dm", "no_propensity"))]
            cells += [" / ".join(_f(g.loc[m, "gain %"], 2, True) for m in METHODS)]
            cells += ["—" if b == "none" else " / ".join(_f(g.loc[m, "fraction greedy"]) for m in ("opc", "dm", "no_propensity"))]
            rows.append(cells)
    parts.append("### T2. Learned repair (Stage 2), mean over 3 datasets × 2 seeds\n\nTrue gain over the logger in CTR points: greedy "
                 "OPC / DM-only / no-propensity (the tempered logger's greedy gain is 0), stochastic OPC / DM-only / no-propensity / "
                 "tempered logger, and the greedy fraction of the oracle repair.\n\n"
                 + _md(["bias", "train", "greedy gain: OPC / DM / no-prop", "stochastic gain: OPC / DM / no-prop / tempered",
                        "fraction of oracle repair: OPC / DM / no-prop"], rows))
    # T3 paired aggregate
    rows = []
    agg = tp[(tp["dataset"] == "all") & (tp["measure"] == "stochastic")]
    for b in ("none",) + BIASES:
        for n in SIZES:
            g = agg[(agg["bias"] == b) & (agg["train_size"] == n)].set_index("contrast")
            rows.append([LABEL[b], f"{n // 1000}k"] + [f"{_f(g.loc[c, 'mean'], 2, True)} [{_f(g.loc[c, 'ci_low'], 2, True)}, "
                                                        f"{_f(g.loc[c, 'ci_high'], 2, True)}]"
                                                        for c in ("opc - dm", "opc - no_propensity", "opc - tempered_logger")])
    parts.append("### T3. OPC minus each baseline (Stage 2), true CTR points, mean and 95% t-interval over the 6 paired conditions\n\n"
                 + _md(["bias", "train", "OPC − DM-only", "OPC − no-propensity", "OPC − tempered logger"], rows))
    # T4 per dataset
    rows = []
    frac = t2[t2["dataset"].isin(DATASETS)]
    dif = tp[tp["dataset"].isin(DATASETS) & (tp["measure"] == "stochastic")]
    for b in BIASES:
        for n in SIZES:
            cells = [LABEL[b], f"{n // 1000}k"]
            for d in DATASETS:
                f = frac[(frac["dataset"] == d) & (frac["bias"] == b) & (frac["train_size"] == n)].set_index("method")
                q = dif[(dif["dataset"] == d) & (dif["bias"] == b) & (dif["train_size"] == n)].set_index("contrast")
                cells.append(f"{_f(f.loc['opc', 'fraction greedy'])} / {_f(f.loc['dm', 'fraction greedy'])}; "
                             f"{_f(q.loc['opc - dm', 'mean'], 2, True)} / {_f(q.loc['opc - no_propensity', 'mean'], 2, True)}")
            rows.append(cells)
    parts.append("### T4. Per dataset (Stage 2, mean of 2 seeds)\n\nEach cell: fraction of the oracle ranking repair OPC / DM-only; "
                 "OPC − DM-only / OPC − no-propensity in true CTR points.\n\n"
                 + _md(["bias", "train"] + [DATASET_NAMES[d] for d in DATASETS], rows))
    # T5 gap
    rows = []
    g = tg[tg["dataset"] == "all"]
    for b in BIASES:
        for n in SIZES:
            r = g[(g["bias"] == b) & (g["train_size"] == n)].iloc[0]
            rows.append([LABEL[b], f"{n // 1000}k", _f(r["V_target_best %"]), _f(r["V_logger %"]), _f(r["V_oracle_repair %"]),
                         _f(r["V_OPC %"]), _f(r["representation_loss %"]), _f(r["structural_gap %"]), _f(r["learning_gap %"]),
                         _f(r["learned_repair_gain %"]), f"{r['structural_gap share']:.2f} / {r['learning_gap share']:.2f} / "
                                                          f"{r['learned_repair_gain share']:.2f}"])
    parts.append("### T5. Structural gap vs learning gap (greedy, CTR %, mean over 3 datasets × 2 seeds)\n\n"
                 + _md(["bias", "train", "V_target_best", "V_logger", "V_oracle_repair", "V_OPC", "representation loss", "structural gap",
                        "learning gap", "learned repair", "shares: structural / learning / learned"], rows))
    rows = []
    gd = tg[tg["dataset"].isin(DATASETS) & (tg["train_size"] == 100000)]
    for b in BIASES:
        rows.append([LABEL[b]] + [" / ".join(f"{gd[(gd['dataset'] == d) & (gd['bias'] == b)].iloc[0][c]:.2f}" for c in
                                            ("structural_gap share", "learning_gap share", "learned_repair_gain share")) for d in DATASETS])
    parts.append("Shares at 100k by dataset (structural / learning / learned):\n\n" + _md(["bias"] + [DATASET_NAMES[d] for d in DATASETS], rows))
    # T6 propensity (earlier)
    if not te.empty:
        rows = []
        for st in [s for s in te["setting"].unique()]:
            for b in ("medium", "high"):
                g = te[(te["setting"] == st) & (te["bias"] == b)].set_index("train_size")
                if st.startswith("change"):
                    cells = [st.replace("change: ", "Δ "), LABEL[b]] + [f"DM {_f(g.loc[n, 'change DM-only'], 2, True)}, OPC {_f(g.loc[n, 'change OPC'], 2, True)}"
                                                                         for n in SIZES]
                else:
                    cells = [st, LABEL[b]] + [_ci(g.loc[n], "OPC-DM") for n in SIZES]
                rows.append(cells)
        parts.append("### T6. Earlier reward-model tests (previous defaults: legacy SNDR, log trick, shrink:100, TPE; pre-fix code)\n\n"
                     "OPC − DM-only in true CTR points (mean [95% CI] over ml, kuairand, anime × 2 seeds), and the change in each arm "
                     "between paired settings.\n\n" + _md(["setting", "bias", "5k", "25k", "100k"], rows))
    # T7 weighting study
    if not tw.empty:
        rows = []
        for c, g in tw.groupby("comparison", sort=False):
            pt, sl = g["per-trial"], g["selected"]
            excl = int(((g["per-trial ci_low"] > 0) | (g["per-trial ci_high"] < 0)).sum())
            rows.append([c, f"{pt.min():+.2f} to {pt.max():+.2f}", f"{excl} of {len(g)}", f"{sl.min():+.2f} to {sl.max():+.2f}"])
        parts.append("### T7. Objective / gradient / weighting study (paired, random sampler; OPC only; medium, high × 5k, 25k, 100k)\n\n"
                     "Range over the 6 cells of the mean per-trial difference in true CTR points (identical configurations and seeds), "
                     "cells whose 95% CI excludes 0, and the range of the selected-policy difference.\n\n"
                     + _md(["comparison", "per trial", "CI excludes 0", "selected"], rows))
    # T8 Su slice
    rows = []
    for r in tsu[tsu["measure"].isin(["V %", "V greedy %", "fraction greedy"])].itertuples():
        rows.append([LABEL.get(r.bias, r.bias), r.measure, f"{r.a:.3f}", f"{r.b:.3f}", f"{r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]"])
    parts.append("### T8. Robustness slice: OPC with harmonic:0.1 minus OPC with shrink:100 (25k; ml, kuairand × 2 seeds)\n\n"
                 + _md(["bias", "measure", "harmonic:0.1", "shrink:100", "difference [95% CI]"], rows))
    # T9 Stage 3
    rows = []
    for b in BIASES[:3]:
        for sh in SHARES:
            r = t3[(t3["bias"] == b) & (t3["share"] == sh)].iloc[0]
            rows.append([LABEL[b], f"{sh:g}", _f(r["logger value %"]), _f(r["oracle gain greedy %"]), _ci(r, "OPC fraction", 2, False),
                         _f(r["DM fraction"]), _ci(r, "OPC-DM %"), f"{r['OPC ESS']:.0f}", _f(r["OPC w>10 %"])])
    parts.append("### T9. Logging-support sweep (Stage 3; ml, kuairand × 2 seeds; 25k)\n\n"
                 + _md(["bias", "logger share", "logger value %", "oracle ranking gain", "OPC fraction [95% CI]", "DM fraction",
                        "OPC − DM-only [95% CI]", "OPC raw-weight ESS", "OPC weights > 10 (%)"], rows))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------------- main


def _raw_or_committed(fn, path: Path, runs: Path) -> pd.DataFrame:
    t = fn(runs)
    if not t.empty:
        t.to_csv(path, index=False)
        return t
    if path.exists():
        print(f"[report] {path.name}: run folders not found, using the committed table", flush=True)
        return pd.read_csv(path)
    print(f"[report] {path.name}: no run folders and no table; skipped", flush=True)
    return pd.DataFrame()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--summaries", default=str(SUMMARIES))
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s, runs, out = Path(a.summaries), Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t1, t2, tp, tg, t3, tsu = stage1_table(s), stage2_table(s), paired_table(s), gap_table(s), stage3_table(s), su_table(s)
    te = _raw_or_committed(propensity_earlier_table, out / "table_propensity_earlier.csv", runs)
    tw = _raw_or_committed(weighting_study_table, out / "table_weighting_study.csv", runs)
    for name, t in (("table_stage1", t1), ("table_stage2", t2), ("table_paired", tp), ("table_gap", tg), ("table_stage3", t3)):
        t.to_csv(out / f"{name}.csv", index=False)
    fig_structural(t1, out)
    fig_fraction(t2, out)
    fig_opc_minus_dm(tp, out)
    fig_gap(tg, out)
    fig_logging_support(t3, out)
    fig_per_dataset(t2, tp, out)
    fig_propensity_value(te, tp, out)
    (out / "tables.md").write_text(tables_markdown(t1, t2, tp, tg, t3, te, tw, tsu))
    print(f"[report] wrote tables and figures to {out}", flush=True)


if __name__ == "__main__":
    main()
