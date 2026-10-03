"""Bounded CausE vs OPC comparison: tables and figures from the condition run folders (development stage).

Inputs: the comparison's run folders (``summary_metrics.csv`` per condition, with opc / dm / tempered_logger and
the ``cause_<prediction>_r<rho per mille>`` methods) and the oracle bounds: Stage 1 rows and the validated
per-world bounds of the follow-up (they use no logged data). Per condition (dataset, bias, seed) and method:
    gain (greedy) = V_greedy - V_logger_greedy           gain (stochastic) = V - V_logger
    fraction of the mismatch repaired = gain / (V_target_best - V_logger_greedy)
    fraction of OPC's structural oracle = gain / (V_oracle_linear - V_logger_greedy)
Native CausE's own class contains the true click model, so its structural oracle is the ceiling and its
oracle fraction equals its mismatch fraction.

Usage: python -m training.analyze_cause_comparison --runs artifacts/full_study/run_<tag> ... --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from training.representation_report import DATASET_NAMES, _plt, _save

SUMMARIES = Path("artifacts/full_study/summaries_20260927")
BIAS_ORDER = ("none", "w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high", "high")
BIAS_NAMES = {"none": "no bias", "w-high.g-none.v-none": "warp high", "w-none.g-high.v-none": "group high",
              "w-none.g-none.v-high": "vector high", "high": "combined high", "medium": "combined medium"}
BASE_METHODS = ("opc", "dm", "tempered_logger")
PREDICTIONS = ("prod_c", "prod_t", "avg")
NAMES = {"opc": "OPC", "dm": "DM-only", "tempered_logger": "tempered logger", "prod_c": "CausE-prod-C",
         "prod_t": "CausE-prod-T", "avg": "CausE-avg"}
COLORS = {"opc": "#0072B2", "dm": "#E69F00", "tempered_logger": "#009E73", "prod_c": "#D55E00", "prod_t": "#56B4E9",
          "avg": "#CC79A7"}


def _ci(x) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan, len(x)
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return m, m - h, m + h, len(x)


def oracle_bounds(summaries: Path = SUMMARIES) -> pd.DataFrame:
    """Per (dataset, bias, seed): ceiling, logger greedy value, OPC's linear-class oracle (validated if available)."""
    s1 = pd.read_csv(summaries / "stage1_oracle_rows.csv")
    s1 = s1[np.isclose(s1["logger_share"], 0.8)]
    b = s1[["dataset", "bias", "seed", "V_clean_greedy", "V_logger_greedy", "oracle_repair_greedy"]].rename(
        columns={"V_clean_greedy": "ceiling", "oracle_repair_greedy": "oracle_linear_greedy"})
    val = summaries / "followup" / "oracle_validation_by_world.csv"
    if val.exists():
        v = pd.read_csv(val)[["dataset", "bias", "seed", "new bound greedy"]]
        b = b.merge(v, on=["dataset", "bias", "seed"], how="left")
        b["oracle_linear_greedy"] = b["new bound greedy"].fillna(b["oracle_linear_greedy"])
        b = b.drop(columns=["new bound greedy"])
    return b


def load_conditions(run_dirs, train_size: int = 25_000) -> pd.DataFrame:
    frames = []
    for rd in run_dirs:
        for p in sorted(Path(rd).glob("*/summary_metrics.csv")):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no summary_metrics.csv under {run_dirs}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["train_size"] == train_size].copy()
    df["bias"] = df["noise_level"].astype(str)
    df["arm"] = np.where(df["method"].str.startswith("cause_"), df.get("cause_prediction"), df["method"])
    df["rho"] = np.where(df["method"].str.startswith("cause_"), df.get("cause_rho"), np.nan)
    return df


def condition_table(df: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    """One row per condition × method with values, gains and fractions."""
    keep = ["dataset", "bias", "seed", "method", "arm", "rho", "train_size", "policy_rewards", "policy_rewards_greedy"]
    extra = ["n_control", "n_treatment", "collection_reward_sum", "opc_collection_reward_sum", "exploration_cost_expected",
             "exploration_cost_realised", "treatment_rows_per_item", "treatment_items_covered", "uniform_value",
             "val_nll", "alpha", "lr", "epochs", "l2_pen", "cf_pen", "oracle_selected_value_greedy"]
    t = df[keep + [c for c in extra if c in df.columns]].merge(bounds, on=["dataset", "bias", "seed"], how="left")
    logger = df[df["method"] == "opc"][["dataset", "bias", "seed"]].drop_duplicates()
    # V_logger (stochastic) is the train_size-0 row of each arm; recompute from the CausE rows when present
    if "initial_reward" in df.columns:
        lv = df.dropna(subset=["initial_reward"]).groupby(["dataset", "bias", "seed"])["initial_reward"].first()
        t = t.merge(lv.rename("V_logger").reset_index(), on=["dataset", "bias", "seed"], how="left")
    t["gain_greedy"] = t["policy_rewards_greedy"] - t["V_logger_greedy"]
    t["gain"] = t["policy_rewards"] - t.get("V_logger", np.nan)
    loss = t["ceiling"] - t["V_logger_greedy"]
    t["representation_loss"] = loss
    t["frac_mismatch"] = np.where(loss > 1e-6, t["gain_greedy"] / loss, np.nan)
    rec = t["oracle_linear_greedy"] - t["V_logger_greedy"]
    t["frac_opc_oracle"] = np.where(rec > 1e-6, t["gain_greedy"] / rec, np.nan)
    del logger
    return t


def summary_table(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (bias, arm, rho), g in t.groupby(["bias", "arm", t["rho"].fillna(-1)]):
        r = {"bias": bias, "arm": arm, "rho": None if rho < 0 else rho, "n_conditions": len(g)}
        for col in ("gain_greedy", "gain", "frac_mismatch", "frac_opc_oracle", "policy_rewards_greedy", "policy_rewards",
                    "exploration_cost_expected", "exploration_cost_realised", "collection_reward_sum"):
            if col in g:
                m, lo, hi, n = _ci(g[col])
                r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
        rows.append(r)
    out = pd.DataFrame(rows)
    out["bias_order"] = out["bias"].map({b: i for i, b in enumerate(BIAS_ORDER)})
    return out.sort_values(["bias_order", "arm", "rho"]).drop(columns="bias_order")


def paired_table(t: pd.DataFrame) -> pd.DataFrame:
    """OPC − CausE (greedy gain, CTR points) paired over dataset × seed, per bias, prediction and rho."""
    opc = t[t["arm"] == "opc"].set_index(["dataset", "bias", "seed"])["gain_greedy"]
    rows = []
    for (bias, arm, rho), g in t[t["arm"].isin(PREDICTIONS)].groupby(["bias", "arm", "rho"]):
        d = opc.reindex(g.set_index(["dataset", "bias", "seed"]).index).values - g["gain_greedy"].values
        m, lo, hi, n = _ci(100 * d)
        rows.append({"bias": bias, "cause": arm, "rho": rho, "OPC minus CausE (pts)": m, "ci_lo": lo, "ci_hi": hi, "n": n,
                     "OPC better in": int(np.sum(d > 0))})
    return pd.DataFrame(rows)


def _legend(fig, axes) -> None:
    """One legend for every series drawn in any panel (identity is never color alone)."""
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def fig_rho_curve(s: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    biases = [b for b in BIAS_ORDER if b in set(s["bias"])]
    fig, axes = plt.subplots(1, len(biases), figsize=(2.6 * len(biases), 2.9), sharey=False)
    axes = np.atleast_1d(axes)
    data = []
    for ax, bias in zip(axes, biases):
        sb = s[s["bias"] == bias]
        for arm in PREDICTIONS:
            g = sb[sb["arm"] == arm].sort_values("rho")
            if g.empty:
                continue
            ax.errorbar(g["rho"], 100 * g["gain_greedy"], yerr=[100 * (g["gain_greedy"] - g["gain_greedy_lo"]),
                        100 * (g["gain_greedy_hi"] - g["gain_greedy"])], color=COLORS[arm], marker="o", markersize=3.5,
                        linewidth=1.4, capsize=2, elinewidth=0.8, label=NAMES[arm])
            data.append(g.assign(panel=bias))
        for arm, ls in (("opc", "-"), ("dm", "--"), ("tempered_logger", ":")):
            g = sb[sb["arm"] == arm]
            if len(g):
                r = g.iloc[0]  # one summary row: the mean (and CI) over conditions, no rho
                ax.axhline(100 * float(r["gain_greedy"]), color=COLORS[arm], linestyle=ls, linewidth=1.3, label=NAMES[arm])
                if arm == "opc" and np.isfinite(r.get("gain_greedy_lo", np.nan)):
                    ax.axhspan(100 * float(r["gain_greedy_lo"]), 100 * float(r["gain_greedy_hi"]), color=COLORS[arm],
                               alpha=0.12, linewidth=0, label="OPC 95% CI")
                data.append(g.assign(panel=bias))
        ax.axhline(0, color="black", linewidth=0.8, label="logger (greedy)")
        ax.set_title(BIAS_NAMES.get(bias, bias))
        ax.set_xlim(-0.012, 0.262)
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.25])
        ax.set_xticklabels(["0", ".05", ".10", ".15", ".25"])
    axes[0].set_ylabel("greedy value − logger's greedy value (CTR pts)")
    fig.supxlabel("randomized share of the budget, ρ", fontsize=9)
    _legend(fig, axes)
    fig.suptitle("Target performance vs randomized traffic (25k interactions; mean and 95% CI over 3 datasets × 2 seeds)",
                 y=1.02, fontsize=9.5)
    _save(fig, out, "fig_rho_curve", pd.concat(data, ignore_index=True) if data else pd.DataFrame())


def fig_exploration_cost(t: pd.DataFrame, out: Path) -> None:
    plt = _plt()
    biases = [b for b in BIAS_ORDER if b in set(t["bias"])]
    fig, axes = plt.subplots(1, len(biases), figsize=(2.6 * len(biases), 2.9))
    axes = np.atleast_1d(axes)
    data = []
    for ax, bias in zip(axes, biases):
        tb = t[t["bias"] == bias]
        for arm in PREDICTIONS:
            g = tb[tb["arm"] == arm].groupby("rho").agg(cost=("exploration_cost_expected", "mean"),
                                                         gain=("gain_greedy", "mean")).reset_index()
            ax.plot(g["cost"], 100 * g["gain"], color=COLORS[arm], marker="o", markersize=3.5, linewidth=1.3, label=NAMES[arm])
            data.append(g.assign(panel=bias, arm=arm))
        g = tb[tb["arm"] == "opc"]
        if len(g):
            ax.scatter([0.0], [100 * g["gain_greedy"].mean()], color=COLORS["opc"], s=36, zorder=3, label="OPC (no randomized traffic)")
            data.append(pd.DataFrame({"panel": [bias], "arm": ["opc"], "cost": [0.0], "gain": [g["gain_greedy"].mean()]}))
        ax.axhline(0, color="black", linewidth=0.8, label="logger (greedy)")
        ax.set_title(BIAS_NAMES.get(bias, bias))
    axes[0].set_ylabel("greedy value − logger's greedy value (CTR pts)")
    fig.supxlabel("expected clicks given up while collecting (ρN·(V(π0) − V(uniform)))", fontsize=9)
    _legend(fig, axes)
    fig.suptitle("Final target value vs cumulative exploration cost (cost = ρN·(V(π0) − V(uniform)); means over 6 conditions)",
                 y=1.02, fontsize=9.5)
    _save(fig, out, "fig_exploration_cost", pd.concat(data, ignore_index=True) if data else pd.DataFrame())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-size", type=int, default=25_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = condition_table(load_conditions(args.runs, args.train_size), oracle_bounds())
    t.to_csv(out / "table_conditions.csv", index=False)
    s = summary_table(t)
    s.to_csv(out / "table_summary.csv", index=False)
    p = paired_table(t)
    p.to_csv(out / "table_paired.csv", index=False)
    fig_rho_curve(s, out)
    fig_exploration_cost(t, out)
    print(f"wrote {out}: {len(t)} condition rows, {t[['dataset', 'bias', 'seed']].drop_duplicates().shape[0]} conditions")


if __name__ == "__main__":
    main()
