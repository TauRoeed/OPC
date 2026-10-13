"""Analysis of the OPC adaptive-estimator study (docs/opc_adaptive_estimator_study.md).

``stage_a``: the fixed-policy oracle evaluator diagnostics of Stage A (§8, §11) over the worlds of ``run_ae_stage_a``:
every evaluator's exact and replicate bias, spread and RMSE at every fixed policy, per regime (pooled and per
dataset); the replicates checked against the exact moments; support; the paired against the independent improvement
estimate; the adaptivity of selection from the recorded trials of ``run_ss_main_25k``.

    python -m training.analyze_adaptive stage_a --run artifacts/full_study/run_ae_stage_a --replicates 200 \
        --trials artifacts/full_study/run_ss_main_25k --out artifacts/full_study/opc_adaptive_estimator/stage_a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_cause_fair import _md, _tags, mean_ci
from training.analyze_shared_objectives import _plt, _save
from training.oracle_eval import HARMONIC_GRID, linear_estimators
from utils.structured_shift import parse_structured

REGIMES = {"s-small.r-strong": "R1", "s-moderate.r-strong": "R2", "s-moderate.r-moderate": "R3",
           "s-moderate.r-none": "C1", "s-none.r-strong": "C2", "s-strong.r-strong": "S1"}
REGIME_ORDER = ("R1", "R2", "R3", "C1", "C2", "S1")
DATASETS = ("ml", "kuairand", "anime")
ARMS = ("likelihood", "likelihood_calib", "opc", "opc_raw", "opc_oq", "source", "value_star")
ESTIMATORS = ("dm", "dr_raw") + tuple(f"dr_h{lam:g}" for lam in HARMONIC_GRID) + ("dr_clip10", "sndr", "dr_hA2",
                                                                                  "dr_hA3", "dr_oracle")
BOUNDED = ("dm", "dr_clip10", "dr_h0.03", "dr_h0.1", "dr_h0.2")  # bounded weights: tight replicate checks
PTS = 100.0  # CTR points


def _world_tags(name: str) -> dict:
    t = dict(p.split("=", 1) for p in name.split("__"))
    lv = parse_structured(t["bias"])
    return {"dataset": t["dataset"], "label": t["bias"], "regime": REGIMES.get(t["bias"], t["bias"]),
            "seed": int(t["seed"]), "shift": lv["shift"], "response": lv["response"]}


def load_stage_a(run: Path, replicates: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per world, policy and estimator: the exact and replicate summary; per world and policy: the support and value
    statistics; the replicate rows' per-log statistics)."""
    summ, supp, reps = [], [], []
    for wdir in sorted(Path(run).glob("dataset=*")):
        f = wdir / f"summary_R{replicates}.csv"
        if not f.exists():
            continue
        tags = _world_tags(wdir.name)
        s = pd.read_csv(f)
        summ.append(s.assign(**tags))
        ex = json.loads((wdir / "exact.json").read_text())
        for pol, r in ex.items():
            if pol == "logger":
                continue
            row = {**tags, "policy": pol, **{k: v for k, v in r.items() if k != "estimators"},
                   "qhat_rmse_logger": ex["logger"]["qhat_rmse"]}
            supp.append(row)
        m = pd.read_csv(wdir / f"mc_rows_R{replicates}.csv.gz")
        m = m[m["estimator"] == "dr_raw"].drop(columns=["estimator", "estimate", "se", "delta", "delta_se"])
        reps.append(m.assign(**tags))
    if not summ:
        raise FileNotFoundError(f"no summary_R{replicates}.csv under {run}")
    return pd.concat(summ, ignore_index=True), pd.DataFrame(supp), pd.concat(reps, ignore_index=True)


def _arm(policy: str) -> tuple[str, str]:
    return (policy[:-2], "greedy") if policy.endswith("_g") else (policy, "softmax")


def regime_summary(summ: pd.DataFrame) -> pd.DataFrame:
    """Per regime (and dataset), policy kind, arm and estimator: the mean over worlds of the exact bias, standard
    deviation and RMSE (points; the replicate ones for the nonlinear estimators), the coverage, and the improvement's
    bias and RMSE."""
    d = summ.copy()
    d[["arm", "kind"]] = pd.DataFrame([_arm(p) for p in d["policy"]], index=d.index)
    lin = d["estimator"].isin(list(linear_estimators()))
    d["bias_pts"] = np.where(lin, d.get("exact_bias", np.nan), d["mc_bias"]) * PTS
    d["sd_pts"] = np.sqrt(np.where(lin, d.get("exact_var", np.nan), d["mc_var"])) * PTS
    d["rmse_pts"] = np.sqrt(np.where(lin, d.get("exact_mse", np.nan), d["mc_mse"])) * PTS
    d["delta_bias_pts"] = np.where(lin, d.get("exact_delta_bias", np.nan), d["mc_delta_bias"]) * PTS
    d["delta_rmse_pts"] = np.sqrt(np.where(lin, d.get("exact_delta_mse", np.nan), d["mc_delta_mse"])) * PTS
    rows = []
    for scope_ds in ["all"] + list(DATASETS):
        g0 = d if scope_ds == "all" else d[d["dataset"] == scope_ds]
        for (reg, kind, arm, e), g in g0.groupby(["regime", "kind", "arm", "estimator"]):
            r = {"dataset": scope_ds, "regime": reg, "kind": kind, "arm": arm, "estimator": e, "worlds": len(g)}
            for col in ("bias_pts", "sd_pts", "rmse_pts", "delta_bias_pts", "delta_rmse_pts", "mc_coverage",
                        "mc_delta_coverage"):
                m, lo, hi, _n = mean_ci(g[col])
                r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            r["abs_bias_pts"] = float(np.mean(np.abs(g["bias_pts"])))
            rows.append(r)
    return pd.DataFrame(rows)


def replicate_checks(summ: pd.DataFrame) -> pd.DataFrame:
    """The replicates against the exact moments (§8): per world, policy and linear estimator, the bias in replicate
    standard errors and the variance ratio; ``ok`` for bounded-weight estimators (|z| < 4, the exact variance in the
    replicates' χ² interval), raw-weight ones reported only (heavy tails)."""
    d = summ[summ["estimator"].isin(list(linear_estimators()))].copy()
    d["bounded"] = d["estimator"].isin(BOUNDED)
    d["ok"] = ~d["bounded"] | ((d["bias_z"].abs() < 4) & d["var_in_ci"].astype(bool))
    return d[["dataset", "regime", "seed", "policy", "estimator", "bounded", "bias_z", "var_ratio", "var_in_ci",
              "ok"]]


def delta_table(summ: pd.DataFrame) -> pd.DataFrame:
    """Per world, policy and linear estimator: the paired improvement's exact variance against two independent logs'
    (the ratio < 1 is the cancellation of common noise) and the correlation of the two absolute estimates."""
    d = summ[summ["estimator"].isin(list(linear_estimators()))].copy()
    d = d[~d["policy"].isin(["source", "source_g"])]
    d["paired_over_independent"] = d["exact_delta_var"] / d["exact_delta_var_independent"]
    v1 = d["exact_var_row"]
    v0 = d["exact_delta_var_independent"] * 25_000 - v1  # the source's own per-row variance (n = 25,000)
    d["corr_with_source"] = d["exact_cov_with_source_row"] / np.sqrt(np.clip(v1 * v0, 1e-300, None))
    return d[["dataset", "regime", "seed", "policy", "estimator", "exact_delta_var", "exact_delta_var_independent",
              "paired_over_independent", "corr_with_source"]]


def adaptivity(trials_run: Path, worlds: pd.DataFrame, exact_clip_bias: pd.DataFrame) -> pd.DataFrame:
    """§8.3: per world and arm, the validation estimate (DR with clip-10 weights, the full-data q̂) minus the true
    value, for the selected trial and averaged over all trials, and the evaluator's exact conditional bias at the
    selected policy: the selected trial's excess over it is the adaptivity of selection."""
    from training.run_state import read_trials_long

    rows = []
    for cond in sorted(Path(trials_run).glob("dataset=*")):
        tags = _tags(cond.name)
        key = (tags["dataset"], tags["bias"], int(tags["seed"]))
        if key not in set(map(tuple, worlds[["dataset", "label", "seed"]].to_numpy())):
            continue
        t = read_trials_long(cond / "trials_long.csv")
        for m, g in t.groupby("method"):
            arm = m.replace("shared_lr_", "")
            err = 100 * (g["r_hat"] - g["actual_reward"])
            sel = g[g["is_best_in_run"].astype(bool)].iloc[0]
            b = exact_clip_bias.get((key[0], key[1], key[2], arm), np.nan)
            rows.append({"dataset": key[0], "label": key[1], "regime": REGIMES.get(key[1]), "seed": key[2], "arm": arm,
                         "err_selected_pts": 100 * (sel["r_hat"] - sel["actual_reward"]),
                         "err_all_trials_pts": float(err.mean()), "exact_clip10_bias_selected_pts": 100 * b,
                         "adaptivity_pts": 100 * (sel["r_hat"] - sel["actual_reward"]) - 100 * b})
    return pd.DataFrame(rows)


def fig_lambda_curve(rs: pd.DataFrame, out: Path, arms=("opc", "value_star"), kind="softmax") -> None:
    """Per regime: harmonic DR's exact |bias|, standard deviation and RMSE against λ (raw DR at λ = 0 drawn at the
    left edge) at the given fixed policies, pooled over worlds."""
    plt = _plt()
    lams = [0.0] + list(HARMONIC_GRID)
    names = ["dr_raw"] + [f"dr_h{lam:g}" for lam in HARMONIC_GRID]
    regs = [r for r in REGIME_ORDER if (rs["regime"] == r).any()]
    arms = tuple(a for a in arms if ((rs["arm"] == a) & (rs["kind"] == kind)).any())  # the arms this run evaluated
    if not arms or not regs:
        return
    fig, axes = plt.subplots(len(arms), len(regs), figsize=(3.0 * len(regs), 2.9 * len(arms)), squeeze=False)
    x = np.array([0.002] + list(HARMONIC_GRID))  # λ = 0 placed at 0.002 on the log axis
    rows = []
    colors = {"abs_bias_pts": "#D55E00", "sd_pts": "#0072B2", "rmse_pts": "#333333"}
    labels = {"abs_bias_pts": "|bias|", "sd_pts": "SD", "rmse_pts": "RMSE"}
    for i, arm in enumerate(arms):
        for j, reg in enumerate(regs):
            ax = axes[i][j]
            g = rs[(rs["dataset"] == "all") & (rs["regime"] == reg) & (rs["arm"] == arm) & (rs["kind"] == kind)]
            g = g.set_index("estimator").reindex(names)
            for col in ("abs_bias_pts", "sd_pts", "rmse_pts"):  # log scale: raw DR's SD spans orders of magnitude
                ax.plot(x, np.maximum(g[col].to_numpy(), 1e-3), "-o", color=colors[col], markersize=4, linewidth=1.8,
                        label=labels[col])
                rows += [{"arm": arm, "regime": reg, "lambda": lam, "metric": col, "value": v}
                         for lam, v in zip(lams, g[col].to_numpy())]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(x, ["0"] + [f"{v:g}" for v in HARMONIC_GRID], fontsize=7)
            ax.minorticks_off()
            ax.set_title(f"{reg}, {arm}", fontsize=9)
            if j == 0:
                ax.set_ylabel("CTR points (log; floor 0.001)")
            if i == len(arms) - 1:
                ax.set_xlabel("harmonic λ")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Harmonic DR at fixed {kind} policies: exact |bias|, SD and RMSE against λ (n = 25,000, fixed q̂; "
                 "mean over worlds)", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, f"fig_lambda_curve_{kind}", pd.DataFrame(rows))
    plt.close(fig)


def stage_a_md(rs, checks, supp, delta, adapt, info) -> str:
    out = ["# Stage A: fixed-policy oracle evaluator diagnostics", "",
           f"{info['worlds']} worlds, {info['replicates']} fresh logs of 25,000 rows each. CTR points; means over "
           "worlds [95% CI]. Linear estimators: exact moments; nonlinear: the replicates.", ""]
    out += ["## Replicates against the exact moments", "",
            f"Bounded-weight estimators passing (|z| < 4, exact variance in the χ² interval): "
            f"{int(checks.loc[checks['bounded'], 'ok'].sum())} of {int(checks['bounded'].sum())}.", ""]
    for kind in ("softmax", "greedy"):
        for arm in ("opc", "opc_raw", "likelihood", "likelihood_calib", "value_star", "source"):
            g = rs[(rs["dataset"] == "all") & (rs["kind"] == kind) & (rs["arm"] == arm)]
            if g.empty:
                continue
            out += [f"## {arm}, {kind}: bias / SD / RMSE (points) by regime", "",
                    "| estimator | " + " | ".join(r for r in REGIME_ORDER if (g["regime"] == r).any()) + " |",
                    "|---|" + "---|" * g["regime"].nunique()]
            for e in ESTIMATORS:
                cells = []
                for reg in REGIME_ORDER:
                    v = g[(g["regime"] == reg) & (g["estimator"] == e)]
                    if (g["regime"] == reg).any():
                        cells.append("–" if v.empty else f"{v['bias_pts'].iloc[0]:+.2f} / {v['sd_pts'].iloc[0]:.2f} / "
                                     f"{v['rmse_pts'].iloc[0]:.2f}")
                out.append(f"| {e} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["## Support (mean over worlds)", ""]
    s = supp.copy()
    s[["arm", "kind"]] = pd.DataFrame([_arm(p) for p in s["policy"]], index=s.index)
    t = s.groupby(["regime", "kind", "arm"])[["S2", "ess_fraction", "lowprop_0.001", "mass_w_gt_10", "mass_w_gt_100",
                                              "qhat_rmse", "a2_lambda_population"]].median().reset_index()
    out += ["Medians over worlds.", "", _md(t.round(4)), ""]
    dl = delta.groupby(["regime", "estimator"])["paired_over_independent"].median().unstack().round(3).reset_index()
    out += ["## Paired against independent improvement estimate: variance ratio (median over worlds and policies)", "",
            _md(dl), ""]
    if adapt is not None and not adapt.empty:
        a = adapt.groupby("arm")[["err_selected_pts", "err_all_trials_pts", "exact_clip10_bias_selected_pts",
                                   "adaptivity_pts"]].mean().round(3).reset_index()
        out += ["## Adaptivity of selection (validation estimate − truth, points; mean over worlds)", "", _md(a), ""]
    return "\n".join(out)


def stage_a_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summ, supp, reps = load_stage_a(Path(args.run), args.replicates)
    summ.to_csv(out / "table_world_policy_estimator.csv", index=False, float_format="%.8g")
    supp.to_csv(out / "table_world_policy_support.csv", index=False, float_format="%.8g")
    lam = reps.groupby(["dataset", "regime", "seed", "policy"])[["a2_lambda", "a3_lambda", "w_max", "w_q99", "w_q999",
                                                                 "ess_kish"]].median().reset_index()
    lam.to_csv(out / "table_world_policy_log_stats.csv", index=False, float_format="%.6g")
    rs = regime_summary(summ)
    rs.to_csv(out / "regime_summary.csv", index=False, float_format="%.6g")
    checks = replicate_checks(summ)
    checks.to_csv(out / "replicate_checks.csv", index=False, float_format="%.4g")
    delta = delta_table(summ)
    delta.to_csv(out / "delta_paired_vs_independent.csv", index=False, float_format="%.6g")
    adapt = None
    if args.trials:
        clip = summ[(summ["estimator"] == "dr_clip10") & ~summ["policy"].str.endswith("_g")]
        cb = {(r.dataset, r.label, int(r.seed), r.policy): float(r.exact_bias) for r in clip.itertuples()}
        adapt = adaptivity(Path(args.trials), summ[["dataset", "label", "seed"]].drop_duplicates(), cb)
        adapt.to_csv(out / "adaptivity.csv", index=False, float_format="%.6g")
    fig_lambda_curve(rs, out, kind="softmax")
    fig_lambda_curve(rs, out, arms=("opc", "likelihood_calib", "value_star"), kind="greedy")
    info = {"worlds": int(summ[["dataset", "label", "seed"]].drop_duplicates().shape[0]),
            "replicates": int(args.replicates),
            "bounded_checks_passed": int(checks.loc[checks["bounded"], "ok"].sum()),
            "bounded_checks": int(checks["bounded"].sum())}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    (out / "tables.md").write_text(stage_a_md(rs, checks, supp, delta, adapt, info))
    print(f"wrote {out}: {info}")


# ----------------------------------------------------------------------------------------------------------- Stage B
BASELINES = {"shared_lr_likelihood": "B1", "shared_lr_likelihood_calib": "B2", "shared_lr_opc": "B3",
             "shared_lr_opc_raw": "B4", "shared_lr_opc_oq": "B5"}
COMPARATORS = (("B3", "shared_lr_opc"), ("B1", "shared_lr_likelihood"), ("B2", "shared_lr_likelihood_calib"))
RULE_COLS = {"native": "native_gain", "best": "best_gain", "mean": "trial_mean_gain"}


def family_of(arm: str) -> str:
    if arm in BASELINES:
        return BASELINES[arm]
    short = arm.replace("shared_lr_opc_", "")
    if short.startswith("h") and short[1:2].isdigit() or short in ("a2", "a3"):
        return "A"
    if short.startswith(("s2b", "s2c", "klb")):
        return "B"
    if short.startswith(("crm", "hcrm", "dcrm")):
        return "C"
    if short in ("d1", "d2"):
        return "D"
    return "?"


def stage_b_tables(runs) -> dict:
    """Per world and arm: the selections (native / best / mean of the trials) as gains over the logger's greedy value
    (TrueGain, points), the recovery fraction, divergences and the extension's diagnostics; the paired differences
    against B3, B1 and B2 per world and rule."""
    from training.analyze_shared_objectives import load_summaries, load_trials, selections

    frames = []  # every arm present in the runs (load_trials needs an arm list)
    for run in runs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if (cond / "summary_metrics.csv").exists():
                frames.append(pd.read_csv(cond / "summary_metrics.csv", usecols=["method"])["method"])
    arms = sorted(set(pd.concat(frames)) if frames else set())
    t = load_trials(*runs, arms=arms)
    s = load_summaries(*runs, arms=arms)
    sel = selections(t)
    sel["family"] = sel["arm"].map(family_of)
    sel["regime"] = sel["bias"].map(REGIMES)
    ag = s.drop_duplicates(["dataset", "bias", "seed"]).set_index(["dataset", "bias", "seed"])["available_gain"]
    sel["available_gain_pts"] = [100 * ag.get((r.dataset, r.bias, r.seed), np.nan) for r in sel.itertuples()]
    sel["recovery_native"] = np.where(sel["available_gain_pts"] >= 0.5, sel["native_gain"] / sel["available_gain_pts"],
                                      np.nan)
    diag = t[t["is_best_in_run"].astype(bool)].set_index(["dataset", "bias", "seed", "method"])
    for col in ("diag_ext_lambda_final", "diag_ext_lambda_mean", "diag_ext_s2_train_final", "diag_ext_nu_final",
                "trial_time_s"):
        if col in diag:
            sel[col] = [diag[col].get((r.dataset, r.bias, r.seed, r.arm), np.nan) for r in sel.itertuples()]
    sel["trial_time_mean_s"] = [float(t.loc[(t["dataset"] == r.dataset) & (t["bias"] == r.bias) & (t["seed"] == r.seed)
                                            & (t["method"] == r.arm), "trial_time_s"].mean()) for r in sel.itertuples()]
    keys = ["dataset", "bias", "seed"]
    idx = sel.set_index(keys + ["arm"])
    rows = []
    for (ds, b, seed), g in sel.groupby(keys):
        for arm in g["arm"]:
            for cname, carm in COMPARATORS:
                if arm == carm or (ds, b, seed, carm) not in idx.index:
                    continue
                for rule, col in RULE_COLS.items():
                    rows.append({"dataset": ds, "bias": b, "regime": REGIMES.get(b), "seed": seed, "arm": arm,
                                 "family": family_of(arm), "vs": cname, "rule": rule,
                                 "d": float(idx.at[(ds, b, seed, arm), col]) - float(idx.at[(ds, b, seed, carm), col])})
    return {"worlds": sel, "paired": pd.DataFrame(rows), "trials": t}


def paired_summary(pw: pd.DataFrame) -> pd.DataFrame:
    """Mean [95% CI] over worlds and the worlds where the arm is higher: per arm, comparator and rule, pooled, per
    regime, per dataset and per regime × dataset."""
    out = []
    panels = [("all", "all", pw)] + [("all", reg, g) for reg, g in pw.groupby("regime")] \
        + [(ds, "all", g) for ds, g in pw.groupby("dataset")] \
        + [(ds, reg, g) for (ds, reg), g in pw.groupby(["dataset", "regime"])]
    for ds, reg, g in panels:
        for (arm, vs, rule), gg in g.groupby(["arm", "vs", "rule"]):
            m, lo, hi, n = mean_ci(gg["d"])
            out.append({"dataset": ds, "regime": reg, "arm": arm, "family": family_of(arm), "vs": vs, "rule": rule,
                        "mean": m, "lo": lo, "hi": hi, "worlds": n, "higher": int((gg["d"] > 0).sum())})
    return pd.DataFrame(out)


def family_decision(worlds: pd.DataFrame, paired: pd.DataFrame, family: str) -> dict:
    """§11's rule: among the family's stable variants (≤ 5% diverged trials), the highest mean native TrueGain over
    the pilot worlds; if the best two differ by less than the SE of their paired difference, the earlier in the fixed
    order. The constraint check of the hard S2 arms reads the oracle evaluation's exact S2 (reported separately)."""
    order = {"A": ["shared_lr_opc_a2", "shared_lr_opc_a3", "shared_lr_opc_h0.005", "shared_lr_opc_h0.01",
                   "shared_lr_opc_h0.03", "shared_lr_opc_h0.2"],
             "B": ["shared_lr_opc_s2c50", "shared_lr_opc_s2c20", "shared_lr_opc_s2c10", "shared_lr_opc_s2c5",
                   "shared_lr_opc_s2b0.0003", "shared_lr_opc_s2b0.001", "shared_lr_opc_s2b0.003", "shared_lr_opc_s2b0.01",
                   "shared_lr_opc_klb0.001", "shared_lr_opc_klb0.003"],
             "C": ["shared_lr_opc_crm0.5", "shared_lr_opc_crm1", "shared_lr_opc_crm2"],
             "D": ["shared_lr_opc_d1", "shared_lr_opc_d2"]}[family]
    w = worlds[worlds["family"] == family]
    if w.empty:
        return {"family": family, "decision": None}
    stats_ = w.groupby("arm").agg(gain=("native_gain", "mean"), worlds=("native_gain", "size"),
                                  diverged=("diverged", "sum"), trials=("trials", "sum"))
    stats_["diverged_share"] = stats_["diverged"] / stats_["trials"]
    stable = stats_[stats_["diverged_share"] <= 0.05].sort_values("gain", ascending=False)
    if stable.empty:
        return {"family": family, "decision": None, "reason": "no stable variant"}
    best = stable.index[0]
    pick, reason = best, "highest mean native TrueGain"
    if len(stable) > 1:
        second = stable.index[1]
        a = w[w["arm"] == best].set_index(["dataset", "bias", "seed"])["native_gain"]
        b = w[w["arm"] == second].set_index(["dataset", "bias", "seed"])["native_gain"]
        d = (a - b).dropna()
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.inf
        if float(d.mean()) < se:
            pick = min((best, second), key=lambda m: order.index(m) if m in order else 99)
            reason = f"best two within one paired SE ({float(d.mean()):.3f} < {se:.3f}): the earlier in the fixed order"
    b3 = paired[(paired["arm"] == pick) & (paired["vs"] == "B3") & (paired["rule"] == "native")]["d"]
    m, lo, hi, n = mean_ci(b3)
    return {"family": family, "decision": pick, "reason": reason,
            "ranking": {a: round(float(g), 4) for a, g in stable["gain"].items()},
            "vs_B3_native": {"mean": m, "lo": lo, "hi": hi, "worlds": n}}


def stage_b_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = stage_b_tables(args.runs)
    res["worlds"].to_csv(out / "table_worlds.csv", index=False, float_format="%.6g")
    res["paired"].to_csv(out / "paired_worlds.csv", index=False, float_format="%.6g")
    ps = paired_summary(res["paired"])
    ps.to_csv(out / "paired_summary.csv", index=False, float_format="%.6g")
    fams = sorted(set(res["worlds"]["family"]) & {"A", "B", "C", "D"})
    dec = {f: family_decision(res["worlds"], res["paired"], f) for f in fams}
    (out / "decisions.json").write_text(json.dumps(dec, indent=2, default=float))
    w = res["worlds"]
    lines = ["# Stage B pilots", "", f"{w[['dataset', 'bias', 'seed']].drop_duplicates().shape[0]} worlds; "
             f"{len(res['trials'])} trials; points; TrueGain = the selected policy's greedy gain over the source's.", ""]
    g = w.groupby(["family", "arm"]).agg(native=("native_gain", "mean"), best=("best_gain", "mean"),
                                         mean=("trial_mean_gain", "mean"), recovery=("recovery_native", "mean"),
                                         diverged=("diverged", "sum"), sec_per_trial=("trial_time_mean_s", "mean"))
    lines += ["## Mean over worlds", "", _md(g.round(3).reset_index()), ""]
    for reg in ("R1", "R2", "R3"):
        gr = w[w["regime"] == reg].groupby("arm")["native_gain"].mean().round(2)
        lines += [f"### {reg}: native TrueGain per arm", "", _md(gr.reset_index()), ""]
    for vs in ("B3", "B1", "B2"):
        q = ps[(ps["vs"] == vs) & (ps["rule"] == "native") & (ps["dataset"] == "all")]
        q = q.pivot_table(index="arm", columns="regime", values="mean").round(2)
        lines += [f"## arm − {vs} (native; mean over worlds, by regime)", "", _md(q.reset_index()), ""]
    lines += ["## Decisions (§11)", "", "```", json.dumps(dec, indent=1, default=float), "```", ""]
    (out / "tables.md").write_text("\n".join(lines))
    print(f"wrote {out}: {json.dumps({f: d.get('decision') for f, d in dec.items()})}")


# ------------------------------------------------------------------------------------------------------- gradients
def gradients_main(args) -> None:
    """§10: per world, state and estimator the gradient metrics (``ae_gradients``), and their means per state."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(Path(args.run).glob("dataset=*/gradients_*.json")):
        r = json.loads(f.read_text())
        tags = _world_tags(Path(r["world"]).name)
        for st, v in r["states"].items():
            for e, m in v.items():
                if isinstance(m, dict):
                    rows.append({**tags, "arm": r["arm"], "t_mid": r["t_mid"], "state": st, "V": v["V"],
                                 "lam_a2_median": v.get("lam_a2_median"), "estimator": e, **m})
    d = pd.DataFrame(rows)
    d.to_csv(out / "table_world_state_estimator.csv", index=False, float_format="%.6g")
    cols = ["rel_bias", "variance", "mse", "snr", "cos_mean", "cos_single"]
    g = d.groupby(["state", "estimator"])[cols].mean().reset_index()
    g.to_csv(out / "summary_by_state.csv", index=False, float_format="%.6g")
    lines = ["# Oracle gradient diagnostics (§10, §10a)", "",
             f"{d[['dataset', 'label', 'seed']].drop_duplicates().shape[0]} worlds; harmonic OPC's (B3) selected pilot "
             "policy, the intermediate point and the source; means over worlds. rel_bias = ‖E ĝ − g*‖ / ‖g*(source)‖; "
             "snr = ‖E ĝ‖ / sqrt(trace Var ĝ); cos_mean = cos(E ĝ, g*); cos_single = mean cos(ĝ, g*) over datasets.", ""]
    for st in ("source", "intermediate", "final"):
        lines += [f"## {st}", "", _md(g[g["state"] == st].drop(columns="state").round(4)), ""]
    (out / "tables.md").write_text("\n".join(lines))
    print(f"wrote {out}: {len(d)} rows")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage_a")
    a.add_argument("--run", required=True)
    a.add_argument("--replicates", type=int, required=True)
    a.add_argument("--trials", default=None)
    a.add_argument("--out", required=True)
    b = sub.add_parser("stage_b")
    b.add_argument("--runs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    gr = sub.add_parser("gradients")
    gr.add_argument("--run", required=True)
    gr.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"stage_a": stage_a_main, "stage_b": stage_b_main, "gradients": gradients_main}[args.cmd](args)


if __name__ == "__main__":
    main()
