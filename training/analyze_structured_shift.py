"""Analysis of the structured scenario-shift study (docs/structured_scenario_shift_study.md §6, §8-§10, §7's gate).

- ``population``: the population optima of every world (training/ss_population.py) as one table, the sanity checks
  A-C, and the mismatches M_L, M_calib, M_harm by shift × response, per dataset and pooled.
- ``compare``: the 25k grid. Per world and arm the gains under every rule (native, common, oracle-best, the mean of
  the 20 trials), OPC − likelihood and the other paired differences under every rule, the decomposition
  M + (finite sample + optimization) + S of the regime study §6 against each arm's population optimum, the weight and
  ESS diagnostics, the correction's size, data identity and divergences; the mechanism (OPC − likelihood against M_L
  and M_harm) and the expansion gate of §7.

All values are greedy CTR points unless named otherwise. A world is the unit of every interval (paired by world).

    python -m training.analyze_structured_shift population --run artifacts/full_study/run_ss_population --out OUT
    python -m training.analyze_structured_shift compare --population artifacts/full_study/run_ss_population \
        --runs artifacts/full_study/run_ss_main_25k --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from training.analyze_cause_fair import _md, mean_ci
from training.analyze_shared_objectives import _plt, _save, load_summaries, load_trials, selections
from training.ss_population import WARN_POINTS, world_dir_name
from utils.structured_shift import parse_structured

LEVELS = ("none", "moderate", "strong")
DATASETS = ("ml", "kuairand", "anime")
WORLD = ["dataset", "bias", "seed"]
SHORT = ("likelihood", "likelihood_calib", "opc", "opc_raw", "opc_oq")
ARMS = {fam: tuple(f"shared_{fam}_{s}" for s in SHORT) for fam in ("lr", "lrg")}
ARM_NAMES = {"likelihood": "likelihood (ordinary head)", "likelihood_calib": "likelihood (calibration-aware head)",
             "opc": "harmonic OPC", "opc_raw": "raw DR", "opc_oq": "harmonic OPC, oracle q"}
# the population optimum each arm's objective estimates (§6): harmonic DR with the oracle q is V itself
POPULATION_OF = {"likelihood": "likelihood", "likelihood_calib": "calib", "opc": "harmonic", "opc_raw": "value",
                 "opc_oq": "value"}
OPTIMA = ("source", "truth", "value", "likelihood", "calib", "harmonic")
RULES = {"native": "native_gain", "common": "common_gain", "best": "best_gain", "mean": "trial_mean_gain"}
PAIRS = (("opc", "likelihood"), ("opc", "likelihood_calib"), ("opc_raw", "likelihood"), ("opc_oq", "likelihood"),
         ("likelihood_calib", "likelihood"), ("opc_oq", "opc"), ("opc_raw", "opc"))
DIVERGENCE_LIMIT = 0.05  # §11: more than 5% of a cell's trials
GATE_MLC = 0.5  # §7 gate (a): pooled M_L − M_calib, points
CONFIG_COLUMNS = ("param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay", "param_anchor_lambda")
WORLD_COLUMNS = ("gamma", "shift_corr", "top1_agreement", "top10_overlap", "shift_frobenius", "available_gain",
                 "source_greedy_ctr", "target_greedy_ctr", "sigma_alpha", "sigma_beta", "kappa", "click_intercept",
                 "best_item_ctr", "reference_ctr", "logging_temperature", "logger_greedy_share", "logger_greedy_ctr")
DIAG_COLUMNS = ("diag_w_max", "diag_w_share_gt1", "diag_w_share_gt10", "diag_w_share_gt100", "diag_dm_mean",
                "diag_correction_mean", "diag_norm_UV", "diag_norm_U", "diag_norm_V", "diag_norm_w_alpha",
                "diag_norm_w_beta", "diag_calib_gamma", "train_rows_sha1", "val_rows_sha1", "train_click_sum",
                "val_click_sum")
# Okabe-Ito, as the repository's other reports (validated for colour-vision deficiency): the ordinary likelihood
# pink, the calibration-aware likelihood orange, harmonic OPC blue, raw DR sky blue, the oracle-q arm bluish green
COLORS = {"likelihood": "#CC79A7", "likelihood_calib": "#E69F00", "opc": "#0072B2", "opc_raw": "#56B4E9",
          "opc_oq": "#009E73"}
MISMATCH_COLORS = {"M_L": COLORS["likelihood"], "M_calib": COLORS["likelihood_calib"], "M_harm": COLORS["opc"]}
MISMATCH_NAMES = {"M_L": "M_L (ordinary likelihood)", "M_calib": "M_calib (calibration-aware)",
                  "M_harm": "M_harm (harmonic DR, q̂_∞)"}
MARKERS = {"ml": "o", "kuairand": "s", "anime": "^"}


def _levels(label: str) -> dict:
    p = parse_structured(label)
    return {"shift": p["shift"], "response": p["response"], "gated": p["gated"]}


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in the study's order: dataset, shift, response, seed."""
    out = df.copy()
    for col, order in (("dataset", DATASETS), ("shift", LEVELS), ("response", LEVELS)):
        if col in out:
            out[col] = pd.Categorical(out[col], [v for v in order if v in set(out[col])] +
                                      sorted(set(out[col]) - set(order)), ordered=True)
    keys = [c for c in ("dataset", "shift", "response", "seed") if c in out]
    out = out.sort_values(keys, kind="stable")
    for col in ("dataset", "shift", "response"):
        if col in out:
            out[col] = out[col].astype(str)
    return out.reset_index(drop=True)


def _ci_row(x) -> dict:
    m, lo, hi, n = mean_ci(x)
    return {"mean": m, "lo": lo, "hi": hi, "n": n}


# ---------------------------------------------------------------------------------------------------- population
def population_table(run: Path) -> pd.DataFrame:
    """One row per finished world of the population run: the optima's greedy and stochastic values, the mismatches,
    q̂_∞'s error, the checks and the world's statistics."""
    rows = []
    for f in sorted(Path(run).glob("dataset=*/population.json")):
        tags = dict(p.split("=", 1) for p in f.parent.name.split("__"))
        r = json.loads(f.read_text())
        w = json.loads((f.parent / "world.json").read_text())
        row = {"dataset": tags["dataset"], "bias": tags["bias"], "seed": int(tags["seed"]),
               "logger_greedy_share": float(tags["lgs"]), **_levels(tags["bias"])}
        for k in OPTIMA:
            row[f"Vg_{k}"] = float(r[k]["greedy"])
            row[f"V_{k}"] = float(r[k]["V"])
        row.update({k: float(v) for k, v in r["mismatch"].items()})
        row["M_L_minus_M_calib"] = row["M_L"] - row["M_calib"]
        row["M_harm_minus_M_L"] = row["M_harm"] - row["M_L"]
        row["qinf_rmse_logging"] = float(r["qhat_inf_rmse"]["logging"])
        row["qinf_rmse_target"] = float(r["qhat_inf_rmse"]["target"])
        for k, v in r["checks"].items():
            row[f"check_{k}"] = v
        row["lik_click_intercept"] = float(r["likelihood"].get("click_intercept", np.nan))
        for k in ("norm_UV", "norm_w_alpha", "norm_w_beta", "gamma"):
            if k in r["calib"]:
                row[f"calib_{k}"] = float(r["calib"][k])
        for k in ("gamma", "shift_corr", "score_corr_q10", "score_corr_min", "top1_agreement", "top10_overlap",
                  "shift_frobenius", "sigma_alpha", "sigma_beta", "kappa", "click_intercept",
                  "click_intercept_homogeneous", "best_item_ctr", "best_item_ctr_homogeneous", "reference_ctr",
                  "logging_temperature", "logger_effective_items", "logger_greedy_ctr", "logging_ctr"):
            if k in w:
                row[f"world_{k}"] = float(w[k])
        row["world_shift_singular_values"] = " ".join(f"{s:.4g}" for s in w.get("shift_singular_values", []))
        for stat, key in (("log_alpha", "sd"), ("alpha_stats", "q10"), ("alpha_stats", "q90"), ("beta_stats", "sd"),
                          ("beta_stats", "q10"), ("beta_stats", "q90")):
            if isinstance(w.get(stat), dict):
                row[f"world_{stat}_{key}"] = float(w[stat][key])
        for k, v in (w.get("click_sanity") or {}).items():
            if k != "pathological":
                row[f"world_{k}"] = float(v)
        fin = f.parent / "qhat_finite_n25000.json"  # the 25k grid's reward model (ss_population --finite-qhat)
        if fin.exists():
            q = json.loads(fin.read_text())
            for k in ("crossfit_logging", "crossfit_target", "full_logging", "full_target"):
                row[f"qhat25k_rmse_{k}"] = float(q[k])
        row["seconds"] = float(r.get("seconds", np.nan))
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no population.json under {run}")
    return _ordered(pd.DataFrame(rows))


POPULATION_CELL_COLUMNS = ("M_L", "M_calib", "M_harm", "M_L_minus_M_calib", "M_harm_minus_M_L", "optimizer_gap",
                           "available_gain", "qinf_rmse_logging", "qinf_rmse_target", "qhat25k_rmse_crossfit_logging",
                           "qhat25k_rmse_crossfit_target")


def population_cells(pop: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over worlds per shift × response, pooled over datasets ("all") and per dataset, and per shift and
    per response pooled over the other factor."""
    rows = []

    def add(scope, ds, shift, response, g):
        r = {"scope": scope, "dataset": ds, "shift": shift, "response": response, "worlds": len(g)}
        for col in POPULATION_CELL_COLUMNS:
            if col not in g:
                continue
            c = _ci_row(g[col])
            r[col], r[col + "_lo"], r[col + "_hi"] = c["mean"], c["lo"], c["hi"]
        r["share_M_harm_ge_M_L"] = float((g["M_harm"] >= g["M_L"]).mean())
        rows.append(r)

    for (shift, response), g in pop.groupby(["shift", "response"]):
        add("cell", "all", shift, response, g)
        for ds, gd in g.groupby("dataset"):
            add("cell", ds, shift, response, gd)
    for shift, g in pop.groupby("shift"):
        add("shift", "all", shift, "all", g)
    for response, g in pop.groupby("response"):
        add("response", "all", "all", response, g)
    for ds, g in pop.groupby("dataset"):
        add("dataset", ds, "all", "all", g)
    add("all", "all", "all", "all", pop)
    return pd.DataFrame(rows)


def population_checks(pop: pd.DataFrame) -> pd.DataFrame:
    """The worlds where a §6 check fails (an empty frame: every check passes)."""
    cols = [c for c in pop if c.startswith("check_")]
    bad = pop[cols].apply(lambda s: s.map(lambda v: v is not None and not pd.isna(v) and not bool(v)))
    return pop.loc[bad.any(axis=1), WORLD + ["shift", "response"] + cols + ["M_L", "M_calib", "optimizer_gap"]]


def population_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pop = population_table(Path(args.run))
    pop.to_csv(out / "table_population.csv", index=False, float_format="%.6g")
    cells = population_cells(pop)
    cells.to_csv(out / "population_cells.csv", index=False, float_format="%.6g")
    fails = population_checks(pop)
    fails.to_csv(out / "population_check_failures.csv", index=False)
    fig_population(pop, cells, out)
    (out / "population.md").write_text(population_md(pop, cells, fails))
    print(f"wrote {out}: {len(pop)} worlds, {len(fails)} with a failed check")


# ------------------------------------------------------------------------------------------------------- 25k grid
def _short(arm: str) -> str:
    return arm.split("_", 2)[2]


def world_table(t: pd.DataFrame, s: pd.DataFrame, pop: pd.DataFrame | None) -> pd.DataFrame:
    """Per world and arm: the selections, the world statistics, the diagnostics of the selected trial, the arm's
    population optimum and the decomposition (§10)."""
    sel = selections(t)
    sel["short"] = sel["arm"].map(_short)
    sel = pd.concat([sel, pd.DataFrame([_levels(b) for b in sel["bias"]], index=sel.index)], axis=1)
    native = t[t["is_best_in_run"].astype(bool)].set_index(WORLD + ["method"])
    for col in ("ess", "ess_train", "ess_raw", "diag_norm_UV", "diag_anchor_R", "diag_norm_w_alpha", "diag_norm_w_beta",
                "diag_calib_gamma"):
        if col in native:
            sel[f"native_{col.replace('diag_', '')}"] = [native[col].get(tuple(k), np.nan)
                                                         for k in sel[WORLD + ["arm"]].itertuples(index=False)]
    if not s.empty:
        keep = [c for c in list(WORLD_COLUMNS) + list(DIAG_COLUMNS) if c in s]
        srow = s.set_index(WORLD + ["method"])[keep]
        srow.columns = [c if c in WORLD_COLUMNS else f"arm_{c}" for c in keep]
        sel = sel.join(srow, on=WORLD + ["arm"])
    if pop is not None and not pop.empty:
        p = pop.set_index(WORLD)
        cols = [c for c in [f"Vg_{k}" for k in OPTIMA] + ["M_L", "M_calib", "M_harm", "M_L_minus_M_calib",
                                                          "optimizer_gap", "qinf_rmse_logging", "qinf_rmse_target",
                                                          "qhat25k_rmse_crossfit_logging",
                                                          "qhat25k_rmse_crossfit_target"] if c in p]
        sel = sel.join(p[cols].add_prefix("pop_"), on=WORLD)
        obj = np.array([sel.at[i, f"pop_Vg_{POPULATION_OF[a]}"] if f"pop_Vg_{POPULATION_OF[a]}" in sel else np.nan
                        for i, a in zip(sel.index, sel["short"])], dtype=float)
        vstar = sel["pop_Vg_value"].to_numpy(dtype=float)
        sel["V_obj_star"] = obj
        sel["M"] = 100 * (vstar - obj)
        sel["EO"] = 100 * (obj - sel["best_V_greedy"])
        sel["S_native"] = 100 * (sel["best_V_greedy"] - sel["native_V_greedy"])
        sel["S_common"] = 100 * (sel["best_V_greedy"] - sel["common_V_greedy"])
        sel["gap_native"] = 100 * (vstar - sel["native_V_greedy"])
        sel["gap_common"] = 100 * (vstar - sel["common_V_greedy"])
        sel["value_star_gain"] = 100 * (vstar - sel["V_logger_greedy"])
        # the population source (the logger's ranking) against the trainer's logger greedy value: the identity
        # that lets population and trial values be subtracted
        sel["logger_check"] = 100 * (sel["pop_Vg_source"] - sel["V_logger_greedy"])
    return _ordered(sel)


def data_identity(t: pd.DataFrame, s: pd.DataFrame) -> pd.DataFrame:
    """Per world: the arms present, whether they saw identical training and validation rows (row hashes), and whether
    trial k trained the same configuration in every arm (§12 test 9)."""
    rows = []
    for key, g in t.groupby(WORLD):
        cfg = g.groupby("trial_number")[[c for c in CONFIG_COLUMNS if c in g]].nunique(dropna=False)
        r = dict(zip(WORLD, key))
        r.update({"arms": g["method"].nunique(), "trials_per_arm_min": int(g.groupby("method").size().min()),
                  "configs_paired": bool((cfg <= 1).all().all())})
        if not s.empty:
            sw = s[(s["dataset"] == key[0]) & (s["bias"] == key[1]) & (s["seed"] == key[2])]
            for col in ("train_rows_sha1", "val_rows_sha1"):
                if col in sw:
                    r[f"{col}_unique"] = int(sw[col].nunique())
        rows.append(r)
    out = pd.DataFrame(rows)
    out["identical"] = out["configs_paired"] & out.get("train_rows_sha1_unique", 1).eq(1) \
        & out.get("val_rows_sha1_unique", 1).eq(1)
    return _ordered(pd.concat([out, pd.DataFrame([_levels(b) for b in out["bias"]])], axis=1))


def divergence_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per shift × response cell and arm: the share of diverged trials; ``over`` flags §11's 5% limit."""
    t = t.assign(**pd.DataFrame([_levels(b) for b in t["bias"]], index=t.index)[["shift", "response"]])
    d = t.get("diverged", pd.Series(False, index=t.index)).astype(bool)
    g = t.assign(div=d).groupby(["shift", "response", "method"])["div"].agg(["sum", "size"]).reset_index()
    g["share"] = g["sum"] / g["size"]
    g["over"] = g["share"] > DIVERGENCE_LIMIT
    return _ordered(g.rename(columns={"sum": "diverged", "size": "trials"}))


def paired_worlds(wt: pd.DataFrame) -> pd.DataFrame:
    """Per world, pair and rule: a − b (CTR points), with the world's mismatches."""
    rows = []
    idx = wt.set_index(WORLD + ["short"])
    for key, g in wt.groupby(WORLD):
        base = {**dict(zip(WORLD, key)), "shift": g["shift"].iloc[0], "response": g["response"].iloc[0]}
        for col in ("pop_M_L", "pop_M_calib", "pop_M_harm", "available_gain"):
            base[col] = float(g[col].iloc[0]) if col in g else np.nan
        present = set(g["short"])
        for a, b in PAIRS:
            if a not in present or b not in present:
                continue
            for rule, col in RULES.items():
                rows.append({**base, "a": a, "b": b, "rule": rule,
                             "d": float(idx.loc[(*key, a), col]) - float(idx.loc[(*key, b), col])})
    return _ordered(pd.DataFrame(rows))


def _scopes(df: pd.DataFrame):
    """The panels every summary is given for: (scope, dataset, shift, response, rows)."""
    for (shift, response), g in df.groupby(["shift", "response"]):
        yield "cell", "all", shift, response, g
        for ds, gd in g.groupby("dataset"):
            yield "cell", ds, shift, response, gd
    for shift, g in df.groupby("shift"):
        yield "shift", "all", shift, "all", g
        for ds, gd in g.groupby("dataset"):
            yield "shift", ds, shift, "all", gd
    for response, g in df.groupby("response"):
        yield "response", "all", "all", response, g
        for ds, gd in g.groupby("dataset"):
            yield "response", ds, "all", response, gd
    for ds, g in df.groupby("dataset"):
        yield "dataset", ds, "all", "all", g
    yield "all", "all", "all", "all", df


def paired_cells(pw: pd.DataFrame) -> pd.DataFrame:
    """a − b paired by world: mean, 95% CI, worlds and the worlds where a is higher, per panel of ``_scopes``."""
    rows = []
    for scope, ds, shift, response, g in _scopes(pw):
        for (a, b, rule), gp in g.groupby(["a", "b", "rule"]):
            c = _ci_row(gp["d"])
            rows.append({"scope": scope, "dataset": ds, "shift": shift, "response": response, "a": a, "b": b,
                         "rule": rule, "a_minus_b": c["mean"], "ci_lo": c["lo"], "ci_hi": c["hi"], "worlds": c["n"],
                         "a_higher": int((gp["d"] > 0).sum())})
    return pd.DataFrame(rows)


ARM_CELL_COLUMNS = ("native_gain", "common_gain", "best_gain", "trial_mean_gain", "value_star_gain", "M", "EO",
                    "S_native", "S_common", "gap_native", "gap_common", "native_anchor_R", "native_norm_UV",
                    "native_changed_share", "native_ess", "regret_native")


def arm_cells(wt: pd.DataFrame) -> pd.DataFrame:
    """Per panel of ``_scopes`` and arm: mean and 95% CI of the gains, the decomposition and the diagnostics."""
    rows = []
    for scope, ds, shift, response, g in _scopes(wt):
        for arm, ga in g.groupby("short"):
            r = {"scope": scope, "dataset": ds, "shift": shift, "response": response, "arm": arm, "worlds": len(ga),
                 "diverged_trials": int(ga["diverged"].sum())}
            for col in ARM_CELL_COLUMNS:
                if col in ga and ga[col].notna().any():
                    c = _ci_row(ga[col])
                    r[col], r[col + "_lo"], r[col + "_hi"] = c["mean"], c["lo"], c["hi"]
            rows.append(r)
    return pd.DataFrame(rows)


def mechanism_table(pw: pd.DataFrame) -> pd.DataFrame:
    """§10 mechanism: per rule, OPC − likelihood against M_L, M_harm and M_L − M_harm over worlds (pooled and per
    dataset): Spearman ρ with its p-value, and the OLS slope with a 95% CI."""
    rows = []
    d = pw[(pw["a"] == "opc") & (pw["b"] == "likelihood")].copy()
    d["pop_M_L_minus_M_harm"] = d["pop_M_L"] - d["pop_M_harm"]
    for rule, gr in d.groupby("rule"):
        for ds, g in [("all", gr)] + list(gr.groupby("dataset")):
            for x in ("pop_M_L", "pop_M_harm", "pop_M_L_minus_M_harm", "available_gain"):
                ok = g[[x, "d"]].dropna()
                if len(ok) < 4 or ok[x].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(ok[x], ok["d"])
                lr = stats.linregress(ok[x], ok["d"])
                h = float(stats.t.ppf(0.975, len(ok) - 2) * lr.stderr)
                rows.append({"rule": rule, "dataset": ds, "x": x.replace("pop_", ""), "worlds": len(ok),
                             "spearman": float(rho), "spearman_p": float(p), "slope": float(lr.slope),
                             "slope_lo": float(lr.slope - h), "slope_hi": float(lr.slope + h),
                             "intercept": float(lr.intercept)})
    return pd.DataFrame(rows)


def expansion_gate(pc: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """§7's expansion gate per pooled shift × response cell: criteria (a)-(d) and whether the cell is expanded. (b) is a
    property of a shift level, so it marks every cell of that level."""
    pcell = pc[(pc["scope"] == "cell") & (pc["dataset"] == "all")]
    popc = cells[(cells["scope"] == "cell") & (cells["dataset"] == "all")].set_index(["shift", "response"])

    def diff(shift, response, a, b, rule="native"):
        g = pcell[(pcell["shift"] == shift) & (pcell["response"] == response) & (pcell["a"] == a)
                  & (pcell["b"] == b) & (pcell["rule"] == rule)]
        return g.iloc[0] if len(g) else None

    signs = {}
    for shift in LEVELS:
        s = [np.sign(r["a_minus_b"]) for resp in LEVELS if (r := diff(shift, resp, "opc", "likelihood")) is not None]
        signs[shift] = len(set(v for v in s if v != 0)) > 1
    rows = []
    for (shift, response), pr in popc.iterrows():
        ol = diff(shift, response, "opc", "likelihood")
        cl = diff(shift, response, "likelihood_calib", "likelihood")
        r = {"shift": shift, "response": response, "M_L": pr["M_L"], "M_calib": pr["M_calib"], "M_harm": pr["M_harm"],
             "M_L_minus_M_calib": pr["M_L_minus_M_calib"],
             "opc_minus_lik": np.nan if ol is None else ol["a_minus_b"],
             "calib_minus_lik": np.nan if cl is None else cl["a_minus_b"],
             "calib_minus_lik_lo": np.nan if cl is None else cl["ci_lo"],
             "calib_minus_lik_hi": np.nan if cl is None else cl["ci_hi"]}
        r["a"] = bool(pr["M_L_minus_M_calib"] >= GATE_MLC)
        r["b"] = bool(signs.get(shift, False))
        r["c"] = bool(pr["M_harm"] < pr["M_L"] and ol is not None and ol["a_minus_b"] < 0)
        r["d"] = bool(cl is not None and (cl["ci_lo"] > 0 or cl["ci_hi"] < 0))
        r["expand"] = r["a"] or r["b"] or r["c"] or r["d"]
        rows.append(r)
    return _ordered(pd.DataFrame(rows))


def harmonic_dominance(pop: pd.DataFrame, ac: pd.DataFrame) -> dict:
    """The gate's stop criterion: M_harm ≥ M_L in most worlds, and harmonic OPC's pooled 25k gap to θ_value* (native)
    the largest of the arms trained on the data (the oracle-q arm excluded)."""
    share = float((pop["M_harm"] >= pop["M_L"]).mean()) if len(pop) else np.nan
    pooled = ac[(ac["scope"] == "all") & (ac["dataset"] == "all")].set_index("arm")
    gaps = {a: float(pooled.at[a, "gap_native"]) for a in pooled.index if a != "opc_oq" and "gap_native" in pooled}
    largest = max(gaps, key=gaps.get) if gaps else None
    return {"share_worlds_M_harm_ge_M_L": share, "gap_native_pooled": gaps, "largest_gap_arm": largest,
            "dominant": bool(share > 0.5 and largest == "opc")}


def compare_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = list(args.arms) if args.arms else list(ARMS["lr"])
    t = load_trials(*args.runs, arms=arms)
    s = load_summaries(*args.runs, arms=arms)
    pop = population_table(Path(args.population)) if args.population else None
    if pop is not None:
        pop = pop[pop.set_index(WORLD).index.isin(t.set_index(WORLD).index.unique())]
    wt = world_table(t, s, pop)
    wt.to_csv(out / "table_worlds.csv", index=False, float_format="%.6g")
    ident = data_identity(t, s)
    ident.to_csv(out / "data_identity.csv", index=False)
    div = divergence_table(t)
    div.to_csv(out / "divergence.csv", index=False, float_format="%.6g")
    pw = paired_worlds(wt)
    pw.to_csv(out / "paired_worlds.csv", index=False, float_format="%.6g")
    pc = paired_cells(pw)
    pc.to_csv(out / "paired_cells.csv", index=False, float_format="%.6g")
    ac = arm_cells(wt)
    ac.to_csv(out / "arm_cells.csv", index=False, float_format="%.6g")
    info = {"worlds": int(t[WORLD].drop_duplicates().shape[0]), "trials": int(len(t)),
            "identical_worlds": int(ident["identical"].sum()), "cells_over_divergence": int(div["over"].sum())}
    gate = None
    if pop is not None and not pop.empty:
        cells = population_cells(pop)
        mech = mechanism_table(pw)
        mech.to_csv(out / "mechanism.csv", index=False, float_format="%.6g")
        gate = expansion_gate(pc, cells)
        gate.to_csv(out / "expansion_gate.csv", index=False, float_format="%.6g")
        info["harmonic_dominance"] = harmonic_dominance(pop, ac)
        info["logger_check_max_abs"] = float(np.nanmax(np.abs(wt["logger_check"]))) if "logger_check" in wt else None
        fig_paired(pc, out)
        fig_mechanism(pw, out)
        fig_decomposition(ac, out)
    (out / "compare_info.json").write_text(json.dumps(info, indent=2, default=float))
    (out / "tables.md").write_text(compare_md(wt, pc, ac, ident, div, gate, info))
    print(f"wrote {out}: {info['worlds']} worlds, {info['trials']} trials; identical {info['identical_worlds']}; "
          f"cells over the divergence limit {info['cells_over_divergence']}")


# --------------------------------------------------------------------------------------------------------- figures
def fig_population(pop: pd.DataFrame, cells: pd.DataFrame, out: Path) -> None:
    """One panel per shift level: the three mismatches against the response level, pooled mean and 95% CI over the
    worlds (dots: the worlds)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    c = cells[(cells["scope"] == "cell") & (cells["dataset"] == "all")]
    rows = []
    x = np.arange(len(LEVELS))
    for ax, shift in zip(axes, LEVELS):
        for j, (m, color) in enumerate(MISMATCH_COLORS.items()):
            off = (j - 1) * 0.16
            g = c[c["shift"] == shift].set_index("response").reindex(LEVELS)
            w = pop[pop["shift"] == shift]
            for i, resp in enumerate(LEVELS):
                vals = w.loc[w["response"] == resp, m]
                ax.scatter(np.full(len(vals), i + off), vals, s=9, color=color, alpha=0.35, linewidths=0)
            ok = g[m].notna().to_numpy()
            err = np.vstack([(g[m] - g[m + "_lo"]).to_numpy(), (g[m + "_hi"] - g[m]).to_numpy()])
            ax.errorbar(x[ok] + off, g[m].to_numpy()[ok], yerr=np.nan_to_num(err[:, ok]), fmt="-o", color=color,
                        markersize=5, linewidth=2, capsize=2, label=MISMATCH_NAMES[m])
            for resp, r in g.iterrows():
                rows.append({"shift": shift, "response": resp, "mismatch": m, "mean": r[m], "lo": r[m + "_lo"],
                             "hi": r[m + "_hi"], "worlds": r["worlds"]})
        ax.axhline(0, color="#8A8A8A", linewidth=0.8)
        ax.axhline(WARN_POINTS, color="#8A8A8A", linewidth=0.6, linestyle=":")
        ax.set_xticks(x, LEVELS)
        ax.set_xlabel("response heterogeneity")
        ax.set_title(f"shift {shift}", fontsize=9)
    axes[0].set_ylabel("population mismatch (greedy CTR points)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Population mismatch to the value optimum, correctly specified rank-4 adapter", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "fig_population_mismatch", pd.DataFrame(rows))
    plt.close(fig)


def fig_paired(pc: pd.DataFrame, out: Path, rules=("native", "best")) -> None:
    """Rows: rules; panels: shift levels. OPC − likelihood (and the diagnostics' differences) against the response
    level, pooled mean and 95% CI over the worlds."""
    plt = _plt()
    pairs = [("opc", "likelihood"), ("opc_raw", "likelihood"), ("opc_oq", "likelihood"),
             ("likelihood_calib", "likelihood")]
    names = {("opc", "likelihood"): "harmonic OPC − likelihood", ("opc_raw", "likelihood"): "raw DR − likelihood",
             ("opc_oq", "likelihood"): "harmonic OPC (oracle q) − likelihood",
             ("likelihood_calib", "likelihood"): "calibration-aware − ordinary likelihood"}
    c = pc[(pc["scope"] == "cell") & (pc["dataset"] == "all")]
    fig, axes = plt.subplots(len(rules), 3, figsize=(10.5, 3.2 * len(rules)), sharey="row", squeeze=False)
    rows = []
    x = np.arange(len(LEVELS))
    for ri, rule in enumerate(rules):
        for ax, shift in zip(axes[ri], LEVELS):
            for j, pair in enumerate(pairs):
                g = c[(c["shift"] == shift) & (c["rule"] == rule) & (c["a"] == pair[0]) & (c["b"] == pair[1])]
                g = g.set_index("response").reindex(LEVELS)
                ok = g["a_minus_b"].notna().to_numpy()
                if not ok.any():
                    continue
                off = (j - 1.5) * 0.12
                err = np.vstack([(g["a_minus_b"] - g["ci_lo"]).to_numpy(), (g["ci_hi"] - g["a_minus_b"]).to_numpy()])
                ax.errorbar(x[ok] + off, g["a_minus_b"].to_numpy()[ok], yerr=np.nan_to_num(err[:, ok]), fmt="-o",
                            color=COLORS[pair[0]], markersize=5, linewidth=2, capsize=2, label=names[pair])
                for resp, r in g.iterrows():
                    rows.append({"rule": rule, "shift": shift, "response": resp, "pair": names[pair],
                                 "mean": r["a_minus_b"], "lo": r["ci_lo"], "hi": r["ci_hi"], "worlds": r["worlds"]})
            ax.axhline(0, color="#8A8A8A", linewidth=0.8)
            ax.set_xticks(x, LEVELS)
            ax.set_title(f"shift {shift}, {rule} rule", fontsize=9)
            if ri == len(rules) - 1:
                ax.set_xlabel("response heterogeneity")
        axes[ri][0].set_ylabel("difference at 25k (greedy CTR points)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Paired differences at 25k by shift × response (mean and 95% CI over worlds)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out, "fig_paired_25k", pd.DataFrame(rows))
    plt.close(fig)


def fig_mechanism(pw: pd.DataFrame, out: Path) -> None:
    """Per world: OPC − likelihood (native) against M_L and against M_L − M_harm; marker shape = dataset."""
    plt = _plt()
    d = pw[(pw["a"] == "opc") & (pw["b"] == "likelihood") & (pw["rule"] == "native")].copy()
    d["M_L_minus_M_harm"] = d["pop_M_L"] - d["pop_M_harm"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, xcol, xlabel in ((axes[0], "pop_M_L", "M_L (points)"),
                             (axes[1], "M_L_minus_M_harm", "M_L − M_harm (points)")):
        for ds, g in d.groupby("dataset"):
            ax.scatter(g[xcol], g["d"], s=22, marker=MARKERS.get(ds, "o"), color=COLORS["opc"], alpha=0.75,
                       edgecolors="white", linewidths=0.6, label=ds)
        ax.axhline(0, color="#8A8A8A", linewidth=0.8)
        ax.axvline(0, color="#8A8A8A", linewidth=0.8)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("harmonic OPC − likelihood, native (points)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does OPC's 25k advantage follow the population mismatches? (one point per world)", fontsize=10)
    fig.tight_layout()
    _save(fig, out, "fig_mechanism", d[WORLD + ["shift", "response", "pop_M_L", "pop_M_harm", "M_L_minus_M_harm", "d"]])
    plt.close(fig)


def fig_decomposition(ac: pd.DataFrame, out: Path) -> None:
    """Per shift × response cell (pooled over datasets) and arm: the gap to θ_value* at 25k (native) split into the
    objective mismatch M, finite sample + optimization E+O and selection S, stacked (negative parts below 0)."""
    plt = _plt()
    arms = ["likelihood", "likelihood_calib", "opc", "opc_raw", "opc_oq"]
    parts = {"M": ("objective mismatch M", "#4D4D4D"), "EO": ("finite sample + optimization", "#A6A6A6"),
             "S_native": ("selection (native)", "#D9D9D9")}
    c = ac[(ac["scope"] == "cell") & (ac["dataset"] == "all")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.4), sharey=True)
    rows = []
    for i, shift in enumerate(LEVELS):
        for j, resp in enumerate(LEVELS):
            ax = axes[i][j]
            g = c[(c["shift"] == shift) & (c["response"] == resp)].set_index("arm").reindex(arms)
            for k, arm in enumerate(arms):
                if arm not in g.index or pd.isna(g.loc[arm].get("M")):
                    continue
                pos = neg = 0.0
                for part, (name, color) in parts.items():
                    v = float(g.loc[arm, part])
                    bottom = pos if v >= 0 else neg
                    ax.bar(k, v, bottom=bottom, color=color, edgecolor=COLORS[arm], linewidth=1.2, width=0.7,
                           label=name if (i, j, k) == (0, 0, 0) else None)
                    if v >= 0:
                        pos += v
                    else:
                        neg += v
                    rows.append({"shift": shift, "response": resp, "arm": arm, "part": part, "mean": v})
                ax.scatter([k], [float(g.loc[arm, "gap_native"])], color=COLORS[arm], s=16, zorder=3)
            ax.axhline(0, color="#8A8A8A", linewidth=0.8)
            ax.set_xticks(range(len(arms)), ["lik", "calib", "OPC", "raw", "OPC-oq"], fontsize=8)
            ax.set_title(f"shift {shift}, response {resp}", fontsize=9)
        axes[i][0].set_ylabel("gap to θ_value* (points)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("25k gap to the value optimum, native rule: M + (E+O) + S (dot: the total; bar edge: the arm)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out, "fig_decomposition_25k", pd.DataFrame(rows))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------- tables
def _fmt(m, lo=None, hi=None, digits=2, signed=True) -> str:
    if m is None or pd.isna(m):
        return "–"
    s = f"{m:+.{digits}f}" if signed else f"{m:.{digits}f}"
    if lo is not None and not pd.isna(lo):
        s += f" [{lo:+.{digits}f}, {hi:+.{digits}f}]" if signed else f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    return s


def _grid(value) -> list[str]:
    """A shift × response markdown grid; ``value(shift, response)`` returns the cell's text."""
    lines = ["| shift \\ response | " + " | ".join(LEVELS) + " |", "|---|" + "---|" * len(LEVELS)]
    for shift in LEVELS:
        lines.append(f"| {shift} | " + " | ".join(value(shift, r) for r in LEVELS) + " |")
    return lines


def population_md(pop: pd.DataFrame, cells: pd.DataFrame, fails: pd.DataFrame) -> str:
    out = ["# Structured scenario shift: population optima", "",
           f"{len(pop)} worlds. Greedy CTR points; mean [95% CI] over worlds (paired by world). M = V_g(θ_value*) − "
           "V_g(θ_obj*): positive is a loss of the objective's population optimum against the value optimum.", ""]

    def cell(ds, col, digits=2):
        c = cells[(cells["scope"] == "cell") & (cells["dataset"] == ds)].set_index(["shift", "response"])

        def f(shift, resp):
            if (shift, resp) not in c.index:
                return "–"
            r = c.loc[(shift, resp)]
            return _fmt(r[col], r[col + "_lo"], r[col + "_hi"], digits) + f" ({int(r['worlds'])})"
        return f

    for col, title in (("M_L", "M_L, ordinary likelihood"), ("M_calib", "M_calib, calibration-aware likelihood"),
                       ("M_harm", "M_harm, harmonic DR with q̂_∞"), ("M_L_minus_M_calib", "M_L − M_calib"),
                       ("available_gain", "available gain V_g(truth) − V_g(source)"),
                       ("optimizer_gap", "the class optimizer's gap V_g(truth) − V_g(θ_value*)"),
                       ("qinf_rmse_target", "q̂_∞ RMSE under θ_value*'s policy"),
                       ("qhat25k_rmse_crossfit_logging", "the 25k cross-fitted q̂'s RMSE under the logger"),
                       ("qhat25k_rmse_crossfit_target", "the 25k cross-fitted q̂'s RMSE under θ_value*'s policy")):
        if col not in cells:
            continue
        out += [f"## {title}", "", "Pooled over datasets and seeds (worlds):", ""]
        digits = 3 if col.startswith("q") else 2
        out += _grid(cell("all", col, digits)) + [""]
        for ds in DATASETS:
            if ((cells["dataset"] == ds) & (cells["scope"] == "cell")).any():
                out += [f"{ds}:", ""] + _grid(cell(ds, col, digits)) + [""]
    out += ["## Sanity checks (§6)", ""]
    out += ["Every check passes." if fails.empty else _md(fails), ""]
    out += ["## Per world", "", pop[WORLD + ["shift", "response", "M_L", "M_calib", "M_harm", "optimizer_gap",
                                           "available_gain", "qinf_rmse_logging", "qinf_rmse_target"]]
            .round(3).pipe(_md), ""]
    return "\n".join(out)


def compare_md(wt, pc, ac, ident, div, gate, info) -> str:
    out = ["# Structured scenario shift: the 25k grid", "",
           f"{info['worlds']} worlds, {info['trials']} trials. Greedy CTR points; mean [95% CI] over worlds, paired by "
           "world. Gains are over the logger's greedy value.", ""]

    def pcell(ds, a, b, rule):
        c = pc[(pc["scope"] == "cell") & (pc["dataset"] == ds) & (pc["a"] == a) & (pc["b"] == b)
               & (pc["rule"] == rule)].set_index(["shift", "response"])

        def f(shift, resp):
            if (shift, resp) not in c.index:
                return "–"
            r = c.loc[(shift, resp)]
            return _fmt(r["a_minus_b"], r["ci_lo"], r["ci_hi"]) + f" ({int(r['a_higher'])}/{int(r['worlds'])})"
        return f

    def acell(ds, arm, col):
        c = ac[(ac["scope"] == "cell") & (ac["dataset"] == ds) & (ac["arm"] == arm)].set_index(["shift", "response"])

        def f(shift, resp):
            if (shift, resp) not in c.index or col not in c or pd.isna(c.loc[(shift, resp)].get(col)):
                return "–"
            r = c.loc[(shift, resp)]
            return _fmt(r[col], r.get(col + "_lo"), r.get(col + "_hi"))
        return f

    out += ["## 1. Harmonic OPC − likelihood (ordinary head), by rule", "",
            "Cells: mean [95% CI] (worlds where OPC is higher / worlds).", ""]
    for rule in RULES:
        out += [f"### {rule} rule", "", "Pooled:", ""] + _grid(pcell("all", "opc", "likelihood", rule)) + [""]
        for ds in DATASETS:
            if (pc["dataset"] == ds).any():
                out += [f"{ds}:", ""] + _grid(pcell(ds, "opc", "likelihood", rule)) + [""]
    out += ["## 2. The other paired differences (native rule, pooled)", ""]
    for a, b in PAIRS[1:]:
        out += [f"### {ARM_NAMES[a]} − {ARM_NAMES[b]}", ""] + _grid(pcell("all", a, b, "native")) + [""]
    out += ["## 3. Per dataset, pooled over cells (native and best rules)", "",
            "| pair | rule | " + " | ".join(DATASETS) + " | all |", "|---|---|" + "---|" * (len(DATASETS) + 1)]
    for a, b in PAIRS:
        for rule in ("native", "best"):
            cells_ = []
            for ds in list(DATASETS) + ["all"]:
                g = pc[(pc["scope"] == ("dataset" if ds != "all" else "all")) & (pc["dataset"] == ds)
                       & (pc["a"] == a) & (pc["b"] == b) & (pc["rule"] == rule)]
                cells_.append("–" if g.empty else _fmt(g["a_minus_b"].iloc[0], g["ci_lo"].iloc[0], g["ci_hi"].iloc[0])
                              + f" ({int(g['a_higher'].iloc[0])}/{int(g['worlds'].iloc[0])})")
            out.append(f"| {a} − {b} | {rule} | " + " | ".join(cells_) + " |")
    out += ["", "## 4. Gains over the logger (native rule, pooled)", ""]
    for arm in SHORT:
        if (ac["arm"] == arm).any():
            out += [f"### {ARM_NAMES[arm]}", ""] + _grid(acell("all", arm, "native_gain")) + [""]
    if "M" in ac:
        out += ["## 5. Decomposition of the gap to θ_value* (native rule, pooled)", "",
                "gap = M (objective) + E+O (finite sample + optimization, to the best of 20) + S (selection).", "",
                "| shift | response | arm | gap | M | E+O | S |", "|---|---|---|---|---|---|---|"]
        c = ac[(ac["scope"] == "cell") & (ac["dataset"] == "all")]
        for shift in LEVELS:
            for resp in LEVELS:
                for arm in SHORT:
                    g = c[(c["shift"] == shift) & (c["response"] == resp) & (c["arm"] == arm)]
                    if g.empty:
                        continue
                    r = g.iloc[0]
                    out.append(f"| {shift} | {resp} | {arm} | " + " | ".join(
                        _fmt(r.get(k), r.get(k + "_lo"), r.get(k + "_hi")) for k in ("gap_native", "M", "EO",
                                                                                     "S_native")) + " |")
        out.append("")
    if gate is not None:
        out += ["## 6. Expansion gate (§7)", "", _md(gate.round(2)), "",
                f"Harmonic dominance: {json.dumps(info.get('harmonic_dominance'), default=float)}", ""]
    out += ["## 7. Data identity and divergences", "",
            f"Worlds with identical rows and paired configurations: {info['identical_worlds']} of {info['worlds']}.",
            f"Cells over the 5% divergence limit: {info['cells_over_divergence']}.",
            f"Largest |population source − trainer logger| greedy value: {info.get('logger_check_max_abs')} points.", ""]
    bad = ident[~ident["identical"]]
    if not bad.empty:
        out += [_md(bad), ""]
    over = div[div["over"]]
    if not over.empty:
        out += [_md(over.round(3)), ""]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("population")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--population", default=None)
    c.add_argument("--arms", nargs="+", default=None)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"population": population_main, "compare": compare_main}[args.cmd](args)


if __name__ == "__main__":
    main()
