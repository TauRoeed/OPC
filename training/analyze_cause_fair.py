"""CausE vs the revalidated OPC at 25k (docs/cause_fair_comparison_25k.md): the tuning stage of the new CausE
variants (§3) and the main comparison (§5, §6). Development stage.

Tuning (``tune``): every trial of the CausE-warm / CausE-capacity-matched studies on the tuning worlds, one row per
(trial, prediction), with its true greedy gain over the logger. The 20-trial protocol of the main grid is simulated on
candidate sub-spaces by resampling the logged trials: per (world, rho), a random k of the trials inside the sub-space
(all when fewer), the one with the lowest validation NLL among the finite ones, its true gain. The draws are shared by
the two predictions (one trained model gives both).

Usage: python -m training.analyze_cause_fair tune --runs artifacts/full_study/run_cause_tune_<family>_s200 ... --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from training.run_state import read_trials_long

DIVERGED_NLL = 1e9  # training/cause_trials.py: a non-finite trial's validation NLL
WORLD = ["dataset", "bias", "seed"]
CELL = WORLD + ["rho"]
PER_PREDICTION = ("val_nll", "val_auc", "value", "value_greedy", "temper_scale", "value_tempered", "val_dr_greedy",
                  "val_dr_greedy_low", "val_dr_tempered", "val_dr_tempered_low")
FAMILY_OF_PREFIX = {"cause": "native", "causewarm": "warm", "causecap": "cap"}
DIMENSIONS = ("lr", "epochs", "l2_pen", "cf_pen", "tie", "bias_init")


def _tags(folder: str) -> dict:
    return dict(part.split("=", 1) for part in folder.split("__") if "=" in part)


def mean_ci(x) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, n
    h = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h, n


def load_cause_trials(*run_dirs) -> pd.DataFrame:
    """Every trial of the warm-started CausE studies of finished conditions: one row per (trial, prediction), with the
    world's tags, the logger's values and the gains over the logger in CTR points (greedy: over the logger's greedy
    value; stochastic and tempered: over the logger's own value)."""
    frames = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            summary = cond / "summary_metrics.csv"
            if not summary.exists():  # written when the condition finishes
                continue
            s = pd.read_csv(summary)
            v_greedy, v_logger = float(s["logger_greedy"].iloc[0]), float(s["initial_reward"].iloc[0])
            tags = _tags(cond.name)
            for path in sorted(cond.glob("cause*_c_r*_trials.csv")):  # the _t_ file holds the same trials
                t = pd.read_csv(path)
                family = FAMILY_OF_PREFIX[path.name.split("_")[0]]
                base = t[[c for c in t.columns if not c.startswith(("c_", "t_")) and c not in ("method", "prediction")]]
                for p in ("c", "t"):
                    cols = {f"{p}_{k}": k for k in PER_PREDICTION if f"{p}_{k}" in t.columns}
                    f = pd.concat([base, t[list(cols)].rename(columns=cols)], axis=1)
                    f["prediction"], f["family"], f["run"] = p, family, Path(run).name
                    f["dataset"], f["bias"], f["seed"] = tags["dataset"], tags["bias"], int(tags["seed"])
                    f["logger_greedy"], f["initial_reward"] = v_greedy, v_logger
                    frames.append(f)
    if not frames:
        raise FileNotFoundError(f"no finished CausE conditions under {run_dirs}")
    t = pd.concat(frames, ignore_index=True)
    t["usable"] = t["finite"].astype(bool) & (t["val_nll"] < DIVERGED_NLL)
    t["gain_greedy"] = 100.0 * (t["value_greedy"] - t["logger_greedy"])
    t["gain"] = 100.0 * (t["value"] - t["initial_reward"])
    if "value_tempered" in t:
        t["gain_tempered"] = 100.0 * (t["value_tempered"] - t["initial_reward"])
    # total step length of momentum SGD with a linear decay to 0: lr x steps / 2 (steps = epochs x batches per epoch)
    t["log10_lr_steps"] = np.log10(t["lr"] * t["steps"] / 2.0)
    return t


def in_space(t: pd.DataFrame, space: dict | None) -> np.ndarray:
    """Rows inside a sub-space: ``{"lr": (lo, hi), "epochs": {...}, "l2_pen": {...}, "cf_pen": {...}, "tie": {...},
    "bias_init": {...}}``; a missing key leaves that dimension unrestricted."""
    mask = np.ones(len(t), dtype=bool)
    for key, allowed in (space or {}).items():
        if key == "lr":
            mask &= (t["lr"] >= allowed[0] * (1 - 1e-9)) & (t["lr"] <= allowed[1] * (1 + 1e-9))
        else:
            mask &= t[key].isin(list(allowed))
    return np.asarray(mask)


def simulate_protocol(t: pd.DataFrame, space: dict | None = None, *, k: int = 20, resamples: int = 200,
                      seed: int = 0) -> pd.DataFrame:
    """Per (world, rho, prediction): the mean true greedy gain of the trial the k-trial protocol selects inside
    ``space``, the mean of the best true gain among the same draws, and the share of draws without a usable trial
    (counted as the logger's own value, gain 0). The draws are shared by the two predictions."""
    rng = np.random.default_rng(seed)
    sub = t[in_space(t, space)]
    rows = []
    for key, g in sub.groupby(CELL, sort=True):
        wide = g.pivot_table(index="trial", columns="prediction", values=["val_nll", "gain_greedy", "usable"],
                             aggfunc="first")
        n = len(wide)
        draws = (np.tile(np.arange(n), (resamples, 1)) if n <= k
                 else rng.random((resamples, n)).argsort(axis=1)[:, :k])
        for p in wide["val_nll"].columns:
            usable = wide[("usable", p)].astype(bool).to_numpy()
            nll = np.where(usable, wide[("val_nll", p)].to_numpy(dtype=float), np.inf)[draws]
            gain = np.where(usable, wide[("gain_greedy", p)].to_numpy(dtype=float), -np.inf)[draws]
            ok = np.isfinite(nll).any(axis=1)
            selected = np.where(ok, np.take_along_axis(gain, nll.argmin(axis=1)[:, None], axis=1)[:, 0], 0.0)
            best = np.where(ok, gain.max(axis=1), 0.0)
            rows.append(dict(zip(CELL, key), prediction=p, available=n, gain_greedy=float(selected.mean()),
                             best_greedy=float(best.mean()), regret=float(best.mean() - selected.mean()),
                             no_usable_trial=float(1 - ok.mean())))
    return pd.DataFrame(rows)


def candidate_table(t: pd.DataFrame, candidates: dict, reference: str, ks=(10, 20), resamples: int = 200) -> pd.DataFrame:
    """Per family, candidate and k: the mean selected greedy gain over (world, rho, prediction), its regret, the mean
    number of trials available, and the paired difference from ``reference`` with a 95% CI over worlds (each world's
    rho × prediction cells averaged first)."""
    rows = []
    for family, tf in t.groupby("family"):
        for k in ks:
            sims = {name: simulate_protocol(tf, space, k=k, resamples=resamples) for name, space in candidates.items()}
            ref = sims[reference].groupby(WORLD)["gain_greedy"].mean()
            for name, sim in sims.items():
                if sim.empty:
                    continue
                per_world = sim.groupby(WORLD)["gain_greedy"].mean()
                d = (per_world - ref.reindex(per_world.index)).values
                m, lo, hi, n = mean_ci(d)
                rows.append({"family": family, "candidate": name, "k": k, "worlds": int(per_world.size),
                             "available": float(sim["available"].mean()), "gain_greedy": float(sim["gain_greedy"].mean()),
                             "gain_greedy_c": float(sim.loc[sim["prediction"] == "c", "gain_greedy"].mean()),
                             "gain_greedy_t": float(sim.loc[sim["prediction"] == "t", "gain_greedy"].mean()),
                             "best_greedy": float(sim["best_greedy"].mean()), "regret": float(sim["regret"].mean()),
                             "no_usable_trial": float(sim["no_usable_trial"].mean()),
                             f"minus_{reference}": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


# The decision rule (docs/cause_fair_comparison_25k.md §3): the candidate sub-space with the best mean selected greedy
# value over the tuning worlds. Candidates: the wide space; the tie direction and the intercept initialization fixed or
# searched; then, on the best of those, 2-decade lr windows, epoch windows, both, and both without the extreme l2 / cf
# values. A candidate needs at least MIN_AVAILABLE trials per cell on average.
STRUCTURES = {"wide": {}, "tie one-way": {"tie": {"one_way"}}, "tie symmetric": {"tie": {"symmetric"}},
              "init zero": {"bias_init": {"zero"}}, "init base rate": {"bias_init": {"base_rate"}},
              "one-way, zero": {"tie": {"one_way"}, "bias_init": {"zero"}},
              "one-way, base rate": {"tie": {"one_way"}, "bias_init": {"base_rate"}},
              "symmetric, zero": {"tie": {"symmetric"}, "bias_init": {"zero"}},
              "symmetric, base rate": {"tie": {"symmetric"}, "bias_init": {"base_rate"}}}
LR_WINDOWS = ((1e-4, 1e-2), (3e-4, 3e-2), (1e-3, 1e-1), (3e-3, 3e-1), (1e-2, 1.0), (3e-2, 3.0))
EPOCH_WINDOWS = ((3, 10, 30, 100, 300), (10, 30, 100, 300), (30, 100, 300), (1, 3, 10, 30), (3, 10, 30, 100),
                 (10, 30, 100))
L2_INNER = (0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)  # without 1e-2
CF_INNER = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)  # without 1000
MIN_AVAILABLE = 5.0


def _score(t: pd.DataFrame, space: dict, k: int, resamples: int) -> dict:
    sim = simulate_protocol(t, space, k=k, resamples=resamples)
    if sim.empty:
        return {"available": 0.0, "gain_greedy": np.nan}
    per_world = sim.groupby(WORLD)["gain_greedy"].mean()
    return {"available": float(sim["available"].mean()), "gain_greedy": float(sim["gain_greedy"].mean()),
            "gain_greedy_c": float(sim.loc[sim["prediction"] == "c", "gain_greedy"].mean()),
            "gain_greedy_t": float(sim.loc[sim["prediction"] == "t", "gain_greedy"].mean()),
            "best_greedy": float(sim["best_greedy"].mean()), "regret": float(sim["regret"].mean()),
            "no_usable_trial": float(sim["no_usable_trial"].mean()), "_per_world": per_world}


def tuning_decision(t: pd.DataFrame, *, k: int = 10, resamples: int = 300) -> pd.DataFrame:
    """Per family: every candidate's mean selected greedy gain under the k-trial protocol, its paired difference from
    the wide space (95% CI over worlds), and the chosen candidate (``chosen``)."""
    out = []
    for family, tf in t.groupby("family"):
        rows = []

        def add(stage, name, space):
            r = _score(tf, space, k, resamples)
            r.update({"family": family, "stage": stage, "candidate": name, "space": repr(space)})
            rows.append(r)
            return r

        wide = add("structure", "wide", {})
        for name, space in list(STRUCTURES.items())[1:]:
            add("structure", name, space)
        eligible = [r for r in rows if r["available"] >= MIN_AVAILABLE]
        base = max(eligible, key=lambda r: r["gain_greedy"])
        base_space = STRUCTURES[base["candidate"]]
        best_lr = max((add("lr window", f"{base['candidate']}; lr {lo:g}-{hi:g}", {**base_space, "lr": (lo, hi)})
                       for lo, hi in LR_WINDOWS), key=lambda r: (r["available"] >= MIN_AVAILABLE, r["gain_greedy"]))
        best_ep = max((add("epoch window", f"{base['candidate']}; epochs {','.join(map(str, ep))}",
                           {**base_space, "epochs": set(ep)}) for ep in EPOCH_WINDOWS),
                      key=lambda r: (r["available"] >= MIN_AVAILABLE, r["gain_greedy"]))
        both = {**eval(best_lr["space"]), **{"epochs": eval(best_ep["space"])["epochs"]}}
        add("lr and epochs", f"{best_lr['candidate']}; {best_ep['candidate'].split('; ')[-1]}", both)
        add("lr and epochs", f"{best_lr['candidate']}; {best_ep['candidate'].split('; ')[-1]}; inner l2 / cf",
            {**both, "l2_pen": set(L2_INNER), "cf_pen": set(CF_INNER)})
        ref = wide["_per_world"]
        for r in rows:
            d = (r["_per_world"] - ref.reindex(r["_per_world"].index)).values if "_per_world" in r else np.array([])
            m, lo, hi, n = mean_ci(d)
            r.update({"minus_wide": m, "ci_lo": lo, "ci_hi": hi, "worlds": n})
            r.pop("_per_world", None)
        eligible = [r for r in rows if r["available"] >= MIN_AVAILABLE]
        chosen = max(eligible, key=lambda r: r["gain_greedy"])
        for r in rows:
            r["eligible"] = r["available"] >= MIN_AVAILABLE
            r["chosen"] = r is chosen
        out += rows
    cols = ["family", "stage", "candidate", "available", "gain_greedy", "gain_greedy_c", "gain_greedy_t", "best_greedy",
            "regret", "no_usable_trial", "minus_wide", "ci_lo", "ci_hi", "worlds", "eligible", "chosen", "space"]
    return pd.DataFrame(out)[cols]


def boundary_table(t: pd.DataFrame, space: dict | None = None) -> pd.DataFrame:
    """Where the NLL-selected trials (over every trial inside ``space``) and the best-true trials sit on each searched
    dimension's range: the shares at its lowest and highest value (lr: the lowest and highest half-decade)."""
    sub = t[in_space(t, space)]
    usable = sub[sub["usable"]]
    keys = ["family"] + CELL + ["prediction"]
    picks = {"selected": usable.loc[usable.groupby(keys)["val_nll"].idxmin()],
             "best true": usable.loc[usable.groupby(keys)["gain_greedy"].idxmax()]}
    rows = []
    for family in sorted(sub["family"].unique()):
        for which, g in picks.items():
            g = g[g["family"] == family]
            for dim in ("lr", "epochs", "l2_pen", "cf_pen"):
                values = sub.loc[sub["family"] == family, dim]
                lo, hi = values.min(), values.max()
                if dim == "lr":
                    at_lo = (np.log10(g[dim]) < np.log10(lo) + 0.5).mean()
                    at_hi = (np.log10(g[dim]) > np.log10(hi) - 0.5).mean()
                else:
                    at_lo, at_hi = (g[dim] == lo).mean(), (g[dim] == hi).mean()
                rows.append({"family": family, "trials": which, "dimension": dim, "lowest": lo, "highest": hi,
                             "share_at_lowest": float(at_lo), "share_at_highest": float(at_hi),
                             "median": float(g[dim].median())})
    return pd.DataFrame(rows)


def _dimension_values(t: pd.DataFrame, dim: str) -> pd.Series:
    if dim == "lr":
        return pd.cut(np.log10(t["lr"]), [-4.0, -3.0, -2.0, -1.0, 0.0, np.log10(3.0)], include_lowest=True,
                      labels=["1e-4-1e-3", "1e-3-1e-2", "1e-2-1e-1", "1e-1-1", "1-3"]).astype(str)
    if dim == "log10_lr_steps":
        return pd.cut(t["log10_lr_steps"], [-np.inf, -1, 0, 1, 2, 3, np.inf]).astype(str)
    return t[dim].astype(str)


def marginal_table(t: pd.DataFrame, dims=DIMENSIONS + ("log10_lr_steps",)) -> pd.DataFrame:
    """Per family, dimension and value: trials, the share that diverged, the mean true greedy gain of the usable
    trials minus their cell's best (0 = as good as the cell's best trial), and the shares of cells whose
    NLL-selected trial (over all trials) and whose best-true trial have this value."""
    rows = []
    for family, tf in t.groupby("family"):
        tf = tf.copy()
        tf["below_best"] = tf["gain_greedy"] - tf[tf["usable"]].groupby(CELL + ["prediction"])["gain_greedy"].transform("max")
        usable = tf[tf["usable"]]
        sel = usable.loc[usable.groupby(CELL + ["prediction"])["val_nll"].idxmin()]
        best = usable.loc[usable.groupby(CELL + ["prediction"])["gain_greedy"].idxmax()]
        n_cells = len(sel)
        for dim in dims:
            v_all, v_sel, v_best = (_dimension_values(x, dim) for x in (tf, sel, best))
            for value in sorted(v_all.unique(), key=lambda s: (len(s), s)):
                g = tf[v_all == value]
                gu = g[g["usable"]]
                rows.append({"family": family, "dimension": dim, "value": value, "trials": len(g),
                             "diverged": float(1 - g["usable"].mean()),
                             "below_best_mean": float(gu["below_best"].mean()) if len(gu) else np.nan,
                             "below_best_median": float(gu["below_best"].median()) if len(gu) else np.nan,
                             "selected_share": float((v_sel == value).sum() / n_cells),
                             "best_share": float((v_best == value).sum() / n_cells)})
    return pd.DataFrame(rows)


def selected_table(t: pd.DataFrame) -> pd.DataFrame:
    """The NLL-selected trial over all trials of each (world, rho, prediction), with its gains and regret."""
    usable = t[t["usable"]]
    sel = usable.loc[usable.groupby(["family"] + CELL + ["prediction"])["val_nll"].idxmin()].copy()
    best = usable.groupby(["family"] + CELL + ["prediction"])["gain_greedy"].max().rename("best_gain_greedy")
    sel = sel.merge(best.reset_index(), on=["family"] + CELL + ["prediction"])
    sel["regret"] = sel["best_gain_greedy"] - sel["gain_greedy"]
    return sel


# ---------------------------------------------------------------------------------------------- main comparison
ORACLE_ROOT = Path("artifacts/full_study/run_oracle_repair_20260927")
REVAL = Path("artifacts/full_study/opc_revalidation_20261004/summaries")
STAGE2_RUNS = tuple(Path("artifacts/full_study") / r for r in (
    "run_reval_stage2_opc_mlkr", "run_reval_stage2_opc_anime", "run_reval_stage2_base_mlkr", "run_reval_stage2_base_anime"))
OLDSPACE_RUNS = tuple(Path("artifacts/full_study") / r for r in ("run_reval_stage2_oldspace_dm_mlkr",))
NATIVE_RUN = Path("artifacts/full_study/run_cause_dev_25k_cause_20261004")
BEST_ITEM = Path("artifacts/full_study/cause_dev_25k_20261004/table_best_single_item.csv")
TRAIN_SIZE = 25_000
BIAS_ORDER = ("none", "w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high", "high")
BIAS_NAMES = {"none": "no bias", "w-high.g-none.v-none": "warp high", "w-none.g-high.v-none": "group high",
              "w-none.g-none.v-high": "vector high", "high": "combined high"}
DATASETS = ("ml", "kuairand", "anime")
# arms: the references, then CausE as <family>_<prediction> (one arm per rho)
REFERENCE_ARMS = ("opc", "opc_raw", "dm_own", "dm", "no_prop", "tempered_logger")
NATIVE_ARMS = ("native_prod_c", "native_prod_t", "native_avg")
FAIR_ARMS = ("warm_c", "warm_t", "cap_c", "cap_t")
NAMES = {"opc": "OPC (harmonic:0.1)", "opc_raw": "OPC (raw DR)", "dm_own": "DM-only (own range)",
         "dm": "DM-only (OPC's range)", "no_prop": "no-propensity", "tempered_logger": "tempered logger", "native_prod_c": "native CausE-prod-C",
         "native_prod_t": "native CausE-prod-T", "native_avg": "native CausE-avg", "warm_c": "CausE-warm-C",
         "warm_t": "CausE-warm-T", "cap_c": "CausE-cap-C", "cap_t": "CausE-cap-T"}
# the arm's structural ceiling (§4): OPC's linear-repair oracle for the linear family, the target-best value for the
# free-vector family (its class contains the true click model)
OWN_ORACLE = {"opc": "linear", "opc_raw": "linear", "dm_own": "linear", "dm": "linear", "no_prop": "linear",
              "tempered_logger": "linear",
              "cap_c": "linear", "cap_t": "linear", "warm_c": "ceiling", "warm_t": "ceiling",
              "native_prod_c": "ceiling", "native_prod_t": "ceiling", "native_avg": "ceiling"}


def world_references() -> pd.DataFrame:
    """Per world: the logger's values, the target-best (clean greedy) value and the Stage 1 linear-repair oracle
    (the best of its two fits, as in the revalidation's Stage 2 rows)."""
    from training.analyze_recoverability import derive, load_oracle

    o = derive(load_oracle(ORACLE_ROOT))
    o = o[o["bias"].isin(BIAS_ORDER)]
    return o[["dataset", "bias", "seed", "V_logger", "V_logger_greedy", "ceiling", "oracle_repair_value",
              "oracle_repair_greedy"]].rename(columns={"oracle_repair_value": "oracle_linear_value",
                                                       "oracle_repair_greedy": "oracle_linear_greedy"})


def _study_rows(run_dirs, methods, train_size=TRAIN_SIZE) -> pd.DataFrame:
    """The selected policy of each study arm (``methods``) per world: true values, the selection estimate (DR point
    and lower bound for OPC; the q-hat estimate for DM-only) and the true selection regret over the arm's own trials,
    greedy and stochastic."""
    rows = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "summary_metrics.csv").exists():
                continue
            tags = _tags(cond.name)
            if tags["bias"] not in BIAS_ORDER or int(tags["seed"]) not in (100, 101):
                continue
            s = pd.read_csv(cond / "summary_metrics.csv")
            t = read_trials_long(cond / "trials_long.csv")
            for _, r in s[(s["train_size"] == train_size) & s["method"].isin(methods)].iterrows():
                g = t[(t["method"] == r["method"]) & (t["train_size"] == train_size)]
                best = g[g["is_best_in_run"].astype(bool)].iloc[0]
                assert abs(float(best["actual_reward"]) - float(r["policy_rewards"])) < 1e-9
                rows.append({"run": Path(run).name, "dataset": tags["dataset"], "bias": tags["bias"],
                             "seed": int(tags["seed"]), "method": r["method"], "V": float(r["policy_rewards"]),
                             "V_greedy": float(r["policy_rewards_greedy"]), "logit_scale": float(best.get("logit_scale", np.nan)),
                             "sel_estimate": float(best["r_hat"]), "sel_estimate_low": float(best["value"]),
                             "regret": float(g["actual_reward"].max() - best["actual_reward"]),
                             "regret_greedy": float(g["actual_reward_greedy"].max() - best["actual_reward_greedy"]),
                             "n_trials": int(len(g)), "train_weights": r.get("train_weights", np.nan),
                             "lr": float(best["param_lr"]), "epochs": int(best["param_num_epochs"])})
    return pd.DataFrame(rows)


def reference_rows(raw_runs=(), dm_own_runs=()) -> pd.DataFrame:
    """OPC (harmonic:0.1), DM-only in OPC's range, no-propensity and the tempered logger: the corrected Stage 2's 25k
    selections on the 30 worlds (not rerun). DM-only in its own range: the old-space decomposition run (ml / kuairand) and
    ``dm_own_runs`` (anime). Raw-DR OPC: ``raw_runs``."""
    parts = [_study_rows(STAGE2_RUNS[:2], ("opc",)).assign(arm="opc"),
             _study_rows(STAGE2_RUNS[2:], ("dm",)).assign(arm="dm"),
             _study_rows(STAGE2_RUNS[2:], ("no_propensity",)).assign(arm="no_prop"),
             _study_rows(STAGE2_RUNS[2:], ("tempered_logger",)).assign(arm="tempered_logger")]
    if dm_own_runs:
        parts.append(_study_rows(OLDSPACE_RUNS + tuple(dm_own_runs), ("dm",)).assign(arm="dm_own"))
    if raw_runs:
        parts.append(_study_rows(raw_runs, ("opc",)).assign(arm="opc_raw"))
    return pd.concat(parts, ignore_index=True)


def cause_rows(run_dirs) -> pd.DataFrame:
    """Every CausE selection (native M5 and the fair variants) per world and rho."""
    frames = []
    for run in run_dirs:
        for p in sorted(Path(run).glob("dataset=*/summary_metrics.csv")):
            tags = _tags(p.parent.name)
            if tags["bias"] not in BIAS_ORDER or int(tags["seed"]) not in (100, 101):
                continue
            s = pd.read_csv(p)
            s = s[(s["train_size"] == TRAIN_SIZE) & s["method"].str.startswith("cause")].copy()
            s["run"], s["dataset"], s["bias"], s["seed"] = Path(run).name, tags["dataset"], tags["bias"], int(tags["seed"])
            frames.append(s)
    s = pd.concat(frames, ignore_index=True)
    family = s["method"].str.split("_").str[0].map(FAMILY_OF_PREFIX)
    s["arm"] = family + "_" + s["cause_prediction"].astype(str)
    s["rho"] = s["cause_rho"].astype(float)
    s = s.rename(columns={"policy_rewards": "V", "policy_rewards_greedy": "V_greedy", "policy_rewards_tempered": "V_tempered"})
    s["regret_greedy"] = s["oracle_selected_value_greedy"] - s["V_greedy"]
    s["regret"] = s["oracle_selected_value"] - s["V"]
    return s


def condition_table(cause: pd.DataFrame, refs: pd.DataFrame, best_item: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per world × arm (× rho for CausE): values, gains (CTR points) and fractions.

    gain_greedy = V_greedy − V_logger_greedy; gain = V − V_logger (stochastic; the tempered softmax for CausE in
    gain_tempered); frac_loss = gain_greedy / (V_target_best − V_logger_greedy), the share of the representation
    loss repaired; frac_oracle = gain_greedy / (the arm's structural ceiling − V_logger_greedy)."""
    keep = ["dataset", "bias", "seed", "arm", "rho", "V", "V_greedy", "V_tempered", "temper_scale", "regret",
            "regret_greedy", "sel_estimate", "sel_estimate_low", "val_nll", "val_auc", "val_dr_greedy",
            "val_dr_greedy_low", "val_dr_tempered", "val_dr_tempered_low", "n_trials", "n_finite_trials", "lr", "epochs",
            "l2_pen", "cf_pen", "cause_tie", "cause_bias_init", "alpha", "logit_scale", "n_control", "n_treatment",
            "control_reward_sum", "treatment_reward_sum", "collection_reward_sum", "opc_collection_reward_sum",
            "exploration_cost_expected", "exploration_cost_realised", "treatment_items_covered", "initial_reward",
            "logger_greedy"]
    t = pd.concat([d[[c for c in keep if c in d.columns]] for d in (refs, cause)], ignore_index=True).reindex(columns=keep)
    w = world_references()
    t = t.merge(w, on=["dataset", "bias", "seed"], how="left", validate="many_to_one")
    # the runs' own logger values are the Stage 1 world's (the logger and the truth do not depend on the logs)
    for own, ref in (("initial_reward", "V_logger"), ("logger_greedy", "V_logger_greedy")):
        ok = t[own].notna()
        assert np.allclose(t.loc[ok, own], t.loc[ok, ref], atol=1e-6), own
    t["family"] = t["arm"].map(lambda a: a.split("_")[0] if a.split("_")[0] in ("native", "warm", "cap") else "reference")
    t["gain_greedy"] = 100 * (t["V_greedy"] - t["V_logger_greedy"])
    t["gain"] = 100 * (t["V"] - t["V_logger"])
    t["gain_tempered"] = 100 * (t["V_tempered"] - t["V_logger"])
    loss = t["ceiling"] - t["V_logger_greedy"]
    t["representation_loss"] = 100 * loss
    t["frac_loss"] = np.where(loss > 1e-3, t["gain_greedy"] / 100 / loss, np.nan)
    ceiling = np.where(t["arm"].map(OWN_ORACLE) == "linear", t["oracle_linear_greedy"], t["ceiling"])
    rec = ceiling - t["V_logger_greedy"]
    t["own_oracle"] = t["arm"].map(OWN_ORACLE)
    t["frac_oracle"] = np.where(rec > 1e-3, t["gain_greedy"] / 100 / rec, np.nan)
    t["oracle_linear_gain"] = 100 * (t["oracle_linear_greedy"] - t["V_logger_greedy"])
    t["regret_greedy"] = 100 * t["regret_greedy"]
    t["regret"] = 100 * t["regret"]
    for c in ("sel_estimate", "sel_estimate_low"):  # selection estimate minus the truth (OPC / DM-only)
        if c in t:
            t[c.replace("estimate", "error")] = 100 * (t[c] - t["V"])
    for c in ("val_dr_greedy", "val_dr_greedy_low"):  # CausE: DR estimate of its greedy policy minus the truth
        if c in t:
            t[c + "_error"] = 100 * (t[c] - t["V_greedy"])
    if best_item is not None:
        t = t.merge(best_item[["dataset", "seed", "best_single_item"]], on=["dataset", "seed"], how="left")
        t["best_item_gain"] = 100 * (t["best_single_item"] - t["V_logger_greedy"])
    t["bias_order"] = t["bias"].map({b: i for i, b in enumerate(BIAS_ORDER)})
    return t.sort_values(["bias_order", "dataset", "seed", "arm", "rho"]).drop(columns="bias_order").reset_index(drop=True)


SUMMARY_COLUMNS = ("gain_greedy", "gain", "gain_tempered", "frac_loss", "frac_oracle", "regret_greedy", "regret",
                   "sel_error", "sel_error_low", "val_dr_greedy_error", "val_dr_greedy_low_error", "temper_scale",
                   "exploration_cost_expected", "exploration_cost_realised", "collection_reward_sum", "n_treatment",
                   "best_item_gain", "representation_loss", "oracle_linear_gain")


def summary_table(t: pd.DataFrame, pool_biased: bool = True) -> pd.DataFrame:
    """Mean and 95% CI over worlds per bias × arm × rho; with ``pool_biased`` also over the 24 biased worlds."""
    groups = [(b, t[t["bias"] == b]) for b in BIAS_ORDER if (t["bias"] == b).any()]
    if pool_biased:
        groups.append(("biased (pooled)", t[t["bias"] != "none"]))
    rows = []
    for bias, tb in groups:
        for (arm, rho), g in tb.groupby(["arm", tb["rho"].fillna(-1.0)]):
            r = {"bias": bias, "arm": arm, "rho": None if rho < 0 else rho, "worlds": len(g)}
            for col in SUMMARY_COLUMNS:
                if col in g and g[col].notna().any():
                    m, lo, hi, _n = mean_ci(g[col])
                    r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def paired_table(t: pd.DataFrame, a: str, b_arms, *, col: str = "gain_greedy", a_rho=None,
                 b_col: str | None = None) -> pd.DataFrame:
    """``a`` − each arm of ``b_arms`` (per rho), paired by world: mean, 95% CI and the count of worlds where ``a`` is
    higher, per bias and pooled over the biased worlds. ``a_rho``: the rho of ``a`` when it is a CausE arm. ``b_col``:
    the b arms' column when it differs from ``col`` (OPC's stochastic value against CausE's tempered one); the result
    column is then named after ``b_col``."""
    b_col = b_col or col

    def values(arm, rho, c):
        g = t[(t["arm"] == arm) & ((t["rho"] == rho) if rho is not None else t["rho"].isna())]
        return g.set_index(WORLD)[c]

    va = values(a, a_rho, col)
    rows = []
    for arm in b_arms:
        rhos = sorted(t.loc[t["arm"] == arm, "rho"].dropna().unique()) or [None]
        for rho in rhos:
            d = (va - values(arm, rho, b_col)).dropna()
            for bias in list(BIAS_ORDER) + ["biased (pooled)"]:
                db = d[d.index.get_level_values("bias") != "none"] if bias == "biased (pooled)" else d[d.index.get_level_values("bias") == bias]
                if db.empty:
                    continue
                m, lo, hi, n = mean_ci(db.values)
                rows.append({"a": a, "b": arm, "rho": rho, "bias": bias, f"a_minus_b_{b_col}": m, "ci_lo": lo, "ci_hi": hi,
                             "worlds": n, "a_higher": int((db > 0).sum())})
    return pd.DataFrame(rows)


def data_identity_check(cause: pd.DataFrame) -> pd.DataFrame:
    """Every CausE family must train on the same rows at the same (world, rho): the native M5 rows and the new runs
    share the warm prefix, the uniform pool and the validation rows. Per (world, rho): whether each budget field
    differs across the families (counts and click sums exactly; the logger's exact values beyond 1e-6, since they
    are computed in float32 and equal only up to summation order, ~1e-8)."""
    exact = ["n_control", "n_treatment", "control_reward_sum", "treatment_reward_sum", "opc_collection_reward_sum",
             "val_size"]
    floats = ["initial_reward", "logger_greedy"]
    g = cause.groupby(CELL)
    out = (g[[f for f in exact if f in cause]].nunique() > 1)
    for f in floats:
        if f in cause:
            out[f] = (g[f].max() - g[f].min()) > 1e-6
    out = out.astype(int).rename(columns=lambda c: f"{c}_differs")
    out["families"] = g["arm"].apply(lambda a: ",".join(sorted({x.split("_")[0] for x in a})))
    return out.reset_index()


def oracle_check(trials: pd.DataFrame) -> pd.DataFrame:
    """Is the Stage 1 linear-repair oracle a ceiling for the capacity-matched family (§4)? Per world: the best greedy
    value of any CausE-cap trial (any rho, either prediction) and of any OPC trial, against the oracle's greedy value
    (both in CTR points over the logger's greedy value)."""
    w = world_references()
    rows = []
    cap = trials[trials["family"] == "cap"]
    if not cap.empty:
        best = cap.groupby(WORLD)["value_greedy"].max().rename("best_trial").reset_index().assign(arm="cap")
        rows.append(best)
    opc = []
    for run in STAGE2_RUNS[:2]:
        for cond in sorted(Path(run).glob("dataset=*")):
            tags = _tags(cond.name)
            if tags["bias"] not in BIAS_ORDER or int(tags["seed"]) not in (100, 101):
                continue
            t = read_trials_long(cond / "trials_long.csv", usecols=["method", "train_size", "actual_reward_greedy"])
            t = t[(t["method"] == "opc") & (t["train_size"] == TRAIN_SIZE)]
            opc.append({"dataset": tags["dataset"], "bias": tags["bias"], "seed": int(tags["seed"]),
                        "best_trial": float(t["actual_reward_greedy"].max()), "arm": "opc"})
    rows.append(pd.DataFrame(opc))
    out = pd.concat(rows, ignore_index=True).merge(w, on=WORLD, how="left")
    out["best_trial_gain"] = 100 * (out["best_trial"] - out["V_logger_greedy"])
    out["oracle_gain"] = 100 * (out["oracle_linear_greedy"] - out["V_logger_greedy"])
    out["best_minus_oracle"] = out["best_trial_gain"] - out["oracle_gain"]
    return out[["arm", "dataset", "bias", "seed", "best_trial_gain", "oracle_gain", "best_minus_oracle"]]


def selection_rule_table(trials: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic, not CausE's protocol: per family, prediction and rho, the true greedy gain of the trial selected by
    validation NLL (CausE's rule), by the 95% DR lower bound of the trial's greedy policy (OPC's kind of score), and
    of the best trial (the oracle choice among the 20), averaged over the biased worlds and over no bias."""
    keys = ["family", "prediction"] + CELL
    usable = trials[trials["usable"]].copy()
    usable["biased"] = usable["bias"] != "none"
    rows = []
    for key, g in usable.groupby(keys):
        r = dict(zip(keys, key))
        r["biased"] = bool(g["biased"].iloc[0])
        r["nll"] = float(g.loc[g["val_nll"].idxmin(), "gain_greedy"])
        if g["val_dr_greedy_low"].notna().any():
            r["dr_lower_bound"] = float(g.loc[g["val_dr_greedy_low"].idxmax(), "gain_greedy"])
        r["best_of_trials"] = float(g["gain_greedy"].max())
        rows.append(r)
    d = pd.DataFrame(rows)
    out = []
    for (family, p, rho, biased), g in d.groupby(["family", "prediction", "rho", "biased"]):
        r = {"family": family, "prediction": p, "rho": rho, "worlds": "biased" if biased else "no bias", "n": len(g)}
        for col in ("nll", "dr_lower_bound", "best_of_trials"):
            if col in g:
                r[col] = float(g[col].mean())
        if "dr_lower_bound" in g:
            m, lo, hi, _n = mean_ci(g["dr_lower_bound"] - g["nll"])
            r.update({"dr_minus_nll": m, "ci_lo": lo, "ci_hi": hi})
        out.append(r)
    return pd.DataFrame(out)


def rho_effect_table(t: pd.DataFrame, arms=FAIR_ARMS + NATIVE_ARMS, col: str = "gain_greedy") -> pd.DataFrame:
    """Each CausE arm at rho minus the same arm at rho = 0 (no randomized rows), paired by world."""
    rows = []
    for arm in arms:
        g = t[t["arm"] == arm]
        if g.empty:
            continue
        base = g[g["rho"] == 0.0].set_index(WORLD)[col]
        for rho in sorted(g["rho"].unique()):
            if rho == 0.0:
                continue
            d = (g[g["rho"] == rho].set_index(WORLD)[col] - base).dropna()
            for bias in list(BIAS_ORDER) + ["biased (pooled)"]:
                db = d[d.index.get_level_values("bias") != "none"] if bias == "biased (pooled)" else d[d.index.get_level_values("bias") == bias]
                if db.empty:
                    continue
                m, lo, hi, n = mean_ci(db.values)
                rows.append({"arm": arm, "rho": rho, "bias": bias, f"minus_rho0_{col}": m, "ci_lo": lo, "ci_hi": hi,
                             "worlds": n, "higher": int((db > 0).sum())})
    return pd.DataFrame(rows)


def variant_contrasts(t: pd.DataFrame, col: str = "gain_greedy") -> pd.DataFrame:
    """At equal rho and prediction side, paired by world: CausE-warm − native CausE-prod (the source representation,
    given CausE's capacity), CausE-cap − CausE-warm (OPC's linear family instead of free vectors, given the source)
    and CausE-cap − native CausE-prod (both)."""
    pairs = (("warm_c", "native_prod_c"), ("warm_t", "native_prod_t"), ("cap_c", "warm_c"), ("cap_t", "warm_t"),
             ("cap_c", "native_prod_c"), ("cap_t", "native_prod_t"))
    rows = []
    for a, b in pairs:
        for rho in sorted(t.loc[t["arm"] == a, "rho"].dropna().unique()):
            va = t[(t["arm"] == a) & (t["rho"] == rho)].set_index(WORLD)[col]
            vb = t[(t["arm"] == b) & (t["rho"] == rho)].set_index(WORLD)[col]
            d = (va - vb).dropna()
            for bias in list(BIAS_ORDER) + ["biased (pooled)"]:
                db = d[d.index.get_level_values("bias") != "none"] if bias == "biased (pooled)" else d[d.index.get_level_values("bias") == bias]
                if db.empty:
                    continue
                m, lo, hi, n = mean_ci(db.values)
                rows.append({"a": a, "b": b, "rho": rho, "bias": bias, f"a_minus_b_{col}": m, "ci_lo": lo, "ci_hi": hi,
                             "worlds": n, "a_higher": int((db > 0).sum())})
    return pd.DataFrame(rows)


def cause_minus_references(t: pd.DataFrame, col: str = "gain_greedy") -> pd.DataFrame:
    """Each fair CausE arm at each rho minus OPC, DM-only (own range) and no-propensity, paired by world. At rho = 0
    CausE uses neither randomized rows nor propensities; CausE-cap there is a click-likelihood fit in OPC's class."""
    rows = []
    for arm in [a for a in FAIR_ARMS if a in set(t["arm"])]:
        for rho in RHOS:
            d = paired_table(t, arm, [a for a in ("opc", "dm_own", "no_prop") if a in set(t["arm"])], col=col, a_rho=rho)
            rows.append(d.assign(a_rho=rho))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------------------------------------------- figures
# one colour per entity (Okabe-Ito); the prediction side of a CausE family by line style and marker
COLORS = {"opc": "#0072B2", "opc_raw": "#0072B2", "dm_own": "#E69F00", "dm": "#E69F00", "tempered_logger": "#009E73",
          "native": "#56B4E9", "warm": "#D55E00", "cap": "#CC79A7", "best_item": "#7F7F7F"}
SIDE_STYLE = {"c": ("-", "o"), "t": ("--", "s"), "prod_c": ("-", "o"), "prod_t": ("--", "s"), "avg": (":", "^")}
RHO_TICKS = (0.0, 0.05, 0.10, 0.15, 0.25)


def _arm_style(arm: str) -> dict:
    family, side = arm.split("_", 1)
    ls, marker = SIDE_STYLE[side]
    return {"color": COLORS[family], "linestyle": ls, "marker": marker}


def _legend(fig, axes) -> None:
    """One legend for every series drawn in any panel (identity is never colour alone)."""
    seen = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def _rho_axis(ax) -> None:
    ax.set_xlim(-0.012, 0.262)
    ax.set_xticks(RHO_TICKS)
    ax.set_xticklabels(["0", ".05", ".10", ".15", ".25"])


def _panels(s: pd.DataFrame, panels) -> tuple:
    from training.representation_report import _plt

    plt = _plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(max(2.65 * len(panels), 7.2), 3.0), sharey=False)
    return plt, fig, np.atleast_1d(axes)


# horizontal offsets so that the error bars of series at the same rho do not hide each other (C and T often coincide)
DODGE = {"warm_c": -0.0045, "warm_t": -0.0015, "cap_c": 0.0015, "cap_t": 0.0045, "native_prod_c": -0.006,
         "native_prod_t": 0.006}


def _ci_line(ax, g: pd.DataFrame, col: str, label: str, arm: str | None = None, **style) -> None:
    x = g["rho"] + DODGE.get(arm, 0.0)
    ax.errorbar(x, g[col], yerr=[g[col] - g[col + "_lo"], g[col + "_hi"] - g[col]], markersize=3.5,
                linewidth=1.4, capsize=2, elinewidth=0.8, label=label, **style)


def _reference_lines(ax, sb: pd.DataFrame, col: str, arms=("opc", "dm_own"), best_item: bool = True) -> list:
    data = []
    for arm, ls in zip(arms, ("-", "--", "-.")):
        g = sb[sb["arm"] == arm]
        if g.empty or col not in g or pd.isna(g.iloc[0].get(col)):
            continue
        r = g.iloc[0]
        ax.axhline(float(r[col]), color=COLORS[arm], linestyle=ls, linewidth=1.3, label=NAMES[arm])
        if arm == "opc" and np.isfinite(r.get(col + "_lo", np.nan)):
            ax.axhspan(float(r[col + "_lo"]), float(r[col + "_hi"]), color=COLORS[arm], alpha=0.12, linewidth=0,
                       label="OPC 95% CI")
        data.append(g)
    ax.axhline(0, color="black", linewidth=0.8, label="logger")
    if best_item and "best_item_gain" in sb and sb["best_item_gain"].notna().any():
        v = float(sb["best_item_gain"].dropna().iloc[0])
        lo, hi = ax.get_ylim()
        if v >= lo:
            ax.axhline(v, color=COLORS["best_item"], linestyle=(0, (4, 3)), linewidth=1.0,
                       label="best single item (no personalization)")
        else:  # far below every method: name it at the panel's bottom instead of stretching the axis
            ax.annotate(f"best single item: {v:+.1f} ↓", xy=(0.03, 0.02), xycoords="axes fraction", fontsize=7.5,
                        color=COLORS["best_item"], va="bottom")
            ax.plot([], [], color=COLORS["best_item"], linestyle=(0, (4, 3)), linewidth=1.0,
                    label="best single item (no personalization)")
    return data


def fig_rho(s: pd.DataFrame, out: Path, *, col: str = "gain_greedy", name: str = "fig1_rho_greedy",
            ylabel: str = "greedy value − logger's greedy value (CTR pts)", arms=FAIR_ARMS, refs=("opc", "dm_own"),
            title: str = "", ref_col: str | None = None) -> None:
    """``ref_col``: the references' column when it differs from the CausE arms' (their stochastic value against
    CausE's tempered one)."""
    biases = [b for b in BIAS_ORDER if b in set(s["bias"])]
    plt, fig, axes = _panels(s, biases)
    data = []
    for ax, bias in zip(axes, biases):
        sb = s[s["bias"] == bias]
        for arm in arms:
            g = sb[sb["arm"] == arm].sort_values("rho")
            if g.empty or col not in g:
                continue
            _ci_line(ax, g, col, NAMES[arm], arm, **_arm_style(arm))
            data.append(g.assign(panel=bias))
        data += [d.assign(panel=bias) for d in _reference_lines(ax, sb, ref_col or col, refs,
                                                                best_item=col == "gain_greedy")]
        ax.set_title(BIAS_NAMES.get(bias, bias))
        _rho_axis(ax)
    axes[0].set_ylabel(ylabel)
    fig.supxlabel("randomized share of the 25k budget, ρ", fontsize=9)
    _legend(fig, axes)
    fig.suptitle(title or "Target value against the randomized share (mean and 95% CI over 3 datasets × 2 seeds)",
                 y=1.02, fontsize=9.5)
    from training.representation_report import _save

    _save(fig, out, name, pd.concat(data, ignore_index=True) if data else pd.DataFrame())


def fig_cost(t: pd.DataFrame, out: Path, arms=FAIR_ARMS) -> None:
    """Final greedy gain against the expected clicks given up while collecting (OPC and DM-only at zero)."""
    biases = [b for b in BIAS_ORDER if b in set(t["bias"])]
    plt, fig, axes = _panels(t, biases)
    data = []
    for ax, bias in zip(axes, biases):
        tb = t[t["bias"] == bias]
        for arm in arms:
            g = tb[tb["arm"] == arm].groupby("rho").agg(cost=("exploration_cost_expected", "mean"),
                                                         gain=("gain_greedy", "mean")).reset_index()
            if g.empty:
                continue
            st = _arm_style(arm)
            ax.plot(g["cost"], g["gain"], color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                    markersize=3.5, linewidth=1.3, label=NAMES[arm])
            data.append(g.assign(panel=bias, arm=arm))
        for arm, marker in (("opc", "D"), ("dm_own", "v")):
            g = tb[tb["arm"] == arm]
            if len(g):
                m, lo, hi, _ = mean_ci(g["gain_greedy"])
                ax.errorbar([0.0], [m], yerr=[[m - lo], [hi - m]], color=COLORS[arm], marker=marker, markersize=5,
                            capsize=2, elinewidth=0.8, linestyle="none", zorder=3, label=f"{NAMES[arm]} (no randomized rows)")
                data.append(pd.DataFrame({"panel": [bias], "arm": [arm], "cost": [0.0], "gain": [m]}))
        ax.axhline(0, color="black", linewidth=0.8, label="logger")
        ax.set_title(BIAS_NAMES.get(bias, bias))
    axes[0].set_ylabel("greedy value − logger's greedy value (CTR pts)")
    fig.supxlabel("expected clicks given up while collecting, ρN·(V(π0) − V(uniform))", fontsize=9)
    _legend(fig, axes)
    fig.suptitle("Final target value against the exploration cost (means over 3 datasets × 2 seeds; the points along "
                 "each line are ρ = 0, .01, .05, .10, .15, .25)", y=1.02, fontsize=9.5)
    from training.representation_report import _save

    _save(fig, out, "fig2_exploration_cost", pd.concat(data, ignore_index=True) if data else pd.DataFrame())


def fig_variants(s: pd.DataFrame, out: Path) -> None:
    """Native vs warm vs capacity-matched CausE against rho: the biased worlds pooled, and no bias."""
    panels = [p for p in ("biased (pooled)", "none") if p in set(s["bias"])]
    plt, fig, axes = _panels(s, panels)
    data = []
    for ax, panel in zip(axes, panels):
        sb = s[s["bias"] == panel]
        for arm in ("native_prod_c", "native_prod_t", "warm_c", "warm_t", "cap_c", "cap_t"):
            g = sb[sb["arm"] == arm].sort_values("rho")
            if g.empty:
                continue
            _ci_line(ax, g, "gain_greedy", NAMES[arm], arm, **_arm_style(arm))
            data.append(g.assign(panel=panel))
        ax.axhline(0, color="black", linewidth=0.8, label="logger")
        ax.set_title("biased worlds (24)" if panel != "none" else "no bias (6)")
        _rho_axis(ax)
    axes[0].set_ylabel("greedy value − logger's greedy value (CTR pts)")
    fig.supxlabel("randomized share of the 25k budget, ρ", fontsize=9)
    _legend(fig, axes)
    fig.suptitle("Native CausE (from scratch) vs CausE-warm (source vectors, free capacity)\nvs CausE-cap (source "
                 "vectors, OPC's linear family); mean and 95% CI over worlds", y=1.06, fontsize=9.5)
    from training.representation_report import _save

    _save(fig, out, "fig3_cause_variants", pd.concat(data, ignore_index=True) if data else pd.DataFrame())


def fig_opc_minus(p: pd.DataFrame, out: Path, arms=FAIR_ARMS) -> None:
    """OPC − CausE (greedy, paired by world) against rho, per bias and pooled over the biased worlds."""
    panels = [b for b in list(BIAS_ORDER) + ["biased (pooled)"] if b in set(p["bias"])]
    plt, fig, axes = _panels(p, panels)
    col = "a_minus_b_gain_greedy"
    data = []
    for ax, bias in zip(axes, panels):
        pb = p[p["bias"] == bias]
        for arm in arms:
            g = pb[pb["b"] == arm].sort_values("rho").rename(columns={"ci_lo": col + "_lo", "ci_hi": col + "_hi"})
            if g.empty:
                continue
            _ci_line(ax, g, col, NAMES[arm], arm, **_arm_style(arm))
            data.append(g.assign(panel=bias))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(BIAS_NAMES.get(bias, bias))
        _rho_axis(ax)
    axes[0].set_ylabel("OPC − CausE, greedy value (CTR pts)")
    fig.supxlabel("CausE's randomized share of the 25k budget, ρ (OPC: all 25k rows from the logger)", fontsize=9)
    _legend(fig, axes)
    fig.suptitle("OPC minus each fair CausE variant, paired by world (mean and 95% CI; above 0 = OPC better)",
                 y=1.02, fontsize=9.5)
    from training.representation_report import _save

    _save(fig, out, "fig4_opc_minus_cause", pd.concat(data, ignore_index=True) if data else pd.DataFrame())


# ------------------------------------------------------------------------------------------------ report tables
RHOS = (0.0, 0.01, 0.05, 0.10, 0.15, 0.25)
PANELS = ("biased (pooled)",) + BIAS_ORDER


def _fmt(m, lo=np.nan, hi=np.nan, digits: int = 2, sign: bool = True) -> str:
    if m is None or not np.isfinite(m):
        return "—"
    f = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"
    return f.format(m)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _cell(s: pd.DataFrame, panel: str, arm: str, rho, col: str, ci: bool = True, digits: int = 2) -> str:
    g = s[(s["bias"] == panel) & (s["arm"] == arm)]
    g = g[g["rho"].isna()] if rho is None else g[np.isclose(g["rho"].astype(float), rho)]
    if g.empty or col not in g:
        return "—"
    r = g.iloc[0]
    return _fmt(r[col], r.get(col + "_lo", np.nan) if ci else np.nan, r.get(col + "_hi", np.nan) if ci else np.nan,
                digits)


def tables_md(t: pd.DataFrame, s: pd.DataFrame, p: pd.DataFrame) -> str:
    """The report's tables (docs/cause_fair_comparison_25k.md §8) as markdown."""
    out = []
    refs = [a for a in ("opc", "opc_raw", "dm_own", "dm", "no_prop", "tempered_logger") if a in set(t["arm"])]
    cause = [a for a in FAIR_ARMS + NATIVE_ARMS if a in set(t["arm"])]
    for col, title in (("gain_greedy", "Greedy value − the logger's greedy value"),
                       ("gain", "Stochastic value − the logger's value (CausE: its raw softmax, τ = 1)"),
                       ("gain_tempered", "Stochastic value − the logger's value, CausE tempered by the DR lower bound")):
        for panel in ("biased (pooled)", "none"):
            rows = []
            for arm in refs:
                if col == "gain_tempered" and arm not in ("opc", "tempered_logger", "dm_own"):
                    continue
                c = "gain" if col == "gain_tempered" else col
                rows.append({"arm": NAMES[arm], "no randomized rows": _cell(s, panel, arm, None, c),
                             **{f"ρ = {r:g}": "" for r in RHOS}})
            for arm in cause:
                if col == "gain_tempered" and arm not in FAIR_ARMS:
                    continue
                rows.append({"arm": NAMES[arm], "no randomized rows": "",
                             **{f"ρ = {r:g}": _cell(s, panel, arm, r, col, ci=r in (0.0, 0.25)) for r in RHOS}})
            name = "biased worlds (24)" if panel != "none" else "no bias (6)"
            out.append(f"**{title}, {name}** (CTR points; mean over worlds, 95% CI at ρ = 0 and 0.25)\n\n"
                       + _md(pd.DataFrame(rows)))
    # per bias, the strongest comparison: OPC vs each fair CausE arm, paired
    for col, title in (("a_minus_b_gain_greedy", "OPC − CausE, greedy value, paired by world"),
                       ("a_minus_b_gain_tempered", "OPC − CausE, stochastic value (CausE tempered), paired by world")):
        rows = []
        for panel in PANELS:
            for arm in [a for a in FAIR_ARMS if a in set(p["b"])]:
                g = p[(p["bias"] == panel) & (p["b"] == arm) & p[col].notna()] if col in p else pd.DataFrame()
                if g.empty:
                    continue
                r = {"bias": BIAS_NAMES.get(panel, panel), "CausE": NAMES[arm]}
                for rho in RHOS:
                    x = g[np.isclose(g["rho"].astype(float), rho)]
                    r[f"ρ = {rho:g}"] = (f"{_fmt(x[col].iloc[0], x['ci_lo'].iloc[0], x['ci_hi'].iloc[0])} "
                                         f"({int(x['a_higher'].iloc[0])}/{int(x['worlds'].iloc[0])})") if len(x) else "—"
                rows.append(r)
        if rows:
            out.append(f"**{title}** (CTR points; mean [95% CI]; in parentheses the worlds where OPC is higher)\n\n"
                       + _md(pd.DataFrame(rows)))
    # fractions, regret and the structural ceilings (biased worlds)
    rows = []
    for arm in refs + cause:
        for rho in ([None] if arm in refs else [0.0, 0.25]):
            panel = "biased (pooled)"
            rows.append({"arm": NAMES[arm] + ("" if rho is None else f", ρ = {rho:g}"),
                         "share of the representation loss repaired": _cell(s, panel, arm, rho, "frac_loss", digits=2),
                         "share of its own structural oracle": _cell(s, panel, arm, rho, "frac_oracle", digits=2),
                         "own ceiling": "linear-repair oracle" if OWN_ORACLE.get(arm) == "linear" else "target best",
                         "selection regret, greedy (pts)": _cell(s, panel, arm, rho, "regret_greedy", ci=False)})
    out.append("**Shares of the loss repaired, structural ceilings and selection regret, biased worlds** (greedy; "
               "mean [95% CI] over the 24 worlds)\n\n" + _md(pd.DataFrame(rows)))
    # exploration cost
    rows = []
    for rho in RHOS:
        g = t[(t["arm"] == "cap_c") & np.isclose(t["rho"].astype(float), rho)] if "cap_c" in set(t["arm"]) else \
            t[(t["arm"] == "native_prod_c") & np.isclose(t["rho"].astype(float), rho)]
        if g.empty:
            continue
        rows.append({"ρ": f"{rho:g}", "uniform rows N_t": f"{g['n_treatment'].mean():,.0f}",
                     "expected clicks given up (mean over worlds)": f"{g['exploration_cost_expected'].mean():,.0f}",
                     "realized clicks given up": f"{g['exploration_cost_realised'].mean():,.0f}",
                     "share of OPC's collection clicks": f"{(g['exploration_cost_realised'] / g['opc_collection_reward_sum']).mean():.1%}"})
    out.append("**Exploration cost of the CausE budget** (all 30 worlds; OPC and DM-only collect no randomized rows)\n\n"
               + _md(pd.DataFrame(rows)))
    return "\n\n".join(out) + "\n"


def compare_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = reference_rows(args.raw_runs, args.dm_own_runs)
    cause = cause_rows([NATIVE_RUN] + list(args.cause_runs))
    best_item = pd.read_csv(BEST_ITEM)
    ident = data_identity_check(cause)
    ident.to_csv(out / "table_data_identity.csv", index=False)
    differing = ident[[c for c in ident.columns if c.endswith("_differs")]].gt(0).any(axis=1)
    if differing.any():
        raise AssertionError(f"CausE families trained on different data in {int(differing.sum())} (world, rho) cells")
    t = condition_table(cause, refs, best_item)
    t.to_csv(out / "table_conditions.csv", index=False, float_format="%.8g")
    s = summary_table(t)
    s.to_csv(out / "table_summary.csv", index=False, float_format="%.6g")
    cause_arms = [a for a in FAIR_ARMS + NATIVE_ARMS if a in set(t["arm"])]
    p = pd.concat([paired_table(t, "opc", cause_arms), paired_table(t, "opc", cause_arms, col="gain"),
                   paired_table(t, "opc", [a for a in FAIR_ARMS if a in set(t["arm"])], col="gain",
                                b_col="gain_tempered")],
                  ignore_index=True)
    p.to_csv(out / "table_paired_opc.csv", index=False, float_format="%.6g")
    pd.concat([paired_table(t, "opc", ["opc_raw", "dm_own", "dm", "no_prop", "tempered_logger"]),
               paired_table(t, "opc", ["opc_raw", "dm_own", "dm", "no_prop", "tempered_logger"], col="gain")],
              ignore_index=True).to_csv(out / "table_paired_references.csv", index=False, float_format="%.6g")
    rho_effect_table(t).to_csv(out / "table_rho_effect.csv", index=False, float_format="%.6g")
    variant_contrasts(t).to_csv(out / "table_variant_contrasts.csv", index=False, float_format="%.6g")
    cause_minus_references(t).to_csv(out / "table_cause_minus_references.csv", index=False, float_format="%.6g")
    trials = load_cause_trials(*args.cause_runs)
    trials.to_csv(out / "cause_trials_long.csv.gz", index=False, float_format="%.8g")
    oc = oracle_check(trials)
    oc.to_csv(out / "table_oracle_check.csv", index=False, float_format="%.6g")
    selection_rule_table(trials).to_csv(out / "table_selection_rule.csv", index=False, float_format="%.6g")
    fig_rho(s, out)
    fig_rho(s, out, col="gain_tempered", name="fig1b_rho_stochastic_tempered", ref_col="gain",
            ylabel="stochastic value − logger's value (CTR pts)", refs=("opc", "tempered_logger", "dm_own"),
            title="Secondary: stochastic value; CausE's softmax tempered by the DR lower bound (references: OPC's "
                  "learned scale, the tempered logger)")
    fig_cost(t, out)
    fig_variants(s, out)
    fig_opc_minus(p[p["a_minus_b_gain_greedy"].notna()], out)
    (out / "tables.md").write_text(tables_md(t, s, p))
    print(f"wrote {out}: {len(t)} rows; arms {sorted(t['arm'].unique())}")


def tune_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load_cause_trials(*args.runs)
    t.to_csv(out / "tuning_trials_long.csv.gz", index=False, float_format="%.8g")
    marginal_table(t).to_csv(out / "tuning_marginals.csv", index=False, float_format="%.6g")
    selected_table(t).to_csv(out / "tuning_selected.csv", index=False, float_format="%.6g")
    decision = tuning_decision(t, k=args.k, resamples=args.resamples)
    decision.to_csv(out / "tuning_decision.csv", index=False, float_format="%.6g")
    chosen = {r["family"]: eval(r["space"]) for _, r in decision[decision["chosen"]].iterrows()}
    bounds = pd.concat([boundary_table(t[t["family"] == f], space).assign(space="chosen")
                        for f, space in chosen.items()] + [boundary_table(t).assign(space="wide")], ignore_index=True)
    bounds.to_csv(out / "tuning_boundaries.csv", index=False, float_format="%.6g")
    for _, r in decision[decision["chosen"]].iterrows():
        print(f"[{r['family']}] chosen: {r['candidate']} (selected greedy gain {r['gain_greedy']:.3f}; wide "
              f"{decision[(decision['family'] == r['family']) & (decision['candidate'] == 'wide')]['gain_greedy'].iloc[0]:.3f})")
    print(f"wrote {out}: {len(t)} trial rows; families {sorted(t['family'].unique())}; "
          f"{t[WORLD].drop_duplicates().shape[0]} worlds")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune", help="the tuning-stage tables")
    tune.add_argument("--runs", nargs="+", required=True)
    tune.add_argument("--out", required=True)
    tune.add_argument("--k", type=int, default=10, help="trials per simulated study (default 10)")
    tune.add_argument("--resamples", type=int, default=300)
    comp = sub.add_parser("compare", help="the main comparison's tables and figures")
    comp.add_argument("--cause-runs", nargs="+", required=True, help="the CausE-warm / CausE-cap main-grid runs")
    comp.add_argument("--raw-runs", nargs="*", default=[], help="the raw-DR OPC run")
    comp.add_argument("--dm-own-runs", nargs="*", default=[], help="DM-only in its own range on anime")
    comp.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "tune":
        tune_main(args)
    else:
        compare_main(args)


if __name__ == "__main__":
    main()
