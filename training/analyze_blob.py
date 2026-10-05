"""BLOB-supplied-source in the controlled environment (docs/blob_controlled_integration.md): the tuning stage (§3) and
the bounded comparison (§4, §5). Development stage.

Tuning (``tune``): every trial of the BLOB studies on the tuning worlds, one row per (family, world, trial), with its
true greedy gain over the logger's greedy value. The 20-trial protocol of the main grid is simulated on candidate
sub-spaces by resampling the logged trials: per (family, world), a random k of the trials inside the sub-space (all
when fewer), the finite one with the lowest validation NLL, its true gain.

Usage: python -m training.analyze_blob tune --runs artifacts/full_study/run_blob_tune_s200 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_cause_fair import _tags, in_space, mean_ci

DIVERGED_NLL = 1e9  # training/blob_trials.py: a non-finite trial's validation NLL
WORLD = ["dataset", "bias", "seed"]
CELL = ["family"] + WORLD
DIMENSIONS = ("lr", "epochs", "wa_m", "wb_m", "kappa_s")
WIDE = {"lr": (1e-4, 3e-2), "epochs": (10, 30, 100, 300), "wa_m": (-1.0, 1.0, 3.0), "wb_m": (-6.0, -3.0, 0.0),
        "kappa_s": (0.01, 0.1, 1.0)}


def load_blob_trials(*run_dirs) -> pd.DataFrame:
    """Every trial of the BLOB studies of finished conditions, with the world's tags, the logger's values and the gains
    over the logger in CTR points (greedy: over the logger's greedy value; stochastic: over the logger's value)."""
    frames = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            summary = cond / "summary_metrics.csv"
            if not summary.exists():  # written when the condition finishes
                continue
            s = pd.read_csv(summary)
            s = s[s["method"].astype(str).str.startswith("blob")]
            tags = _tags(cond.name)
            for path in sorted(cond.glob("blob_*_trials.csv")):
                t = pd.read_csv(path)
                label = path.name[: -len("_trials.csv")]
                row = s[s["method"] == label].iloc[0]
                t["family"], t["run"] = label.split("_", 1)[1], Path(run).name
                t["dataset"], t["bias"], t["seed"] = tags["dataset"], tags["bias"], int(tags["seed"])
                t["logger_greedy"], t["initial_reward"] = float(row["logger_greedy"]), float(row["initial_reward"])
                t["selected_trial"] = int(row["selected_trial"])
                frames.append(t)
    if not frames:
        raise FileNotFoundError(f"no finished BLOB conditions under {run_dirs}")
    t = pd.concat(frames, ignore_index=True)
    t["usable"] = t["finite"].astype(bool) & (t["val_nll"] < DIVERGED_NLL)
    t["gain_greedy"] = 100.0 * (t["value_greedy"] - t["logger_greedy"])
    t["gain"] = 100.0 * (t["value"] - t["initial_reward"])
    t["log10_lr_steps"] = np.log10(t["lr"] * t["steps"])  # Adam: about lr per step, no decay
    return t


def simulate_protocol(t: pd.DataFrame, space: dict | None = None, *, k: int = 10, resamples: int = 300,
                      seed: int = 0) -> pd.DataFrame:
    """Per (family, world): the mean true greedy gain of the trial the k-trial protocol selects inside ``space``, the
    mean best true gain among the same draws, and the share of draws without a usable trial (counted as the logger's
    own greedy value, gain 0)."""
    rng = np.random.default_rng(seed)
    sub = t[in_space(t, space)]
    rows = []
    for key, g in sub.groupby(CELL, sort=True):
        g = g.sort_values("trial")
        n = len(g)
        draws = (np.tile(np.arange(n), (resamples, 1)) if n <= k
                 else rng.random((resamples, n)).argsort(axis=1)[:, :k])
        usable = g["usable"].to_numpy(bool)
        nll = np.where(usable, g["val_nll"].to_numpy(float), np.inf)[draws]
        gain = np.where(usable, g["gain_greedy"].to_numpy(float), -np.inf)[draws]
        ok = np.isfinite(nll).any(axis=1)
        selected = np.where(ok, np.take_along_axis(gain, nll.argmin(axis=1)[:, None], axis=1)[:, 0], 0.0)
        best = np.where(ok, gain.max(axis=1), 0.0)
        rows.append(dict(zip(CELL, key), available=n, gain_greedy=float(selected.mean()),
                         best_greedy=float(best.mean()), regret=float(best.mean() - selected.mean()),
                         no_usable_trial=float(1 - ok.mean())))
    return pd.DataFrame(rows)


# The decision rule (docs/blob_controlled_integration.md §3), per family: the candidate sub-space with the best mean
# selected greedy gain over the tuning worlds. Candidates: the wide space and each prior dimension fixed at one value
# (the others searched); then, on the best of those, 1.5-decade lr windows, epoch windows and the best of each
# combined. A candidate needs at least MIN_AVAILABLE trials per cell on average.
STRUCTURES = {"wide": {}, **{f"{dim} {v:g}": {dim: {v}} for dim in ("kappa_s", "wa_m", "wb_m") for v in WIDE[dim]}}
LR_WINDOWS = ((1e-4, 3e-3), (3e-4, 1e-2), (1e-3, 3e-2))
EPOCH_WINDOWS = ((10, 30, 100), (30, 100, 300), (100, 300))
MIN_AVAILABLE = 5.0
EDGE_MARGIN = 0.25  # CTR points: an edge value must beat its neighbour by more than this to extend the range


def _score(t: pd.DataFrame, space: dict, k: int, resamples: int) -> dict:
    sim = simulate_protocol(t, space, k=k, resamples=resamples)
    if sim.empty:
        return {"available": 0.0, "gain_greedy": np.nan}
    return {"available": float(sim["available"].mean()), "gain_greedy": float(sim["gain_greedy"].mean()),
            "best_greedy": float(sim["best_greedy"].mean()), "regret": float(sim["regret"].mean()),
            "no_usable_trial": float(sim["no_usable_trial"].mean()),
            "_per_world": sim.groupby(WORLD)["gain_greedy"].mean()}


def tuning_decision(t: pd.DataFrame, *, k: int = 10, resamples: int = 300) -> pd.DataFrame:
    """Per family: every candidate's mean selected greedy gain under the k-trial protocol, its paired difference from
    the wide space (95% CI over worlds) and the chosen candidate (``chosen``)."""
    out = []
    for family, tf in t.groupby("family"):
        rows = []

        def add(stage, name, space):
            r = _score(tf, space, k, resamples)
            r.update({"family": family, "stage": stage, "candidate": name, "space": repr(space)})
            rows.append(r)
            return r

        def best_of(rs):
            return max(rs, key=lambda r: (r["available"] >= MIN_AVAILABLE, r["gain_greedy"]))

        wide = add("structure", "wide", {})
        for name, space in list(STRUCTURES.items())[1:]:
            add("structure", name, space)
        base = best_of([r for r in rows if r["stage"] == "structure"])
        base_space = STRUCTURES[base["candidate"]]
        best_lr = best_of([add("lr window", f"{base['candidate']}; lr {lo:g}-{hi:g}", {**base_space, "lr": (lo, hi)})
                           for lo, hi in LR_WINDOWS])
        best_ep = best_of([add("epoch window", f"{base['candidate']}; epochs {','.join(map(str, ep))}",
                               {**base_space, "epochs": set(ep)}) for ep in EPOCH_WINDOWS])
        both = {**eval(best_lr["space"]), "epochs": eval(best_ep["space"])["epochs"]}
        add("lr and epochs", f"{best_lr['candidate']}; {best_ep['candidate'].split('; ')[-1]}", both)
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
    cols = ["family", "stage", "candidate", "available", "gain_greedy", "best_greedy", "regret", "no_usable_trial",
            "minus_wide", "ci_lo", "ci_hi", "worlds", "eligible", "chosen", "space"]
    return pd.DataFrame(out)[cols]


LR_EDGES = (-4.0, -3.5, -3.0, -2.5, -2.0, np.log10(3e-2))  # half-decades of the wide lr range


def _values(t: pd.DataFrame, dim: str) -> pd.Series:
    if dim == "lr":
        return pd.cut(np.log10(t["lr"]), LR_EDGES, include_lowest=True,
                      labels=["1e-4-3e-4", "3e-4-1e-3", "1e-3-3e-3", "3e-3-1e-2", "1e-2-3e-2"]).astype(str)
    if dim == "log10_lr_steps":
        return pd.cut(t["log10_lr_steps"], [-np.inf, -1, 0, 1, 2, np.inf]).astype(str)
    return t[dim].astype(float).map(lambda v: f"{v:g}")


def _ordered_values(dim: str) -> list[str]:
    if dim == "lr":
        return ["1e-4-3e-4", "3e-4-1e-3", "1e-3-3e-3", "3e-3-1e-2", "1e-2-3e-2"]
    return [f"{v:g}" for v in WIDE[dim]]


def marginal_table(t: pd.DataFrame, space: dict | None = None, dims=DIMENSIONS + ("log10_lr_steps",)) -> pd.DataFrame:
    """Per family, dimension and value, inside ``space``: trials, the share that diverged, the mean and median gap
    between a usable trial's true greedy gain and its cell's best trial in the space (0 = as good as the best), and
    the shares of cells whose NLL-selected and best-true trials have this value."""
    sub = t[in_space(t, space)].copy()
    rows = []
    for family, tf in sub.groupby("family"):
        tf = tf.copy()
        tf["below_best"] = tf["gain_greedy"] - tf[tf["usable"]].groupby(WORLD)["gain_greedy"].transform("max")
        usable = tf[tf["usable"]]
        sel = usable.loc[usable.groupby(WORLD)["val_nll"].idxmin()]
        best = usable.loc[usable.groupby(WORLD)["gain_greedy"].idxmax()]
        n_cells = len(sel)
        for dim in dims:
            v_all, v_sel, v_best = (_values(x, dim) for x in (tf, sel, best))
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


def edge_check(t: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    """The edge rule (§3): in each family's chosen space, per searched dimension, the value with the smallest mean gap
    to the cell's best; ``extend`` when it lies at an edge of the wide range and beats its neighbour by more than
    EDGE_MARGIN points."""
    rows = []
    for family, space in chosen.items():
        m = marginal_table(t[t["family"] == family], space, dims=DIMENSIONS)
        for dim in DIMENSIONS:
            g = m[m["dimension"] == dim].set_index("value")["below_best_mean"].dropna()
            if len(g) < 2:  # fixed in the chosen space
                continue
            order = [v for v in _ordered_values(dim) if v in g.index]
            best = g.idxmax()
            i = order.index(best)
            wide_order = _ordered_values(dim)
            at_edge = best in (wide_order[0], wide_order[-1])
            neighbour = order[i + 1] if i == 0 else order[i - 1] if i == len(order) - 1 else None
            margin = float(g[best] - g[neighbour]) if neighbour is not None else np.nan
            rows.append({"family": family, "dimension": dim, "best_value": best, "best_below_best": float(g[best]),
                         "neighbour": neighbour, "margin": margin, "at_wide_edge": at_edge,
                         "extend": bool(at_edge and neighbour is not None and margin > EDGE_MARGIN)})
    return pd.DataFrame(rows)


def selected_table(t: pd.DataFrame) -> pd.DataFrame:
    """The NLL-selected trial over all trials of each (family, world), with its gains and regret."""
    usable = t[t["usable"]]
    sel = usable.loc[usable.groupby(CELL)["val_nll"].idxmin()].copy()
    best = usable.groupby(CELL)["gain_greedy"].max().rename("best_gain_greedy")
    sel = sel.merge(best.reset_index(), on=CELL)
    sel["regret"] = sel["best_gain_greedy"] - sel["gain_greedy"]
    assert (sel["trial"] == sel["selected_trial"]).all(), "the run's selection differs from the NLL argmin"
    return sel


def tune_main(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load_blob_trials(*args.runs)
    t.to_csv(out / "tuning_trials_long.csv.gz", index=False, float_format="%.8g")
    marginal_table(t).to_csv(out / "tuning_marginals_wide.csv", index=False, float_format="%.6g")
    selected_table(t).to_csv(out / "tuning_selected.csv", index=False, float_format="%.6g")
    decision = tuning_decision(t, k=args.k, resamples=args.resamples)
    decision.to_csv(out / "tuning_decision.csv", index=False, float_format="%.6g")
    chosen = {r["family"]: eval(r["space"]) for _, r in decision[decision["chosen"]].iterrows()}
    pd.concat([marginal_table(t[t["family"] == f], space).assign(space="chosen") for f, space in chosen.items()],
              ignore_index=True).to_csv(out / "tuning_marginals_chosen.csv", index=False, float_format="%.6g")
    edges = edge_check(t, chosen)
    edges.to_csv(out / "tuning_edges.csv", index=False, float_format="%.6g")
    for _, r in decision[decision["chosen"]].iterrows():
        wide = decision[(decision["family"] == r["family"]) & (decision["candidate"] == "wide")]["gain_greedy"].iloc[0]
        print(f"[{r['family']}] chosen: {r['candidate']} (selected greedy gain {r['gain_greedy']:.3f}; wide {wide:.3f})")
    if len(edges) and edges["extend"].any():
        print("edge rule: extend " + ", ".join(f"{r.family}:{r.dimension} past {r.best_value}"
                                                for r in edges[edges["extend"]].itertuples()))
    print(f"wrote {out}: {len(t)} trial rows; families {sorted(t['family'].unique())}; "
          f"{t[WORLD].drop_duplicates().shape[0]} worlds")


# ---------------------------------------------------------------------------------------------- main comparison
TRAIN_SIZE = 25_000
FS = Path("artifacts/full_study")
CAP_RUN, WARM_RUN = FS / "run_cause_fair_cap_25k", FS / "run_cause_fair_warm_25k"
DM_OWN_ANIME = FS / "run_cause_fair_dm_oldspace_anime"
ORACLE_RUN = FS / "run_class_oracles_20261005"
BLOB_ARMS = ("blob_nq", "blob_mnq")
REFERENCE_ARMS = ("opc", "dm_own", "dm", "tempered_logger")
LIKELIHOOD_ARMS = ("cap_c", "cap_t", "warm_c")
ARMS = BLOB_ARMS + LIKELIHOOD_ARMS + REFERENCE_ARMS
NAMES = {"blob_nq": "BLOB-NQ (supplied source)", "blob_mnq": "BLOB-MNQ (supplied source)",
         "cap_c": "CausE-cap-C, ρ = 0 (plain likelihood, OPC's class)", "cap_t": "CausE-cap-T, ρ = 0",
         "warm_c": "CausE-warm-C, ρ = 0 (likelihood, free vectors)", "opc": "OPC (harmonic:0.1)",
         "dm_own": "DM-only (own range)", "dm": "DM-only (OPC's range)", "tempered_logger": "tempered logger"}
# each arm's ranking family over the logger's vectors (§1.6, §3): its value oracle is its structural ceiling
OWN_CLASS = {"blob_nq": "blob", "blob_mnq": "blob", "cap_c": "affine_bilinear", "cap_t": "affine_bilinear",
             "opc": "affine_bilinear", "dm_own": "affine_bilinear", "dm": "affine_bilinear"}
ORACLE_CLASSES = ("affine_bilinear", "blob", "bilinear")


def blob_rows(run_dirs, seeds=(100, 101)) -> pd.DataFrame:
    """The selected BLOB policy per world and family."""
    from training.analyze_cause_fair import BIAS_ORDER

    frames = []
    for run in run_dirs:
        for p in sorted(Path(run).glob("dataset=*/summary_metrics.csv")):
            tags = _tags(p.parent.name)
            if tags["bias"] not in BIAS_ORDER or int(tags["seed"]) not in seeds:
                continue
            s = pd.read_csv(p)
            s = s[(s["train_size"] == TRAIN_SIZE) & s["method"].astype(str).str.startswith("blob")].copy()
            s["run"], s["dataset"], s["bias"], s["seed"] = Path(run).name, tags["dataset"], tags["bias"], int(tags["seed"])
            frames.append(s)
    if not frames:
        raise FileNotFoundError(f"no finished BLOB conditions under {run_dirs}")
    s = pd.concat(frames, ignore_index=True)
    s["arm"] = s["method"]
    s = s.rename(columns={"policy_rewards": "V", "policy_rewards_greedy": "V_greedy",
                          "policy_rewards_tempered": "V_tempered"})
    s["regret_greedy"] = s["oracle_selected_value_greedy"] - s["V_greedy"]
    s["regret"] = s["oracle_selected_value"] - s["V"]
    return s


def likelihood_rows() -> pd.DataFrame:
    """CausE-capacity-matched (C and T) and CausE-warm-C at ρ = 0 from the fair comparison (not rerun): no
    randomized rows, so they are click-likelihood fits on the same N warm rows."""
    from training.analyze_cause_fair import cause_rows

    c = cause_rows([CAP_RUN, WARM_RUN])
    return c[(c["rho"] == 0.0) & c["arm"].isin(LIKELIHOOD_ARMS)].copy()


def class_oracle_table(root: Path = ORACLE_RUN) -> pd.DataFrame:
    """Per world: each class's value-oracle and likelihood-oracle greedy values, the value oracle's stochastic value
    and the likelihood oracle's objective (the class's infinite-data NLL under π0)."""
    frames = [pd.read_csv(p) for p in sorted(Path(root).glob("*/class_oracles.csv"))]
    if not frames:
        raise FileNotFoundError(f"no class_oracles.csv under {root}")
    o = pd.concat(frames, ignore_index=True)
    o["bias"] = o["bias_label"]
    o["key"] = o["objective"] + "_" + o["class"]
    wide = o.pivot_table(index=WORLD, columns="key", values=["greedy", "value", "fit_objective"], aggfunc="first")
    wide.columns = [f"{k}_{v}" for v, k in wide.columns]
    keep = [c for c in wide.columns if c.endswith("_greedy") or (c.startswith("value_") and c.endswith("_value"))
            or (c.startswith("likelihood_") and c.endswith("_fit_objective"))]
    return wide[keep].reset_index()


def condition_table(blob: pd.DataFrame, refs: pd.DataFrame, lik: pd.DataFrame, oracles: pd.DataFrame | None,
                    best_item: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per world × arm: values, gains (CTR points), the shares of the representation loss and of the arm's
    own value oracle repaired, the selection regret, and the class oracles of the world.

    gain_greedy = V_greedy − V_logger_greedy; gain = V − V_logger (stochastic; for BLOB and CausE the tempered softmax
    is gain_tempered); frac_loss = gain_greedy / (V_target_best − V_logger_greedy); frac_oracle = gain_greedy / (the
    value oracle of the arm's class − V_logger_greedy)."""
    from training.analyze_cause_fair import BIAS_ORDER, world_references

    keep = ["dataset", "bias", "seed", "arm", "V", "V_greedy", "V_tempered", "temper_scale", "regret", "regret_greedy",
            "sel_estimate", "sel_estimate_low", "val_nll", "val_auc", "val_dr_greedy", "val_dr_greedy_low",
            "n_trials", "n_finite_trials", "lr", "epochs", "wa_m", "wb_m", "kappa_s", "sp_wa", "sp_wb", "wc",
            "zeta_norm", "kappa_rms", "kappa_sd_post", "deviation_ratio", "l2_pen", "cf_pen", "cause_tie",
            "logit_scale", "initial_reward", "logger_greedy"]
    t = pd.concat([d[[c for c in keep if c in d.columns]] for d in (blob, lik, refs)], ignore_index=True)
    t = t.reindex(columns=keep)
    t = t.merge(world_references(), on=WORLD, how="left", validate="many_to_one")
    for own, ref in (("initial_reward", "V_logger"), ("logger_greedy", "V_logger_greedy")):
        ok = t[own].notna()
        assert np.allclose(t.loc[ok, own], t.loc[ok, ref], atol=1e-6), own
    t["gain_greedy"] = 100 * (t["V_greedy"] - t["V_logger_greedy"])
    t["gain"] = 100 * (t["V"] - t["V_logger"])
    t["gain_tempered"] = 100 * (t["V_tempered"] - t["V_logger"])
    loss = t["ceiling"] - t["V_logger_greedy"]
    t["representation_loss"] = 100 * loss
    t["frac_loss"] = np.where(loss > 1e-3, t["gain_greedy"] / 100 / loss, np.nan)
    t["regret_greedy"] = 100 * t["regret_greedy"]
    t["regret"] = 100 * t["regret"]
    t["best_trial_gain"] = t["gain_greedy"] + t["regret_greedy"]
    t["own_class"] = t["arm"].map(OWN_CLASS)
    if oracles is not None:
        t = t.merge(oracles, on=WORLD, how="left", validate="many_to_one")
        for cls in ORACLE_CLASSES:
            for obj in ("value", "likelihood"):
                col = f"{obj}_{cls}_greedy"
                if col in t:
                    t[f"oracle_{obj}_{cls}_gain"] = 100 * (t[col] - t["V_logger_greedy"])
        own = np.full(len(t), np.nan)
        own_lik = np.full(len(t), np.nan)
        for cls in ORACLE_CLASSES:
            m = (t["own_class"] == cls).to_numpy()
            if f"oracle_value_{cls}_gain" in t:
                own[m] = t.loc[m, f"oracle_value_{cls}_gain"]
            if f"oracle_likelihood_{cls}_gain" in t:
                own_lik[m] = t.loc[m, f"oracle_likelihood_{cls}_gain"]
        t["own_value_oracle_gain"] = own
        t["own_likelihood_oracle_gain"] = own_lik
        t["frac_oracle"] = np.where(own > 0.1, t["gain_greedy"] / own, np.nan)
        # the accounting of §5: gain = own value oracle − (oracle − the best of the arm's trials) − selection regret
        t["training_gap"] = t["own_value_oracle_gain"] - t["best_trial_gain"]
    t["oracle_linear_gain"] = 100 * (t["oracle_linear_greedy"] - t["V_logger_greedy"])
    if best_item is not None:
        t = t.merge(best_item[["dataset", "seed", "best_single_item"]], on=["dataset", "seed"], how="left")
        t["best_item_gain"] = 100 * (t["best_single_item"] - t["V_logger_greedy"])
    t["bias_order"] = t["bias"].map({b: i for i, b in enumerate(BIAS_ORDER)})
    t["arm_order"] = t["arm"].map({a: i for i, a in enumerate(ARMS)})
    return t.sort_values(["bias_order", "dataset", "seed", "arm_order"]).drop(columns=["bias_order", "arm_order"]) \
        .reset_index(drop=True)


SUMMARY_COLUMNS = ("gain_greedy", "gain", "gain_tempered", "frac_loss", "frac_oracle", "regret_greedy",
                   "best_trial_gain", "training_gap", "own_value_oracle_gain", "own_likelihood_oracle_gain",
                   "val_nll", "val_auc", "kappa_rms", "deviation_ratio", "sp_wa", "sp_wb", "temper_scale",
                   "representation_loss", "best_item_gain")


def _panels(t: pd.DataFrame):
    from training.analyze_cause_fair import BIAS_ORDER

    for b in BIAS_ORDER:
        if (t["bias"] == b).any():
            yield b, t[t["bias"] == b]
    yield "biased (pooled)", t[t["bias"] != "none"]


def summary_table(t: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% CI over worlds per bias × arm, and over the 24 biased worlds pooled."""
    rows = []
    for bias, tb in _panels(t):
        for arm, g in tb.groupby("arm"):
            r = {"bias": bias, "arm": arm, "worlds": len(g)}
            for col in SUMMARY_COLUMNS:
                if col in g and g[col].notna().any():
                    m, lo, hi, _n = mean_ci(g[col])
                    r[col], r[col + "_lo"], r[col + "_hi"] = m, lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def paired_table(t: pd.DataFrame, a: str, b_arms, col: str = "gain_greedy", b_col: str | None = None) -> pd.DataFrame:
    """``a`` − each arm of ``b_arms``, paired by world: mean, 95% CI and the worlds where ``a`` is higher, per bias
    and pooled over the biased worlds. ``b_col``: the b arms' column when it differs (a tempered softmax against
    OPC's own stochastic policy)."""
    b_col = b_col or col
    va = t[t["arm"] == a].set_index(WORLD)[col]
    rows = []
    for arm in b_arms:
        d = (va - t[t["arm"] == arm].set_index(WORLD)[b_col]).dropna()
        frame = d.reset_index()
        frame.columns = WORLD + ["d"]
        for bias, g in _panels(frame):
            if g.empty:
                continue
            m, lo, hi, n = mean_ci(g["d"])
            rows.append({"a": a, "b": arm, "col": col, "b_col": b_col, "bias": bias, "a_minus_b": m, "ci_lo": lo,
                         "ci_hi": hi, "worlds": n, "a_higher": int((g["d"] > 0).sum())})
    return pd.DataFrame(rows)


def oracle_summary(oracles: pd.DataFrame) -> pd.DataFrame:
    """Per bias: each class's value and likelihood oracle greedy gains over the logger (CTR points), and the paired
    differences that separate the class from the objective: value − likelihood within a class, and blob − affine
    and affine − bilinear within an objective."""
    from training.analyze_cause_fair import world_references

    o = oracles.merge(world_references()[WORLD + ["V_logger_greedy"]], on=WORLD, how="left")
    for cls in ORACLE_CLASSES:
        for obj in ("value", "likelihood"):
            o[f"{obj}:{cls}"] = 100 * (o[f"{obj}_{cls}_greedy"] - o["V_logger_greedy"])
        o[f"value−likelihood:{cls}"] = o[f"value:{cls}"] - o[f"likelihood:{cls}"]
    for obj in ("value", "likelihood"):
        o[f"{obj}:blob−affine"] = o[f"{obj}:blob"] - o[f"{obj}:affine_bilinear"]
        o[f"{obj}:affine−bilinear"] = o[f"{obj}:affine_bilinear"] - o[f"{obj}:bilinear"]
    cols = [c for c in o.columns if ":" in c]
    rows = []
    for bias, g in _panels(o):
        for c in cols:
            m, lo, hi, n = mean_ci(g[c])
            rows.append({"bias": bias, "quantity": c, "mean": m, "ci_lo": lo, "ci_hi": hi, "worlds": n})
    return pd.DataFrame(rows)


def accounting_table(t: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """a − b in greedy gain split, per world, into its class's ceiling (value oracles), training (the ceiling minus the
    best of the arm's own 20 trials) and selection (the best trial minus the selected one):
    Δgain = Δceiling − Δtraining − Δselection."""
    cols = ["gain_greedy", "own_value_oracle_gain", "training_gap", "regret_greedy"]
    wa = t[t["arm"] == a].set_index(WORLD)[cols]
    wb = t[t["arm"] == b].set_index(WORLD)[cols]
    d = (wa - wb).dropna().reset_index()
    rows = []
    for bias, g in _panels(d):
        r = {"a": a, "b": b, "bias": bias, "worlds": len(g)}
        for c, name in zip(cols, ("gain", "ceiling", "training", "selection")):
            m, lo, hi, _ = mean_ci(g[c])
            r[name], r[name + "_lo"], r[name + "_hi"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def diagnostics_table(diag: pd.DataFrame) -> pd.DataFrame:
    """The pick diagnostics (training/policy_diagnostics.py) per arm and bias: means over worlds."""
    d = diag.copy()
    d["arm"] = d["arm"].str.replace(r"_n\d+(_r\d+)?$", "", regex=True)
    num = [c for c in d.columns if c not in WORLD + ["run", "arm", "world_seconds"] and pd.api.types.is_numeric_dtype(d[c])]
    rows = []
    for bias, g in _panels(d):
        for arm, ga in g.groupby("arm"):
            r = {"bias": bias, "arm": arm, "worlds": ga[WORLD].drop_duplicates().shape[0]}
            r.update({c: float(ga[c].mean()) for c in num})
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------- figures
# one colour per entity (Okabe-Ito), as in the CausE comparison; the two BLOB families share a hue, NQ solid
COLORS = {"blob_nq": "#D55E00", "blob_mnq": "#D55E00", "cap_c": "#CC79A7", "cap_t": "#CC79A7", "warm_c": "#56B4E9",
          "opc": "#0072B2", "dm_own": "#E69F00", "dm": "#E69F00", "tempered_logger": "#009E73"}
MARKERS = {"blob_nq": "o", "blob_mnq": "s", "cap_c": "D", "cap_t": "d", "warm_c": "^", "opc": "P", "dm_own": "v",
           "dm": "v", "tempered_logger": "x"}
FIG_ARMS = ("blob_nq", "blob_mnq", "cap_c", "warm_c", "opc", "dm_own")


def fig_gains(s: pd.DataFrame, out: Path, col: str = "gain_greedy", name: str = "fig1_greedy_gain",
              xlabel: str = "greedy value − logger's greedy value (CTR pts)", arms=FIG_ARMS) -> None:
    """Per bias panel: each arm's mean and 95% CI over worlds."""
    from training.analyze_cause_fair import BIAS_NAMES
    from training.representation_report import _plt, _save

    plt = _plt()
    panels = [p for p in ["biased (pooled)", "none", "w-high.g-none.v-none", "w-none.g-high.v-none",
                          "w-none.g-none.v-high", "high"] if p in set(s["bias"])]
    arms = [a for a in arms if a in set(s["arm"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels) + 1.6, 0.36 * len(arms) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    data = []
    for ax, panel in zip(axes, panels):
        g = s[s["bias"] == panel].set_index("arm")
        for i, arm in enumerate(arms):
            if arm not in g.index or col not in g or not np.isfinite(g.loc[arm, col]):
                continue
            r = g.loc[arm]
            ax.errorbar([r[col]], [i], xerr=[[r[col] - r[col + "_lo"]], [r[col + "_hi"] - r[col]]], color=COLORS[arm],
                        marker=MARKERS[arm], markersize=5, capsize=2, elinewidth=1, linestyle="none")
            data.append({"panel": panel, "arm": arm, "mean": r[col], "lo": r[col + "_lo"], "hi": r[col + "_hi"]})
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("biased worlds (24)" if panel == "biased (pooled)" else BIAS_NAMES.get(panel, panel) + " (6)")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([NAMES[a] for a in arms])
    axes[0].invert_yaxis()  # once: the panels share the y axis
    fig.supxlabel(xlabel, fontsize=9)
    fig.suptitle("Target value at 25k (mean and 95% CI over worlds); BLOB and the likelihood learners ignore "
                 "propensities", y=1.02, fontsize=9.5)
    _save(fig, out, name, pd.DataFrame(data))


def fig_accounting(acc: pd.DataFrame, out: Path) -> None:
    """BLOB − OPC and BLOB − CausE-cap, split into the class ceiling, training and selection (Δgain = Δceiling −
    Δtraining − Δselection), per bias panel."""
    from training.analyze_cause_fair import BIAS_NAMES
    from training.representation_report import _plt, _save

    plt = _plt()
    pairs = [(a, b) for a, b in acc[["a", "b"]].drop_duplicates().itertuples(index=False)]
    panels = [p for p in ["biased (pooled)", "none", "w-high.g-none.v-none", "w-none.g-high.v-none",
                          "w-none.g-none.v-high", "high"] if p in set(acc["bias"])]
    parts = (("gain", "net difference", "#000000"), ("ceiling", "class ceiling (value oracles)", "#0072B2"),
             ("training", "− training gap (ceiling − best of 20 trials)", "#E69F00"),
             ("selection", "− selection regret (best − selected)", "#009E73"))
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    data = []
    for ax, (a, b) in zip(axes, pairs):
        g = acc[(acc["a"] == a) & (acc["b"] == b)].set_index("bias")
        x = np.arange(len(panels))
        for j, (col, label, color) in enumerate(parts):
            sign = -1.0 if col in ("training", "selection") else 1.0
            vals = np.array([sign * g.loc[p, col] if p in g.index else np.nan for p in panels])
            lo = np.array([sign * g.loc[p, col + ("_hi" if sign < 0 else "_lo")] if p in g.index else np.nan for p in panels])
            hi = np.array([sign * g.loc[p, col + ("_lo" if sign < 0 else "_hi")] if p in g.index else np.nan for p in panels])
            xx = x + (j - 1.5) * 0.19
            ax.bar(xx, vals, width=0.17, color=color, label=label, edgecolor="white", linewidth=0.5)
            ax.errorbar(xx, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="#444444", elinewidth=0.7, capsize=1.5)
            data += [{"a": a, "b": b, "panel": p, "part": col, "signed": v} for p, v in zip(panels, vals)]
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["biased\n(24)" if p == "biased (pooled)" else BIAS_NAMES.get(p, p).replace(" ", "\n")
                            for p in panels])
        ax.set_title(f"{NAMES[a]} − {NAMES[b]}")
    axes[0].set_ylabel("CTR points (greedy)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("Where the difference comes from: structural ceiling, training and selection (mean and 95% CI over "
                 "worlds)", y=1.03, fontsize=9.5)
    _save(fig, out, "fig2_accounting", pd.DataFrame(data))


PICK_ARMS = {"logger": ("logger", "#7F7F7F", "x"), "opc": ("OPC (harmonic:0.1)", "#0072B2", "P"),
             "causecap_c_r000": ("CausE-cap-C, ρ = 0", "#CC79A7", "D"),
             "blob_nq": ("BLOB-NQ", "#D55E00", "o"), "blob_mnq": ("BLOB-MNQ", "#D55E00", "s"),
             "oracle_affine_bilinear_value": ("value oracle, OPC's class", "#0072B2", "*"),
             "oracle_blob_value": ("value oracle, BLOB's class", "#D55E00", "*"),
             "oracle_affine_bilinear_likelihood": ("likelihood oracle, OPC's class", "#CC79A7", "*"),
             "oracle_blob_likelihood": ("likelihood oracle, BLOB's class", "#D55E00", "X")}


def fig_picks(dt: pd.DataFrame, out: Path, panel: str = "biased (pooled)") -> None:
    """Where each selected policy recommends and what it is worth there (training/policy_diagnostics.py), over the
    biased worlds: the share of users whose pick is not the logger's top item, the true click probability at those
    picks and at the logger's top item, and for the click models the optimism at the picks."""
    from training.representation_report import _plt, _save

    plt = _plt()
    d = dt[dt["bias"] == panel].set_index("arm")
    arms = [a for a in PICK_ARMS if a in d.index]
    d = d.assign(moved_off_top1=1.0 - d["agree_logger_top1"])
    metrics = (("moved_off_top1", "share of users whose pick is not\nthe logger's top item", 1.0, None),
               ("q_at_picks_not_top1", "true CTR at those picks (filled) and\nat the logger's top item (open), %",
                100.0, "q_at_picks_top1"),
               ("optimism_at_pick", "click model's optimism at its picks,\nΣ prior (σ(f) − q), CTR pts", 100.0, None))
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.5, 0.34 * len(arms) + 1.3), sharey=True)
    data = []
    for ax, (col, label, scale, col2) in zip(axes, metrics):
        for i, arm in enumerate(arms):
            name, color, marker = PICK_ARMS[arm]
            if col in d and np.isfinite(d.loc[arm, col]):
                ax.plot([scale * d.loc[arm, col]], [i], marker=marker, color=color, markersize=6, linestyle="none")
                data.append({"panel": panel, "arm": arm, "metric": col, "value": scale * d.loc[arm, col]})
            if col2 and col2 in d and np.isfinite(d.loc[arm, col2]):
                ax.plot([scale * d.loc[arm, col2]], [i], marker=marker, color=color, markersize=6, linestyle="none",
                        markerfacecolor="none")
                data.append({"panel": panel, "arm": arm, "metric": col2, "value": scale * d.loc[arm, col2]})
        if col == "optimism_at_pick":
            ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(label)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([PICK_ARMS[a][0] for a in arms])
    axes[0].invert_yaxis()
    fig.suptitle("Where the selected policies recommend and what it is worth there (exact; means over the 24 biased "
                 "worlds)", y=1.02, fontsize=9.5)
    _save(fig, out, "fig3_picks", pd.DataFrame(data))


TABLE_PANELS = ("biased (pooled)", "none", "w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high",
                "high")


def _panel_name(p: str) -> str:
    from training.analyze_cause_fair import BIAS_NAMES

    return "biased (24)" if p == "biased (pooled)" else BIAS_NAMES.get(p, p) + " (6)"


def _cell(s: pd.DataFrame, panel: str, arm: str, col: str, ci: bool = True, digits: int = 2, sign: bool = True) -> str:
    from training.analyze_cause_fair import _fmt

    g = s[(s["bias"] == panel) & (s["arm"] == arm)]
    if g.empty or col not in g or not np.isfinite(g[col].iloc[0]):
        return "—"
    r = g.iloc[0]
    return _fmt(r[col], r.get(col + "_lo", np.nan) if ci else np.nan, r.get(col + "_hi", np.nan) if ci else np.nan,
                digits, sign)


def tables_md(t: pd.DataFrame, s: pd.DataFrame, p: pd.DataFrame, acc: pd.DataFrame | None,
              orc: pd.DataFrame | None) -> str:
    """The report's tables (docs/blob_controlled_integration.md §5) as markdown."""
    from training.analyze_cause_fair import _fmt, _md

    arms = [a for a in ARMS if a in set(t["arm"])]
    out = []
    for col, title in (("gain_greedy", "Greedy value − the logger's greedy value (primary)"),
                       ("gain", "Stochastic value − the logger's value (BLOB and CausE: raw softmax, τ = 1)"),
                       ("gain_tempered", "Stochastic value − the logger's value, BLOB and CausE tempered by the DR "
                                         "lower bound (OPC: its learned scale)")):
        rows = []
        for arm in arms:
            c = "gain" if (col == "gain_tempered" and arm not in BLOB_ARMS + LIKELIHOOD_ARMS) else col
            if col == "gain_tempered" and arm not in BLOB_ARMS + LIKELIHOOD_ARMS + ("opc", "tempered_logger"):
                continue
            rows.append({"arm": NAMES[arm], **{_panel_name(pn): _cell(s, pn, arm, c, ci=pn == "biased (pooled)")
                                                for pn in TABLE_PANELS}})
        out.append(f"**{title}** (CTR points; mean over worlds, 95% CI for the pooled biased worlds)\n\n"
                   + _md(pd.DataFrame(rows)))
    for col, title in (("gain_greedy", "Paired differences, greedy value"),
                       ("gain_tempered", "Paired differences, stochastic value (BLOB tempered)")):
        rows = []
        for (a, b), g in p[(p["col"] == col)].groupby(["a", "b"], sort=False):
            r = {"a − b": f"{NAMES[a]} − {NAMES[b]}"}
            for pn in TABLE_PANELS:
                x = g[g["bias"] == pn]
                r[_panel_name(pn)] = (f"{_fmt(x['a_minus_b'].iloc[0], x['ci_lo'].iloc[0], x['ci_hi'].iloc[0])} "
                                      f"({int(x['a_higher'].iloc[0])}/{int(x['worlds'].iloc[0])})") if len(x) else "—"
            rows.append(r)
        if rows:
            out.append(f"**{title}** (CTR points; mean [95% CI] over worlds; in parentheses the worlds where a is "
                       "higher)\n\n" + _md(pd.DataFrame(rows)))
    rows = []
    for arm in arms:
        rows.append({"arm": NAMES[arm],
                     "share of the representation loss": _cell(s, "biased (pooled)", arm, "frac_loss", sign=False),
                     "share of its class's value oracle": _cell(s, "biased (pooled)", arm, "frac_oracle", sign=False),
                     "class ceiling (pts)": _cell(s, "biased (pooled)", arm, "own_value_oracle_gain", ci=False),
                     "best of its 20 trials (pts)": _cell(s, "biased (pooled)", arm, "best_trial_gain", ci=False),
                     "selection regret (pts)": _cell(s, "biased (pooled)", arm, "regret_greedy", ci=False,
                                                     sign=False)})
    out.append("**Shares repaired, ceilings, best trials and selection regret, biased worlds** (greedy; mean [95% CI] "
               "over the 24 worlds)\n\n" + _md(pd.DataFrame(rows)))
    if orc is not None and len(orc):
        rows = []
        for q in orc["quantity"].unique():
            g = orc[orc["quantity"] == q]
            r = {"quantity": q}
            for pn in TABLE_PANELS:
                x = g[g["bias"] == pn]
                r[_panel_name(pn)] = _fmt(x["mean"].iloc[0], x["ci_lo"].iloc[0] if pn == "biased (pooled)" else np.nan,
                                          x["ci_hi"].iloc[0] if pn == "biased (pooled)" else np.nan) if len(x) else "—"
            rows.append(r)
        out.append("**Class oracles** (truth-trained, no logged data; greedy gain over the logger's greedy value, CTR "
                   "points; value = the value oracle, likelihood = the infinite-data likelihood fit under π0)\n\n"
                   + _md(pd.DataFrame(rows)))
    if acc is not None and len(acc):
        rows = []
        for (a, b), g in acc.groupby(["a", "b"], sort=False):
            for pn in TABLE_PANELS:
                x = g[g["bias"] == pn]
                if x.empty:
                    continue
                x = x.iloc[0]
                rows.append({"a − b": f"{NAMES[a]} − {NAMES[b]}", "worlds": _panel_name(pn),
                             **{k: _fmt(x[k], x[k + "_lo"], x[k + "_hi"]) for k in ("gain", "ceiling", "training",
                                                                                    "selection")}})
        out.append("**Accounting: Δgain = Δceiling − Δtraining − Δselection** (greedy, CTR points; mean [95% CI] over "
                   "worlds)\n\n" + _md(pd.DataFrame(rows)))
    return "\n\n".join(out) + "\n"


def data_identity_table(blob: pd.DataFrame, lik: pd.DataFrame) -> pd.DataFrame:
    """Per world and BLOB family: the click sum of its N training rows against CausE-cap's warm rows at ρ = 0 (the
    rows OPC trains on), and its validation size against CausE-cap's."""
    cap = lik[lik["arm"] == "cap_c"].set_index(WORLD)[["opc_collection_reward_sum", "val_size"]]
    b = blob.set_index(WORLD)[["arm", "train_click_sum", "val_size"]].join(cap, rsuffix="_cap")
    b["train_rows_differ"] = (b["train_click_sum"] - b["opc_collection_reward_sum"]).abs() > 1e-6
    b["val_rows_differ"] = b["val_size"] != b["val_size_cap"]
    return b.reset_index()


def compare_main(args) -> None:
    from training.analyze_cause_fair import BEST_ITEM, reference_rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = blob_rows(args.blob_runs)
    refs = reference_rows(dm_own_runs=[DM_OWN_ANIME])
    refs = refs[refs["arm"].isin(REFERENCE_ARMS)]
    lik = likelihood_rows()
    ident = data_identity_table(blob, lik)
    ident.to_csv(out / "table_data_identity.csv", index=False)
    if ident["train_rows_differ"].any() or ident["val_rows_differ"].any():
        raise AssertionError("BLOB trained or selected on different rows than the other arms")
    oracles = class_oracle_table(Path(args.oracles)) if args.oracles else None
    t = condition_table(blob, refs, lik, oracles, pd.read_csv(BEST_ITEM))
    t.to_csv(out / "table_conditions.csv", index=False, float_format="%.8g")
    s = summary_table(t)
    s.to_csv(out / "table_summary.csv", index=False, float_format="%.6g")
    others = [a for a in ARMS if a in set(t["arm"])]
    pairs = []
    for a in [x for x in BLOB_ARMS if x in set(t["arm"])]:
        pairs += [paired_table(t, a, [b for b in others if b != a]),
                  paired_table(t, a, [b for b in others if b != a], col="gain"),
                  paired_table(t, a, [b for b in ("cap_c", "warm_c") if b in set(t["arm"])], col="gain_tempered"),
                  paired_table(t, a, [b for b in ("opc", "tempered_logger") if b in set(t["arm"])], col="gain_tempered",
                               b_col="gain")]
    pairs.append(paired_table(t, "opc", [b for b in ("cap_c", "dm_own") if b in set(t["arm"])]))
    p = pd.concat(pairs, ignore_index=True)
    p.to_csv(out / "table_paired.csv", index=False, float_format="%.6g")
    acc, orc = [], None
    if oracles is not None:
        orc = oracle_summary(oracles)
        orc.to_csv(out / "table_class_oracles.csv", index=False, float_format="%.6g")
        acc = [accounting_table(t, a, b) for a in BLOB_ARMS for b in ("opc", "cap_c") if {a, b} <= set(t["arm"])]
        acc += [accounting_table(t, "opc", "cap_c")] if {"opc", "cap_c"} <= set(t["arm"]) else []
        if acc:
            pd.concat(acc, ignore_index=True).to_csv(out / "table_accounting.csv", index=False, float_format="%.6g")
    (out / "tables.md").write_text(tables_md(t, s, p, pd.concat(acc, ignore_index=True) if acc else None, orc))
    fig_gains(s, out)
    fig_gains(s, out, col="gain_tempered", name="fig1b_tempered_gain",
              xlabel="stochastic value − logger's value (CTR pts); BLOB and CausE tempered", arms=("blob_nq", "blob_mnq",
                                                                                                   "cap_c", "warm_c"))
    if oracles is not None and acc:
        fig_accounting(pd.concat(acc, ignore_index=True), out)
    if args.diagnostics:
        diag = pd.read_csv(Path(args.diagnostics) / "policy_diagnostics.csv")
        dt = diagnostics_table(diag)
        dt.to_csv(out / "table_pick_diagnostics.csv", index=False, float_format="%.6g")
        fig_picks(dt, out)
        pp = pd.read_csv(Path(args.diagnostics) / "policy_pairs.csv")
        pp.to_csv(out / "policy_pairs.csv", index=False, float_format="%.8g")
    print(f"wrote {out}: {len(t)} rows; arms {sorted(t['arm'].unique())}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune", help="the tuning-stage tables")
    tune.add_argument("--runs", nargs="+", required=True)
    tune.add_argument("--out", required=True)
    tune.add_argument("--k", type=int, default=10, help="trials per simulated study (default 10)")
    tune.add_argument("--resamples", type=int, default=300)
    comp = sub.add_parser("compare", help="the bounded comparison's tables")
    comp.add_argument("--blob-runs", nargs="+", required=True)
    comp.add_argument("--oracles", default=str(ORACLE_RUN), help="the class-oracle run ('' to skip)")
    comp.add_argument("--diagnostics", default=None, help="the policy_diagnostics output folder")
    comp.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.command == "tune":
        tune_main(args)
    else:
        compare_main(args)


if __name__ == "__main__":
    main()
