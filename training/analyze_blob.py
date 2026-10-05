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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    tune = sub.add_parser("tune", help="the tuning-stage tables")
    tune.add_argument("--runs", nargs="+", required=True)
    tune.add_argument("--out", required=True)
    tune.add_argument("--k", type=int, default=10, help="trials per simulated study (default 10)")
    tune.add_argument("--resamples", type=int, default=300)
    args = ap.parse_args(argv)
    if args.command == "tune":
        tune_main(args)


if __name__ == "__main__":
    main()
