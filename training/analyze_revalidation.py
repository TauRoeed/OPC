"""Analysis of the OPC revalidation's tuning runs (docs/simulator_fix_opc_revalidation_20261004.md, Phase 2).

Every run of the revalidation uses the paired random sampler: within a run the trained arms (OPC, DM-only,
no-propensity) train the same configurations with the same seeds, and runs of the same grid and search space
draw the same configurations, so trials pair across runs and arms. Every trial's true value is known (development
runs only), which these tools use to study the search space and the weights. Values are CTR points.

Subcommands:
  range    the true value against the optimization budget (lr x steps), and the selection protocol simulated on
           candidate sub-ranges of a wide search (each candidate keeps the trials inside it and selects among a
           random 20 of them by the arm's own score, averaged over resamples)
  paired   per-trial and selected differences between runs (same trials), with 95% t-intervals over conditions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ARMS = ("opc", "dm", "no_propensity")
KEYS = ["dataset", "bias", "seed", "train_size"]


def _tags(folder: str) -> dict:
    return dict(part.split("=", 1) for part in folder.split("__") if "=" in part)


def load_trials(*run_dirs, methods=ARMS) -> pd.DataFrame:
    """Every trial of the given runs, with the condition's tags, the logger's value (initial_reward) and the
    optimization budget: steps = epochs x ceil(n / batch); move = lr x steps per epoch x sum_e decay^e."""
    frames = []
    for run in run_dirs:
        for cond in sorted(Path(run).glob("dataset=*")):
            path = cond / "trials_long.csv"
            if not path.exists():
                continue
            t = pd.read_csv(path)
            t = t[t["method"].isin(methods) & (t["train_size"] > 0)].copy()
            tags = _tags(cond.name)
            t["run"] = Path(run).name
            t["dataset"], t["bias"], t["seed"] = tags["dataset"], tags["bias"], int(tags["seed"])
            frames.append(t)
    if not frames:
        raise FileNotFoundError(f"no trials_long.csv under {run_dirs}")
    t = pd.concat(frames, ignore_index=True)
    per_epoch = np.ceil(t["train_size"] / t["param_batch_size"])
    decay = t["param_lr_decay"].clip(upper=1.0)
    epochs = t["param_num_epochs"]
    geometric = np.where(np.isclose(decay, 1.0), epochs, (1.0 - decay ** epochs) / (1.0 - decay))
    t["steps"] = epochs * per_epoch
    t["move"] = t["param_lr"] * per_epoch * geometric
    t["gain"] = 100.0 * (t["actual_reward"] - t["initial_reward"])  # true gain over the logger, points
    return t


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


def response_table(t: pd.DataFrame, bins=(-np.inf, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf)) -> pd.DataFrame:
    """Mean true gain per arm × train size × log10(move) bin, and the Spearman correlation of gain with move."""
    t = t.copy()
    t["log10_move"] = np.log10(t["move"])
    t["move_bin"] = pd.cut(t["log10_move"], bins)
    rows = []
    for (arm, n), g in t.groupby(["method", "train_size"]):
        rho = g.groupby(["dataset", "bias", "seed"]).apply(
            lambda x: stats.spearmanr(x["gain"], x["move"]).statistic, include_groups=False)
        for b, gb in g.groupby("move_bin", observed=True):
            rows.append(dict(method=arm, train_size=n, move_bin=str(b), trials=len(gb), gain=gb["gain"].mean(),
                             gain_q90=gb["gain"].quantile(0.9), spearman_gain_move=float(rho.mean())))
    return pd.DataFrame(rows)


def simulate_protocol(t: pd.DataFrame, lr=None, epochs=None, *, k=20, resamples=200, seed=0) -> pd.DataFrame:
    """The selected trial's true gain when the search space is the sub-range ``lr`` × ``epochs`` of the run's
    wider space: per condition, arm and size, the trials inside it, a random ``k`` of them (all when fewer),
    the one with the best selection score (``value``), its gain; averaged over ``resamples`` draws (the same
    draws for every arm, so the arms stay paired)."""
    rng = np.random.default_rng(seed)
    inside = np.ones(len(t), dtype=bool)
    if lr is not None:
        inside &= (t["param_lr"] >= lr[0]) & (t["param_lr"] <= lr[1])
    if epochs is not None:
        inside &= (t["param_num_epochs"] >= epochs[0]) & (t["param_num_epochs"] <= epochs[1])
    sub = t[inside]
    rows = []
    for key, g in sub.groupby(KEYS):
        trials = np.sort(g["trial_number"].unique())
        draws = [trials if len(trials) <= k else rng.choice(trials, size=k, replace=False) for _ in range(resamples)]
        for arm, ga in g.groupby("method"):
            ga = ga.set_index("trial_number")
            sel = [ga.loc[d, "value"].idxmax() for d in draws]
            best = [ga.loc[d, "gain"].max() for d in draws]
            rows.append(dict(zip(KEYS, key), method=arm, available=len(trials), gain=ga.loc[sel, "gain"].mean(),
                             oracle_best=float(np.mean(best)), regret=float(np.mean(best) - ga.loc[sel, "gain"].mean())))
    return pd.DataFrame(rows)


def paired_runs(a: pd.DataFrame, b: pd.DataFrame, method: str = "opc") -> pd.DataFrame:
    """Run b minus run a for ``method``: per trial (identical configurations and seeds) and for each run's own
    selected trial, mean and 95% interval over the conditions of each bias × train size cell."""
    ka = a[a["method"] == method].set_index(KEYS + ["trial_number"])
    kb = b[b["method"] == method].set_index(KEYS + ["trial_number"])
    common = ka.index.intersection(kb.index)
    if len(common) == 0:
        raise ValueError("the runs share no trials")
    for c in ("param_lr", "param_num_epochs", "param_batch_size"):
        if not np.allclose(ka.loc[common, c], kb.loc[common, c]):
            raise ValueError(f"the runs' trials differ in {c}: not the same paired grid")
    d = (kb.loc[common, "gain"] - ka.loc[common, "gain"]).rename("diff").reset_index()
    per_cond = d.groupby(KEYS)["diff"].agg(["mean", lambda x: (x > 0).sum(), "size"]).reset_index()
    per_cond.columns = KEYS + ["trial_diff", "trials_better", "trials"]
    sa = a[(a["method"] == method) & a["is_best_in_run"].astype(bool)].set_index(KEYS)["gain"]
    sb = b[(b["method"] == method) & b["is_best_in_run"].astype(bool)].set_index(KEYS)["gain"]
    per_cond = per_cond.set_index(KEYS)
    per_cond["selected_diff"] = (sb - sa).reindex(per_cond.index)
    per_cond = per_cond.reset_index()
    rows = []
    for (bias, n), g in per_cond.groupby(["bias", "train_size"]):
        mt, lt, ht, k = mean_ci(g["trial_diff"])
        ms, ls, hs, _ = mean_ci(g["selected_diff"])
        rows.append(dict(bias=bias, train_size=n, conditions=k, trial_diff=mt, trial_lo=lt, trial_hi=ht,
                         trials_better=int(g["trials_better"].sum()), trials=int(g["trials"].sum()),
                         selected_diff=ms, selected_lo=ls, selected_hi=hs))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("range")
    r.add_argument("runs", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=20)
    p = sub.add_parser("paired")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--method", default="opc")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "range":
        t = load_trials(*args.runs)
        t.to_csv(out / "trials.csv", index=False)
        response_table(t).to_csv(out / "response_by_move.csv", index=False)
        print(response_table(t).to_string())
    else:
        a, b = load_trials(args.a), load_trials(args.b)
        res = paired_runs(a, b, args.method)
        res.to_csv(out / f"paired_{Path(args.b).name}_minus_{Path(args.a).name}.csv", index=False)
        print(res.to_string())


if __name__ == "__main__":
    main()
