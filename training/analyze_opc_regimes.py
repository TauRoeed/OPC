"""The regime map, the OPC-favorability margin and the 25k decomposition (docs/opc_gradient_regime_study.md §6, §8-§10).

    python -m training.analyze_opc_regimes map --runs RUN... --states STATES_RUN... --gradients GRAD_TABLE... --out DIR
    python -m training.analyze_opc_regimes decompose --runs RUN... --states STATES_RUN --empirical EMP_RUN --out DIR

Greedy CTR points throughout (the primary metric); a trial's gain is over its world's logger greedy value.

- Selections per world, support level (logger greedy share), N and arm: native (the trainer's choice), common (the DR
  lower bound of the greedy policy), best of 20 (diagnostic) and the mean of the 20 trials.
- The margin F = M_L − (T_O − T_L) − (S_O − S_L), with M_L = V* − V(θ_log*), T_L = V(θ_log*) − V(best_L),
  T_O = V* − V(best_O), S = V(best) − V(selected). F is V(selected_O) − V(selected_L) exactly; the identity is checked
  per world.
- The 25k decomposition per arm: V* − V(selected) = M + E + O + S, the objective (surrogate) mismatch, the
  empirical-objective gap, the optimization / minibatch / search gap and the selection gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_cause_fair import _tags, mean_ci
from training.run_state import read_trials_long

WORLD = ["dataset", "bias", "seed"]
CELL = ["share", "train_size"]
COMMON = "diag_dr_greedy_low"
LIKELIHOOD = "shared_likelihood"
OPC_ARMS = ("shared_opc_raw", "shared_opc")
BIAS_ORDER = ("none", "w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high", "high")
BIAS_NAMES = {"none": "no bias", "w-high.g-none.v-none": "warp", "w-none.g-high.v-none": "group",
              "w-none.g-none.v-high": "vector", "high": "combined"}
# arm -> its empirical-objective reference and its population optimum (§6)
ARM_OBJECTIVE = {"shared_likelihood": ("likelihood", "likelihood"), "shared_opc_raw": ("dr_raw", "value"),
                 "shared_opc": ("dr_harmonic", "harmonic"), "shared_opc_raw_oq": ("dr_raw_oq", "value"),
                 "shared_opc_oq": ("dr_harmonic_oq", "value"), "shared_opc_b8192": ("dr_harmonic", "harmonic"),
                 "shared_opc_raw_b8192": ("dr_raw", "value"), "shared_opc_bfull": ("dr_harmonic", "harmonic"),
                 "shared_opc_raw_bfull": ("dr_raw", "value")}


# ----------------------------------------------------------------------------------------------------- trials
def load_trials(*runs) -> pd.DataFrame:
    """Every shared-arm trial of the runs, with the world, the support level (the condition's logger greedy share),
    the logger's greedy value and each trial's greedy gain."""
    frames = []
    for run in runs:
        for cond in sorted(Path(run).glob("dataset=*")):
            if not (cond / "trials_long.csv").exists() or not (cond / "summary_metrics.csv").exists():
                continue
            t = read_trials_long(cond / "trials_long.csv")
            t = t[t["method"].astype(str).str.startswith("shared_")].copy()
            if t.empty:
                continue
            s = pd.read_csv(cond / "summary_metrics.csv")
            v0 = float(s.loc[s["train_size"] == 0, "policy_rewards_greedy"].iloc[0])
            meta = json.loads((cond / "run_meta.json").read_text())
            share = float(meta["world"].get("logger_greedy_share", 0.8))
            tags = _tags(cond.name)
            t = t.assign(dataset=tags["dataset"], bias=tags["bias"], seed=int(tags["seed"]), share=share,
                         run_tag=Path(run).name, V_logger_greedy=v0)
            t["gain_greedy"] = 100 * (t["actual_reward_greedy"] - v0)
            frames.append(t)
    if not frames:
        raise FileNotFoundError(f"no shared-arm trials under {runs}")
    return pd.concat(frames, ignore_index=True)


def select(t: pd.DataFrame) -> pd.DataFrame:
    """Per world, support level, N and arm: the native, common and best trials and the mean of the trials."""
    rows = []
    for key, g in t.groupby(WORLD + CELL + ["method"]):
        r = dict(zip(WORLD + CELL + ["arm"], key))
        native = g[g["is_best_in_run"].astype(bool)]
        if len(native) != 1:
            raise ValueError(f"{key}: {len(native)} natively selected trials")
        picks = {"native": native.iloc[0], "common": g.loc[g[COMMON].idxmax()],
                 "best": g.loc[g["actual_reward_greedy"].idxmax()]}
        for name, tr in picks.items():
            r[f"{name}_V_greedy"] = float(tr["actual_reward_greedy"])
            r[f"{name}_gain"] = float(tr["gain_greedy"])
            r[f"{name}_trial"] = int(tr["trial_number"])
        r.update(trials=len(g), mean_gain=float(g["gain_greedy"].mean()), V_logger_greedy=float(g["V_logger_greedy"].iloc[0]),
                 diverged=int(g.get("diverged", pd.Series(False, index=g.index)).astype(bool).sum()))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- population optima
def load_states(*runs) -> pd.DataFrame:
    """Per world and support level: the greedy values of the population states (``opc_gradient_benchmark`` runs)."""
    rows = []
    for run in runs:
        for wdir in sorted(Path(run).glob("dataset=*")):
            if not (wdir / "states.json").exists():
                continue
            tags = _tags(wdir.name)
            meta = json.loads((wdir / "states.json").read_text())
            r = {"dataset": tags["dataset"], "bias": tags["bias"], "seed": int(tags["seed"]), "share": float(tags["lgs"])}
            for k, v in meta.items():
                if isinstance(v, dict):
                    r[f"{k}_greedy"] = v.get("greedy")
                    r[f"{k}_V"] = v.get("value")
            rows.append(r)
    return pd.DataFrame(rows).drop_duplicates(WORLD + ["share"])


def margins(sel: pd.DataFrame, states: pd.DataFrame, opc_arms=OPC_ARMS) -> pd.DataFrame:
    """Per world, support level, N, OPC arm and selection rule: M_L, T_L, T_O, S_L, S_O, F and the observed
    difference V(selected_O) − V(selected_L) (greedy CTR points)."""
    pop = states.set_index(WORLD + ["share"])
    vstar = states.groupby(WORLD)["value_greedy"].max()  # θ_value* does not depend on the logger: its best fit
    lik = sel[sel["arm"] == LIKELIHOOD].set_index(WORLD + CELL)
    rows = []
    for arm in opc_arms:
        o = sel[sel["arm"] == arm].set_index(WORLD + CELL)
        for idx in o.index.intersection(lik.index):
            world, share = idx[:3], idx[3]
            if (*world, share) not in pop.index:
                continue
            v_star, v_log = float(vstar.loc[world]), float(pop.loc[(*world, share), "likelihood_greedy"])
            L, O = lik.loc[idx], o.loc[idx]
            for rule in ("native", "common"):
                pts = lambda a, b: 100 * (a - b)
                m_l = pts(v_star, v_log)
                t_l = pts(v_log, L["best_V_greedy"])
                t_o = pts(v_star, O["best_V_greedy"])
                s_l = pts(L["best_V_greedy"], L[f"{rule}_V_greedy"])
                s_o = pts(O["best_V_greedy"], O[f"{rule}_V_greedy"])
                f = m_l - (t_o - t_l) - (s_o - s_l)
                obs = pts(O[f"{rule}_V_greedy"], L[f"{rule}_V_greedy"])
                rows.append({**dict(zip(WORLD + CELL, idx)), "opc_arm": arm, "rule": rule, "M_L": m_l, "T_L": t_l,
                             "T_O": t_o, "S_L": s_l, "S_O": s_o, "dT": t_o - t_l, "dS": s_o - s_l, "F": f,
                             "observed": obs, "identity_error": f - obs,
                             "best_diff": pts(O["best_V_greedy"], L["best_V_greedy"]),
                             "mean_diff": O["mean_gain"] - L["mean_gain"]})
    out = pd.DataFrame(rows)
    if not out.empty and float(out["identity_error"].abs().max()) > 1e-9:
        raise AssertionError(f"F differs from the observed difference by {out['identity_error'].abs().max()}")
    return out


def _panels(df: pd.DataFrame):
    for b in BIAS_ORDER:
        if (df["bias"] == b).any():
            yield BIAS_NAMES[b], df[df["bias"] == b]
    yield "biased (pooled)", df[df["bias"] != "none"]
    yield "misspecified (group, vector, combined)", df[df["bias"].isin(["w-none.g-high.v-none", "w-none.g-none.v-high", "high"])]


def regime_summary(m: pd.DataFrame) -> pd.DataFrame:
    """Per support level, N, OPC arm, rule and bias panel: the mean and 95% t-interval over worlds of F's parts and the
    observed difference, and the worlds where OPC is ahead."""
    rows = []
    for key, g in m.groupby(CELL + ["opc_arm", "rule"]):
        for panel, h in _panels(g):
            if h.empty:
                continue
            r = {**dict(zip(CELL + ["opc_arm", "rule"], key)), "panel": panel, "worlds": len(h),
                 "opc_ahead": int((h["observed"] > 0).sum())}
            for col in ("observed", "F", "M_L", "dT", "dS", "T_L", "T_O", "S_L", "S_O", "best_diff", "mean_diff"):
                mm, lo, hi, _ = mean_ci(h[col])
                r[col], r[f"{col}_lo"], r[f"{col}_hi"] = mm, lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ gradient join
GRAD_COLS = ("snr", "cos_mean", "p_positive", "rel_bias", "rel_mse", "w_ess_share", "w_max", "qhat_rmse_logging",
             "pop_ess_share", "cond_rel_bias")


def gradient_features(tables) -> pd.DataFrame:
    """Per world, support level, N and state: G3's and G5's gradient quality (one column per estimator and metric)."""
    df = pd.concat([pd.read_csv(p) for p in tables], ignore_index=True)
    df = df[df["estimator"].isin(["G3", "G5", "G5-G3 (paired bias)"])].copy()
    df["seed"] = df["seed"].astype(int)
    keep = [c for c in GRAD_COLS if c in df]
    wide = df.pivot_table(index=WORLD + CELL + ["state"], columns="estimator", values=keep)
    wide.columns = [f"{e}_{c}" for c, e in wide.columns]
    return wide.reset_index()


def correlations(m: pd.DataFrame, feats: pd.DataFrame, state: str = "mid_greedy") -> pd.DataFrame:
    """Spearman correlations, over worlds × cells, between the observed OPC − likelihood difference (native) and each
    gradient feature at ``state`` (and M_L); biased worlds only."""
    from scipy.stats import spearmanr

    f = feats[feats["state"] == state].drop(columns=["state"])
    d = m[(m["rule"] == "native") & (m["bias"] != "none")].merge(f, on=WORLD + CELL, how="inner")
    rows = []
    for arm, g in d.groupby("opc_arm"):
        for col in [c for c in f.columns if c not in WORLD + CELL] + ["M_L", "dT"]:
            x = pd.to_numeric(g[col], errors="coerce")
            ok = x.notna() & g["observed"].notna()
            if ok.sum() < 5:
                continue
            rho, p = spearmanr(x[ok], g.loc[ok, "observed"])
            rows.append({"opc_arm": arm, "state": state, "feature": col, "rho": float(rho), "p": float(p),
                         "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ the 25k decomposition
def decomposition(sel: pd.DataFrame, states: pd.DataFrame, empirical_run: Path, *, train_size: int = 25_000,
                  share: float = 0.8) -> pd.DataFrame:
    """Per world and arm at (N, share): M, E, O, S (native and common), each a difference of greedy values in points;
    they add up to V* − V(selected) exactly."""
    pop = states[np.isclose(states["share"], share)].set_index(WORLD)
    s = sel[(sel["train_size"] == train_size) & np.isclose(sel["share"], share)]
    rows = []
    for _, r in s.iterrows():
        world = tuple(r[c] for c in WORLD)
        arm = r["arm"]
        if arm not in ARM_OBJECTIVE or world not in pop.index:
            continue
        objective, target = ARM_OBJECTIVE[arm]
        wdir = Path(empirical_run) / f"dataset={world[0]}__bias={world[1]}__seed={world[2]}__lgs={share:g}"
        ep = wdir / f"empirical_{objective}.json"
        if not ep.exists():
            continue
        emp = json.loads(ep.read_text())
        v_star = float(pop.loc[world, "value_greedy"])
        if target == "likelihood":
            v_obj = float(pop.loc[world, "likelihood_greedy"])
        elif target == "harmonic":
            hp = wdir / "population_harmonic.json"
            if not hp.exists():
                continue
            v_obj = float(json.loads(hp.read_text())["greedy"])
        else:
            v_obj = v_star
        pts = lambda a, b: 100 * (a - b)
        base = {**dict(zip(WORLD, world)), "arm": arm, "objective": objective, "M": pts(v_star, v_obj),
                "E": pts(v_obj, emp["greedy"]), "O": pts(emp["greedy"], r["best_V_greedy"]),
                "emp_lr": emp["lr"], "emp_grad_norm": emp["grad_norm"],
                "emp_path_best_gain": 100 * (emp["path_best_greedy_sample"] - r["V_logger_greedy"]),
                "emp_gain": 100 * (emp["greedy"] - r["V_logger_greedy"]), "best_gain": r["best_gain"],
                "V_star_gain": 100 * (v_star - r["V_logger_greedy"])}
        for rule in ("native", "common"):
            row = {**base, "rule": rule, "S": pts(r["best_V_greedy"], r[f"{rule}_V_greedy"]),
                   "total": pts(v_star, r[f"{rule}_V_greedy"]), "selected_gain": r[f"{rule}_gain"]}
            row["sum_error"] = row["M"] + row["E"] + row["O"] + row["S"] - row["total"]
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and float(out["sum_error"].abs().max()) > 1e-9:
        raise AssertionError("the decomposition does not add up")
    return out


def decomposition_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in d.groupby(["arm", "rule"]):
        for panel, h in _panels(g):
            if h.empty:
                continue
            r = {"arm": key[0], "rule": key[1], "panel": panel, "worlds": len(h)}
            for col in ("M", "E", "O", "S", "total", "selected_gain", "best_gain", "emp_gain", "emp_path_best_gain",
                        "V_star_gain"):
                mm, lo, hi, _ = mean_ci(h[col])
                r[col], r[f"{col}_lo"], r[f"{col}_hi"] = mm, lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("map")
    mp.add_argument("--runs", nargs="+", required=True)
    mp.add_argument("--states", nargs="+", required=True)
    mp.add_argument("--gradients", nargs="*", default=[])
    mp.add_argument("--out", required=True)
    dp = sub.add_parser("decompose")
    dp.add_argument("--runs", nargs="+", required=True)
    dp.add_argument("--states", nargs="+", required=True)
    dp.add_argument("--empirical", required=True)
    dp.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load_trials(*args.runs)
    sel = select(t)
    sel.to_csv(out / "table_selected.csv", index=False, float_format="%.8g")
    states = load_states(*args.states)
    states.to_csv(out / "table_population_states.csv", index=False, float_format="%.8g")
    if args.cmd == "map":
        m = margins(sel, states)
        m.to_csv(out / "table_margins.csv", index=False, float_format="%.6g")
        regime_summary(m).to_csv(out / "table_regime_summary.csv", index=False, float_format="%.6g")
        if args.gradients:
            feats = gradient_features(args.gradients)
            feats.to_csv(out / "table_gradient_features.csv", index=False, float_format="%.6g")
            pd.concat([correlations(m, feats, s) for s in ("source", "mid", "mid_greedy")], ignore_index=True).to_csv(
                out / "table_correlations.csv", index=False, float_format="%.4g")
        print(f"wrote {out}: {len(m)} margin rows")
    else:
        d = decomposition(sel, states, Path(args.empirical))
        d.to_csv(out / "table_decomposition.csv", index=False, float_format="%.6g")
        decomposition_summary(d).to_csv(out / "table_decomposition_summary.csv", index=False, float_format="%.6g")
        print(f"wrote {out}: {len(d)} decomposition rows")


if __name__ == "__main__":
    main()
