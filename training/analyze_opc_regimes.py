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
DATASETS = ("ml", "kuairand", "anime")
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
    """Per world and support level: the greedy values of the population states (``opc_gradient_benchmark`` runs) and
    their overlap with the logger (population ESS share, target mass where π0 < 1e-4; §12.3)."""
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
            if (wdir / "gstar.json").exists():
                for k, v in json.loads((wdir / "gstar.json").read_text()).items():
                    r[f"{k}_ess"] = v.get("pop_ess_share")
                    r[f"{k}_low_p0"] = v.get("target_mass_low_p0")
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
                             "mean_diff": O["mean_gain"] - L["mean_gain"],
                             # θ_value*'s overlap with this logger (§12.3, exploratory): N × ESS share, low-π0 mass
                             "n_eff_value": idx[4] * float(pop.loc[(*world, share)].get("value_ess", np.nan)),
                             "low_p0_value": float(pop.loc[(*world, share)].get("value_low_p0", np.nan))})
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
    for ds in DATASETS:  # each dataset's biased worlds (the dataset × corruption cells: dataset_corruption)
        if (df["dataset"] == ds).any():
            yield f"{ds} (biased)", df[(df["dataset"] == ds) & (df["bias"] != "none")]


def dataset_corruption(df: pd.DataFrame, keys, cols) -> pd.DataFrame:
    """Per dataset × corruption cell (and the other ``keys``): the mean of ``cols`` over the cell's worlds (seeds) and
    their number; one row per cell, so every pooled number can be traced to its datasets and corruptions."""
    g = df.groupby(list(keys) + ["dataset", "bias"], dropna=False)
    out = g[list(cols)].mean()
    out["worlds"] = g.size()
    out = out.reset_index()
    out["corruption"] = out["bias"].map(BIAS_NAMES).fillna(out["bias"])
    return out


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
        for col in [c for c in f.columns if c not in WORLD + CELL] + ["M_L", "dT", "n_eff_value", "low_p0_value"]:
            x = pd.to_numeric(g[col], errors="coerce")
            ok = x.notna() & g["observed"].notna()
            if ok.sum() < 5:
                continue
            rho, p = spearmanr(x[ok], g.loc[ok, "observed"])
            rows.append({"opc_arm": arm, "state": state, "feature": col, "rho": float(rho), "p": float(p),
                         "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- the figures
# The support levels are ordinal: one hue, light (poor) to dark (better) (ColorBrewer Blues), each with its own marker,
# so identity never rests on colour; the margin's parts in the Okabe-Ito pink (the likelihood's misspecification) and
# blue (OPC's extra training and selection gap).
SUPPORT_STYLE = {"poor": ("#6BAED6", "o"), "current": ("#2171B5", "s"), "better": ("#08306B", "D")}
FIG_PANELS = ("warp", "group", "vector", "combined", "biased (pooled)")


def _support_names(shares, levels: dict | None) -> dict:
    """share -> 'poor' / 'current' / 'better' from support_levels.json's levels (the current share alone without it)."""
    names = {float(v): k for k, v in (levels or {}).items()}
    return {float(x): names.get(float(x), "current" if np.isclose(float(x), 0.8) else f"share {x:g}") for x in shares}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6})
    return plt


def regime_figures(summary: pd.DataFrame, out: Path, levels: dict | None = None, rule: str = "native") -> None:
    """fig_r1_<arm>: OPC − likelihood (the observed difference = F, greedy CTR points, 95% t-interval over worlds)
    against N per bias panel, one series per support level; fig_r2_<arm>: the margin's parts on the pooled biased
    worlds, M_L against dT + dS: OPC is ahead where M_L is above."""
    plt = _plt()
    s = summary[summary["rule"] == rule]
    names = _support_names(s["share"].unique(), levels)
    sizes = sorted(s["train_size"].unique())
    for arm, g in s.groupby("opc_arm"):
        fig, axes = plt.subplots(1, len(FIG_PANELS), figsize=(2.3 * len(FIG_PANELS), 2.6), sharey=True)
        rows = []
        for ax, panel in zip(axes, FIG_PANELS):
            h = g[g["panel"] == panel]
            for k, (share, hs) in enumerate(sorted(h.groupby("share"), key=lambda t: list(SUPPORT_STYLE).index(
                    names[float(t[0])]) if names[float(t[0])] in SUPPORT_STYLE else 9)):
                name = names[float(share)]
                c, mk = SUPPORT_STYLE.get(name, ("#6B6B6B", "x"))
                hs = hs.sort_values("train_size")
                x = np.array([sizes.index(n) for n in hs["train_size"]]) + (k - 1) * 0.12
                ax.errorbar(x, hs["observed"], yerr=[hs["observed"] - hs["observed_lo"], hs["observed_hi"] - hs["observed"]],
                            color=c, marker=mk, markersize=4.5, linewidth=1.2, capsize=2, label=name)
                rows += [{"opc_arm": arm, "panel": panel, "support": name, "share": float(share), **r}
                         for r in hs[["train_size", "observed", "observed_lo", "observed_hi", "worlds", "opc_ahead"]].to_dict("records")]
            ax.axhline(0.0, color="#555555", linewidth=0.8)
            ax.set_xticks(range(len(sizes)), [f"{n // 1000}k" for n in sizes])
            ax.set_title(panel, fontsize=8)
            ax.set_xlabel("N (training rows)")
        axes[0].set_ylabel("OPC − likelihood (CTR points)")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
        _save(fig, out, f"fig_r1_{arm}", pd.DataFrame(rows))
        fig, ax = plt.subplots(figsize=(3.6, 2.8))
        h = g[g["panel"] == "biased (pooled)"]
        rows = []
        for k, (share, hs) in enumerate(sorted(h.groupby("share"), key=lambda t: float(t[0]), reverse=True)):
            name = names[float(share)]
            c, mk = SUPPORT_STYLE.get(name, ("#6B6B6B", "x"))
            hs = hs.sort_values("train_size")
            x = np.array([sizes.index(n) for n in hs["train_size"]])
            ax.plot(x, hs["dT"] + hs["dS"], color=c, marker=mk, markersize=4.5, linewidth=1.4, label=f"dT + dS, {name}")
            ax.plot(x, hs["M_L"], color=c, linestyle="--", linewidth=1.0)
            rows += [{"opc_arm": arm, "support": name, "share": float(share), **r}
                     for r in hs[["train_size", "M_L", "dT", "dS", "observed"]].to_dict("records")]
        ax.plot([], [], color="#555555", linestyle="--", linewidth=1.0, label="M_L (dashed, per level)")
        ax.set_xticks(range(len(sizes)), [f"{n // 1000}k" for n in sizes])
        ax.set_xlabel("N (training rows)")
        ax.set_ylabel("CTR points (biased worlds)")
        ax.set_title("OPC ahead where M_L > dT + dS", fontsize=8)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))  # the parts are distances: from 0, not magnified
        fig.tight_layout()
        fig.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
        _save(fig, out, f"fig_r2_{arm}", pd.DataFrame(rows))


def _save(fig, out: Path, name: str, data: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    data.to_csv(Path(out) / f"{name}.csv", index=False, float_format="%.6g")
    for ext in ("png", "pdf"):
        fig.savefig(Path(out) / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ the interventions (§7)
INTERVENTION_PAIRS = (  # (a, b, what a − b isolates)
    ("shared_opc_oq", "shared_opc", "oracle q vs q-hat (harmonic)"),
    ("shared_opc_raw_oq", "shared_opc_raw", "oracle q vs q-hat (raw)"),
    ("shared_opc_b8192", "shared_opc", "batch 8192 vs standard (harmonic)"),
    ("shared_opc_raw_b8192", "shared_opc_raw", "batch 8192 vs standard (raw)"),
    ("shared_opc_bfull", "shared_opc", "full batch vs standard (harmonic)"),
    ("shared_opc_raw_bfull", "shared_opc_raw", "full batch vs standard (raw)"),
    ("shared_opc_raw", "shared_opc", "raw vs harmonic (q-hat)"),
    ("shared_opc_raw_oq", "shared_opc_oq", "raw vs harmonic (oracle q)"),
    ("shared_opc", "shared_likelihood", "OPC vs likelihood"),
    ("shared_opc_raw", "shared_likelihood", "raw DR vs likelihood"),
    ("shared_opc_oq", "shared_likelihood", "OPC with oracle q vs likelihood"),
    ("shared_opc_raw_oq", "shared_likelihood", "raw DR with oracle q vs likelihood"),
    ("shared_opc_b8192", "shared_likelihood", "OPC, batch 8192 vs likelihood"),
    ("shared_opc_bfull", "shared_likelihood", "OPC, full batch vs likelihood"),
)
RULES = ("native", "common", "best", "mean")


def intervention_worlds(sel: pd.DataFrame, *, train_size: int = 25_000, share: float = 0.8,
                        pairs=INTERVENTION_PAIRS) -> pd.DataFrame:
    """a − b per world at (N, share), in greedy CTR points, per selection rule (native, common, best of the trials,
    mean over the trials): the dataset × corruption cells of the interventions."""
    s = sel[(sel["train_size"] == train_size) & np.isclose(sel["share"], share)]
    col = {"native": "native_gain", "common": "common_gain", "best": "best_gain", "mean": "mean_gain"}
    rows = []
    for a, b, what in pairs:
        va, vb = (s[s["arm"] == x].set_index(WORLD) for x in (a, b))
        common_worlds = va.index.intersection(vb.index)
        for rule in RULES if not common_worlds.empty else ():
            d = (va.loc[common_worlds, col[rule]] - vb.loc[common_worlds, col[rule]]).reset_index()
            d.columns = WORLD + ["d"]
            rows += [{"a": a, "b": b, "contrast": what, "rule": rule, **r} for r in d.to_dict("records")]
    return pd.DataFrame(rows, columns=["a", "b", "contrast", "rule"] + WORLD + ["d"])


def interventions(sel: pd.DataFrame, **kw) -> pd.DataFrame:
    """intervention_worlds summarized per panel: mean, 95% t-interval and the worlds where a is higher."""
    w = intervention_worlds(sel, **kw)
    rows = []
    for (a, b, what, rule), d in w.groupby(["a", "b", "contrast", "rule"], sort=False):
        for panel, g in _panels(d):
            if g.empty:
                continue
            m, lo, hi, n = mean_ci(g["d"])
            rows.append({"a": a, "b": b, "contrast": what, "rule": rule, "panel": panel, "a_minus_b": m,
                         "ci_lo": lo, "ci_hi": hi, "worlds": n, "a_higher": int((g["d"] > 0).sum())})
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
    mp.add_argument("--levels", help="support_levels.json (names the support levels in the figures)")
    mp.add_argument("--out", required=True)
    dp = sub.add_parser("decompose")
    dp.add_argument("--runs", nargs="+", required=True)
    dp.add_argument("--states", nargs="+", required=True)
    dp.add_argument("--empirical", required=True)
    dp.add_argument("--out", required=True)
    ip = sub.add_parser("interventions")
    ip.add_argument("--runs", nargs="+", required=True)
    ip.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = load_trials(*args.runs)
    sel = select(t)
    sel.to_csv(out / "table_selected.csv", index=False, float_format="%.8g")
    if args.cmd == "interventions":
        iv = interventions(sel)
        iv.to_csv(out / "table_interventions.csv", index=False, float_format="%.6g")
        dataset_corruption(intervention_worlds(sel), ["contrast", "a", "b", "rule"], ("d",)).to_csv(
            out / "table_interventions_dataset_corruption.csv", index=False, float_format="%.6g")
        print(f"wrote {out}: {len(iv)} intervention rows")
        return
    states = load_states(*args.states)
    states.to_csv(out / "table_population_states.csv", index=False, float_format="%.8g")
    if args.cmd == "map":
        m = margins(sel, states)
        m.to_csv(out / "table_margins.csv", index=False, float_format="%.6g")
        summary = regime_summary(m)
        summary.to_csv(out / "table_regime_summary.csv", index=False, float_format="%.6g")
        dataset_corruption(m, CELL + ["opc_arm", "rule"], ("observed", "M_L", "dT", "dS", "best_diff", "mean_diff",
                                                           "n_eff_value", "low_p0_value")).to_csv(
            out / "table_dataset_corruption.csv", index=False, float_format="%.6g")
        levels = json.loads(Path(args.levels).read_text())["levels"] if args.levels else None
        regime_figures(summary, out, levels)
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
        dataset_corruption(d, ["arm", "rule"], ("M", "E", "O", "S", "total", "selected_gain", "emp_path_best_gain")).to_csv(
            out / "table_decomposition_dataset_corruption.csv", index=False, float_format="%.6g")
        print(f"wrote {out}: {len(d)} decomposition rows")


if __name__ == "__main__":
    main()
