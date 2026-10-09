"""Tables of the gradient benchmark (docs/opc_gradient_regime_study.md §4), from ``training.opc_gradient_benchmark``
run folders.

    python -m training.analyze_opc_gradients --runs RUN... --out DIR

Per world, policy state and estimator, over the R logged datasets:
- the relative bias ‖ḡ − g*‖ / ‖g*‖, raw and with its Monte Carlo floor tr(Ĉ)/R removed, and the bias ratio
  ‖ḡ − g*‖² / (tr Ĉ / R) (about 1 without bias) with a t-test along g*;
- G5's bias three ways: direct, paired (G5 − G3) and conditional (the exact bias given each dataset's q̂);
- the SNR ‖E ĝ‖² / E‖ĝ − E ĝ‖² (unbiased plug-ins), the cosines with g* (mean, median, P(⟨ĝ, g*⟩ > 0) with a
  Wilson interval), the norm ratio, the relative total variance and the relative MSE;
- the minibatch gradients' error against the full-data gradient's; the weights' profile and the population overlap;
  the reward model's error.
Intervals: a percentile bootstrap over datasets per world; a t-interval over worlds when pooled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import f as f_dist
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits

ESTIMATORS = ("G1", "G2", "G3", "G4", "G5")
REFERENCE_STATE = {"value": "source"}  # at θ_value* g* ≈ 0: relative measures use ‖g*(source)‖ (§3)
N_BOOT = 400


def _tags(name: str) -> dict:
    out = {}
    for part in name.split("__"):
        k, _, v = part.partition("=")
        out[k] = v
    return out


def load_world(wdir: Path) -> dict:
    wdir = Path(wdir)
    cfg = json.loads((wdir / "config.json").read_text())
    states_meta = json.loads((wdir / "states.json").read_text())
    gz = np.load(wdir / "gstar.npz")
    gmeta = json.loads((wdir / "gstar.json").read_text())
    reps = sorted(p for p in (wdir / "rep").glob("rep_*.json"))
    grads, bias5, diags = {}, {}, []
    for p in reps:
        d = json.loads(p.read_text())
        z = np.load(p.with_suffix(".npz"))
        diags.append(d)
        for k in z.files:
            name, _, kind = k.partition("__")
            (grads if kind == "grads" else bias5).setdefault(name, []).append(z[k].astype(np.float64))
    gvis, gvis_meta = {}, {}
    if (wdir / "gstar_visible.npz").exists():  # §12.3: the low-overlap states' gradient over the visible pairs
        zv = np.load(wdir / "gstar_visible.npz")
        gvis = {k: zv[k] for k in zv.files}
        gvis_meta = json.loads((wdir / "gstar_visible.json").read_text())
    return {"dir": wdir, "tags": _tags(wdir.name), "config": cfg, "states": states_meta,
            "gstar": {k: gz[k] for k in gz.files}, "gmeta": gmeta, "diags": diags, "gvis": gvis, "gvis_meta": gvis_meta,
            "grads": {k: np.stack(v) for k, v in grads.items()},  # state -> (R, E, P)
            "bias5": {k: np.stack(v) for k, v in bias5.items()}}  # state -> (R, P)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def gradient_metrics(G: np.ndarray, gstar: np.ndarray, ref: float) -> dict:
    """The §4 metrics of R gradient samples ``G`` (R × P) against ``gstar``, relative to the norm ``ref``."""
    R = G.shape[0]
    gbar = G.mean(axis=0)
    resid = G - gbar
    trc = float((resid ** 2).sum() / max(R - 1, 1))
    floor = trc / R
    bias2 = float(((gbar - gstar) ** 2).sum())
    deb = bias2 - floor
    gn = float(np.linalg.norm(gstar))
    u = gstar / max(gn, 1e-300)
    along = (G - gstar) @ u
    t_along = float(along.mean() / max(along.std(ddof=1) / np.sqrt(R), 1e-300)) if R > 1 else np.nan
    norms = np.linalg.norm(G, axis=1)
    cos = (G @ gstar) / np.maximum(norms * gn, 1e-300)
    k_pos = int(((G @ gstar) > 0).sum())
    lo, hi = _wilson(k_pos, R)
    signal = float((gbar ** 2).sum()) - floor
    return {
        "R": R, "gstar_norm": gn, "ref_norm": ref,
        "rel_bias_raw": np.sqrt(bias2) / ref, "rel_bias": np.sign(deb) * np.sqrt(abs(deb)) / ref,
        "rel_bias_floor": np.sqrt(floor) / ref, "bias_ratio": bias2 / max(floor, 1e-300),
        "t_along_gstar": t_along, "p_along_gstar": float(2 * student_t.sf(abs(t_along), R - 1)) if R > 1 else np.nan,
        "snr": signal / max(trc, 1e-300), "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
        "p_positive": k_pos / R, "p_positive_lo": lo, "p_positive_hi": hi,
        "norm_ratio": float(norms.mean() / max(gn, 1e-300)), "rel_total_var": trc / ref ** 2,
        "rel_mse": float(((G - gstar) ** 2).sum(axis=1).mean()) / ref ** 2,
    }


def bias_tests(G: np.ndarray, gstar: np.ndarray, k: int = 10) -> dict:
    """Calibrated tests of E ĝ = g* (§4's check). Under no bias R‖ḡ − g*‖² is Σ λ_i χ²_1 over Ĉ's eigenvalues, so the
    bias ratio is about χ²_ν / ν with ν = (tr Ĉ)² / tr Ĉ² (Satterthwaite): near 1 when the noise has one dominant
    direction, where ratios of 3-4 are common without bias. Second, Hotelling's T² on k principal directions taken
    from the odd replicates and tested on the even ones (in-sample directions inflate their own variances when P > R,
    which makes the test far too conservative)."""
    R = G.shape[0]
    gbar = G.mean(axis=0)
    resid = G - gbar
    lam = np.clip(np.linalg.eigvalsh(resid @ resid.T / (R - 1)), 0, None)  # Ĉ's nonzero eigenvalues
    trc = float(lam.sum())
    nu = trc ** 2 / max(float((lam ** 2).sum()), 1e-300)
    ratio = float(((gbar - gstar) ** 2).sum()) / max(trc / R, 1e-300)
    fit, test = G[1::2], G[0::2]
    n = test.shape[0]
    k = min(k, n - 2, fit.shape[0] - 1)
    _, _, Vt = np.linalg.svd(fit - fit.mean(axis=0), full_matrices=False)
    z = (test - gstar) @ Vt[:k].T
    zbar = z.mean(axis=0)
    t2 = n * float(zbar @ np.linalg.solve(np.cov(z, rowvar=False).reshape(k, k), zbar))
    sq = (resid ** 2).sum(axis=1)
    return {"noise_rank_eff": nu, "p_bias_ratio": float(chi2.sf(ratio * nu, nu)),
            "p_bias_hotelling": float(f_dist.sf(t2 * (n - k) / (k * (n - 1)), k, n - k)),
            "top5_var_share": float(np.sort(sq)[-5:].sum() / max(sq.sum(), 1e-300))}  # heavy tails (diagnostic)


def bootstrap_ci(G: np.ndarray, gstar: np.ndarray, ref: float, keys=("rel_bias", "snr", "cos_mean", "rel_mse"),
                 n_boot: int = N_BOOT, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    R = G.shape[0]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        m = gradient_metrics(G[rng.integers(0, R, R)], gstar, ref)
        for k in keys:
            draws[k].append(m[k])
    return {f"{k}_{s}": float(np.percentile(v, q)) for k, v in draws.items() for s, q in (("lo", 2.5), ("hi", 97.5))}


def world_rows(w: dict, boot: bool = True) -> list[dict]:
    rows = []
    est = w["config"]["estimators"]
    for state, G_all in w["grads"].items():
        gstar = w["gstar"][state]
        ref = float(np.linalg.norm(w["gstar"][REFERENCE_STATE.get(state, state)]))
        diag_states = [d["states"].get(state, {}) for d in w["diags"]]
        common = {**w["tags"], "train_size": int(w["config"]["train_size"]),
                  "share": float(w["config"]["world_options"].get("logger_greedy_share", 0.8)),
                  "state": state, "V": w["gmeta"][state]["V"],
                  "greedy": w["states"][state]["greedy"] if state in w["states"] else np.nan,
                  **{k: w["gmeta"][state][k] for k in ("pop_ess_share", "target_mass_low_p0", "pop_w_max")}}
        for k in ("w_ess_share", "w_max", "w_q99.9", "w_q50", "qhat_rmse_logging", "qhat_rmse_target"):
            vals = [d[k] for d in diag_states if k in d]
            common[k] = float(np.mean(vals)) if vals else np.nan
        ref_src = float(np.linalg.norm(w["gstar"]["source"]))
        common["gstar_rel_source"] = float(np.linalg.norm(gstar)) / max(ref_src, 1e-300)
        for j, e in enumerate(est):
            G = G_all[:, j, :]
            r = {**common, "estimator": e, **gradient_metrics(G, gstar, ref), **bias_tests(G, gstar)}
            # the split-sample Hotelling test without the 3 most extreme replicates: its robustness to heavy tails
            # (a diagnostic only: dropping replicates biases the mean, so it is never a test of unbiasedness)
            keep = np.sort(np.argsort(-((G - G.mean(axis=0)) ** 2).sum(axis=1))[3:])
            r["p_bias_hotelling_drop3"] = bias_tests(G[keep], gstar)["p_bias_hotelling"]
            vis_meta = w["gvis_meta"].get(state)
            if state in w["gvis"] and vis_meta is not None and vis_meta["replicates"] == G.shape[0]:
                gv = w["gvis"][state]
                r.update({f"{k}_vis": v for k, v in bias_tests(G, gv).items() if k not in ("noise_rank_eff", "top5_var_share")})
                r.update(bias_ratio_vis=gradient_metrics(G, gv, ref)["bias_ratio"],
                         hidden_mass=vis_meta["hidden_mass"], gvis_rel_gstar=float(
                             np.linalg.norm(gv - gstar) / max(np.linalg.norm(gstar), 1e-300)))
                for key in (k for k in w["gvis"] if k.startswith(f"{state}@")):  # the edge's sharpness (diagnostic)
                    c = key.partition("@")[2]
                    t = bias_tests(G, w["gvis"][key])
                    r.update({f"p_bias_ratio_vis@{c}": t["p_bias_ratio"], f"p_bias_hotelling_vis@{c}": t["p_bias_hotelling"],
                              f"hidden_mass@{c}": w["gvis_meta"][key]["hidden_mass"]})
            # the same errors on one scale for every state: relative to the source's ‖g*‖ (g* ≈ 0 near an optimum)
            m_src = gradient_metrics(G, gstar, ref_src)
            r.update({f"{k}_src": m_src[k] for k in ("rel_bias", "rel_bias_floor", "rel_total_var", "rel_mse")})
            if boot:
                r.update(bootstrap_ci(G, gstar, ref))
            if e in ("G3", "G5"):
                mbf = [d.get(f"mb_{e}_sq_to_full") for d in diag_states if d.get(f"mb_{e}_sq_to_full") is not None]
                mbg = [d.get(f"mb_{e}_sq_to_gstar") for d in diag_states if d.get(f"mb_{e}_sq_to_gstar") is not None]
                if mbf:
                    r["mb_rel_sq_to_full"] = float(np.mean(mbf)) / ref ** 2
                    r["mb_rel_mse"] = float(np.mean(mbg)) / ref ** 2
                    r["mb_noise_share"] = r["mb_rel_sq_to_full"] / max(r["mb_rel_mse"], 1e-300)
            rows.append(r)
        if "G3" in est and "G5" in est:  # G5's bias: paired against the unbiased G3, and conditional on q̂
            D = G_all[:, est.index("G5"), :] - G_all[:, est.index("G3"), :]
            m = gradient_metrics(D + gstar, gstar, ref)
            row = {**common, "estimator": "G5-G3 (paired bias)", "rel_bias": m["rel_bias"],
                   "rel_bias_raw": m["rel_bias_raw"], "rel_bias_floor": m["rel_bias_floor"],
                   "bias_ratio": m["bias_ratio"], **bias_tests(D + gstar, gstar)}
            if state in w["bias5"]:
                B = w["bias5"][state]
                mb = gradient_metrics(B + gstar, gstar, ref)
                row.update({"cond_rel_bias": mb["rel_bias"], "cond_rel_bias_raw": mb["rel_bias_raw"],
                            "cond_rel_bias_floor": mb["rel_bias_floor"],
                            "cond_cos_with_paired": float(B.mean(0) @ D.mean(0) / max(
                                np.linalg.norm(B.mean(0)) * np.linalg.norm(D.mean(0)), 1e-300)),
                            "cond_bias_cos_gstar": float(B.mean(0) @ gstar / max(
                                np.linalg.norm(B.mean(0)) * np.linalg.norm(gstar), 1e-300))})
            rows.append(row)
    return rows


def pooled(df: pd.DataFrame, cols, by=("state", "estimator")) -> pd.DataFrame:
    """Mean and 95% t-interval over worlds per bias type, over the biased worlds, and over all worlds."""
    out = []
    panels = [(b, g) for b, g in df.groupby("bias")] + [("biased", df[df["bias"] != "none"]), ("all", df)]
    for panel, g in panels:
        for key, h in g.groupby(list(by)):
            r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
            r.update(panel=panel, worlds=len(h))
            for c in cols:
                v = pd.to_numeric(h[c], errors="coerce").dropna()
                if len(v) == 0:
                    continue
                m = float(v.mean())
                half = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan
                r[c], r[f"{c}_lo"], r[f"{c}_hi"] = m, m - half, m + half
            out.append(r)
    return pd.DataFrame(out)


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment over the finite p-values (NaN stays NaN)."""
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    out[order] = np.minimum(1.0, np.maximum.accumulate((len(ok) - np.arange(len(ok))) * p[order]))
    return out


ESS_BELOW = 1e-4  # §12.3: below this population ESS share a state's cell is also tested against the visible gradient


def unbiasedness_check(df: pd.DataFrame, estimators=("G1", "G2", "G3", "G4"), alpha: float = 0.05) -> pd.DataFrame:
    """§11's first stop condition (with §12.2-§12.3), per analytically unbiased estimator. Every (world, state) cell is
    tested against g*, Holm-adjusted over the cells. A rejected cell whose state's population ESS share is below 1e-4 is
    a *practical support failure* when neither test rejects against the supported-region gradient (the pairs expected
    at least once in the R × N rows; Holm over those cells), and *low overlap, unresolved* when one does: the region's
    edge is soft, and the analytical unbiasedness (tested by enumeration) is not in question either way. Elsewhere a
    rejected cell is *biased*. ``stop`` marks a raw DR (G2, G3) cell that is biased or unresolved."""
    out = []
    for e in estimators:
        h = df[df["estimator"] == e].copy()
        if h.empty:
            continue
        for col in ("p_bias_ratio", "p_bias_hotelling", "p_bias_ratio_vis", "p_bias_hotelling_vis"):
            h[f"{col}_holm"] = _holm(h[col].to_numpy(dtype=float)) if col in h else np.nan
        rejected = (h["p_bias_ratio_holm"] < alpha) | (h["p_bias_hotelling_holm"] < alpha)
        ess = pd.to_numeric(h["pop_ess_share"], errors="coerce") if "pop_ess_share" in h else np.nan
        low = ess < ESS_BELOW
        visible_ok = (h["p_bias_ratio_vis_holm"] >= alpha) & (h["p_bias_hotelling_vis_holm"] >= alpha)
        h["verdict"] = np.where(~rejected, "consistent", np.where(
            low & visible_ok, "practical support failure", np.where(low, "low overlap, unresolved", "biased")))
        h["stop"] = h["estimator"].isin(["G2", "G3"]) & h["verdict"].isin(["biased", "low overlap, unresolved"])
        out.append(h)
    cols = ["dataset", "bias", "seed", "train_size", "share", "state", "estimator", "R", "pop_ess_share", "bias_ratio",
            "noise_rank_eff", "p_bias_ratio", "p_bias_hotelling", "p_bias_ratio_holm", "p_bias_hotelling_holm",
            "t_along_gstar", "p_along_gstar", "top5_var_share", "p_bias_hotelling_drop3", "hidden_mass", "gvis_rel_gstar", "bias_ratio_vis", "p_bias_ratio_vis",
            "p_bias_hotelling_vis", "p_bias_ratio_vis_holm", "p_bias_hotelling_vis_holm", "verdict", "stop"]
    res = pd.concat(out) if out else pd.DataFrame(columns=cols)
    cols += sorted(c for c in res.columns if "@" in c)  # the supported region's edge, other boundaries
    return res[[c for c in cols if c in res]]


SUMMARY_COLS = ("gstar_rel_source", "rel_bias_src", "rel_total_var_src", "rel_mse_src",
                "rel_bias", "rel_bias_raw", "rel_bias_floor", "bias_ratio", "noise_rank_eff", "snr", "cos_mean", "cos_median",
                "p_positive", "norm_ratio", "rel_total_var", "rel_mse", "mb_rel_sq_to_full", "mb_rel_mse",
                "mb_noise_share", "cond_rel_bias", "w_ess_share", "w_max", "pop_ess_share", "qhat_rmse_logging",
                "qhat_rmse_target")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-boot", action="store_true")
    ap.add_argument("--threads", type=int, default=4, help="BLAS threads (the bootstrap uses every core otherwise)")
    args = ap.parse_args(argv)
    threadpool_limits(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in args.runs:
        for wdir in sorted(Path(run).glob("dataset=*")):
            if not (wdir / "gstar.npz").exists() or not list((wdir / "rep").glob("rep_*.json")):
                continue
            rows += world_rows(load_world(wdir), boot=not args.no_boot)
    df = pd.DataFrame(rows)
    df.to_csv(out / "table_gradients_world.csv", index=False, float_format="%.6g")
    pooled(df, [c for c in SUMMARY_COLS if c in df]).to_csv(out / "table_gradients_summary.csv", index=False,
                                                            float_format="%.6g")
    check = unbiasedness_check(df)
    check.to_csv(out / "table_unbiasedness.csv", index=False, float_format="%.6g")
    for e, h in check.groupby("estimator"):
        print(f"{e}: {len(h)} cells, min p (ratio / Hotelling) {h['p_bias_ratio'].min():.3g} / "
              f"{h['p_bias_hotelling'].min():.3g}, min Holm {h['p_bias_ratio_holm'].min():.3g} / "
              f"{h['p_bias_hotelling_holm'].min():.3g}")
    for (e, v), h in check.groupby(["estimator", "verdict"]):
        if v != "consistent":
            print(f"{e} {v}: " + ", ".join(f"{r.dataset}/{r.bias}/{r.state}" for r in h.itertuples()))
    if check["stop"].any():
        print("STOP (§11): raw DR biased beyond its Monte Carlo floor in", int(check["stop"].sum()), "cells")
    figures(df, out)
    print(f"wrote {out}: {df[['dataset', 'bias', 'seed']].drop_duplicates().shape[0]} worlds, {len(df)} rows")


# ------------------------------------------------------------------------------------------------------- figures
# One hue per estimator, raw in blues and harmonic in oranges (oracle q lighter), from the Okabe-Ito palette; every
# point also carries a marker per estimator, so identity is never color alone.
EST_STYLE = {"G1": ("#6B6B6B", "o"), "G2": ("#56B4E9", "s"), "G3": ("#0072B2", "D"), "G4": ("#E69F00", "^"),
             "G5": ("#D55E00", "v")}
EST_NAMES = {"G1": "G1 raw IPS", "G2": "G2 raw DR, oracle q", "G3": "G3 raw DR, q̂", "G4": "G4 harmonic DR, oracle q",
             "G5": "G5 harmonic DR, q̂ (OPC)"}
STATE_NAMES = {"source": "source (logger)", "mid": "mid (softmax halfway)", "mid_greedy": "mid_greedy (ranking halfway)",
               "likelihood": "θ_log* (value-optimal scale)", "value": "θ_value*"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.edgecolor": "#8A8A8A",
                         "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#333333"})
    return plt


def figures(df: pd.DataFrame, out: Path, states=("source", "mid", "mid_greedy"), biased_only: bool = True) -> None:
    """fig_g1: SNR per estimator and state (log scale; per world, and the median); fig_g2: P(⟨ĝ, g*⟩ > 0) and the mean
    cosine; fig_g3: each estimator's bias against its noise, both relative to ‖g*(state)‖ (debiased bias floored at
    the Monte Carlo floor)."""
    plt = _plt()
    d = df[df["estimator"].isin(EST_STYLE)].copy()
    if biased_only:
        d = d[d["bias"] != "none"]
    rng = np.random.default_rng(0)
    for name, cols, ylab, log in (("fig_g1_snr", ["snr"], "SNR  ‖E ĝ‖² / E‖ĝ − E ĝ‖²", True),
                                  ("fig_g2_alignment", ["p_positive", "cos_mean"], None, False)):
        fig, axes = plt.subplots(len(cols), len(states), figsize=(3.4 * len(states), 2.9 * len(cols)), squeeze=False)
        rows = []
        for i, col in enumerate(cols):
            for j, st in enumerate(states):
                ax = axes[i, j]
                g = d[d["state"] == st]
                for k, e in enumerate(EST_STYLE):
                    v = pd.to_numeric(g.loc[g["estimator"] == e, col], errors="coerce").dropna()
                    if log:
                        v = v.clip(lower=1e-3)
                    c, mk = EST_STYLE[e]
                    ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=c, marker=mk, alpha=0.75,
                               edgecolors="none")
                    if len(v):
                        ax.plot([k - 0.3, k + 0.3], [v.median()] * 2, color="#333333", linewidth=1.4)
                        rows.append({"state": st, "metric": col, "estimator": e, "median": float(v.median()),
                                     "worlds": int(len(v))})
                ax.set_xticks(range(len(EST_STYLE)), list(EST_STYLE))
                if log:
                    ax.set_yscale("log")
                    ax.axhline(1.0, color="#8A8A8A", linewidth=0.8, linestyle="--")
                else:  # one range per row: the panels' values are comparable, and a probability stays below 1
                    v_all = pd.to_numeric(d.loc[d["state"].isin(states), col], errors="coerce")
                    ax.set_ylim(min(0.4 if col == "p_positive" else 0.0, float(v_all.min()) - 0.05), 1.02)
                ax.set_title(STATE_NAMES.get(st, st), fontsize=9)
                ax.set_ylabel(ylab or {"p_positive": "P(⟨ĝ, g*⟩ > 0)", "cos_mean": "mean cos(ĝ, g*)"}[col])
        handles = [plt.Line2D([], [], color=EST_STYLE[e][0], marker=EST_STYLE[e][1], linestyle="", label=EST_NAMES[e])
                   for e in EST_STYLE]
        fig.tight_layout()
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False, float_format="%.6g")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, len(states), figsize=(3.4 * len(states), 3.1), squeeze=False, sharex=True, sharey=True)
    rows = []
    for j, st in enumerate(states):
        ax = axes[0, j]
        g = d[d["state"] == st]
        for e in EST_STYLE:
            h = g[g["estimator"] == e]
            x = np.sqrt(pd.to_numeric(h["rel_total_var"], errors="coerce")).to_numpy()
            y = np.maximum(pd.to_numeric(h["rel_bias"], errors="coerce"),
                           pd.to_numeric(h["rel_bias_floor"], errors="coerce")).to_numpy()
            sig = (pd.to_numeric(h["bias_ratio"], errors="coerce") > 3.0).to_numpy()  # bias well above its MC floor
            c, mk = EST_STYLE[e]
            ax.scatter(x[sig], y[sig], s=18, color=c, marker=mk, alpha=0.85, edgecolors="none")
            ax.scatter(x[~sig], y[~sig], s=18, facecolors="none", edgecolors=c, marker=mk, alpha=0.85, linewidths=0.9)
            rows += [{"state": st, "estimator": e, "noise": float(a), "bias_or_floor": float(b), "bias_significant": bool(q)}
                     for a, b, q in zip(x, y, sig)]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise  √tr Ĉ / ‖g*‖")
        ax.set_title(STATE_NAMES.get(st, st), fontsize=9)
    v = np.array([[r["noise"], r["bias_or_floor"]] for r in rows], dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    lim = [10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max()))] if v.size else [1e-3, 1e1]
    for ax in axes[0]:
        ax.plot(lim, lim, color="#8A8A8A", linewidth=0.8, linestyle="--")  # bias = noise
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    axes[0, 0].set_ylabel("bias  ‖E ĝ − g*‖ / ‖g*‖")
    handles = [plt.Line2D([], [], color=EST_STYLE[e][0], marker=EST_STYLE[e][1], linestyle="", label=EST_NAMES[e])
               for e in EST_STYLE]
    handles += [plt.Line2D([], [], color="#6B6B6B", marker="o", linestyle="", label="bias > 3× its MC floor"),
                plt.Line2D([], [], color="#6B6B6B", marker="o", markerfacecolor="none", linestyle="",
                           label="bias at its MC floor (upper bound)")]
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    pd.DataFrame(rows).to_csv(out / "fig_g3_bias_noise.csv", index=False, float_format="%.6g")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_g3_bias_noise.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
