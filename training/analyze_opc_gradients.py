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
from scipy.stats import t as student_t

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
    return {"dir": wdir, "tags": _tags(wdir.name), "config": cfg, "states": states_meta,
            "gstar": {k: gz[k] for k in gz.files}, "gmeta": gmeta, "diags": diags,
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
        common = {**w["tags"], "state": state, "V": w["gmeta"][state]["V"],
                  "greedy": w["states"][state]["greedy"] if state in w["states"] else np.nan,
                  **{k: w["gmeta"][state][k] for k in ("pop_ess_share", "target_mass_low_p0", "pop_w_max")}}
        for k in ("w_ess_share", "w_max", "w_q99.9", "w_q50", "qhat_rmse_logging", "qhat_rmse_target"):
            vals = [d[k] for d in diag_states if k in d]
            common[k] = float(np.mean(vals)) if vals else np.nan
        for j, e in enumerate(est):
            G = G_all[:, j, :]
            r = {**common, "estimator": e, **gradient_metrics(G, gstar, ref)}
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
                   "bias_ratio": m["bias_ratio"]}
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


SUMMARY_COLS = ("rel_bias", "rel_bias_raw", "rel_bias_floor", "bias_ratio", "snr", "cos_mean", "cos_median",
                "p_positive", "norm_ratio", "rel_total_var", "rel_mse", "mb_rel_sq_to_full", "mb_rel_mse",
                "mb_noise_share", "cond_rel_bias", "w_ess_share", "w_max", "pop_ess_share", "qhat_rmse_logging",
                "qhat_rmse_target")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-boot", action="store_true")
    args = ap.parse_args(argv)
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
    print(f"wrote {out}: {df[['dataset', 'bias', 'seed']].drop_duplicates().shape[0]} worlds, {len(df)} rows")


if __name__ == "__main__":
    main()
