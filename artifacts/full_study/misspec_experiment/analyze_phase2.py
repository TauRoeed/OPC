"""Phase-2: OPC-mlp64 vs CausE-cap-mlp64 DiD + ref rows (CausE-linear, BLOB)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("artifacts/full_study")
OUT = Path("artifacts/full_study/misspec_experiment/phase2")
TAGS = {
    0.0: ("nlres_p2_rstar_g0_mlp64", "nlres_p2_rstar_g0_refs"),
    0.25: ("nlres_p2_rstar_g0.25_mlp64", "nlres_p2_rstar_g0.25_refs"),
    0.5: ("nlres_p2_rstar_g0.5_mlp64", "nlres_p2_rstar_g0.5_refs"),
}
CAUSE_C = "causecap_c_r000"
CAUSE_T = "causecap_t_r000"
BLOB = "blob_l10_nq"
G0, GSTAR = 0.0, 0.5


def load(tag: str) -> pd.DataFrame:
    rows = []
    for sm in (ROOT / f"run_{tag}").glob("dataset=*/summary_metrics.csv"):
        df = pd.read_csv(sm)
        seed = int(str(sm.parent.name).split("seed=")[1].split("__")[0])
        for method, g in df.groupby("method", sort=False):
            r = g.iloc[-1]
            rows.append(dict(
                tag=tag, method=str(method), seed=seed,
                greedy=float(r["policy_rewards_greedy"]),
            ))
    return pd.DataFrame(rows)


def paired_did(a0: pd.Series, a_star: pd.Series, b0: pd.Series, b_star: pd.Series) -> dict:
    common = sorted(set(a0.index) & set(a_star.index) & set(b0.index) & set(b_star.index))
    did = 100.0 * ((a_star.loc[common] - b_star.loc[common]) - (a0.loc[common] - b0.loc[common]))
    n = len(did)
    mean = float(did.mean())
    se = float(did.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    return dict(
        n_seeds=n, DiD_pt=mean, DiD_se=se,
        DiD_ci95=[mean - tcrit * se, mean + tcrit * se],
        sign_positive=int((did > 0).sum()), sign_total=n,
        per_seed_DiD_pt=did.loc[common].to_dict(),
        D0_pt=float(100.0 * (a0.loc[common] - b0.loc[common]).mean()),
        Dstar_pt=float(100.0 * (a_star.loc[common] - b_star.loc[common]).mean()),
        level0_a=float(100.0 * a0.loc[common].mean()),
        level0_b=float(100.0 * b0.loc[common].mean()),
        level_star_a=float(100.0 * a_star.loc[common].mean()),
        level_star_b=float(100.0 * b_star.loc[common].mean()),
    )


def series(df: pd.DataFrame, method: str, tier: str, gamma: float) -> pd.Series:
    s = df[(df.method == method) & (df.tier == tier) & (df.gamma == gamma)]
    return s.set_index("seed")["greedy"].sort_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for g, (mlp, refs) in TAGS.items():
        a = load(mlp)
        a["gamma"] = g
        a["tier"] = "mlp64"
        b = load(refs)
        b["gamma"] = g
        b["tier"] = "refs"
        frames.extend([a, b])
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "arm_values.csv", index=False)

    opc0 = series(df, "opc", "mlp64", G0)
    opc_star = series(df, "opc", "mlp64", GSTAR)
    primary = paired_did(opc0, opc_star, series(df, CAUSE_C, "mlp64", G0), series(df, CAUSE_C, "mlp64", GSTAR))

    # D by gamma for primary contrast
    keep = df[df.method.isin(["opc", CAUSE_C]) & (df.tier == "mlp64")].copy()
    wide = keep.pivot_table(index=["seed", "gamma"], columns="method", values="greedy").reset_index()
    wide["D"] = wide["opc"] - wide[CAUSE_C]
    wide["D_pt"] = 100.0 * wide["D"]
    wide.to_csv(OUT / "D_opc_vs_causec_mlp.csv", index=False)
    by = wide.groupby("gamma")["D_pt"].agg(D_mean="mean", D_sd="std", n="count")

    refs = {
        "vs_causec_c_mlp": primary,
        "vs_causec_t_mlp": paired_did(
            opc0, opc_star, series(df, CAUSE_T, "mlp64", G0), series(df, CAUSE_T, "mlp64", GSTAR)),
        "vs_causec_c_linear": paired_did(
            opc0, opc_star, series(df, CAUSE_C, "refs", G0), series(df, CAUSE_C, "refs", GSTAR)),
        "vs_causec_t_linear": paired_did(
            opc0, opc_star, series(df, CAUSE_T, "refs", G0), series(df, CAUSE_T, "refs", GSTAR)),
        "vs_blob_l10_nq": paired_did(
            opc0, opc_star, series(df, BLOB, "refs", G0), series(df, BLOB, "refs", GSTAR)),
    }

    means = []
    for g in sorted(df.gamma.unique()):
        for tier in ("mlp64", "refs"):
            for m in sorted(df[(df.gamma == g) & (df.tier == tier)].method.unique()):
                s = df[(df.gamma == g) & (df.tier == tier) & (df.method == m)]["greedy"]
                means.append(dict(
                    gamma=float(g), method=m, tier=tier,
                    mean_pct=float(100 * s.mean()), sd_pct=float(100 * s.std()), n=int(len(s)),
                ))

    summary = dict(
        primary_opc_vs_causec_c_mlp=primary,
        D_by_gamma_pt=by.reset_index().to_dict(orient="records"),
        refs_did=refs,
        arm_means=means,
        # backward-compatible top-level keys (primary)
        n_seeds=primary["n_seeds"], DiD_pt=primary["DiD_pt"], DiD_se=primary["DiD_se"],
        DiD_ci95=primary["DiD_ci95"], sign_positive=primary["sign_positive"],
        sign_total=primary["sign_total"], per_seed_DiD_pt=primary["per_seed_DiD_pt"],
    )
    (OUT / "phase2_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "Phase-2 (misspec-DR-v1)",
        "D(gamma) = V(OPC-mlp64) - V(CausE-cap-C-mlp64)   [exact greedy CTR]",
        "DiD = D(0.5) - D(0)",
        "",
        "Mean D by gamma (CTR points):",
    ]
    for _, r in by.reset_index().iterrows():
        lines.append(f"  gamma={r.gamma:g}:  D={r.D_mean:+.2f} pt  (sd={r.D_sd:.2f}, n={int(r.n)})")
    lines += [
        "",
        f"DiD = {primary['DiD_pt']:+.2f} pt   95% CI "
        f"[{primary['DiD_ci95'][0]:+.2f}, {primary['DiD_ci95'][1]:+.2f}]   "
        f"signs {primary['sign_positive']}/{primary['sign_total']}",
        "",
        "Per-seed DiD (pt):",
        *[f"  seed {s}: {float(v):+.2f}" for s, v in primary["per_seed_DiD_pt"].items()],
        "",
        "Refs DiD (OPC-mlp - ref):",
    ]
    for name, r in refs.items():
        if name == "vs_causec_c_mlp":
            continue
        lines.append(
            f"  {name}: DiD={r['DiD_pt']:+.2f}  CI[{r['DiD_ci95'][0]:+.2f}, {r['DiD_ci95'][1]:+.2f}]  "
            f"signs {r['sign_positive']}/{r['sign_total']}  "
            f"level@γ* OPC={r['level_star_a']:.2f} ref={r['level_star_b']:.2f}"
        )
    lines += ["", "Arm means (greedy CTR %):"]
    for row in means:
        lines.append(
            f"  gamma={row['gamma']:g}  {row['method']:22s} [{row['tier']}]  "
            f"{row['mean_pct']:+.2f}  (sd {row['sd_pct']:.2f}, n={row['n']})"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "phase2_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
