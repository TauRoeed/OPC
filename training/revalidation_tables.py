"""Markdown tables of the revalidation document (docs/simulator_fix_opc_revalidation_20261004.md, Phases 3-4).

Rendered only from committed tables: the old summaries (artifacts/full_study/summaries_20260927), the corrected
summaries (opc_revalidation_20261004/summaries, training.revalidation_phase3), the report's old-vs-new tables
(opc_revalidation_20261004/report, training.revalidation_report) and the Phase 0 coupling tables (phase0/). The
document's result tables are copied from the output, so no number in them is typed by hand.

Usage: python -m training.revalidation_tables   (writes opc_revalidation_20261004/report/tables.md)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.representation_report import BIASES, DATASET_NAMES, DATASETS, LABEL, SIZES

ROOT = Path("artifacts/full_study")
REVAL = ROOT / "opc_revalidation_20261004"
ALL_BIASES = ("none",) + BIASES
ARMS = {"opc": "OPC", "dm": "DM-only", "no_propensity": "no-propensity", "tempered_logger": "tempered logger"}
SINGLES = ("w-high.g-none.v-none", "w-none.g-high.v-none", "w-none.g-none.v-high")


def _f(x, d: int = 2, sign: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _ci(m, lo, hi, d: int = 2) -> str:
    if not np.isfinite(m):
        return "—"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _f(m, d)
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def _k(n) -> str:
    return f"{int(n) // 1000}k"


def _table(header: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def _cells(df: pd.DataFrame, biases=ALL_BIASES, sizes=SIZES):
    for b in biases:
        for n in sizes:
            yield b, n, df[(df["bias"] == b) & (df["train_size"] == n)]


def t_stage2(fr: pd.DataFrame) -> str:
    """Corrected Stage 2: gains and fractions per bias × size (means over the worlds)."""
    rows = []
    for b, n, g in _cells(fr):
        if g.empty:
            continue
        v = lambda m, c: g.loc[g["method"] == m, c].mean() if (g["method"] == m).any() else np.nan
        rows.append([LABEL[b], _k(n),
                     " / ".join(_f(v(m, "gain greedy %")) for m in ("opc", "dm", "no_propensity")),
                     " / ".join(_f(v(m, "gain %")) for m in ARMS),
                     " / ".join(_f(v(m, "fraction greedy"), 2, False) if b != "none" else "—" for m in ("opc", "dm", "no_propensity"))])
    return _table(["bias", "train", "greedy gain: OPC / DM / no-prop", "stochastic gain: OPC / DM / no-prop / tempered",
                   "fraction of oracle repair: OPC / DM / no-prop"], rows)


def t_contrasts(paired: pd.DataFrame) -> str:
    """Corrected Stage 2: OPC minus each baseline and minus the robustness arm (stochastic, points)."""
    p = paired[paired["measure"] == "stochastic"]
    contrasts = [("opc - dm", "OPC − DM-only"), ("opc - no_propensity", "OPC − no-propensity"),
                 ("opc - tempered_logger", "OPC − tempered logger")]
    extra = sorted(c for c in p["contrast"].unique() if c.startswith("opc - opc_"))
    contrasts += [(c, "OPC − OPC with " + c.split("opc_", 1)[1].replace("shrink", "shrink:")) for c in extra]
    rows = []
    for b, n, g in _cells(p):
        if g.empty:
            continue
        cell = lambda c: g[g["contrast"] == c]
        rows.append([LABEL[b], _k(n)] + [_ci(*cell(c)[["mean", "ci_low", "ci_high"]].iloc[0]) if len(cell(c)) else "—"
                                         for c, _ in contrasts])
    return _table(["bias", "train"] + [name for _, name in contrasts], rows)


def t_old_new(t: pd.DataFrame, contrast: str, d: int = 2) -> str:
    g0 = t[t["contrast"] == contrast]
    rows = []
    for b, n, g in _cells(g0):
        if g.empty or not g["worlds"].iloc[0]:  # e.g. no fractions without a bias
            continue
        r = g.iloc[0]
        rows.append([LABEL[b], _k(n), str(int(r["worlds"])), _ci(r["old"], r["old_lo"], r["old_hi"], d),
                     _ci(r["new"], r["new_lo"], r["new_hi"], d), _ci(r["change"], r["change_lo"], r["change_hi"], d),
                     r["verdict"]])
    return _table(["bias", "train", "worlds", "old (buggy logs)", "corrected", "change (paired)", "classification"], rows)


def t_pooled(t: pd.DataFrame) -> str:
    """Old vs corrected, biased worlds pooled, per contrast × size."""
    rows = []
    for c in t["contrast"].unique():
        for n in SIZES:
            g = t[(t["contrast"] == c) & (t["train_size"] == n)]
            if g.empty:
                continue
            r = g.iloc[0]
            d = 3 if c.endswith("(greedy)") and "fraction" in c else 2
            rows.append([c.replace(" - ", " − "), _k(n), str(int(r["worlds"])), _ci(r["old"], r["old_lo"], r["old_hi"], d),
                         _ci(r["new"], r["new_lo"], r["new_hi"], d), _ci(r["change"], r["change_lo"], r["change_hi"], d),
                         r["verdict"]])
    return _table(["quantity (biased worlds)", "train", "worlds", "old", "corrected", "change", "classification"], rows)


def t_gap(old: pd.DataFrame, new: pd.DataFrame) -> str:
    rows = []
    for b in BIASES:
        for n in (5000, 100000):
            o = old[(old["bias"] == b) & (old["train_size"] == n)]
            w = new[(new["bias"] == b) & (new["train_size"] == n)]
            if o.empty or w.empty:
                continue
            o, w = o.iloc[0], w.iloc[0]
            sh = lambda r: " / ".join(_f(r[c], 2, False) for c in ("structural_gap share", "learning_gap share",
                                                                   "learned_repair_gain share"))
            rows.append([LABEL[b], _k(n), _f(w["representation_loss %"], 2, False), _f(w["structural_gap %"], 2, False),
                         f"{_f(o['learning_gap %'], 2, False)} → {_f(w['learning_gap %'], 2, False)}",
                         f"{_f(o['learned_repair_gain %'], 2, False)} → {_f(w['learned_repair_gain %'], 2, False)}",
                         f"{sh(o)} → {sh(w)}"])
    return _table(["bias", "train", "representation loss", "structural gap (unchanged)", "learning gap: old → corrected",
                   "learned repair: old → corrected", "shares structural / learning / learned: old → corrected"], rows)


def t_by_dataset(byds: pd.DataFrame) -> str:
    rows = []
    for b, n, g in _cells(byds, biases=BIASES):
        cells = []
        for d in DATASETS:
            r = g[g["dataset"] == d]
            if r.empty:
                cells.append("—")
                continue
            r = r.iloc[0]
            cells.append(f"{_f(r['OPC fraction greedy'], 2, False)} / {_f(r['DM fraction greedy'], 2, False)}; "
                         f"{_f(r['OPC-DM'])} / {_f(r['OPC-no-prop'])}")
        if any(c != "—" for c in cells):
            rows.append([LABEL[b], _k(n)] + cells)
    return _table(["bias", "train"] + [DATASET_NAMES[d] for d in DATASETS], rows)


def t_diagnostics(old: pd.DataFrame, new: pd.DataFrame) -> str:
    """Selected policies' support and selection diagnostics, biased worlds, old → corrected."""
    rows = []
    for m in ("opc", "dm"):  # no-propensity's naive score is not an estimate of the policy's value
        for n in SIZES:
            sel = lambda r: r[(r["method"] == m) & (r["train_size"] == n) & (r["bias"] != "none")]
            o, w = sel(old), sel(new)
            if w.empty:
                continue
            arrow = lambda f, d=2, s=True: f"{_f(f(o), d, s)} → {_f(f(w), d, s)}"
            rows.append([ARMS[m], _k(n), arrow(lambda r: r["ess_raw"].mean(), 0, False),
                         arrow(lambda r: 100 * r["w_share_gt10"].mean(), 2, False),
                         arrow(lambda r: 100 * r["sel_error_point"].mean()),
                         arrow(lambda r: 100 * r["regret"].mean(), 2, False),
                         _f(100 * w["qhat_error"].mean()) if "qhat_error" in w else "—"])
    return _table(["arm", "train", "raw-weight ESS (mean)", "weights > 10 (%)", "selection estimate − truth (points)",
                   "true selection regret (points)", "q̂ estimate − truth, corrected (points)"], rows)


def t_reward(old: pd.DataFrame, new: pd.DataFrame) -> str:
    rows = []
    for setting in ("external 50k-row q̂", "misspecified (concat) q̂"):
        for b in ("medium", "high"):
            for n in SIZES:
                sel = lambda t: t[(t["setting"] == setting) & (t["bias"] == b) & (t["train_size"] == n)]
                o, w = sel(old), sel(new)
                if o.empty and w.empty:
                    continue
                g = lambda t, c: _ci(*t[[c, f"{c} lo", f"{c} hi"]].iloc[0]) if len(t) and c in t else "—"
                v = lambda t, c: _f(t[c].iloc[0]) if len(t) and c in t else "—"
                rows.append([setting, LABEL[b], _k(n), f"{g(o, 'OPC-DM reference')} → {g(w, 'OPC-DM reference')}",
                             f"{g(o, 'OPC-DM altered')} → {g(w, 'OPC-DM altered')}",
                             f"{g(o, 'change dm')} → {g(w, 'change dm')}", f"{g(o, 'change opc')} → {g(w, 'change opc')}",
                             f"{v(o, 'qhat error dm altered')} → {v(w, 'qhat error dm altered')}"])
    return _table(["q̂ setting (altered)", "bias", "train", "OPC − DM, reference q̂: old → corrected",
                   "OPC − DM, altered q̂: old → corrected", "DM-only change (altered − reference)",
                   "OPC change (altered − reference)", "DM-only: q̂ estimate − truth, altered (points)"], rows)


def coupling_by_share(phase0: Path) -> pd.DataFrame:
    """Per logger share and single-type bias: the corrected logger's concentration (Σπ0², i.e. the chance that two
    draws for one user agree, and its inverse, the effective number of actions per user) and the buggy logs'
    effective one; means over ml and kuairand at seed 100."""
    out = []
    for share in (0.6, 0.8, 0.95):
        p = phase0 / f"coupling_exact_lgs{share}.csv"
        if not p.exists():
            continue
        c = pd.read_csv(p)
        c = c[c["dataset"].isin(["ml", "kuairand"]) & c["bias"].isin(SINGLES)]
        for b, g in c.groupby("bias"):
            out.append(dict(share=share, bias=b, coll_pi0=g["coll_pi0"].mean(), coll_eff=g["coll_eff"].mean(),
                            eff_actions_pi0=(1 / g["coll_pi0"]).mean(), eff_actions_old=(1 / g["coll_eff"]).mean(),
                            v_pi0_pct=100 * g["v_pi0"].mean()))
    return pd.DataFrame(out)


def t_support(old: pd.DataFrame, new: pd.DataFrame, coupling: pd.DataFrame) -> str:
    rows = []
    for b in SINGLES:
        for share in (0.6, 0.8, 0.95):
            sel = lambda t: t[(t["bias"] == b) & (np.isclose(t["share"].astype(float), share))]
            o, w, c = sel(old), sel(new), sel(coupling) if len(coupling) else coupling
            if w.empty:
                continue
            o, w = (o.iloc[0] if len(o) else None), w.iloc[0]
            ov = lambda col, d=2, s=True: _f(o[col], d, s) if o is not None and col in o else "—"
            ci = lambda r, col: _ci(r[col], r[f"{col}_lo"], r[f"{col}_hi"])
            conc = (f"{_f(c['eff_actions_pi0'].iloc[0], 0, False)} ({_f(c['eff_actions_old'].iloc[0], 2, False)})"
                    if len(c) else "—")
            rows.append([LABEL[b], f"{share:g}", _f(w["logger_value_pct"], 2, False), conc,
                         _f(w["oracle_gain_greedy_pct"], 2, False),
                         f"{ov('opc_fraction', 2, False)} → {ci(w, 'opc_fraction')}",
                         f"{ov('dm_fraction', 2, False)} → {_f(w['dm_fraction'], 2, False)}",
                         f"{ci(o, 'opc_minus_dm') if o is not None else '—'} → {ci(w, 'opc_minus_dm')}",
                         f"{ov('opc_ess', 0, False)} → {_f(w['opc_ess'], 0, False)}",
                         f"{ov('opc_w_gt10_pct', 2, False)} → {_f(w['opc_w_gt10_pct'], 2, False)}",
                         f"{ov('opc_regret_pct', 2, False)} → {_f(w['opc_regret_pct'], 2, False)}"])
    return _table(["bias", "logger share", "logger value %", "effective actions per user: corrected (old logs)",
                   "oracle ranking gain (points)", "OPC fraction: old → corrected [95% CI]", "DM-only fraction",
                   "OPC − DM-only (points)", "OPC raw-weight ESS (mean)", "OPC weights > 10 (%)",
                   "OPC selection regret (points)"], rows)


def t_robust(rob: pd.DataFrame) -> str:
    """The default (harmonic:0.1) minus the robustness arm (shrink:100), the sign of R2 and of the old Su slice; the
    summary file holds robust − default."""
    rows = []
    for b, n, g in _cells(rob):
        if g.empty:
            continue
        r = lambda meas: g[g["measure"] == meas]
        flip = lambda t: (-t["diff"].iloc[0], -t["ci_high"].iloc[0], -t["ci_low"].iloc[0])
        v, fr = r("V %"), r("fraction greedy")
        rows.append([LABEL[b], _k(n), str(int(v["n"].iloc[0])) if len(v) else "—", _ci(*flip(v)) if len(v) else "—",
                     _ci(*flip(fr), d=3) if len(fr) else "—"])
    return _table(["bias", "train", "worlds", "V: harmonic:0.1 − shrink:100 (points)", "fraction (greedy)"], rows)


def t_decomposition(dec: pd.DataFrame) -> str:
    rows = []
    for kind in ("biased", "no bias"):
        d0 = dec[dec["worlds_kind"] == kind]
        for q in d0["quantity"].unique():
            for n in SIZES:
                g = d0[(d0["quantity"] == q) & (d0["train_size"] == n)]
                if g.empty:
                    continue
                r = g.iloc[0]
                rows.append([kind, q.replace(" - ", " − "), _k(n), str(int(r["worlds"])),
                             " / ".join(_f(r[k]) for k in ("old", "mid", "new")),
                             _ci(r["simulator"], r["simulator_lo"], r["simulator_hi"]),
                             _ci(r["configuration"], r["configuration_lo"], r["configuration_hi"]),
                             _ci(r["total"], r["total_lo"], r["total_hi"])])
    return _table(["worlds", "quantity (points)", "train", "n", "old / old config on corrected logs / corrected",
                   "simulator fix (old config)", "retuning (corrected logs)", "total"], rows)


def t_m5(t: pd.DataFrame) -> str:
    """Phase 5: the M5 OPC side (pre-revalidation configuration) vs the revalidated configuration, same logs, 25k."""
    rows = []
    order = [b for b in ALL_BIASES] + ["biased", "all"]
    for arm in ("opc", "dm", "tempered_logger"):
        for b in order:
            g = t[(t["method"] == arm) & (t["bias"] == b)]
            if g.empty:
                continue
            r = g.iloc[0]
            rows.append([ARMS[arm], LABEL.get(b, f"pooled: {b}"), str(int(r["worlds"])), _f(r["m5"], 2, False),
                         _f(r["revalidated"], 2, False), _ci(r["diff"], r["diff_lo"], r["diff_hi"]),
                         _f(r["max_abs_diff"], 3, False)])
    return _table(["arm", "bias", "worlds", "M5 (V %)", "revalidated (V %)", "revalidated − M5 (points)",
                   "largest absolute difference"], rows)


GROUP_SHORT = (("Stage 2 recovery", "Stage 2 recovery (fraction)"), ("Stage 2: OPC minus", "OPC − baseline (points)"),
               ("No-bias", "no-bias control (points)"), ("Reward-model", "reward-model tests (points)"),
               ("Logging support", "logging support (fraction)"), ("Importance weighting", "weighting (points)"),
               ("Selection", "selection (points)"), ("Selected OPC policy's raw-weight ESS", "OPC ESS (log10)"),
               ("Selected OPC policy's share", "OPC weights > 10 (%)"))


def _short(group: str) -> str:
    return next((short for prefix, short in GROUP_SHORT if group.startswith(prefix)), group)


def t_findings(f: pd.DataFrame) -> str:
    rows = []
    for _, r in f.iterrows():
        d = 3 if "fraction" in str(r.get("group", "")) else 2
        rows.append([_short(str(r.get("group", ""))), r["finding"], str(int(r["worlds"])),
                     _ci(r["old"], r["old_lo"], r["old_hi"], d), _ci(r["new"], r["new_lo"], r["new_hi"], d),
                     _ci(r["change"], r["change_lo"], r["change_hi"], d), r["verdict"]])
    return _table(["group", "finding", "worlds", "old", "corrected", "change", "classification"], rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old", default=str(ROOT / "summaries_20260927"))
    ap.add_argument("--new", default=str(REVAL / "summaries"))
    ap.add_argument("--report", default=str(REVAL / "report"))
    ap.add_argument("--phase0", default=str(REVAL / "phase0"))
    a = ap.parse_args(argv)
    old, new, rep = Path(a.old), Path(a.new), Path(a.report)
    rd = lambda p: pd.read_csv(p) if Path(p).exists() else pd.DataFrame()
    learned = lambda p: rd(p).query("train_size > 0") if Path(p).exists() else pd.DataFrame()
    t2 = rd(rep / "old_new_stage2.csv")
    parts = [("R1. Corrected Stage 2: learned repair (mean over 3 datasets × 2 seeds; gains in true CTR points)",
              t_stage2(rd(new / "stage2" / "stage2_fractions.csv"))),
             ("R2. Corrected Stage 2: OPC minus each baseline and minus the robustness arm (stochastic true CTR points, "
              "mean [95% CI] over the paired worlds)", t_contrasts(rd(new / "stage2" / "stage2_paired.csv"))),
             ("R3. Old vs corrected, biased worlds pooled (points; fractions greedy)",
              t_pooled(rd(rep / "old_new_stage2_biased_pooled.csv")))]
    for c in ("OPC - DM-only", "OPC - no-propensity", "OPC - tempered logger"):
        parts.append((f"R4. Old vs corrected: {c.replace(' - ', ' − ')} per bias × size (points)", t_old_new(t2, c)))
    for c in ("OPC fraction (greedy)", "DM-only fraction (greedy)"):
        parts.append((f"R5. Old vs corrected: {c} of the oracle repair per bias × size", t_old_new(t2, c, d=3)))
    parts += [("R6. Structural gap, learning gap and learned repair (greedy, CTR points; mean over 3 datasets × 2 seeds)",
               t_gap(rd(old / "followup" / "gap_decomposition.csv"), rd(new / "followup" / "gap_decomposition.csv"))),
              ("R7. Corrected Stage 2 per dataset (mean of 2 seeds): fraction OPC / DM-only; OPC − DM-only / OPC − "
               "no-propensity (stochastic points)", t_by_dataset(rd(new / "followup" / "stage2_by_dataset.csv"))),
              ("R8. Selected policies' weights and selection, biased worlds: old → corrected",
               t_diagnostics(learned(old / "stage2_learned_rows.csv"), learned(new / "stage2" / "stage2_learned_rows.csv"))),
              ("R9. Reward-model tests: old (older pipeline, buggy logs) → corrected (mean [95% CI] over ml, kuairand, "
               "anime × 2 seeds)", t_reward(rd(rep / "reward_model_tests_old.csv"), rd(rep / "reward_model_tests_new.csv"))),
              ("R10. Logging support at 25k (ml, kuairand × 2 seeds): old → corrected",
               t_support(rd(rep / "support_old.csv"), rd(rep / "support_new.csv"), coupling_by_share(Path(a.phase0)))),
              ("R11. Robustness arm: OPC with harmonic:0.1 (default) minus OPC with shrink:100 training weights "
               "(paired trial by trial)", t_robust(rd(new / "stage2" / "stage2_robust_shrink100_minus_default.csv"))),
              ("R12. The simulator fix vs the retuning (ml, kuairand; paired by world; points)",
               t_decomposition(rd(rep / "decomposition_simulator_vs_retuning.csv"))
               if (rep / "decomposition_simulator_vs_retuning.csv").exists() else "(the decomposition runs are missing)\n"),
              ("R13. Every major old finding, old vs corrected (classification: training/revalidation_compare.py)",
               t_findings(rd(rep / "findings_old_vs_new.csv"))),
              ("R14. Phase 5: the OPC side of the CausE M5 comparison vs the revalidated configuration (25k; the same "
               "worlds, logs and splits)", t_m5(rd(rep / "phase5_m5_opc_side_vs_revalidated.csv"))
               if (rep / "phase5_m5_opc_side_vs_revalidated.csv").exists() else "(the M5 run folder is missing)\n")]
    text = "# Revalidation tables (generated by training/revalidation_tables.py; do not edit)\n\n" + "\n".join(
        f"### {title}\n\n{body}" for title, body in parts)
    (rep / "tables.md").write_text(text)
    print(f"wrote {rep / 'tables.md'} ({len(parts)} tables)")


if __name__ == "__main__":
    main()
