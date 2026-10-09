"""The regime analysis's accounting (docs/opc_gradient_regime_study.md §6, §9, §13 item 11): the selections, the margin
F (equal to the observed OPC − likelihood difference by construction) and the 25k decomposition adding up."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training.analyze_opc_regimes import WORLD, decomposition, margins, regime_summary, select


def _trials(seed=0, worlds=(("ml", "high", 100), ("ml", "w-none.g-high.v-none", 100), ("kuairand", "high", 100)),
            arms=("shared_likelihood", "shared_opc_raw", "shared_opc"), n=6):
    rng = np.random.default_rng(seed)
    rows = []
    for ds, bias, s in worlds:
        for share in (0.8, 0.95):
            for size in (5000, 25000):
                for arm in arms:
                    vals = 0.2 + 0.03 * rng.random(n)
                    native = int(rng.integers(n))
                    for k in range(n):
                        rows.append({"dataset": ds, "bias": bias, "seed": s, "share": share, "train_size": size,
                                     "method": arm, "trial_number": k, "actual_reward_greedy": vals[k],
                                     "diag_dr_greedy_low": rng.random(), "is_best_in_run": k == native,
                                     "V_logger_greedy": 0.19, "gain_greedy": 100 * (vals[k] - 0.19)})
    return pd.DataFrame(rows)


def _states(t):
    rng = np.random.default_rng(1)
    w = t[WORLD + ["share"]].drop_duplicates()
    return w.assign(value_greedy=0.26 + 0.01 * rng.random(len(w)), likelihood_greedy=0.24 + 0.01 * rng.random(len(w)),
                    value_ess=10 ** rng.uniform(-6, -1, len(w)), value_low_p0=rng.random(len(w)))


def test_the_margin_is_the_observed_difference_and_its_parts_add_up():
    t = _trials()
    sel = select(t)
    assert len(sel) == 3 * 2 * 2 * 3
    m = margins(sel, _states(t))
    assert len(m) == 3 * 2 * 2 * 2 * 2  # worlds × shares × sizes × OPC arms × rules
    np.testing.assert_allclose(m["F"], m["observed"], atol=1e-12)
    np.testing.assert_allclose(m["F"], m["M_L"] - m["dT"] - m["dS"], atol=1e-12)
    states = _states(t).set_index(WORLD + ["share"])
    r = m.iloc[0]
    vstar = _states(t).groupby(WORLD)["value_greedy"].max().loc[tuple(r[c] for c in WORLD)]
    v_log = states.loc[(*[r[c] for c in WORLD], r["share"]), "likelihood_greedy"]
    assert r["M_L"] == pytest.approx(100 * (vstar - v_log))
    ess = states.loc[(*[r[c] for c in WORLD], r["share"]), "value_ess"]
    assert r["n_eff_value"] == pytest.approx(r["train_size"] * ess)  # θ_value*'s effective sample size at N
    s = regime_summary(m)
    assert {"observed", "F", "M_L", "dT", "dS"} <= set(s.columns) and (s["worlds"] > 0).all()


def test_the_decomposition_adds_up_to_the_selected_trials_distance(tmp_path):
    t = _trials(arms=("shared_likelihood", "shared_opc_raw", "shared_opc"))
    t = t[(t["share"] == 0.8) & (t["train_size"] == 25000)].assign(train_size=25000)
    sel = select(t)
    states = _states(t)
    for ds, bias, s in t[WORLD].drop_duplicates().itertuples(index=False):
        wdir = tmp_path / f"dataset={ds}__bias={bias}__seed={s}__lgs=0.8"
        wdir.mkdir()
        for obj, g in (("likelihood", 0.235), ("dr_raw", 0.21), ("dr_harmonic", 0.225)):
            (wdir / f"empirical_{obj}.json").write_text(json.dumps({"greedy": g, "lr": 1e-3, "grad_norm": 0.1,
                                                                     "path_best_greedy_sample": g + 0.01}))
        (wdir / "population_harmonic.json").write_text(json.dumps({"greedy": 0.25}))
    d = decomposition(sel, states, tmp_path)
    assert set(d["arm"]) == {"shared_likelihood", "shared_opc_raw", "shared_opc"}
    np.testing.assert_allclose(d["M"] + d["E"] + d["O"] + d["S"], d["total"], atol=1e-12)
    assert (d.loc[d["arm"] == "shared_opc_raw", "M"] == 0).all()  # raw DR's population objective is the value
