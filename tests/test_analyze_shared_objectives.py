"""training/analyze_shared_objectives.py on synthetic trials: the three selections, the edge rule and the
decomposition (docs/shared_objective_study.md §5, §8, §10)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.analyze_shared_objectives import (
    ARMS,
    NATIVE,
    WORLD,
    condition_table,
    edge_table,
    marginal_table,
    oracle_profile,
    oracle_summary,
    selections,
)


def _trials(seed=0, n=8, worlds=(("ml", "high", 100), ("ml", "none", 100))):
    rng = np.random.default_rng(seed)
    rows = []
    for ds, bias, s in worlds:
        for arm in ARMS:
            for k in range(n):
                rows.append({"dataset": ds, "bias": bias, "seed": s, "method": arm, "train_size": 1000, "run": 0,
                             "trial_number": k, "actual_reward_greedy": 0.2 + 0.01 * rng.random(),
                             "actual_reward": 0.15 + 0.01 * rng.random(), "value": rng.random(),
                             "diag_val_nll": rng.random(), "diag_val_iw_nll": rng.random(),
                             "diag_val_iw_nll_clip": rng.random(), "diag_dr_greedy_low": rng.random(),
                             "param_anchor_lambda": rng.choice([0.0, 0.01, 1.0]),
                             "param_lr": 10 ** rng.uniform(-4, np.log10(2e-3)),
                             "param_num_epochs": int(rng.integers(5, 31)), "param_batch_size": 1024,
                             "param_lr_decay": rng.uniform(0.8, 1.0), "V_logger_greedy": 0.2, "diverged": False})
    t = pd.DataFrame(rows)
    t["gain_greedy"] = 100 * (t["actual_reward_greedy"] - t["V_logger_greedy"])
    t["is_best_in_run"] = False
    for key, g in t.groupby(WORLD + ["method"]):  # the trainer's choice: the native column's best
        col, how = NATIVE[key[-1]]
        t.loc[g[col].idxmin() if how == "min" else g[col].idxmax(), "is_best_in_run"] = True
    return t


def test_the_three_selections():
    t = _trials()
    sel = selections(t)
    assert len(sel) == 2 * len(ARMS)
    for _, r in sel.iterrows():
        g = t[(t["dataset"] == r["dataset"]) & (t["bias"] == r["bias"]) & (t["method"] == r["arm"])]
        assert r["common_trial"] == g.loc[g["diag_dr_greedy_low"].idxmax(), "trial_number"]
        assert r["best_trial"] == g.loc[g["actual_reward_greedy"].idxmax(), "trial_number"]
        assert r["native_trial"] == g.loc[g["is_best_in_run"], "trial_number"].item()
        assert r["regret_native"] == pytest.approx(r["best_gain"] - r["native_gain"]) and r["regret_native"] >= 0
        assert r["regret_common"] >= 0
    bad = t.copy()  # a trainer choice that disagrees with the native column is an error
    g = bad[(bad["method"] == "shared_likelihood") & (bad["bias"] == "high")]
    bad.loc[g.index, "is_best_in_run"] = False
    bad.loc[g["diag_val_nll"].idxmax(), "is_best_in_run"] = True
    with pytest.raises(ValueError, match="native column"):
        selections(bad)


def test_the_edge_rule_fires_only_past_the_margin_at_an_extendable_edge():
    t = _trials(n=40, worlds=tuple(("ml", b, s) for b in ("high", "none") for s in (100, 101)))
    t = t[t["method"] == "shared_opc"].copy()
    top = t["param_lr"] >= 1e-3
    t.loc[top, "actual_reward_greedy"] += 0.02  # the top lr bin is 2 points better: past the 0.25 margin
    t["gain_greedy"] = 100 * (t["actual_reward_greedy"] - t["V_logger_greedy"])
    e = edge_table(marginal_table(t)).set_index("dimension")
    assert e.loc["lr", "edge"] == "high" and bool(e.loc["lr", "extend"])
    t2 = t.copy()
    t2.loc[top, "actual_reward_greedy"] -= 0.02 - 0.001  # now only 0.1 points: within the margin
    t2["gain_greedy"] = 100 * (t2["actual_reward_greedy"] - t2["V_logger_greedy"])
    assert not bool(edge_table(marginal_table(t2)).set_index("dimension").loc["lr", "extend"])
    t3 = t2.copy()
    zero = t3["param_anchor_lambda"] == 0.0
    t3.loc[zero, "actual_reward_greedy"] += 0.05  # λ = 0 best: an edge, but the bottom of λ cannot be extended
    t3["gain_greedy"] = 100 * (t3["actual_reward_greedy"] - t3["V_logger_greedy"])
    lam = edge_table(marginal_table(t3)).set_index("dimension").loc["anchor_lambda"]
    assert lam["edge"] == "low" and not bool(lam["extendable"]) and not bool(lam["extend"])


def test_the_decomposition_adds_up():
    t = _trials()
    sel = selections(t)
    oracles = []
    for ds, bias, s in t[WORLD].drop_duplicates().itertuples(index=False):
        for obj, v in (("value", 0.25), ("likelihood", 0.23), ("uniform_likelihood", 0.235), ("clip10_likelihood", 0.232)):
            oracles.append({"dataset": ds, "bias": bias, "seed": s, "optimum": obj, "V_greedy": v})
    summaries = pd.DataFrame([{**dict(zip(WORLD, w)), "method": a, "train_w_ess": 1.0}
                              for w in t[WORLD].drop_duplicates().itertuples(index=False) for a in ARMS])
    comps = pd.DataFrame(columns=WORLD + ["arm", "native_V_greedy", "native_V", "native_gain", "V_logger_greedy",
                                          "V_logger"])
    c = condition_table(sel, summaries, pd.DataFrame(oracles), comps)
    total = c["objective_mismatch"] + c["training_gap"] + c["selection_gap_native"]
    np.testing.assert_allclose(total, c["total_gap_native"], atol=1e-9)
    assert (c.loc[c["arm"] == "shared_opc", "objective_mismatch"] == 0).all()
    np.testing.assert_allclose(c.loc[c["arm"] == "shared_likelihood", "objective_mismatch"], 2.0, atol=1e-9)
    frac = (c["native_V_greedy"] - c["V_logger_greedy"]) / (0.25 - c["V_logger_greedy"])
    biased = c["bias"] != "none"
    np.testing.assert_allclose(c.loc[biased, "frac_value_gap_native"], frac[biased])
    assert c.loc[~biased, "frac_value_gap_native"].isna().all()  # no gap to recover without bias


def test_a_supplementary_round_is_selected_per_run_and_pooled_for_the_edge_rule():
    first, supp = _trials(seed=1).assign(run_tag="first"), _trials(seed=2).assign(run_tag="supp")
    supp["param_lr"] *= 10 ** 0.5  # the extended range: half a decade above the first round's
    t = pd.concat([first, supp], ignore_index=True)
    sel = selections(t)
    assert len(sel) == 2 * 2 * len(ARMS) and set(sel["run_tag"]) == {"first", "supp"}  # one native choice per run
    m = marginal_table(t)
    lr = m[(m["arm"] == "shared_opc") & (m["dimension"] == "lr")]
    assert lr["trials"].sum() == 2 * 2 * 8  # both rounds, both worlds
    assert lr["value"].iloc[-1].startswith("[0.0032")  # a bin past the first round's top
    g = t[t["method"] == "shared_opc"]
    gap = g["gain_greedy"] - g.groupby(WORLD)["gain_greedy"].transform("max")  # the best over both rounds
    assert (lr["below_best_mean"] * lr["trials"]).sum() == pytest.approx(gap.sum())


def test_the_population_tables_average_over_worlds_per_bias():
    rng = np.random.default_rng(3)
    rows, pairs = [], []
    for bias in ("none", "high", "w-high.g-none.v-none"):
        for s in (100, 101):
            for obj, v in (("likelihood", 0.25), ("uniform_likelihood", 0.24), ("clip10_likelihood", 0.245),
                           ("value", 0.26)):
                rows.append({"dataset": "ml", "bias": bias, "seed": s, "optimum": obj, "V_greedy": v + 0.01 * s / 100,
                             "V_logger": 0.2, "V_stochastic": 0.255 if obj == "value" else np.nan,
                             "gain_greedy": 100 * (v - 0.22), **{c: rng.random() for c in (
                                 "L_likelihood", "L_uniform_likelihood", "L_clip10_likelihood", "M_dev", "item_share")}})
            pairs.append({"dataset": "ml", "bias": bias, "seed": s, "a": "likelihood", "b": "value",
                          "M_cosine_distance": 0.1 * s / 100, "top1_agreement": 0.5})
    o, d = pd.DataFrame(rows), pd.DataFrame(pairs)
    summ = oracle_summary(o).set_index(["bias", "quantity"])
    assert summ.loc[("high", "V(θ_value*) − V(θ_log*)"), "mean"] == pytest.approx(1.0)
    assert summ.loc[("biased (pooled)", "V(θ_uniform*) − V(θ_log*)"), "worlds"] == 4
    profile, dist = oracle_profile(o, d)
    v = profile.set_index(["bias", "optimum"]).loc[("none", "value")]
    assert v["stochastic_gain"] == pytest.approx(5.5) and v["worlds"] == 2
    lik = o[(o["bias"] == "high") & (o["optimum"] == "likelihood")]
    assert profile.set_index(["bias", "optimum"]).loc[("high", "likelihood"), "M_dev"] == pytest.approx(lik["M_dev"].mean())
    assert dist.set_index("bias").loc["biased (pooled)", "M_cosine_distance"] == pytest.approx(0.1005)
