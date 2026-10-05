"""The fair-comparison analysis (training/analyze_cause_fair.py): the resampled k-trial protocol, sub-space filters and
paired contrasts, on synthetic trials."""
import numpy as np
import pandas as pd
import pytest

from training.analyze_cause_fair import CELL, in_space, paired_table, simulate_protocol


def _trials(n_trials=12, seed=0):
    """Two worlds × one rho × predictions c and t; one trial diverged."""
    rng = np.random.default_rng(seed)
    rows = []
    for world_seed in (200, 201):
        for trial in range(n_trials):
            lr = float(10 ** rng.uniform(-4, 0))
            for p in ("c", "t"):
                rows.append({"dataset": "ml", "bias": "high", "seed": world_seed, "rho": 0.05, "prediction": p,
                             "trial": trial, "lr": lr, "epochs": int(rng.choice([1, 10, 100])),
                             "tie": "one_way" if trial % 2 else "symmetric", "val_nll": float(rng.uniform(0.3, 0.5)),
                             "gain_greedy": float(rng.normal(2.0, 1.0)), "usable": trial != 3})
    return pd.DataFrame(rows)


def test_protocol_with_every_trial_selects_the_lowest_usable_nll():
    t = _trials()
    sim = simulate_protocol(t, k=50, resamples=3)
    for _, r in sim.iterrows():
        g = t[(t["seed"] == r["seed"]) & (t["prediction"] == r["prediction"]) & t["usable"]]
        assert r["gain_greedy"] == pytest.approx(g.loc[g["val_nll"].idxmin(), "gain_greedy"])
        assert r["best_greedy"] == pytest.approx(g["gain_greedy"].max())
        assert r["available"] == 12 and r["no_usable_trial"] == 0.0


def test_protocol_draws_k_trials_and_shares_them_across_predictions():
    t = _trials()
    t.loc[t["prediction"] == "t", "val_nll"] = t.loc[t["prediction"] == "c", "val_nll"].to_numpy()
    t.loc[t["prediction"] == "t", "gain_greedy"] = t.loc[t["prediction"] == "c", "gain_greedy"].to_numpy()
    sim = simulate_protocol(t, k=4, resamples=300, seed=1).set_index(["seed", "prediction"])
    for s in (200, 201):  # identical predictions on identical draws give identical results
        assert sim.loc[(s, "c"), "gain_greedy"] == sim.loc[(s, "t"), "gain_greedy"]
        assert sim.loc[(s, "c"), "regret"] >= 0
    full = simulate_protocol(t, k=50, resamples=1).set_index(["seed", "prediction"])
    assert (sim["best_greedy"] <= full["best_greedy"] + 1e-12).all()  # fewer trials, a lower best


def test_draws_without_a_usable_trial_count_as_the_logger():
    t = _trials()
    t["usable"] = t["trial"] == 0
    sim = simulate_protocol(t, k=1, resamples=400, seed=2)
    assert sim["no_usable_trial"].between(0.85, 0.98).all()  # 11 of 12 single draws hit a diverged trial


def test_sub_space_filters():
    t = _trials()
    inside = t[in_space(t, {"lr": (1e-3, 1e-1), "tie": {"symmetric"}, "epochs": {10, 100}})]
    assert inside["lr"].between(1e-3, 1e-1).all() and set(inside["tie"]) == {"symmetric"}
    assert set(inside["epochs"]) <= {10, 100}
    sim = simulate_protocol(t, {"tie": {"one_way"}}, k=50, resamples=1)
    assert (sim["available"] == 6).all()


def test_paired_contrast_by_world():
    rows = []
    for b, base in (("high", 1.0), ("none", 0.0)):
        for d in ("ml", "anime"):
            for s in (100, 101):
                rows.append({"dataset": d, "bias": b, "seed": s, "arm": "opc", "rho": np.nan, "gain_greedy": base + 2.0})
                for rho in (0.0, 0.25):
                    rows.append({"dataset": d, "bias": b, "seed": s, "arm": "cap_c", "rho": rho,
                                 "gain_greedy": base + rho})
    p = paired_table(pd.DataFrame(rows), "opc", ["cap_c"]).set_index(["rho", "bias"])
    assert p.loc[(0.0, "high"), "a_minus_b_gain_greedy"] == pytest.approx(2.0)
    assert p.loc[(0.25, "biased (pooled)"), "a_minus_b_gain_greedy"] == pytest.approx(1.75)
    assert p.loc[(0.25, "none"), "worlds"] == 4 and p.loc[(0.25, "none"), "a_higher"] == 4
    assert CELL == ["dataset", "bias", "seed", "rho"]
