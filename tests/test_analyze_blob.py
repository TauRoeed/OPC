"""training/analyze_blob.py: the simulated protocol, the tuning decision rule and the edge rule on synthetic trials."""
import numpy as np
import pandas as pd
import pytest

from training.analyze_blob import WIDE, edge_check, simulate_protocol, tuning_decision


def _trials(n_trials=40, worlds=4, seed=0):
    """Synthetic trials: the true gain rises with kappa_s and peaks at lr 3e-3; the validation NLL tracks it with
    noise. One trial per cell diverges."""
    rng = np.random.default_rng(seed)
    rows = []
    for family in ("nq", "mnq"):
        for w in range(worlds):
            for i in range(n_trials):
                lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(3e-2))))
                r = {"family": family, "dataset": "toy", "bias": f"b{w}", "seed": 200, "trial": i, "lr": lr,
                     "epochs": int(rng.choice(WIDE["epochs"])), "wa_m": float(rng.choice(WIDE["wa_m"])),
                     "wb_m": float(rng.choice(WIDE["wb_m"])), "kappa_s": float(rng.choice(WIDE["kappa_s"]))}
                gain = 2.0 * np.log10(r["kappa_s"] / 0.01) - 3.0 * (np.log10(lr) + 2.5) ** 2 + rng.normal(0, 0.1)
                r.update({"gain_greedy": gain, "val_nll": 0.5 - 0.01 * gain + rng.normal(0, 0.001),
                          "usable": i != 7, "finite": i != 7, "steps": 25 * r["epochs"]})
                rows.append(r)
    t = pd.DataFrame(rows)
    t.loc[~t["usable"], "val_nll"] = 1e9
    return t


def test_protocol_with_every_trial_selects_the_nll_argmin():
    t = _trials()
    sim = simulate_protocol(t, k=1000, resamples=3)
    for (family, bias), g in t[t["usable"]].groupby(["family", "bias"]):
        row = sim[(sim["family"] == family) & (sim["bias"] == bias)].iloc[0]
        assert row["gain_greedy"] == pytest.approx(g.loc[g["val_nll"].idxmin(), "gain_greedy"])
        assert row["best_greedy"] == pytest.approx(g["gain_greedy"].max())
        assert row["available"] == 40 and row["no_usable_trial"] == 0.0


def test_decision_takes_the_best_eligible_structure_then_windows():
    t = _trials(n_trials=60)
    d = tuning_decision(t, k=10, resamples=50)
    for family, g in d.groupby("family"):
        s = g[g["stage"] == "structure"]
        assert s.loc[s["gain_greedy"].idxmax(), "candidate"] == "kappa_s 1"
        chosen = g[g["chosen"]]
        assert len(chosen) == 1 and bool(chosen["eligible"].iloc[0])
        assert chosen["gain_greedy"].iloc[0] == g.loc[g["eligible"], "gain_greedy"].max()
        assert (g.loc[g["stage"] != "structure", "candidate"].str.startswith("kappa_s 1")).all()
        assert g.loc[g["candidate"] == "wide", "minus_wide"].iloc[0] == pytest.approx(0.0)


def test_edge_rule_extends_only_a_dominant_edge_value():
    t = _trials(n_trials=200)
    e = edge_check(t, {"nq": {}})
    k = e[e["dimension"] == "kappa_s"].iloc[0]
    assert k["best_value"] == "1" and k["at_wide_edge"] and k["extend"]  # +2 points per decade
    lr = e[e["dimension"] == "lr"].iloc[0]
    assert lr["best_value"] in ("1e-3-3e-3", "3e-3-1e-2") and not lr["extend"]  # interior optimum
