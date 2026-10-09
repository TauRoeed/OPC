"""The gradient benchmark's metrics (docs/opc_gradient_regime_study.md §4): the debiased bias, the bias ratio and the SNR
on Gaussian gradient samples with a known bias and noise; the figures from a table of world rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.analyze_opc_gradients import EST_STYLE, bias_tests, figures, gradient_metrics, unbiasedness_check


def _samples(bias: float, sigma: float = 0.5, R: int = 4000, P: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    gstar = rng.normal(size=P)
    gstar /= np.linalg.norm(gstar)
    b = rng.normal(size=P)
    b *= bias / np.linalg.norm(b)
    return gstar + b + sigma * rng.normal(size=(R, P)) / np.sqrt(P), gstar


def test_an_unbiased_estimator_has_its_bias_at_the_monte_carlo_floor():
    G, gstar = _samples(bias=0.0)
    m = gradient_metrics(G, gstar, ref=1.0)
    assert abs(m["rel_bias"]) < 3 * m["rel_bias_floor"] and m["bias_ratio"] < 3
    assert m["rel_total_var"] == pytest.approx(0.25, rel=0.05)  # tr C = σ²
    assert m["snr"] == pytest.approx(1 / 0.25, rel=0.05)  # ‖g*‖² / tr C
    assert m["p_positive"] > 0.95 and m["p_positive_lo"] < m["p_positive"] <= m["p_positive_hi"] + 1e-12


def test_the_debiased_bias_recovers_a_known_bias_and_scales_with_the_reference_norm():
    G, gstar = _samples(bias=0.2)
    m = gradient_metrics(G, gstar, ref=1.0)
    assert m["rel_bias"] == pytest.approx(0.2, rel=0.05) and m["bias_ratio"] > 100
    assert m["rel_bias_raw"] > m["rel_bias"]  # the raw norm keeps the Monte Carlo floor
    assert gradient_metrics(G, gstar, ref=2.0)["rel_bias"] == pytest.approx(m["rel_bias"] / 2)


def _one_direction(bias: float, seed: int, R: int = 200, P: int = 40):
    """Noise with one dominant direction (effective rank about 1.3, as at the source states)."""
    rng = np.random.default_rng(seed)
    gstar = np.ones(P) / np.sqrt(P)
    u = np.eye(P)[0]
    G = gstar + bias * u + rng.normal(size=(R, 1)) * u + 0.05 * rng.normal(size=(R, P))
    return G, gstar


def test_the_calibrated_bias_tests_hold_their_level_with_one_dominant_noise_direction_and_detect_a_bias():
    p_ratio, p_hot, ratio_gt3 = [], [], []
    for seed in range(300):
        G, gstar = _one_direction(0.0, seed)
        t = bias_tests(G, gstar)
        p_ratio.append(t["p_bias_ratio"])
        p_hot.append(t["p_bias_hotelling"])
        ratio_gt3.append(gradient_metrics(G, gstar, 1.0)["bias_ratio"] > 3)
    assert 1.0 < t["noise_rank_eff"] < 1.6
    assert 0.02 < np.mean(np.array(p_ratio) < 0.05) < 0.09 and 0.02 < np.mean(np.array(p_hot) < 0.05) < 0.09
    assert np.mean(ratio_gt3) > 0.05  # the raw ratio alone would flag unbiased estimators too often
    G, gstar = _one_direction(0.6, 0)  # a bias of 0.6 sd along the noisy direction, R = 200
    t = bias_tests(G, gstar)
    assert t["p_bias_ratio"] < 1e-3 and t["p_bias_hotelling"] < 1e-3


def test_the_unbiasedness_check_stops_only_on_raw_dr_beyond_the_holm_adjusted_level():
    rows = [{"dataset": "ml", "bias": f"b{i}", "seed": 100, "state": "mid", "estimator": e, "R": 300,
             "p_bias_ratio": p, "p_bias_hotelling": 0.5}
            for e, ps in (("G1", [0.001] * 3), ("G2", [0.2, 0.5, 0.9]), ("G3", [0.04, 0.6, 0.0001]))
            for i, p in enumerate(ps)]
    c = unbiasedness_check(pd.DataFrame(rows))
    assert c.loc[c["estimator"] == "G3", "p_bias_ratio_holm"].tolist() == pytest.approx([0.08, 0.6, 0.0003])
    assert c["stop"].tolist() == [False] * 6 + [False, False, True]  # G1 is never a stop, G2 not significant


def test_the_figures_are_written(tmp_path):
    rng = np.random.default_rng(0)
    rows = [{"estimator": e, "bias": b, "state": st, "snr": 10 ** rng.normal(), "p_positive": rng.random(),
             "cos_mean": rng.uniform(-0.2, 1), "rel_total_var": 10 ** rng.normal(-2, 1),
             "rel_bias": 10 ** rng.normal(-2, 1), "rel_bias_floor": 1e-3, "bias_ratio": 10 ** rng.normal(1, 1)}
            for e in EST_STYLE for b in ("none", "w-high.g-none.v-none", "high") for st in ("source", "mid", "mid_greedy")]
    figures(pd.DataFrame(rows), tmp_path)
    for name in ("fig_g1_snr", "fig_g2_alignment", "fig_g3_bias_noise"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.csv").exists()
    g3 = pd.read_csv(tmp_path / "fig_g3_bias_noise.csv")
    assert len(g3) == len(EST_STYLE) * 2 * 3  # the unbiased control worlds are left out


def test_the_tail_share_separates_gaussian_noise_from_a_few_extreme_replicates():
    G, gstar = _samples(bias=0.0, R=300)
    assert bias_tests(G, gstar)["top5_var_share"] < 0.03  # about 5 / 300 without heavy tails
    G[:3] += 40 * np.eye(G.shape[1])[:3]  # three replicates far out
    assert bias_tests(G, gstar)["top5_var_share"] > 0.9
