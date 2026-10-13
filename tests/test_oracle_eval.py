"""The fixed-policy oracle evaluator (training/oracle_eval.py, docs/opc_adaptive_estimator_study.md §8): exact moments
against brute-force enumeration, the replicates against the exact moments, the source policy's support."""
import math

import numpy as np
import pytest
import torch

from training.oracle_eval import (
    DenseWorld,
    FixedPolicy,
    exact_world,
    linear_estimators,
    mc_world,
    summarize_mc,
    transform,
)
from utils.representation_bias import WorldConfig
from utils.seeding import seed_everything
from utils.structured_shift import build_structured_world


class _Lookup:
    def __init__(self, m):
        self.m = torch.as_tensor(m, dtype=torch.float32)

    def __getitem__(self, u):
        return self.m[torch.as_tensor(u).cpu()]


def _world(seed=3):
    """The structured-shift tests' vectors (700 users, 600 items, K = 12), a strong-response world."""
    rng = np.random.default_rng(0)
    scales = np.linspace(1.0, 0.25, 12)
    x = (rng.normal(size=(700, 12)) * scales + 0.3).astype(np.float32)
    a = (rng.normal(size=(600, 12)) * scales[::-1] + 0.2).astype(np.float32)
    seed_everything(seed)
    ds = build_structured_world(x, a, "s-moderate.r-strong", seed=seed, config=WorldConfig(target_ctr=0.05))
    n_users, n_items = ds["n_users"], ds["n_actions"]
    q = np.asarray(ds["env"].reward_prob_block(np.arange(n_users), 0, n_items), dtype=np.float64)
    qhat = np.clip(q + rng.normal(scale=0.05, size=q.shape), 0.01, 0.99)  # a fixed, wrong reward model
    world = DenseWorld(ds, _Lookup(qhat), torch.device("cpu"))
    T = world.T
    ux = ds["our_x"] + rng.normal(scale=0.3, size=ds["our_x"].shape)
    pols = [FixedPolicy("source", ds["our_x"], ds["our_a"], T, "softmax", "cpu"),
            FixedPolicy("source_g", ds["our_x"], ds["our_a"], T, "greedy", "cpu"),
            FixedPolicy("pi", 1.5 * ux, ds["our_a"], T, "softmax", "cpu"),
            FixedPolicy("pi_g", 1.5 * ux, ds["our_a"], T, "greedy", "cpu")]
    for p in pols:
        p.prepare(world.n_users)
    return ds, world, pols, q, qhat


SOURCE = {"softmax": "source", "greedy": "source_g"}


def test_exact_moments_equal_brute_force_enumeration():
    ds, world, pols, q, qhat = _world()
    ex = exact_world(world, pols, n=1, source=SOURCE)
    prior = world.prior.numpy()
    mu = world.mu.double().numpy()
    mu = mu / mu.sum(axis=1, keepdims=True)
    q, qhat = world.q.double().numpy(), world.qhat.double().numpy()  # the stored (float32) matrices the sums read
    probs = {p.name: p.probs(torch.arange(world.n_users)).double().numpy() for p in pols}
    for p in pols:
        pi, pi0 = probs[p.name], probs[SOURCE[p.kind]]
        assert ex[p.name]["V"] == pytest.approx(float(prior @ (pi * q).sum(1)), rel=1e-9)
        assert ex[p.name]["S2"] == pytest.approx(float(prior @ (pi ** 2 / mu).sum(1)), rel=1e-9)
        for e, (kind, param, src) in linear_estimators().items():
            base = q if src == "q" else qhat
            dm, dm0 = (pi * base).sum(1), (pi0 * base).sum(1)
            g = transform(torch.as_tensor(pi / mu), kind, param).numpy()
            g0 = transform(torch.as_tensor(pi0 / mu), kind, param).numpy()
            m1 = m2 = d1 = d2 = 0.0
            for r, pr in ((1.0, q), (0.0, 1 - q)):  # every (x, a, r) outcome with its probability
                prob = prior[:, None] * mu * pr
                f = dm[:, None] + g * (r - base)
                fd = f - (dm0[:, None] + g0 * (r - base))
                m1, m2 = m1 + (prob * f).sum(), m2 + (prob * f * f).sum()
                d1, d2 = d1 + (prob * fd).sum(), d2 + (prob * fd * fd).sum()
            r_ = ex[p.name]["estimators"][e]
            assert r_["mean"] == pytest.approx(m1, rel=1e-8, abs=1e-12)
            assert r_["var_row"] == pytest.approx(m2 - m1 * m1, rel=1e-7, abs=1e-12)
            assert r_["delta_mean"] == pytest.approx(d1, rel=1e-8, abs=1e-12)
            assert r_["delta_var_row"] == pytest.approx(d2 - d1 * d1, rel=1e-7, abs=1e-12)


def test_raw_and_oracle_dr_are_unbiased_and_the_source_has_unit_weights():
    ds, world, pols, q, qhat = _world()
    ex = exact_world(world, pols, n=1000, source=SOURCE)
    for p in pols:
        est = ex[p.name]["estimators"]
        assert abs(est["dr_raw"]["bias"]) < 1e-9 and abs(est["dr_oracle"]["bias"]) < 1e-9
        assert abs(est["dr_raw"]["delta_bias"]) < 1e-9
    assert ex["source"]["S2"] == pytest.approx(1.0, abs=1e-9)  # the source is the logger: w ≡ 1
    for e in ("dr_h0.1", "dr_clip10", "dr_raw"):
        assert abs(ex["source"]["estimators"][e]["bias"]) < 1e-9  # h(1) = 1: no shrinkage at the logger


def test_replicates_agree_with_the_exact_moments():
    ds, world, pols, q, qhat = _world()
    n, R = 400, 1000
    ex = exact_world(world, pols, n=n, source=SOURCE)
    rows = mc_world(world, pols, n=n, replicates=R, seed=11, source=SOURCE)
    s = summarize_mc(rows, ex)
    lin = s[s["estimator"].isin(list(linear_estimators()))]
    assert (lin["bias_z"].abs() < 4.5).all(), lin.loc[lin["bias_z"].abs() >= 4.5, ["policy", "estimator", "bias_z"]]
    # bounded weights: the replicates' variance is close to the exact one; raw weights are heavy-tailed, and a sample
    # variance of heavy-tailed replicates usually falls short of its expectation (a looser check)
    bounded = lin["estimator"].isin(["dm", "dr_clip10", "dr_h0.03", "dr_h0.1", "dr_h0.2"])
    assert lin.loc[bounded, "var_ratio"].between(0.8, 1.25).all(), lin.loc[bounded, ["policy", "estimator", "var_ratio"]]
    assert lin.loc[~bounded, "var_ratio"].between(0.6, 1.6).all(), lin.loc[~bounded, ["policy", "estimator", "var_ratio"]]
    raw = s[(s["estimator"] == "dr_raw") & (s["kind"] == "softmax")]
    assert raw["mc_coverage"].between(0.9, 0.98).all()
