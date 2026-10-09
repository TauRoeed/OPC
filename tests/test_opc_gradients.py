"""The OPC gradient diagnostics (docs/opc_gradient_regime_study.md §2-§4, §13): the exact population gradient against
finite differences and the study's value function; the estimators' expectations by enumeration of every row
outcome on a tiny world (raw IPS and raw DR unbiased with the oracle q or any fixed q̂, harmonic DR unbiased with
the oracle q and biased by the conditional-bias formula otherwise); the estimators against the training loss; the
minibatch accounting; reproducibility."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.custom_losses import DRPolicyLoss
from training.opc_gradients import (
    ESTIMATOR_NAMES,
    WorldTensors,
    build_model,
    estimator_gradients,
    exact_value,
    get_theta,
    logged_rows,
    minibatch_gradients,
    policy_params,
    set_theta,
)
from training.training_utils import minibatch_loss
from utils.simulation_utils import SyntheticBanditEnv

D64 = torch.float64


def tiny_world(seed=0, n_users=4, n_items=5, k=3, temperature=0.7):
    """A world small enough to enumerate every logged row (user, item, click)."""
    rng = np.random.default_rng(seed)
    clean_x, clean_a = rng.normal(size=(n_users, k)), rng.normal(size=(n_items, k))
    env = SyntheticBanditEnv(emb_x=clean_x, emb_a=clean_a, scale=0.9, offset=-1.2)
    our_x = (clean_x + 0.6 * rng.normal(size=clean_x.shape)).astype(np.float32)
    our_a = (clean_a + 0.6 * rng.normal(size=clean_a.shape)).astype(np.float32)
    return {"n_users": n_users, "n_actions": n_items, "emb_dim": k, "our_x": our_x, "our_a": our_a, "env": env,
            "user_prior": rng.dirichlet(np.ones(n_users)), "policy_temperature": temperature}


def toy_dataset():
    """The toy world of tests/test_reproducibility.py (its BPR-like embeddings), through generate_dataset."""
    from test_reproducibility import DIM, N_ITEMS, N_USERS
    from utils.simulation_utils import generate_dataset

    rng = np.random.default_rng(0)
    emb_x = rng.standard_normal((N_USERS, DIM)).astype(np.float32)
    emb_a = rng.standard_normal((N_ITEMS, DIM)).astype(np.float32)
    return generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=emb_x, emb_a=emb_a)


def _model(ds, theta_seed=1, scale=0.3):
    model = build_model(ds, device="cpu", dtype=D64)
    rng = np.random.default_rng(theta_seed)
    set_theta(model, scale * rng.normal(size=get_theta(model).size))
    return model


class FixedQ:
    """A fixed q̂: rows of a (users × items) matrix."""

    def __init__(self, mat):
        self.mat = torch.as_tensor(mat, dtype=D64)

    def __getitem__(self, u):
        return self.mat[u.cpu()]


def enumerate_rows(ds, world):
    """Every one-row outcome (u, a, r) with its probability prior(u) π0(a|u) P(r | u, a): an estimator that is a
    mean of per-row terms has the expectation Σ prob × term."""
    users, items, clicks, pscore, prob = [], [], [], [], []
    prior = np.asarray(ds["user_prior"])
    with torch.no_grad():
        p0 = world.logger(torch.arange(ds["n_users"])).numpy()
        q = world.q(torch.arange(ds["n_users"])).numpy()
    for u in range(ds["n_users"]):
        for a in range(ds["n_actions"]):
            for r in (0.0, 1.0):
                users.append(u), items.append(a), clicks.append(r), pscore.append(p0[u, a])
                prob.append(prior[u] * p0[u, a] * (q[u, a] if r else 1.0 - q[u, a]))
    rows = {"x_idx": np.array(users), "a": np.array(items), "r": np.array(clicks), "pscore": np.array(pscore)}
    return rows, np.array(prob), q


# ---------------------------------------------------------------------------------- the exact population gradient
def test_the_exact_gradient_matches_central_finite_differences():
    ds = tiny_world()
    model = _model(ds)
    world = WorldTensors(ds, "cpu", dtype=D64)
    v, g = exact_value(model, ds, world=world)
    theta = get_theta(model)
    h = 1e-6
    for k in [0, 5, 9, 12, 15, 20, len(theta) - 1]:  # D_u, b_u, D_a, b_a and θ_s coordinates
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        set_theta(model, tp)
        vp, _ = exact_value(model, ds, world=world, grad=False)
        set_theta(model, tm)
        vm, _ = exact_value(model, ds, world=world, grad=False)
        assert g[k] == pytest.approx((vp - vm) / (2 * h), rel=1e-6, abs=1e-9)
    set_theta(model, theta)
    assert exact_value(model, ds, world=world, grad=False)[0] == pytest.approx(v, abs=1e-15)


def test_the_exact_value_is_the_study_value_function():
    """V(θ) here is calc_reward's value of the same policy (the vectors with the scale folded in, the logger's T)."""
    from training.trainer_trials import _policy_reward_from_embeddings

    rng = np.random.default_rng(3)
    ds = toy_dataset()
    model = build_model(ds, device="cpu")
    set_theta(model, 0.05 * rng.normal(size=get_theta(model).size))
    v, _ = exact_value(model, ds, world=WorldTensors(ds, "cpu"), grad=False)
    x, a = model.get_params()
    ref = float(np.asarray(_policy_reward_from_embeddings(ds, x.detach().numpy(), a.detach().numpy())).reshape(-1)[0])
    assert v == pytest.approx(ref, rel=1e-5)


# ---------------------------------------------------------------------- the estimators' expectations, enumerated
@pytest.mark.parametrize("state", ["source", "moved"])
def test_raw_ips_and_raw_dr_are_unbiased_and_harmonic_dr_only_with_the_oracle_q(state):
    ds = tiny_world()
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = build_model(ds, device="cpu", dtype=D64) if state == "source" else _model(ds)
    _, g_star = exact_value(model, ds, world=world)
    rows, prob, q = enumerate_rows(ds, world)
    assert prob.sum() == pytest.approx(1.0)
    wrong = FixedQ(np.clip(q + np.random.default_rng(7).normal(scale=0.15, size=q.shape), 0.01, 0.99))
    exp_wrong = estimator_gradients(model, rows, world=world, qhat=wrong, row_weight=prob, conditional_bias=True)
    for e in ("G1", "G2", "G3", "G4"):  # G3 with an arbitrary fixed q̂: still unbiased
        np.testing.assert_allclose(exp_wrong[e], g_star, rtol=1e-9, atol=1e-12, err_msg=e)
    bias = exp_wrong["G5"] - g_star
    assert np.linalg.norm(bias) > 1e-4 * np.linalg.norm(g_star)  # harmonic DR with a wrong q̂: biased...
    np.testing.assert_allclose(bias, exp_wrong["bias5"], rtol=1e-8, atol=1e-12)  # ... by the conditional-bias formula
    exact_q = estimator_gradients(model, rows, world=world, qhat=FixedQ(q), row_weight=prob, conditional_bias=True)
    np.testing.assert_allclose(exact_q["G5"], g_star, rtol=1e-9, atol=1e-12)  # with q̂ = q, unbiased
    np.testing.assert_allclose(exact_q["bias5"], 0.0, atol=1e-14)


def test_at_the_source_the_harmonic_correction_is_scaled_by_h_prime_of_one():
    """Every weight is 1 at the logger: the harmonic correction's gradient is 0.9 times the raw one, h'(1) = 0.9."""
    ds = tiny_world()
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = build_model(ds, device="cpu", dtype=D64)
    rows, prob, q = enumerate_rows(ds, world)
    qh = FixedQ(np.full_like(q, 0.2))
    g = estimator_gradients(model, rows, world=world, qhat=qh, row_weight=prob, estimators=("G3", "G5"))
    dm = estimator_gradients(model, {**rows, "pscore": rows["pscore"] * 1e12}, world=world, qhat=qh, row_weight=prob,
                             estimators=("G3",))["G3"]  # weights ≈ 0: the DM term alone
    np.testing.assert_allclose(g["G5"] - dm, 0.9 * (g["G3"] - dm), rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------------------------ against the training loss
@pytest.mark.parametrize("spec,source", [("none", "zero"), ("none", "qhat"), ("harmonic:0.1", "qhat")])
def test_the_estimators_are_the_training_losses_gradient(spec, source):
    ds = tiny_world(n_users=30, n_items=12)
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = _model(ds, scale=0.2)
    rng = np.random.default_rng(11)
    n = 37
    users = rng.integers(0, ds["n_users"], n)
    with torch.no_grad():
        p0 = world.logger(torch.as_tensor(users)).numpy()
    acts = np.array([rng.choice(ds["n_actions"], p=row / row.sum()) for row in p0])
    rows = {"x_idx": users, "a": acts, "r": (rng.random(n) < 0.3).astype(float), "pscore": p0[np.arange(n), acts]}
    qh = FixedQ(rng.uniform(0.05, 0.5, size=(ds["n_users"], ds["n_actions"])))
    name = {("none", "zero"): "G1", ("none", "qhat"): "G3", ("harmonic:0.1", "qhat"): "G5"}[(spec, source)]
    ours = estimator_gradients(model, rows, world=world, qhat=qh, estimators=(name,), chunk=10)[name]
    u, a = torch.as_tensor(users), torch.as_tensor(acts)
    prob = model(u)[:, :, 0]
    scores = torch.zeros_like(prob) if source == "zero" else qh[u]
    loss = DRPolicyLoss(use_log_trick=False, weights=spec)(torch.as_tensor(rows["pscore"]), scores, prob,
                                                          torch.as_tensor(rows["r"]), a)
    ref = -torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss, policy_params(model))]).numpy()
    np.testing.assert_allclose(ours, ref, rtol=1e-10, atol=1e-14)


def test_an_epochs_minibatch_losses_add_up_to_the_full_batch_loss():
    """training_utils.minibatch_loss weights a short final batch by its rows, so the minibatch losses of a partition,
    each times batch / n, sum to the full-data loss: the estimators' full-data gradient is the epoch's gradient."""
    ds = tiny_world(n_users=30, n_items=12)
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = _model(ds, scale=0.2)
    rng = np.random.default_rng(5)
    n, b = 23, 8
    users = rng.integers(0, ds["n_users"], n)
    with torch.no_grad():
        p0 = world.logger(torch.as_tensor(users)).numpy()
    acts = np.array([rng.choice(ds["n_actions"], p=row / row.sum()) for row in p0])
    r = (rng.random(n) < 0.3).astype(float)
    crit = DRPolicyLoss(use_log_trick=False, weights="harmonic:0.1")
    qh = FixedQ(rng.uniform(0.05, 0.5, size=(ds["n_users"], ds["n_actions"])))
    full = crit(torch.as_tensor(p0[np.arange(n), acts]), qh[torch.as_tensor(users)], model(torch.as_tensor(users))[:, :, 0],
                torch.as_tensor(r), torch.as_tensor(acts))
    total = 0.0
    for s in range(0, n, b):
        u = torch.as_tensor(users[s:s + b])
        total = total + minibatch_loss(crit, torch.as_tensor(p0[np.arange(n), acts][s:s + b]), qh[u], model(u)[:, :, 0],
                                       torch.as_tensor(r[s:s + b]), torch.as_tensor(acts[s:s + b]), b) * (b / n)
    assert float(total) == pytest.approx(float(full), rel=1e-12)
    g_full = torch.autograd.grad(full, policy_params(model), retain_graph=True)
    g_sum = torch.autograd.grad(total, policy_params(model))
    for x, y in zip(g_full, g_sum):
        torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-14)


# ------------------------------------------------------------------------------------------------- reproducibility
def test_logged_rows_and_gradients_are_reproducible():
    rng = np.random.default_rng(4)
    ds = toy_dataset()
    a, b, c = logged_rows(ds, 500, seed=9), logged_rows(ds, 500, seed=9), logged_rows(ds, 500, seed=10)
    for k in ("x_idx", "a", "r", "pscore"):
        np.testing.assert_array_equal(a[k], b[k])
    assert not np.array_equal(a["a"], c["a"])
    world = WorldTensors(ds, "cpu")
    model = build_model(ds, device="cpu")
    set_theta(model, 0.03 * rng.normal(size=get_theta(model).size))
    qh = FixedQ(np.full((ds["n_users"], ds["n_actions"]), 0.05))
    g1 = estimator_gradients(model, a, world=world, qhat=qh)
    g2 = estimator_gradients(model, b, world=world, qhat=qh)
    for e in ESTIMATOR_NAMES:
        np.testing.assert_array_equal(g1[e], g2[e])
    mb1 = minibatch_gradients(model, a, world=world, qhat=qh, estimators=("G5",), batch=64, count=2, seed=1)
    mb2 = minibatch_gradients(model, a, world=world, qhat=qh, estimators=("G5",), batch=64, count=2, seed=1)
    np.testing.assert_array_equal(np.array(mb1["G5"]), np.array(mb2["G5"]))
