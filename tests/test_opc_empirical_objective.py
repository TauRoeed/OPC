"""The 25k decomposition's references (docs/opc_gradient_regime_study.md §6, §13): the full-data objectives against the
training loss, the population reward model and the harmonic population objective."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from models.custom_losses import DRPolicyLoss
from test_opc_gradients import D64, FixedQ, _model, tiny_world
from training.opc_empirical_objective import (
    _objective_backward,
    _Rows,
    population_reward_model,
    qinf_rows,
)
from training.opc_gradients import WorldTensors, build_model, exact_value, policy_params, set_theta


def _rows(ds, world, n=41, seed=2):
    rng = np.random.default_rng(seed)
    users = rng.integers(0, ds["n_users"], n)
    with torch.no_grad():
        p0 = world.logger(torch.as_tensor(users)).numpy()
    acts = np.array([rng.choice(ds["n_actions"], p=row / row.sum()) for row in p0])
    return {"x_idx": users, "a": acts, "r": (rng.random(n) < 0.3).astype(float), "pscore": p0[np.arange(n), acts]}


@pytest.mark.parametrize("spec", ["none", "harmonic:0.1"])
def test_the_full_data_dr_objective_is_the_training_loss_on_every_row(spec):
    ds = tiny_world(n_users=30, n_items=12)
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = _model(ds, scale=0.2)
    rows = _rows(ds, world)
    qh = FixedQ(np.random.default_rng(3).uniform(0.05, 0.5, size=(ds["n_users"], ds["n_actions"])))
    r = _Rows(rows, world, lookup=qh, source="qhat", chunk=7)  # several chunks; q̂ rows kept in float32, as training
    qh = FixedQ(qh.mat.float().double())  # the reference reads the same float32 values
    val = _objective_backward(model, r, "dr", spec)
    ours = [p.grad.clone() for p in policy_params(model)]
    model.zero_grad()
    u = torch.as_tensor(rows["x_idx"])
    loss = DRPolicyLoss(use_log_trick=False, weights=spec)(torch.as_tensor(rows["pscore"]), qh[u], model(u)[:, :, 0],
                                                          torch.as_tensor(rows["r"]), torch.as_tensor(rows["a"]))
    loss.backward()
    assert val == pytest.approx(float(loss), rel=1e-12)
    for g, p in zip(ours, policy_params(model)):
        torch.testing.assert_close(g, p.grad, rtol=1e-10, atol=1e-14)


def test_the_full_data_likelihood_is_the_click_nll_at_the_logged_pairs():
    ds = tiny_world(n_users=30, n_items=12)
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = build_model(ds, mode="click", device="cpu", dtype=D64)
    set_theta(model, 0.2 * np.random.default_rng(1).normal(size=sum(p.numel() for p in policy_params(model))))
    with torch.no_grad():
        model.click_intercept.fill_(-1.3)
    rows = _rows(ds, world)
    val = _objective_backward(model, _Rows(rows, world), "likelihood", None, backward=False)
    z = model(torch.as_tensor(rows["x_idx"]))[:, :, 0][torch.arange(len(rows["a"])), torch.as_tensor(rows["a"])]
    ref = torch.nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(rows["r"]))
    assert val == pytest.approx(float(ref), rel=1e-12)


def test_the_population_reward_model_recovers_a_representable_click_model_and_harmonic_dr_is_then_the_value():
    """When the true q is a logistic model of [x, a, x ⊙ a] on the logger's own vectors, q̂_∞ = q; with q̂ = q the
    harmonic DR population objective Σ π q̂ + π0 h(π/π0)(q − q̂) is the value V."""
    ds = tiny_world(n_users=40, n_items=15)
    ds["env"].emb_x = np.asarray(ds["our_x"], dtype=np.float64)  # the truth is the logger's view: representable
    ds["env"].emb_a = np.asarray(ds["our_a"], dtype=np.float64)
    world = WorldTensors(ds, "cpu", dtype=D64)
    users = np.arange(ds["n_users"])
    qinf = population_reward_model(ds, world, users)
    u = torch.as_tensor(users)
    err = qinf_rows(qinf, world, u).numpy() - world.q(u).numpy()
    with torch.no_grad():
        p0 = world.logger(u).numpy()
    assert np.sqrt((p0 * err ** 2).sum(axis=1).mean()) < 1e-4 and np.abs(err).max() < 1e-3  # q̂_∞ = q
    model = _model(ds, scale=0.2)
    v, _ = exact_value(model, ds, world=world, grad=False)
    with torch.no_grad():
        pi, p0, q = model(u)[:, :, 0], world.logger(u), world.q(u)
        w = pi / p0
        vh = ((pi * q + p0 * (w / (0.9 + 0.1 * w)) * (q - q)).sum(dim=1) * torch.as_tensor(ds["user_prior"])).sum()
    assert float(vh) == pytest.approx(v, rel=1e-12)
