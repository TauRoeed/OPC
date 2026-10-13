"""The oracle gradient diagnostics (training/ae_gradients.py, docs/opc_adaptive_estimator_study.md §10): θ from the
exported vectors, raw DR's expected gradient is the true one for any fixed q̂, the replicates average to the exact
expected gradient, the intermediate state's value is halfway."""
import numpy as np
import pytest
import torch

from training.ae_gradients import (
    dataset_gradients,
    expected_gradient,
    line_state,
    sample_rows,
    theta_from_export,
    true_gradient,
)
from training.opc_gradients import WorldTensors, build_model, exact_value, get_theta, set_theta
from utils.representation_bias import WorldConfig
from utils.seeding import seed_everything
from utils.structured_shift import build_structured_world

_W = {}


def _world():
    if "w" not in _W:
        rng = np.random.default_rng(0)
        scales = np.linspace(1.0, 0.25, 12)
        x = (rng.normal(size=(700, 12)) * scales + 0.3).astype(np.float32)
        a = (rng.normal(size=(600, 12)) * scales[::-1] + 0.2).astype(np.float32)
        seed_everything(3)
        ds = build_structured_world(x, a, "s-moderate.r-strong", seed=3, config=WorldConfig(target_ctr=0.05))
        world = WorldTensors(ds, torch.device("cpu"))
        m = build_model(ds, device=torch.device("cpu"))
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            m.action_transform.U.copy_(0.3 * torch.randn(m.action_transform.U.shape, generator=g))
            m.log_logit_scale.copy_(torch.tensor(0.02))
        _W["w"] = (ds, world, m)
    return _W["w"]


def test_theta_from_the_exported_vectors_reproduces_the_policy():
    ds, world, m = _world()
    ux, ia = (t.detach().numpy() for t in m.get_params())
    m2 = build_model(ds, device=torch.device("cpu"))
    set_theta(m2, theta_from_export(ds, ux, ia, torch.device("cpu")))
    u = torch.arange(50)
    with torch.no_grad():
        assert torch.allclose(m(u), m2(u), atol=1e-6)


def test_raw_dr_expected_gradient_is_the_true_gradient_for_any_fixed_qhat():
    ds, world, m = _world()
    wrong = lambda u: (0.5 * world.q(u) + 0.03).clamp(max=1.0)
    g_star = true_gradient(m, ds, world)
    g_raw = expected_gradient(m, ds, world, wrong, 0.0)
    assert np.allclose(g_raw, g_star, rtol=1e-4, atol=1e-8)
    g_h = expected_gradient(m, ds, world, wrong, 0.1)  # shrinkage with a wrong q̂ is biased
    assert np.linalg.norm(g_h - g_star) > 1e-3 * np.linalg.norm(g_star)


def test_replicate_gradients_average_to_the_exact_expectation():
    ds, world, m = _world()
    q = {"w": lambda u: (0.5 * world.q(u) + 0.03).clamp(max=1.0)}
    est = [("h", "w", "lam", 0.1)]
    G = np.stack([dataset_gradients(m, sample_rows(world, ds, 4000, 100 + r), world, q, est, 4000)["h"]
                  for r in range(300)])
    exact = expected_gradient(m, ds, world, q["w"], 0.1)
    se = G.std(axis=0, ddof=1) / np.sqrt(len(G))
    z = (G.mean(axis=0) - exact) / np.maximum(se, 1e-12)
    assert np.mean(np.abs(z) < 3.5) > 0.97


def test_the_intermediate_state_is_halfway_in_value():
    ds, world, m = _world()
    th, t = line_state(ds, get_theta(m), world)
    m_mid = build_model(ds, device=torch.device("cpu"))
    set_theta(m_mid, th)
    v_src = exact_value(build_model(ds, device=torch.device("cpu")), ds, world=world, grad=False)[0]
    v_fin = exact_value(m, ds, world=world, grad=False)[0]
    v_mid = exact_value(m_mid, ds, world=world, grad=False)[0]
    assert 0.0 < t < 1.0
    assert v_mid == pytest.approx(0.5 * (v_src + v_fin), abs=1e-4 * abs(v_fin - v_src) + 1e-7)
