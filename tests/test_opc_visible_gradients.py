"""The value gradient over the logged data's visible pairs (docs/opc_gradient_regime_study.md §12.3) and the
unbiasedness check's verdicts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_opc_gradients import D64, _model, tiny_world
from training.analyze_opc_gradients import unbiasedness_check
from training.opc_gradients import WorldTensors, exact_value, policy_params, visible_value_gradient
from training.opc_visible_gradients import low_overlap_states


def test_the_visible_gradient_is_g_star_without_a_limit_and_the_masked_gradient_with_one():
    ds = tiny_world(n_users=30, n_items=12)
    world = WorldTensors(ds, "cpu", dtype=D64)
    model = _model(ds, scale=0.4)
    _, gstar = exact_value(model, ds, world=world, chunk=7)
    g0, hidden0 = visible_value_gradient(model, ds, world=world, rows=1e6, min_count=0.0, chunk=7)
    np.testing.assert_array_equal(g0, gstar)  # exact_value's own arithmetic
    assert hidden0 == 0.0
    rows, prior = 400.0, torch.as_tensor(ds["user_prior"], dtype=D64)
    g, hidden = visible_value_gradient(model, ds, world=world, rows=rows, min_count=1.0, chunk=7)
    u = torch.arange(ds["n_users"])
    with torch.no_grad():
        seen = (rows * prior[:, None] * world.logger(u) >= 1.0).double()
    assert 0.0 < seen.mean() < 1.0  # the limit binds on some pairs, not all
    pi = model(u)[:, :, 0]
    ref = torch.autograd.grad(((pi * world.q(u) * seen).sum(dim=1) * prior).sum(), policy_params(model))
    np.testing.assert_allclose(g, torch.cat([r.reshape(-1) for r in ref]).numpy(), rtol=1e-10, atol=1e-14)
    assert hidden == pytest.approx(float(((pi.detach() * (1 - seen)).sum(dim=1) * prior).sum()), rel=1e-12)


def test_only_the_low_overlap_states_are_selected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"states": ["source", "mid", "value"]}))
    (tmp_path / "gstar.json").write_text(json.dumps({s: {"pop_ess_share": e} for s, e in
                                                     (("source", 1.0), ("mid", 5e-5), ("value", 3e-6),
                                                      ("likelihood", 1e-6))}))
    assert low_overlap_states(tmp_path) == ["mid", "value"]  # likelihood was not benchmarked here


def test_a_rejection_is_a_practical_support_failure_only_with_low_overlap_and_a_consistent_visible_gradient():
    cells = [  # (bias, ess, p vs g*, p vs the visible gradient)
        ("a", 3e-6, 1e-40, 0.6),    # low overlap, consistent with the visible pairs: practical support failure
        ("b", 3e-6, 1e-40, 1e-30),  # low overlap, rejected against the visible pairs too: unresolved
        ("c", 0.2, 1e-40, np.nan),  # good overlap: biased
        ("d", 0.2, 0.5, np.nan),    # consistent
    ]
    rows = [{"dataset": "ml", "bias": b, "seed": 100, "state": "value", "estimator": e, "R": 300,
             "pop_ess_share": ess, "p_bias_ratio": 0.5, "p_bias_hotelling": p,
             "p_bias_ratio_vis": 0.5 if np.isfinite(pv) else np.nan, "p_bias_hotelling_vis": pv}
            for e in ("G1", "G3") for b, ess, p, pv in cells]
    c = unbiasedness_check(pd.DataFrame(rows))
    verdicts = ["practical support failure", "low overlap, unresolved", "biased", "consistent"]
    assert c["verdict"].tolist() == verdicts * 2
    assert c["stop"].tolist() == [False] * 4 + [False, True, True, False]  # only raw DR stops
