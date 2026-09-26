"""--opc-gradient: what OPC's DR loss optimizes under the log trick and under the direct gradient, for the
weight transforms the study uses (none, clip:10, shrink:100, shrink:10000).

The log trick puts the transformed weight g(w) as a detached coefficient on grad log pi(a|x). Since
grad log pi = grad w / w, it follows the exact gradient of DM + H(w) (r - q_hat) with
H(w) = int_0^w g(t)/t dt, not of the nominal transformed estimate DM + g(w) (r - q_hat), whose exact
(direct, pathwise) gradient carries g'(w). The two agree only for raw weights.
"""

import math
import sys

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from models.custom_losses import DRPolicyLoss
from test_objective_gradients import _H, _close, _epoch_direction, _far, _g, _grad, _literal

STUDY_SPECS = ["none", "clip:10", "shrink:100", "shrink:10000"]


@pytest.fixture(scope="module")
def heavy():
    """Weights that pass 10 and 100, so clip:10, shrink:100 and shrink:10000 all act: a flat logger over 200
    actions (propensities near 1/200), a sharp policy, and rows drawn partly where the policy is confident
    (the identities tested hold for any logged rows)."""
    g = torch.Generator().manual_seed(5)
    n, n_users, n_actions, d = 3000, 150, 200, 6
    X = torch.randn(n_users, d, generator=g, dtype=torch.float64)
    theta0 = torch.randn(d, n_actions, generator=g, dtype=torch.float64)
    logger = torch.softmax(0.1 * X @ theta0, dim=1)
    theta = 3.0 * (theta0 + 0.3 * torch.randn(d, n_actions, generator=g, dtype=torch.float64))
    users = torch.randint(0, n_users, (n,), generator=g)
    rows = 0.3 * torch.softmax(X @ theta, dim=1) + 0.7 * logger
    actions = torch.multinomial(rows[users], 1, generator=g).squeeze(1)
    q_hat = 0.05 + 0.3 * torch.rand(n_users, n_actions, generator=g, dtype=torch.float64)
    q_true = (q_hat + 0.1 + 0.15 * torch.randn(n_users, n_actions, generator=g, dtype=torch.float64)).clamp(0.01, 0.9)
    w = dict(X=X, theta=theta, q_hat=q_hat, users=users, actions=actions, pscore=logger[users, actions],
             rewards=torch.bernoulli(q_true[users, actions], generator=g))
    _, iw, _ = _literal(w, w["theta"])
    assert float((iw > 10).double().mean()) > 0.2 and float((iw > 100).double().mean()) > 0.02
    return w


@pytest.mark.parametrize("spec", ["clip:10", "shrink:100", "shrink:10000"])
def test_closed_form_of_the_induced_weight(spec):
    """H(w) = int_0^w g(t)/t dt: w up to M then M(1 + ln(w/M)) for clip:M; sqrt(lam) arctan(w / sqrt(lam))
    for shrink:lam."""
    mode, p = spec.split(":")
    p = float(p)
    for w in (0.5, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        num, _ = quad(lambda t: float(_g(torch.tensor(t, dtype=torch.float64), spec)) / t, 0.0, w, limit=200,
                      points=[p] if mode == "clip" and p < w else None)
        assert float(_H(torch.tensor(w, dtype=torch.float64), spec)) == pytest.approx(num, rel=1e-8)
    if mode == "shrink":  # bounded: H rises to sqrt(lam) pi / 2, while g peaks at sqrt(lam) / 2 and falls back to 0
        assert float(_H(torch.tensor(1e9, dtype=torch.float64), spec)) == pytest.approx(math.sqrt(p) * math.pi / 2, rel=1e-6)


def _near(a, b):
    """The log trick clamps pi at 1e-10 inside log, so actions below it (a third of the (row, action) pairs in
    this deliberately sharp world, pi down to 1e-29) drop out of its gradient: ~2e-11 against entries of ~0.1.
    With no probability below the clamp the match is exact to ~1e-16."""
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("spec", STUDY_SPECS)
def test_log_trick_and_direct_gradients(heavy, spec):
    log_trick, _ = _epoch_direction(heavy, DRPolicyLoss(use_log_trick=True, weights=spec), 384)
    direct, _ = _epoch_direction(heavy, DRPolicyLoss(use_log_trick=False, weights=spec), 384)
    _near(log_trick, _grad(lambda dm, w, res: dm.mean() + (_H(w, spec) * res).mean(), heavy))
    _close(direct, _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).mean(), heavy))
    if spec == "none":
        _near(log_trick, direct)
    else:
        _far(log_trick, direct)


@pytest.mark.parametrize("log_trick", [True, False])
def test_the_direct_shrinkage_gradient_turns_against_a_clicked_heavy_row(log_trick):
    """One logged row, clicked (r = 1 > q_hat), weight w = 20 > sqrt(100): the nominal shrunk estimate
    lam w e / (w^2 + lam) falls as w grows past sqrt(lam), so its exact gradient lowers the clicked action's
    probability; the log trick raises it (coefficient g(w) / w > 0 at every w). q_hat is constant, so the DM
    term has no gradient."""
    n_actions, a = 8, 3
    logits = torch.zeros(1, n_actions, dtype=torch.float64)
    logits[0, a] = 2.0
    pi_a = float(torch.softmax(logits, dim=1)[0, a])
    pscore = torch.tensor([pi_a / 20.0], dtype=torch.float64)  # w = 20
    lg = logits.clone().requires_grad_(True)
    loss = DRPolicyLoss(use_log_trick=log_trick, weights="shrink:100")
    val = loss(pscore, torch.full((1, n_actions), 0.1, dtype=torch.float64), torch.softmax(lg, dim=1),
               torch.tensor([1.0], dtype=torch.float64), torch.tensor([a]))
    (grad,) = torch.autograd.grad(val, lg)
    ascent = -float(grad[0, a])
    lam, w, e = 100.0, 20.0, 0.9
    coef = lam / (w * w + lam) if log_trick else lam * (lam - w * w) / (lam + w * w) ** 2  # on grad w
    assert ascent == pytest.approx(coef * e * w * (1 - pi_a), rel=1e-9)  # d w / d logit_a = w (1 - pi_a)
    assert (ascent > 0) if log_trick else (ascent < 0)


def test_opc_gradient_reaches_the_opc_loss_only(tmp_path, monkeypatch):
    from test_reproducibility import _toy_embeddings

    import training.trainer_trials as tt
    from training.run_full_study import OPC_GRADIENTS, _finalize_summary_df, _run_condition

    assert OPC_GRADIENTS == ("log-trick", "direct")
    _toy_embeddings(tmp_path)
    calls = []
    real = tt._policy_loss_from_name
    monkeypatch.setattr(tt, "_policy_loss_from_name",
                        lambda name, **kw: calls.append((name, kw["use_log_trick"])) or real(name, **kw))
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=1,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, policy_loss_types=("dr",),
              train_weights="shrink:100", methods=("opc", "no_propensity", "dm"))
    rewards = {}
    for gradient in ("direct", "log-trick"):
        calls.clear()
        run_dir = tmp_path / gradient
        run_dir.mkdir()
        opc, nop, opc_trials, _, meta, extra = _run_condition(**kw, run_dir=run_dir, opc_gradient=gradient,
                                                              return_extra=True)
        assert sorted(calls) == sorted([("dr", gradient == "log-trick"), ("naive", False), ("dm", False)])
        assert meta["opc_gradient"] == gradient and meta["opc_use_log_trick_fixed"] == (gradient == "log-trick")
        assert (_finalize_summary_df(opc, nop, meta, extra=extra)["opc_gradient"] == gradient).all()
        rewards[gradient] = opc_trials
    with pytest.raises(ValueError, match="opc_gradient"):
        _run_condition(**kw, run_dir=tmp_path, opc_gradient="reinforce")


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli(module, monkeypatch, capsys, tmp_path):
    import argparse

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    text = " ".join(capsys.readouterr().out.split())
    assert "--opc-gradient" in text and "does not change training here" in text
    real_parse, seen = argparse.ArgumentParser.parse_args, {}

    class Parsed(Exception):
        pass

    def parse(self, args=None, namespace=None):
        seen["default"] = real_parse(self, [], namespace)
        seen["set"] = real_parse(self, ["--opc-gradient", "direct"], namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        main()
    assert seen["default"].opc_gradient == "log-trick" and seen["set"].opc_gradient == "direct"


def test_parallel_runner_forwards_the_gradient(monkeypatch, tmp_path):
    import training.run_full_study_parallel as par

    captured = {}
    monkeypatch.setattr(par, "_run_with_memory_cap", lambda configs, **kw: captured.setdefault("configs", list(configs)) and [])
    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", "--datasets", "ml", "--seeds", "7", "--bias-configs",
                                      "medium", "--train-sizes", "5000", "--opc-gradient", "direct", "--out-dir",
                                      str(tmp_path), "--run-tag", "t", "--no-skip-completed"])
    par.main()
    (config,) = captured["configs"]
    assert config["opc_gradient"] == "direct"

    class Called(Exception):
        pass

    def fake_condition(**kwargs):
        captured["kwargs"] = kwargs
        raise Called

    monkeypatch.setattr(par, "_run_condition", fake_condition)
    with pytest.raises(Called):
        par._execute_run(config)
    assert captured["kwargs"]["opc_gradient"] == "direct"
