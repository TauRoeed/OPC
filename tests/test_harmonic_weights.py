"""Harmonic importance-weight correction (Metelli, Russo and Restelli, NeurIPS 2021, Definition 4.1 with s = -1):

    w_lam = ((1 - lam) w^-1 + lam)^-1 = w / (1 - lam + lam w),   lam in [0, 1],

the weighted harmonic mean of w and 1 (weights 1 - lam and lam): lam = 0 is raw IS, lam = 1 gives 1, and
w_lam <= 1 / lam (their Lemma 4.1 (ii)). Its gradient (1 - lam) / (1 - lam + lam w)^2 * grad w (their
Section 5) is positive and bounded, so OPC optimizes DR with these weights by its direct gradient (their
DR-lambda estimator, off-policy learning by gradient ascent)."""

import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from models.custom_losses import DRPolicyLoss, harmonic_importance_weights, transform_importance_weights
from models.estimators import DoublyRobust, InverseProbabilityWeighting
from test_objective_gradients import _close, _epoch_direction, _g, _grad, _literal
from test_opc_gradient import heavy  # noqa: F401  (fixture)
from utils.importance_weights import parse_weight_spec, transform_weights, weight_spec_label
from utils.simulation_utils import _estimator_weight_kwargs

LAMS = [0.05, 0.1, 0.2]
W = np.array([0.0, 1e-6, 0.3, 1.0, 2.0, 10.0, 99.0, 1e4, 1e8])


def _formula(w, lam):
    return w / (1.0 - lam + lam * w)


def test_spec_parsing_and_labels():
    assert parse_weight_spec("harmonic:0.1") == ("harmonic", 0.1)
    assert parse_weight_spec(("harmonic", 0.2)) == ("harmonic", 0.2)
    assert parse_weight_spec("harmonic:1") == ("harmonic", 1.0)
    assert parse_weight_spec("harmonic:0") == ("none", math.inf)  # lam = 0: raw IS
    assert weight_spec_label("harmonic:0.10") == "harmonic:0.1" and weight_spec_label("harmonic:0.05") == "harmonic:0.05"
    for bad in ("harmonic", "harmonic:1.5", "harmonic:-0.1", ("harmonic", 2.0)):
        with pytest.raises(ValueError):
            parse_weight_spec(bad)
    assert _estimator_weight_kwargs("harmonic:0.1") == {"harmonic_lambda": 0.1}


@pytest.mark.parametrize("lam", LAMS)
def test_matches_the_formula_everywhere(lam):
    np.testing.assert_allclose(transform_weights(W, f"harmonic:{lam}"), _formula(W, lam), rtol=1e-15)
    pi = torch.tensor(np.minimum(W, 1e3) * 0.001, dtype=torch.float64)  # pi = w * pscore with pscore 0.001
    p = torch.full_like(pi, 0.001)
    got = transform_importance_weights(pi, p, use_iw=True, iw_mode="harmonic", harmonic_lambda=lam)
    torch.testing.assert_close(got, torch.tensor(_formula(np.minimum(W, 1e3), lam), dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(got, harmonic_importance_weights(pi, p, lam, use_iw=True))
    # the paper's power-mean form, Definition 4.1 with s = -1
    w = np.array([0.3, 1.0, 2.0, 10.0, 99.0])
    np.testing.assert_allclose(transform_weights(w, f"harmonic:{lam}"), ((1 - lam) * w ** -1.0 + lam) ** -1.0, rtol=1e-13)


@pytest.mark.parametrize("lam", LAMS)
def test_limits_and_shape(lam):
    t = transform_weights(W, f"harmonic:{lam}")
    assert t[0] == 0.0 and transform_weights(np.array([1.0]), f"harmonic:{lam}")[0] == pytest.approx(1.0, abs=1e-15)
    assert (t <= 1.0 / lam + 1e-12).all() and t[-1] == pytest.approx(1.0 / lam, rel=1e-6)  # bounded by 1 / lam
    assert (np.diff(t) > 0).all()  # increasing in w
    between = (t >= np.minimum(W, 1.0) - 1e-15) & (t <= np.maximum(W, 1.0) + 1e-15)
    assert between.all()  # a mean of w and 1
    np.testing.assert_allclose(transform_weights(W, "harmonic:1e-12"), W, rtol=1e-3)  # lam -> 0: raw weights
    np.testing.assert_allclose(transform_weights(W[1:], "harmonic:1"), np.ones(len(W) - 1))  # lam = 1: constant 1
    heavy = np.array([2.0, 10.0, 99.0])
    assert (np.diff([transform_weights(heavy, f"harmonic:{x}") for x in (0.05, 0.1, 0.2)], axis=0) < 0).all()


@pytest.mark.parametrize("lam", LAMS)
def test_derivative_against_autograd(lam):
    w = torch.tensor([0.01, 0.5, 1.0, 3.0, 10.0, 100.0, 1000.0], dtype=torch.float64, requires_grad=True)
    (d,) = torch.autograd.grad(_formula(w, lam).sum(), w)
    want = (1 - lam) / (1 - lam + lam * w.detach()) ** 2
    torch.testing.assert_close(d, want, rtol=1e-12, atol=0)
    assert (d > 0).all() and float(d.max()) <= 1.0 / (1 - lam)  # positive and bounded (w = 0 end)
    # the paper's statement: grad w_lam = (1 - lam) w / (1 - lam + lam w)^2 * grad log p, at most 1 / (4 lam) * |grad log p|
    assert float((d * w.detach()).max()) <= 1.0 / (4 * lam) + 1e-12


@pytest.mark.parametrize("lam", LAMS)
def test_dr_loss_direct_gradient_is_the_literal_gradient(heavy, lam):
    spec = f"harmonic:{lam}"
    got, _ = _epoch_direction(heavy, DRPolicyLoss(use_log_trick=False, weights=spec), 384)
    want = _grad(lambda dm, w, res: dm.mean() + (_formula(w, lam) * res).mean(), heavy)
    _close(got, want)
    _, iw, _ = _literal(heavy, heavy["theta"])
    assert float(_formula(iw, lam).max()) < 1.0 / lam  # the heavy rows are capped
    assert float((iw > 1.0 / lam).double().mean()) > 0.02  # and there are rows past the cap


def test_post_hoc_estimators_apply_the_correction():
    rng = np.random.default_rng(0)
    n, k = 400, 5
    pi = rng.dirichlet(np.ones(k), n)
    action = rng.integers(0, k, n)
    pscore = rng.uniform(0.01, 0.3, n)
    reward = rng.binomial(1, 0.2, n).astype(float)
    q = rng.uniform(0, 0.4, (n, k))
    w = pi[np.arange(n), action] / pscore
    dist, qq = pi[:, :, None], q[:, :, None]
    ipw = InverseProbabilityWeighting(harmonic_lambda=0.1).estimate_policy_value(
        reward=reward, action=action, pscore=pscore, action_dist=dist)
    assert ipw == pytest.approx(np.mean(_formula(w, 0.1) * reward), rel=1e-12)
    dr = DoublyRobust(harmonic_lambda=0.1).estimate_policy_value(
        reward=reward, action=action, pscore=pscore, action_dist=dist, estimated_rewards_by_reg_model=qq)
    dm = (pi * q).sum(1)
    assert dr == pytest.approx(np.mean(dm + _formula(w, 0.1) * (reward - q[np.arange(n), action])), rel=1e-12)
    with pytest.raises(ValueError, match="harmonic_lambda"):
        DoublyRobust(harmonic_lambda=1.5)


@pytest.fixture(scope="module")
def toy_harmonic(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _finalize_summary_df, _run_condition

    tmp = tmp_path_factory.mktemp("harmonic")
    _toy_embeddings(tmp)
    kw = dict(dataset_name="toy", emb_dir=tmp, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",), policy_loss_types=("dr",),
              sampler="random", opc_gradient="direct")
    runs = {}
    for spec in ("harmonic:0.1", "shrink:100"):
        run_dir = tmp / spec.replace(":", "_")
        run_dir.mkdir()
        opc, nop, _, _, meta = _run_condition(**kw, run_dir=run_dir, train_weights=spec)
        summary = _finalize_summary_df(opc, nop, meta)
        summary.to_csv(run_dir / "summary_metrics.csv", index=False)
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        runs[spec] = run_dir
    return kw, runs


def test_end_to_end_and_serialization(toy_harmonic):
    """A toy study condition trains with harmonic weights by the direct gradient and records the transform
    and its parameter in run_meta.json and the summary (read back from disk)."""
    _, runs = toy_harmonic
    meta = json.load(open(runs["harmonic:0.1"] / "run_meta.json"))
    assert (meta["train_weights"], meta["train_weight_mode"], meta["train_weight_param"]) == ("harmonic:0.1", "harmonic", 0.1)
    assert (meta["select_weight_mode"], meta["select_weight_param"]) == ("clip", 10.0) and meta["opc_gradient"] == "direct"
    summary = pd.read_csv(runs["harmonic:0.1"] / "summary_metrics.csv")
    assert (summary["train_weights"] == "harmonic:0.1").all() and (summary["train_weight_mode"] == "harmonic").all()
    assert np.allclose(summary["train_weight_param"], 0.1)
    trials = pd.read_csv(runs["harmonic:0.1"] / "trials_long.csv")
    assert len(trials) == 2 and trials["actual_reward"].notna().all()
    other = pd.read_csv(runs["shrink:100"] / "trials_long.csv")
    pd.testing.assert_frame_equal(trials[["param_lr", "param_num_epochs", "param_batch_size"]],
                                  other[["param_lr", "param_num_epochs", "param_batch_size"]])  # a replay pair
    assert not np.allclose(trials["actual_reward"], other["actual_reward"])  # a different training objective
    none_meta = json.load(open(runs["shrink:100"] / "run_meta.json"))
    assert (none_meta["train_weight_mode"], none_meta["train_weight_param"]) == ("shrink", 100.0)


def test_harmonic_needs_the_direct_gradient(toy_harmonic, tmp_path):
    from training.run_full_study import _run_condition

    kw, _ = toy_harmonic
    with pytest.raises(ValueError, match="direct gradient"):
        _run_condition(**{**kw, "opc_gradient": "log-trick"}, run_dir=tmp_path, train_weights="harmonic:0.1")


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_accepts_and_records_harmonic(module, monkeypatch, capsys):
    import argparse

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    assert "harmonic:lambda" in " ".join(capsys.readouterr().out.split())
    real_parse, seen = argparse.ArgumentParser.parse_args, {}

    class Parsed(Exception):
        pass

    def parse(self, args=None, namespace=None):
        seen["args"] = real_parse(self, ["--train-weights", "harmonic:0.10", "--select-weights", "harmonic:0.2"], namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        main()
    assert (seen["args"].train_weights, seen["args"].select_weights) == ("harmonic:0.1", "harmonic:0.2")


def test_parallel_runner_forwards_the_transform(monkeypatch, tmp_path):
    import training.run_full_study_parallel as par

    captured = {}
    monkeypatch.setattr(par, "_run_with_memory_cap", lambda configs, **kw: captured.setdefault("configs", list(configs)) and [])
    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", "--datasets", "ml", "--seeds", "7", "--bias-configs",
                                      "medium", "--train-sizes", "5000", "--train-weights", "harmonic:0.1",
                                      "--opc-gradient", "direct", "--out-dir", str(tmp_path), "--run-tag", "t",
                                      "--no-skip-completed"])
    par.main()
    (config,) = captured["configs"]

    class Called(Exception):
        pass

    def fake_condition(**kwargs):
        captured["kwargs"] = kwargs
        raise Called

    monkeypatch.setattr(par, "_run_condition", fake_condition)
    with pytest.raises(Called):
        par._execute_run(config)
    assert captured["kwargs"]["train_weights"] == "harmonic:0.1" and captured["kwargs"]["opc_gradient"] == "direct"
