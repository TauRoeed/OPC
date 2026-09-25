"""Importance-weight transforms (utils/importance_weights.py): the spec, the numpy / torch
transforms, and their use in the training losses, the selection score, the post-hoc estimators
and the runners."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from models.custom_losses import (
    CRMPolicyLoss,
    IPWPolicyLoss,
    KLPolicyLoss,
    SNDRPolicyLoss,
    batch_mc_kl,
    transform_importance_weights,
)
from models.estimators import DoublyRobust, InverseProbabilityWeighting, SelfNormalizedDoublyRobust, SelfNormalizedInverseProbabilityWeighting
from utils.importance_weights import effective_sample_size, parse_weight_spec, transform_weights, weight_spec_label
from utils.simulation_utils import _estimator_weight_kwargs


# ------------------------------------------------------------------ the spec and the transforms
def test_parse_and_label():
    assert parse_weight_spec("none") == ("none", math.inf) == parse_weight_spec("raw")
    assert parse_weight_spec("clip:100") == ("clip", 100.0)
    assert parse_weight_spec("shrink:1e4") == ("shrink", 1e4)
    assert parse_weight_spec(("clip", 3)) == ("clip", 3.0)
    assert parse_weight_spec("clip:inf") == ("none", math.inf) == parse_weight_spec(("shrink", math.inf))
    assert weight_spec_label("CLIP:100.0") == "clip:100" and weight_spec_label("shrink:10000") == "shrink:10000"
    assert weight_spec_label("none") == "none"
    for bad in ("clip", "clip:0", "clip:-1", "shrink:0", "cap:3", "clip:x"):
        with pytest.raises(ValueError):
            parse_weight_spec(bad)


def test_numpy_transforms():
    w = np.array([0.0, 0.5, 1.0, 10.0, 100.0, 1e4])
    np.testing.assert_array_equal(transform_weights(w, "none"), w)
    np.testing.assert_array_equal(transform_weights(w, "clip:10"), np.minimum(w, 10.0))
    lam = 100.0
    s = transform_weights(w, f"shrink:{lam:g}")
    np.testing.assert_allclose(s, lam * w / (w**2 + lam))
    assert s[3] == pytest.approx(math.sqrt(lam) / 2)  # the maximum, at w = sqrt(lam)
    assert s.max() == pytest.approx(math.sqrt(lam) / 2) and s[-1] < s[3]  # large weights fall back toward 0
    assert transform_weights(np.array([0.01]), "shrink:1e6")[0] == pytest.approx(0.01)  # small weights barely move
    assert effective_sample_size(np.ones(7)) == pytest.approx(7.0)
    assert effective_sample_size(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("spec", ["none", "clip:3", "shrink:5"])
def test_torch_transform_matches_numpy(spec):
    pi = torch.rand(50)
    ps = torch.rand(50).clamp(min=0.01)
    mode, param = parse_weight_spec(spec)
    got = transform_importance_weights(pi, ps, use_iw=True, iw_mode=mode, clip_m=param, shrink_lambda=param)
    np.testing.assert_allclose(got.numpy(), transform_weights((pi / ps).numpy(), spec), rtol=1e-6)
    ones = transform_importance_weights(pi, ps, use_iw=False, iw_mode=mode, clip_m=param, shrink_lambda=param)
    np.testing.assert_array_equal(ones.numpy(), np.ones(50))  # no-propensity: no weights


# ------------------------------------------------------------------ training losses
def _batch(n=64, n_actions=6, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n, n_actions, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    scores = torch.rand(n, n_actions)
    actions = torch.randint(0, n_actions, (n,))
    rewards = (torch.rand(n) < 0.3).float()
    pscore = torch.rand(n).clamp(min=0.02, max=0.5)
    return logits, policy, scores, actions, rewards, pscore


def _equivalent_pscore(policy, actions, pscore, spec):
    """Propensities whose raw weights equal the transformed ones (same loss value)."""
    pi_a = policy.detach()[torch.arange(len(actions)), actions]
    w = transform_weights((pi_a / pscore).numpy(), spec)
    return pi_a / torch.as_tensor(w, dtype=pi_a.dtype)


@pytest.mark.parametrize("spec", ["clip:2", "shrink:4"])
@pytest.mark.parametrize("log_trick", [True, False])
@pytest.mark.parametrize("cls", [SNDRPolicyLoss, IPWPolicyLoss])
def test_losses_use_the_transformed_weights(cls, log_trick, spec):
    _, policy, scores, actions, rewards, pscore = _batch()
    pi_a = policy.detach()[torch.arange(len(actions)), actions]
    assert ((pi_a / pscore) > 2).any()  # some rows are affected
    got = cls(use_log_trick=log_trick, weights=spec)(pscore, scores, policy, rewards, actions)
    want = cls(use_log_trick=log_trick, weights="none")(_equivalent_pscore(policy, actions, pscore, spec), scores, policy, rewards, actions)
    got, want = got.detach().item(), want.detach().item()
    assert got == pytest.approx(want, rel=1e-5)
    raw = cls(use_log_trick=log_trick)(pscore, scores, policy, rewards, actions)
    assert raw.detach().item() != pytest.approx(got, rel=1e-6)
    assert cls(use_log_trick=log_trick, weights="none")(pscore, scores, policy, rewards, actions).item() == raw.item()


def test_kl_loss_weights_its_sndr_part_only():
    _, policy, scores, actions, rewards, pscore = _batch()
    gamma = 0.3
    got = KLPolicyLoss(gamma=gamma, weights="clip:2")(pscore, scores, policy, rewards, actions)
    sndr = SNDRPolicyLoss(weights="clip:2")(pscore, scores, policy, rewards, actions)
    pi_a = policy[torch.arange(len(actions)), actions]
    assert got.detach().item() == pytest.approx((sndr + gamma * batch_mc_kl(pi_a, pscore, 1e-10)).detach().item(), rel=1e-6)


def test_clipped_rows_carry_no_weight_gradient():
    logits, policy, scores, actions, rewards, pscore = _batch()
    pi_a = policy.detach()[torch.arange(len(actions)), actions]
    over = (pi_a / pscore) > 2
    scores = torch.zeros_like(scores)  # no DM term: only the weighted residuals remain
    loss = IPWPolicyLoss(use_log_trick=False, weights="clip:2")(pscore, scores, policy, rewards, actions)
    loss.backward()
    rows_grad = logits.grad.abs().sum(dim=1)
    assert torch.all(rows_grad[over] == 0) and torch.any(rows_grad[~over] > 0)


def test_crm_losses_keep_their_own_clip():
    with pytest.raises(TypeError):
        CRMPolicyLoss(weights="clip:2")  # crm / kl_crm take clip_m / iw_mode, searched by Optuna


# ------------------------------------------------------------------ post-hoc estimators
def _ope_inputs(n=300, n_actions=8, seed=1):
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(n_actions) * 0.3, size=n)[:, :, None]
    a = rng.integers(0, n_actions, n)
    ps = rng.uniform(0.02, 0.5, n)
    r = (rng.random(n) < 0.2).astype(float)
    q = rng.uniform(0, 0.4, size=(n, n_actions, 1))
    return r, a, ps, pi, q


@pytest.mark.parametrize("spec", ["none", "clip:3", "shrink:9"])
def test_estimators_apply_the_transform(spec):
    r, a, ps, pi, q = _ope_inputs()
    n = len(a)
    w = transform_weights(pi[np.arange(n), a, 0] / ps, spec)
    dm = (q[:, :, 0] * pi[:, :, 0]).sum(axis=1)
    resid = r - q[np.arange(n), a, 0]
    kw = _estimator_weight_kwargs(spec)
    assert InverseProbabilityWeighting(**kw).estimate_policy_value(r, a, pi, pscore=ps) == pytest.approx(np.mean(r * w))
    assert SelfNormalizedInverseProbabilityWeighting(**kw).estimate_policy_value(r, a, pi, pscore=ps) == pytest.approx(np.mean(r * w / w.mean()))
    assert DoublyRobust(**kw).estimate_policy_value(r, a, pi, pscore=ps, estimated_rewards_by_reg_model=q) == pytest.approx(np.mean(dm + w * resid))
    assert SelfNormalizedDoublyRobust(**kw).estimate_policy_value(r, a, pi, pscore=ps, estimated_rewards_by_reg_model=q) == pytest.approx(np.mean(dm + w * resid / w.mean()))
    assert _estimator_weight_kwargs(None) == {} == _estimator_weight_kwargs("none")


# ------------------------------------------------------------------ selection score and runners
def test_runner_passes_the_weights_to_losses_selection_and_meta(tmp_path, monkeypatch):
    import training.trainer_trials as tt
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    _toy_embeddings(tmp_path)
    seen_train, seen_select = [], []
    loss_from_name, split_score = tt._policy_loss_from_name, tt._split_dr_vec_and_ess

    def loss_spy(*a, **k):
        seen_train.append(k.get("train_weights"))
        return loss_from_name(*a, **k)

    def score_spy(*a, **k):
        seen_select.append(k.get("weights"))
        return split_score(*a, **k)

    monkeypatch.setattr(tt, "_policy_loss_from_name", loss_spy)
    monkeypatch.setattr(tt, "_split_dr_vec_and_ess", score_spy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    *_, meta = _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="low", ctr=0.05, seed=0, train_sizes=[1000],
                              n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                              policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                              shared_regression_size=2000, train_weights="shrink:50", select_weights="clip:7")
    assert (meta["train_weights"], meta["select_weights"], meta["dr_score_clip_m"]) == ("shrink:50", "clip:7", 7.0)
    assert ("shrink", 50.0) in seen_train  # the OPC losses (no-prop builds naive losses, weights unused)
    assert set(seen_select) == {("clip", 7.0), ("none", math.inf)}  # OPC selection; no-prop has no weights
    trials = pd.read_csv(run_dir / "trials_long.csv")
    opc = trials[trials["method"] == "opc"]
    assert (opc["param_dr_score_clip_m"] == 7.0).all() and opc["ess_raw"].notna().all()
    assert (opc["ess"] > 0).all() and (opc["ess_raw"] > 0).all()


def test_eval_policy_uses_its_weights():
    from utils.simulation_utils import eval_policy

    r, a, ps, pi, q = _ope_inputs()
    n = len(a)

    class Q:  # the reward model eval_policy scores the rows with
        def predict(self, x):
            return q

    data = {"x": np.zeros((n, 2)), "a": a, "r": r, "pscore": ps, "x_idx": np.arange(n)}
    out = eval_policy(Q(), data, None, pi[:, :, 0], weights="clip:3")
    pi_f = np.maximum(pi[:, :, 0], 1e-15)
    pi_f = (pi_f / pi_f.sum(axis=1, keepdims=True)).astype(np.float32).astype(np.float64)
    w = np.minimum(pi_f[np.arange(n), a] / ps, 3.0)
    dm = (q[:, :, 0] * pi_f).sum(axis=1)
    resid = r - q[np.arange(n), a, 0]
    np.testing.assert_allclose(out, [dm.mean(), np.mean(dm + w * resid), np.mean(r * w / w.mean()),
                                     np.mean(dm + w * resid / w.mean())], rtol=1e-5)
    raw = eval_policy(Q(), data, None, pi[:, :, 0])
    assert not np.allclose(raw[1:], out[1:])  # the weighted estimates change; DM does not
    assert raw[0] == pytest.approx(out[0])


def test_post_hoc_estimates_use_the_selection_weights(tmp_path, monkeypatch):
    import training.trainer_trials as tt
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    _toy_embeddings(tmp_path)
    seen = []
    real = tt.eval_policy

    def spy(*a, **k):
        seen.append(k.get("weights"))
        return real(*a, **k)

    monkeypatch.setattr(tt, "eval_policy", spy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="low", ctr=0.05, seed=0, train_sizes=[1000], n_trials=1,
                   batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
                   policy_reward_mc_sim=8, run_dir=run_dir, slim=False, shared_regression_size=2000, select_weights="shrink:40")
    assert len(seen) == 4 and set(seen) == {("shrink", 40.0)}  # baseline + selected policy, for both methods


def test_logged_selection_variants(tmp_path):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import _run_condition

    _toy_embeddings(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    specs = ["clip:7", "none", "shrink:50"]
    *_, meta = _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="low", ctr=0.05, seed=0, train_sizes=[1000],
                              n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                              policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                              shared_regression_size=2000, select_weights="clip:7", log_select_weights=specs)
    assert meta["log_select_weights"] == specs
    trials = pd.read_csv(run_dir / "trials_long.csv")
    opc, nop = trials[trials["method"] == "opc"], trials[trials["method"] == "no_propensity"]
    for spec in specs:
        assert opc[f"sel_r_hat[{spec}]"].notna().all() and nop[f"sel_r_hat[{spec}]"].isna().all()
    # the run's own selection spec reproduces its selection score and objective (ci_low)
    np.testing.assert_allclose(opc["sel_r_hat[clip:7]"], opc["r_hat"], rtol=1e-6)
    np.testing.assert_allclose(opc["sel_ci_low[clip:7]"], opc["value"], rtol=1e-6)
    assert not np.allclose(opc["sel_r_hat[none]"], opc["r_hat"]) or not np.allclose(opc["sel_r_hat[shrink:50]"], opc["r_hat"])
