"""Sanity tests for bandit policy losses (IW, batch MC KL, propensity modes)."""

import numpy as np
import torch

from models.custom_losses import (
    CRMPolicyLoss,
    IPWPolicyLoss,
    KLPolicyLoss,
    SNDRPolicyLoss,
    batch_mc_kl,
    clipped_importance_weights,
    crm_per_sample_u,
    crm_surrogate,
    crm_variance_penalty,
    dr_correction,
    dr_sndr_loss,
    dr_sndr_surrogate,
    importance_weights,
    sndr_r_hat,
    uses_importance_weighting,
)
from training.trainer_trials import _build_cf_dataset, _resolve_logged_pscore


def _batch(n=8, n_actions=5, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(n, n_actions, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    scores = torch.rand(n, n_actions)
    actions = torch.randint(0, n_actions, (n,))
    rewards = torch.rand(n)
    pscore = torch.rand(n).clamp(min=0.05, max=0.9)
    return logits, policy, scores, actions, rewards, pscore


def test_importance_weights_uniform_is_ones():
    pi_e = torch.tensor([0.2, 0.4, 0.1])
    pscore = torch.tensor([0.25, 0.5, 0.05])
    iw = importance_weights(pi_e, pscore, use_iw=False)
    assert torch.allclose(iw, torch.ones(3))
    assert not uses_importance_weighting("uniform")
    assert uses_importance_weighting("logged")


def test_importance_weights_logged():
    pi_e = torch.tensor([0.2, 0.4])
    pscore = torch.tensor([0.25, 0.5])
    iw = importance_weights(pi_e, pscore, use_iw=True)
    assert torch.allclose(iw, pi_e / pscore)


def test_batch_mc_kl_formula():
    pi_e = torch.tensor([0.2, 0.3], requires_grad=True)
    pi_b = torch.tensor([0.25, 0.35])
    kl = batch_mc_kl(pi_e, pi_b)
    expected = (torch.log(pi_b) - torch.log(pi_e)).mean()
    assert torch.allclose(kl, expected)


def test_dataset_keeps_true_pscore_for_uniform_mode():
    train_data = {
        "x_idx": np.array([0, 1, 2]),
        "a": np.array([0, 1, 0]),
        "r": np.array([1.0, 0.0, 1.0]),
        "pscore": np.array([0.2, 0.5, 0.1], dtype=np.float32),
    }
    ds = _build_cf_dataset(train_data, original_policy_prob=None, propensity_mode="uniform")
    assert np.allclose(ds.pscore, train_data["pscore"])
    resolved = _resolve_logged_pscore(train_data, None, mode="uniform")
    assert np.allclose(resolved, train_data["pscore"])


def test_sndr_r_hat_uniform_is_per_row_dm_plus_residual():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.5])
    q = torch.tensor([0.2, 0.3, 0.8, 0.4])
    dm = torch.tensor([0.1, 0.2, 0.3, 0.4])
    iw = torch.ones(4)
    r_hat = sndr_r_hat(iw, rewards, q, dm)
    expected = dm + (rewards - q)
    assert torch.allclose(r_hat, expected)


def test_sndr_r_hat_logged_normalizes_by_mean_iw():
    rewards = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.2, 0.5])
    dm = torch.tensor([0.1, 0.2])
    iw = torch.tensor([2.0, 4.0])
    r_hat = sndr_r_hat(iw, rewards, q, dm)
    expected = dm + iw * (rewards - q) / iw.mean()
    assert torch.allclose(r_hat, expected)


def test_sndr_log_trick_matches_reference_structure():
    _, policy, scores, actions, rewards, pscore = _batch()
    idx = torch.arange(len(actions))
    pi_det = policy.detach()
    log_p = torch.log(policy)
    q = scores[idx, actions]
    iw = torch.ones_like(q)
    correction = (rewards - q) * log_p[idx, actions]
    dm = (scores * pi_det * log_p).sum(dim=1)
    expected = -(correction + dm).mean()

    loss = KLPolicyLoss(gamma=0.0, propensity_mode="uniform", use_log_trick=True)(
        pscore, scores, policy, rewards, actions
    )
    assert torch.allclose(loss, expected, atol=1e-5)


def test_sndr_direct_uses_attached_pi():
    logits = torch.randn(3, 4, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    scores = torch.rand(3, 4)
    actions = torch.randint(0, 4, (3,))
    rewards = torch.rand(3)
    pscore = torch.ones(3) * 0.5

    loss = dr_sndr_loss(
        scores, policy, actions, rewards, pscore, use_iw=True, use_log_trick=False
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.norm() > 0


def test_sndr_log_trick_detaches_pi_in_weights():
    logits = torch.randn(3, 4, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    scores = torch.rand(3, 4)
    actions = torch.randint(0, 4, (3,))
    rewards = torch.rand(3)
    pscore = torch.ones(3) * 0.5

    sur = dr_sndr_surrogate(
        scores, policy, actions, rewards, pscore, use_iw=True, use_log_trick=True
    )
    idx = torch.arange(3)
    pi_det = policy.detach()
    iw = pi_det[idx, actions] / pscore
    corr = dr_correction(iw, rewards, scores[idx, actions]) * torch.log(policy[idx, actions])
    dm = (scores * pi_det * torch.log(policy)).sum(dim=1)
    assert torch.allclose(sur, corr + dm, atol=1e-5)


def test_dr_dm_has_pathwise_grad_when_correction_constant():
    logits = torch.randn(4, 3, requires_grad=True)
    policy = torch.softmax(logits, dim=-1)
    scores = torch.rand(4, 3)
    actions = torch.randint(0, 3, (4,))
    rewards = torch.zeros(4)
    pscore = torch.ones(4) * 0.5
    sndr = SNDRPolicyLoss(use_log_trick=False, propensity_mode="uniform")
    loss = sndr(pscore, scores, policy, rewards, actions)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.norm() > 0


def test_grad_flow_log_trick_vs_direct():
    logits_log, policy, scores, actions, rewards, pscore = _batch(seed=1)
    kl_log = KLPolicyLoss(gamma=0.1, propensity_mode="logged", use_log_trick=True)
    loss_log = kl_log(pscore, scores, policy, rewards, actions)
    loss_log.backward()
    assert logits_log.grad is not None
    assert logits_log.grad.norm() > 0

    logits_dir, policy2, scores2, actions2, rewards2, pscore2 = _batch(seed=1)
    policy2 = torch.softmax(logits_dir, dim=-1)
    kl_dir = KLPolicyLoss(gamma=0.1, propensity_mode="logged", use_log_trick=False)
    loss_dir = kl_dir(pscore2, scores2, policy2, rewards2, actions2)
    loss_dir.backward()
    assert logits_dir.grad is not None
    assert logits_dir.grad.norm() > 0


def test_ipw_no_log_logged_has_grad_through_iw():
    logits, policy, scores, actions, rewards, pscore = _batch(seed=2)
    ipw = IPWPolicyLoss(use_log_trick=False, propensity_mode="logged")
    loss = ipw(pscore, scores, policy, rewards, actions)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.norm() > 0


def test_ipw_no_prop_no_log_is_constant_wrt_policy():
    _, policy, scores, actions, rewards, pscore = _batch(seed=3)
    ipw = IPWPolicyLoss(use_log_trick=False, propensity_mode="uniform")
    loss = ipw(pscore, scores, policy, rewards, actions)
    assert torch.allclose(loss, -rewards.mean())
    assert not loss.requires_grad


def test_clipped_importance_weights_caps_at_m():
    pi_e = torch.tensor([0.5, 0.8, 0.2])
    pscore = torch.tensor([0.1, 0.2, 0.4])
    iw = clipped_importance_weights(pi_e, pscore, clip_m=2.0, use_iw=True)
    raw = pi_e / pscore
    expected = torch.clamp(raw, max=2.0)
    assert torch.allclose(iw, expected)


def test_crm_variance_penalty_formula():
    u = torch.tensor([-0.5, -0.2, -0.8, -0.1])
    lam = 2.0
    pen = crm_variance_penalty(u, lam)
    n = 4
    var = u.var(unbiased=True)
    expected = lam * torch.sqrt(var / n)
    assert torch.allclose(pen, expected, atol=1e-6)


def test_crm_lambda_zero_matches_clipped_ipw_surrogate():
    _, policy, scores, actions, rewards, pscore = _batch(seed=5)
    pi = policy[torch.arange(len(actions)), actions]
    iw = clipped_importance_weights(pi, pscore, clip_m=5.0, use_iw=True)
    expected = -(rewards * iw.detach() * torch.log(pi)).mean()
    loss = CRMPolicyLoss(
        clip_m=5.0, crm_lambda=0.0, propensity_mode="logged", use_log_trick=True
    )(pscore, scores, policy, rewards, actions)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_crm_direct_has_grad():
    logits, policy, scores, actions, rewards, pscore = _batch(seed=6)
    loss = CRMPolicyLoss(
        clip_m=10.0, crm_lambda=0.5, propensity_mode="logged", use_log_trick=False
    )(pscore, scores, policy, rewards, actions)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.norm() > 0


def test_sndr_and_ipw_all_modes_run():
    no_grad_cases = {
        (IPWPolicyLoss, "uniform", False),
        (CRMPolicyLoss, "uniform", False),
    }
    for loss_cls in (IPWPolicyLoss, SNDRPolicyLoss, KLPolicyLoss, CRMPolicyLoss):
        for mode in ("logged", "uniform"):
            for log_trick in (True, False):
                logits, policy, scores, actions, rewards, pscore = _batch(seed=4)
                kwargs = dict(use_log_trick=log_trick, propensity_mode=mode)
                if loss_cls is KLPolicyLoss:
                    loss_fn = loss_cls(gamma=0.05, **kwargs)
                elif loss_cls is CRMPolicyLoss:
                    loss_fn = loss_cls(clip_m=10.0, crm_lambda=0.1, **kwargs)
                else:
                    loss_fn = loss_cls(**kwargs)
                loss = loss_fn(pscore, scores, policy, rewards, actions)
                assert torch.isfinite(loss)
                if (loss_cls, mode, log_trick) in no_grad_cases:
                    continue
                loss.backward()
                assert logits.grad is not None
                assert logits.grad.norm() > 0
