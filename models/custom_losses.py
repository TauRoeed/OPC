import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/code")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch.nn as nn

PROPENSITY_MODES = ("logged", "uniform")


def _align_policy_scores(scores, policy_prob):
    """(batch, n_actions) tensors; squeeze trailing singleton dims."""
    if policy_prob.dim() == 3 and policy_prob.shape[-1] == 1:
        policy_prob = policy_prob.squeeze(-1)
    if scores.dim() == 3 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    return scores, policy_prob


def uses_importance_weighting(propensity_mode: str) -> bool:
    """``uniform`` = no IW (iw=1); ``logged`` = iw = pi_e / pi_b."""
    return str(propensity_mode).lower() != "uniform"


def batch_mc_kl(pi_e_at_action, pi_b_at_action, log_eps=1e-10):
    """Batch MC estimate of E[KL(pi_b||pi_e)] from logged a ~ pi_b.

    Uses mean over batch of (log pi_b - log pi_e) at logged actions only.
    No softmax across batch, no full action sum.
    """
    pi_b = pi_b_at_action.detach().clamp(min=log_eps)
    pi_e = pi_e_at_action.clamp(min=log_eps)
    return (torch.log(pi_b) - torch.log(pi_e)).mean()


def importance_weights(pi_e_at_action, pscore, use_iw: bool, log_eps=1e-10):
    if use_iw:
        return pi_e_at_action / pscore.clamp(min=log_eps)
    return torch.ones_like(pi_e_at_action)


def policy_grad_surrogate(pi_at_action, use_log_trick=True, log_eps=1e-10):
    """Policy-gradient factor at the logged action.

    ``use_log_trick=True``: REINFORCE with ``log(pi)``.
    ``False``: unit factor; grad flows through ``iw`` only.
    """
    pi = pi_at_action.squeeze().clamp(min=log_eps)
    if use_log_trick:
        return torch.log(pi)
    return torch.ones_like(pi)


def grad_importance_weights(iw, use_log_trick: bool):
    return iw.detach() if use_log_trick else iw


def sndr_r_hat(iw, rewards, q_at_action, dm_reward):
    return (iw * (rewards - q_at_action)).sum() / iw.sum() + dm_reward


class _BanditPolicyLossBase(nn.Module):
    def __init__(self, log_eps=1e-10, use_log_trick=True, propensity_mode="logged"):
        super().__init__()
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)
        self.propensity_mode = str(propensity_mode).lower()
        if self.propensity_mode not in PROPENSITY_MODES:
            raise ValueError(
                f"propensity_mode must be one of {PROPENSITY_MODES}, got {propensity_mode!r}"
            )

    def _use_iw(self) -> bool:
        return uses_importance_weighting(self.propensity_mode)

    def _prepare_iw(self, pi_e_at_position, pscore):
        iw = importance_weights(
            pi_e_at_position, pscore, self._use_iw(), self.log_eps
        )
        iw_val = iw.detach()
        iw_grad = grad_importance_weights(iw, self.use_log_trick)
        return iw_val, iw_grad

    def _logged_action_prob(self, policy_prob, actions):
        n = actions.shape[0]
        idx = torch.arange(n, device=policy_prob.device)
        return policy_prob[idx, actions].squeeze()


class IPWPolicyLoss(_BanditPolicyLossBase):
    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        _, iw_grad = self._prepare_iw(pi_e_at_position, pscore)
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        reinforce_grad = iw_grad * original_policy_rewards * grad_term
        return reinforce_grad.mean()


class SNDRPolicyLoss(_BanditPolicyLossBase):
    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        iw_val, iw_grad = self._prepare_iw(pi_e_at_position, pscore)
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        r_hat = sndr_r_hat(iw_val, original_policy_rewards, q_hat_at_position, dm_reward)
        if self.use_log_trick:
            reinforce_grad = r_hat.detach() * grad_term
        else:
            reinforce_grad = r_hat.detach() * iw_grad * grad_term

        return reinforce_grad.mean()


class KLPolicyLoss(_BanditPolicyLossBase):
    """SNDR-style PG + batch MC KL toward logging policy (logged actions only)."""

    def __init__(self, gamma=0.05, log_eps=1e-10, use_log_trick=True, propensity_mode="logged"):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
        self.gamma = gamma

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        iw_val, iw_grad = self._prepare_iw(pi_e_at_position, pscore)
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        r_hat = sndr_r_hat(iw_val, original_policy_rewards, q_hat_at_position, dm_reward)
        if self.use_log_trick:
            pg = r_hat.detach() * grad_term
        else:
            pg = r_hat.detach() * iw_grad * grad_term

        kl = batch_mc_kl(pi_e_at_position, pscore, self.log_eps)
        return (pg + self.gamma * kl).mean()
