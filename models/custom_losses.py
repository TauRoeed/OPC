import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/code")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# import debugpy
import torch.nn as nn
import torch.nn.functional as F


def _align_policy_scores(scores, policy_prob):
    """(batch, n_actions) tensors; squeeze trailing singleton dims."""
    if policy_prob.dim() == 3 and policy_prob.shape[-1] == 1:
        policy_prob = policy_prob.squeeze(-1)
    if scores.dim() == 3 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    return scores, policy_prob


def policy_grad_surrogate(pi_at_action, use_log_trick=True, log_eps=1e-10):
    """Policy-gradient factor at the logged action.

    ``use_log_trick=True``: REINFORCE with ``log(pi)`` (legacy).
    ``False``: direct probability surrogate (matches KL non-log path).
    """
    pi = pi_at_action.squeeze().clamp(min=log_eps)
    if use_log_trick:
        return torch.log(pi)
    return pi


class IPWPolicyLoss(nn.Module):
    def __init__(self, log_eps=1e-10, use_log_trick=True):
        super(IPWPolicyLoss, self).__init__()
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        reinforce_grad = iw * original_policy_rewards * grad_term

        return reinforce_grad.mean()


class SNDRPolicyLoss(nn.Module):
    def __init__(self, log_eps=1e-10, use_log_trick=True):
        super(SNDRPolicyLoss, self).__init__()
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward
        reinforce_grad = r_hat * grad_term

        return reinforce_grad.mean()
    

class KLPolicyLoss(nn.Module):
    """DR-style training loss with optional KL regularizer toward logging policy.

    When ``use_log_trick`` is True (default), the KL term passes ``log(pi_e)`` and
    ``log(pi_b)`` through ``softmax`` (legacy RL-style log-prob parameterization).

    When False, KL is computed directly on the scalar logged-action probabilities.
    """

    def __init__(self, gamma=0.05, log_eps=1e-10, use_log_trick=True):
        super(KLPolicyLoss, self).__init__()
        self.gamma = gamma
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)

    def policy_kl_loss(self, logits_new, logits_old, detach_old=True, reduction='mean'):
        # logits_new: current policy logits (requires grad)
        # logits_old: baseline/previous policy logits
        p_new = F.softmax(logits_new, dim=-1)
        p_old = F.softmax(logits_old, dim=-1)

        if detach_old:
            p_old = p_old.detach()

        kl = torch.sum(p_old * (torch.log(p_old + self.log_eps) - torch.log(p_new + self.log_eps)), dim=-1)
        if reduction == 'mean':
            return kl.mean()
        elif reduction == 'sum':
            return kl.sum()
        else:
            return kl

    def policy_kl_from_probs(self, p_new, p_old, detach_old=True, reduction='mean'):
        """KL(pi_b || pi_e) on probabilities without log-space reparameterization."""
        eps = self.log_eps
        p_new = p_new.clamp(min=eps)
        if detach_old:
            p_old = p_old.detach().clamp(min=eps)
        else:
            p_old = p_old.clamp(min=eps)
        kl = p_old * (torch.log(p_old) - torch.log(p_new))
        if reduction == 'mean':
            return kl.mean()
        if reduction == 'sum':
            return kl.sum()
        return kl

    def _kl_regularizer(self, pi_e_at_position, pscore):
        if self.use_log_trick:
            return self.policy_kl_loss(
                torch.log(pi_e_at_position.clamp(min=self.log_eps)),
                torch.log(pscore.detach().clamp(min=self.log_eps)),
            )
        return self.policy_kl_from_probs(pi_e_at_position, pscore.detach())

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)

        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward

        loss = r_hat + self.gamma * self._kl_regularizer(pi_e_at_position, pscore)

        return loss.mean()
