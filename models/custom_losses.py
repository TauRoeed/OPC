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


class IPWPolicyLoss(nn.Module):
    def __init__(self, log_eps=1e-10):
        super(IPWPolicyLoss, self).__init__()
        self.log_eps = log_eps

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        log_pi = torch.log(pi_e_at_position).squeeze()
        
        # reinforce trick step
        reinforce_grad = iw * original_policy_rewards * log_pi
        
        return reinforce_grad.mean()
    

class SNDRPolicyLoss(nn.Module):
    def __init__(self, log_eps=1e-10):
        super(SNDRPolicyLoss, self).__init__()
        self.log_eps = log_eps

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        log_pi = torch.log(pi_e_at_position).squeeze()
        
        # reinforce trick step
        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward
        reinforce_grad = r_hat * log_pi

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

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)

        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward

        loss = r_hat + self.gamma * self._kl_regularizer(pi_e_at_position, pscore)

        return loss.mean()
