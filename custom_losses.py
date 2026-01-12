import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import sys
sys.path.append("/code")

import torch
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# import debugpy
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import check_random_state

# implementing OPE of the IPWLearner using synthetic bandit data


from scipy.special import softmax


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
    def __init__(self, gamma=0.05, log_eps=1e-10):
        super(KLPolicyLoss, self).__init__()
        self.gamma = gamma
        self.log_eps = log_eps

    def policy_kl_loss(self, logits_new, logits_old, detach_old=True, reduction='mean'):
        # logits_new: current policy logits (requires grad)
        # logits_old: baseline/previous policy logits
        p_new = F.softmax(logits_new, dim=-1)
        p_old = F.softmax(logits_old, dim=-1)

        if detach_old:
            p_old = p_old.detach()

        kl = torch.sum(p_old * (torch.log(p_old + 1e-8) - p_new), dim=-1)
        if reduction == 'mean':
            return kl.mean()
        elif reduction == 'sum':
            return kl.sum()
        else:
            return kl

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]

        pi_e_at_position = policy_prob[torch.arange(n), original_policy_actions].squeeze()
        iw = pi_e_at_position / pscore
        iw = iw.detach()
        q_hat_at_position = scores[torch.arange(n), original_policy_actions].squeeze()
        dm_reward = (scores * policy_prob.detach()).sum(dim=1)
        
        # reinforce trick step
        r_hat = ((iw * (original_policy_rewards - q_hat_at_position)) / iw.sum()) + dm_reward

        loss = r_hat + self.gamma * self.policy_kl_loss(torch.log(pi_e_at_position), torch.log(pscore.detach()))

        return loss.mean()
