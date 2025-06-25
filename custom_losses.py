import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import sys
sys.path.append("/code")

import torch
# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



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


class BPRLoss(nn.Module):
    def __init__(self, log_eps=1e-10):
        super(BPRLoss, self).__init__()

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        num_items = policy_prob.shape[1]
        batch_size = scores.size(0)

        # Filter to only positive-reward samples (reward == 1)
        mask = original_policy_rewards > 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=scores.device)

        pos_idx = torch.arange(batch_size, device=mask.device)[mask]
        pos_actions = original_policy_actions[mask]
        pos_scores = scores[pos_idx, pos_actions]
        pos_pscore = pscore[mask]

        # Sample negative actions not equal to the positive ones
        neg_actions = torch.randint(0, num_items, size=(pos_idx.size(0),), device=scores.device)
        conflict = neg_actions == pos_actions
        
        while conflict.any():
            neg_actions[conflict] = torch.randint(0, num_items, size=(conflict.sum(),), device=scores.device)
            conflict = neg_actions == pos_actions

        neg_scores = scores[pos_idx, neg_actions]

        # Compute pairwise BPR loss
        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)

        # Importance weighting using inverse propensity score
        loss = (bpr / (pos_pscore + 1e-6)).mean()

        return loss