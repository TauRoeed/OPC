import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from sklearn.utils import check_random_state

from sklearn.utils import check_random_state
import matplotlib.pyplot as plt

from scipy.special import softmax
from abc import ABCMeta

# import open bandit pipeline (obp)
from obp.ope import (
    SelfNormalizedInverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR
)

class CustomCFDataset(Dataset):
    def __init__(self, user_idx, action_idx, rewards, original_prob):
        """
        Args:
            np_arrays (list of np.ndarray): List of numpy arrays
        """
        self.user_idx = user_idx
        self.action_idx = action_idx
        self.rewards = rewards
        self.original_prob = original_prob

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, sample_idx):   
        # Convert list to tensor
        user = torch.tensor(self.user_idx[sample_idx].squeeze())
        action =  torch.tensor(self.action_idx[sample_idx].squeeze())
        reward = torch.tensor(self.rewards[sample_idx].squeeze(), dtype=torch.double)
        action_dist = torch.tensor(self.original_prob[user].squeeze())
                    
        return user, action, reward, action_dist
    

class NeighborhoodModel(metaclass=ABCMeta):
    def __init__(self, context, actions, action_emb, context_emb, rewards, num_neighbors=5, gamma=0.5):
        self.gamma = 0.5
        self.num_neighbors = num_neighbors
        self.fit(action_emb, context_emb, actions, context, rewards)

    def fit(self, action_emb, context_emb, actions, context, rewards):
        self.update(action_emb, context_emb, calc=False)
        self.actions = np.int32(actions)
        self.context = np.int32(context)
        self.reward = rewards
        self.calculate_scores()

    def update(self, action_emb, context_emb, calc=True):
        action_emb = np.divide(action_emb.T, np.linalg.norm(action_emb, axis=1))
        self.action_similarity = (action_emb.T @ action_emb + 1) / 2
        
        context_emb = np.divide(context_emb.T, np.linalg.norm(context_emb, axis=1))
        self.context_similarity = (context_emb.T @ context_emb + 1) / 2
        if calc:
            self.calculate_scores()

    def add_data(self, context, actions, rewards):
        self.actions = np.concatenate(self.actions, actions)
        self.context = np.concatenate(self.context, context)
        self.reward = np.concatenate(self.reward, rewards)
        self.calculate_scores()
    
    def calculate_scores(self):
        context = np.arange(self.context_similarity.shape[0])
        self.scores = self.context_convolve(context)

    def convolve(self, test_actions, test_context):
        cosine_context = self.context_similarity[np.int32(test_context)][:, self.context]
        cosine_actions = self.action_similarity[np.int32(test_actions)][:, self.actions]
        
        tot_cosine = self.gamma * cosine_actions + (1 - self.gamma) * cosine_context
        top_n_tot = np.argsort(tot_cosine, axis=1)[:, -self.num_neighbors:]
        similarity = tot_cosine[np.arange(tot_cosine.shape[0])[:, None], top_n_tot]

        eta = self.reward[top_n_tot]
        eta = eta * similarity

        return eta.sum(axis=1) / (similarity.sum(axis=1))

    def context_convolve(self, test_context):
        all_context = test_context.reshape(-1, 1) @ np.ones((1, self.action_similarity.shape[0]))
        all_context = all_context.flatten()

        all_actions = np.arange(self.action_similarity.shape[0]).reshape(-1, 1) @ np.ones((1, test_context.shape[0]))
        all_actions = all_actions.T.flatten()

        eta_all = self.convolve(all_actions, all_context)
        eta_all = eta_all.reshape(test_context.shape[0], self.action_similarity.shape[0], 1)
        return eta_all
    
    def predict(self, test_context):       
        return self.scores[np.int32(test_context)]


class CFModel(nn.Module):
    def __init__(self, num_users, num_actions, embedding_dim, 
                 initial_user_embeddings=None, initial_actions_embeddings=None):

        super(CFModel, self).__init__()
        self.actions = torch.arange(num_actions)
        self.users = torch.arange(num_users)

        
        # Initialize user and actions embeddings
        if initial_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        else:
            # If initial embeddings are provided, set them as the embeddings
            self.user_embeddings = nn.Embedding.from_pretrained(initial_user_embeddings, freeze=False)
        
        if initial_actions_embeddings is None:
            self.actions_embeddings = nn.Embedding(num_actions, embedding_dim)
        else:
            # If initial embeddings are provided, set them as the embeddings
            self.actions_embeddings = nn.Embedding.from_pretrained(initial_actions_embeddings, freeze=False)

    def get_params(self):
        return self.user_embeddings(self.users), self.actions_embeddings(self.actions)
        
    def forward(self, user_ids):
        # Get embeddings for users and actions
        user_embedding = self.user_embeddings(user_ids)
        actions_embedding = self.actions_embeddings
        
        # Calculate dot product between user and actions embeddings
        scores = user_embedding @ actions_embedding(self.actions).T
        
        # Apply softmax to get the predicted probability distribution
        return F.softmax(scores, dim=1).unsqueeze(-1)
    
    def to(self, device):
        # Move the module itself
        super().to(device)
        self.actions = self.actions.to(device)
        self.users = self.users.to(device)
        return self


class BPRModel(nn.Module):
    def __init__(self, num_users, num_actions, embedding_dim, 
                 initial_user_embeddings=None, initial_actions_embeddings=None):
        super(BPRModel, self).__init__()

        self.actions = torch.arange(num_actions)
        self.users = torch.arange(num_users)
        
        # Initialize user and actions embeddings
        if initial_user_embeddings is None:
            self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        else:
            # If initial embeddings are provided, set them as the embeddings
            self.user_embeddings = nn.Embedding.from_pretrained(initial_user_embeddings, freeze=False)
        
        if initial_actions_embeddings is None:
            self.actions_embeddings = nn.Embedding(num_actions, embedding_dim)
        else:
            # If initial embeddings are provided, set them as the embeddings
            self.actions_embeddings = nn.Embedding.from_pretrained(initial_actions_embeddings, freeze=False)


    def forward(self, user_ids, pos_action_ids, neg_action_ids):
        user_embeds = self.user_embeddings(user_ids)
        pos_action_embeds = self.actions_embeddings(pos_action_ids)
        neg_action_embeds = self.actions_embeddings(neg_action_ids)

        # Compute dot product between user and action embeddings
        pos_scores = (user_embeds * pos_action_embeds).sum(dim=1)
        neg_scores = (user_embeds * neg_action_embeds).sum(dim=1)

        return pos_scores, neg_scores


def generate_dataset(params):
    random_ = check_random_state(12345)
    emb_a = random_.normal(size=(params["n_actions"], params["emb_dim"]))
    noise_a = random_.normal(size=(params["emb_dim"]))
    our_a = (1-params["eps"]) * emb_a + params["eps"] * noise_a

    original_a = our_a.copy()

    emb_x = random_.normal(size=(params["n_users"], params["emb_dim"]))
    noise_x = random_.normal(size=(params["emb_dim"])) 
    our_x = (1-params["eps"]) * emb_x + params["eps"] * noise_x
    original_x = our_x.copy()

    score = emb_x @ emb_a.T
    # score = random_.normal(score, scale=params["sigma"])
    q_x_a = (1 / (5.0 + np.exp(-score)))

    return dict(
                emb_a=emb_a,
                our_a=our_a,
                original_a=original_a,
                emb_x=emb_x,
                our_x=our_x,
                original_x=original_x,
                q_x_a=q_x_a,
                n_actions=params["n_actions"],
                n_users=params["n_users"],
                emb_dim=params["emb_dim"]
                )


def create_simluation_data_from_pi(pi: np.ndarray, q_x_a: np.ndarray, n_users: np.int, n_actions: np.int, random_state: int = 12345):
    random_ = check_random_state(random_state)
    simulation_data = {'actions':np.zeros((n_actions, n_users), dtype=np.int32), 
                       'users': np.zeros((n_actions, n_users), dtype=np.int32), 
                       'reward':np.zeros((n_actions, n_users)),
                       'pscore':np.zeros((n_actions, n_users))}
    
    reward = q_x_a  > random_.random(size=q_x_a.shape)
    simulation_data['pi_0'] = pi
    actions = []
    for i in range(n_users):
        user_actions = random_.choice(np.arange(n_actions), size=n_actions, p=pi[i], replace=True)
        actions.append(np.array(user_actions))

    actions = np.vstack(actions)
    for i in range(n_actions):
        simulation_data['actions'][i] = actions[:, i]
        simulation_data['users'][i] = np.arange(n_users)
        qq = q_x_a[np.arange(n_users), simulation_data['actions'][i]]
        simulation_data['reward'][i] = np.squeeze(qq > random_.random(size=qq.shape))
        simulation_data['pscore'][i] = np.squeeze(pi[np.arange(n_users), simulation_data['actions'][i]])
    
    simulation_data['q_x_a'] = reward
    return simulation_data


def get_train_data(n_actions, train_size, sim_data, idx, emb_x):
   return dict(
                num_data=train_size,
                num_actions=n_actions,
                x=emb_x[sim_data['users'][idx].flatten()],
                a=sim_data['actions'][idx].flatten(),
                r=sim_data['reward'][idx].flatten(),
                x_idx=sim_data['users'][idx].flatten(),
                pi_0=sim_data['pi_0'],
                pscore=sim_data['pscore'][idx].flatten(),
                q_x_a=sim_data['q_x_a']
                )


def eval_policy(model, test_data, original_policy_prob, policy):
    dr = DR()
    dm = DM()
    ipw = IPW()
    sndr = SNDR()

    scores = model.predict(test_data['x_idx'])

    policy = policy[test_data['x_idx']]
    
    actions = np.squeeze(np.argmax(policy, axis=1))
    res = []
    # reward = test_data['q_x_a'][test_data['x_idx'], actions]
    # res.append(reward.mean())

    pscore = original_policy_prob[test_data['x_idx'], actions].squeeze()
    
    res.append(dm.estimate_policy_value(policy, scores))
    res.append(dr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))
    res.append(ipw.estimate_policy_value(test_data['r'], test_data['a'], policy, pscore=pscore))
    res.append(sndr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))

    return np.array(res)



if __name__ == '__main__':
    # data, train, val, test = generate_dataset(10, 10, 20, 8)
    # reg_model = train_model(data, train)

    # convolution = neighborhood_model(np.squeeze(np.eye(10)[train['action'].reshape(-1)]), train['context'], reg_model)
    # convolution.convolve(np.squeeze(np.eye(10)[test['action'].reshape(-1)]), test['context'])
    pass
