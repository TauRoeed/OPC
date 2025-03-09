import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim


from sklearn.utils import check_random_state
# implementing OPE of the IPWLearner using synthetic bandit data
from sklearn.linear_model import LogisticRegression, LinearRegression

import matplotlib.pyplot as plt

from scipy.special import softmax
from abc import ABCMeta

# import open bandit pipeline (obp)
from obp.ope import (
    OffPolicyEvaluation,
    RegressionModel,
    InverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR
)


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


def eval_policy(model, test_data, original_policy_prob, policy):
    dr = DR()
    dm = DM()
    ipw = IPW()
    sndr = SNDR()

    scores = model.predict(test_data['x_idx'])

    policy = policy[test_data['x_idx']]
    actions = np.squeeze(np.argmax(policy, axis=1))
    res = []
    reward = test_data['q_x_a'][test_data['x_idx'], actions]
    res.append(reward.mean())

    pscore = original_policy_prob[test_data['x_idx'], actions].squeeze()
    
    res.append(dm.estimate_policy_value(policy, scores))
    res.append(dr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))
    res.append(ipw.estimate_policy_value(test_data['r'], test_data['a'], policy, pscore=pscore))
    res.append(sndr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))

    return np.array(res)


def sample_policy_actions(pi: np.ndarray, q_x_a: np.ndarray, n_users: np.int, n_actions: np.int, random_state: int = 12345):
    # sample actions from Pi
    random_ = check_random_state(random_state)
    x_idx, a_idx = np.mgrid[:n_users, :n_actions]
    x_idx, a_idx = x_idx.flatten(), a_idx.flatten()
    prob = softmax(pi.flatten())
    all_idx = random_.choice(np.arange(len(prob)), size=len(prob), p=prob, replace=False)
    reward = q_x_a.flatten()

    return dict(
        actions=a_idx[all_idx],
        users=x_idx[all_idx],
        reward=reward[all_idx],
        pscore=pi[x_idx[all_idx], a_idx[all_idx]],
        pi_0=pi
    )


def create_simluation_data_from_pi(pi: np.ndarray, q_x_a: np.ndarray, n_users: np.int, n_actions: np.int, random_state: int = 12345):
    random_ = check_random_state(random_state)
    simulation_data = {'actions':np.zeros((n_actions, n_users), dtype=np.int32), 'users': np.zeros((n_actions, n_users), dtype=np.int32), 'reward':np.zeros((n_actions, n_users))}
    actions = []
    for i in range(n_users):
        user_actions = random_.choice(np.arange(n_actions), size=n_actions, p=pi[i], replace=False)
        actions.append(np.array(user_actions))

    actions = np.vstack(actions)
    for i in range(n_actions):
        simulation_data['actions'][i] = actions[:, i]
        simulation_data['users'][i] = np.arange(n_users)
        simulation_data['reward'][i] = np.squeeze(q_x_a[np.arange(n_users), simulation_data['actions'][i]])

    return simulation_data


def train_model(dataset, train_data):
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        base_model=LogisticRegression(),)

    regression_model.fit(
        context=train_data["context"],
        action=train_data["action"],
        reward=train_data["reward"],
        pscore=train_data["pscore"]
    )
    return regression_model


if __name__ == '__main__':
    # data, train, val, test = generate_dataset(10, 10, 20, 8)
    # reg_model = train_model(data, train)

    # convolution = neighborhood_model(np.squeeze(np.eye(10)[train['action'].reshape(-1)]), train['context'], reg_model)
    # convolution.convolve(np.squeeze(np.eye(10)[test['action'].reshape(-1)]), test['context'])
    pass


