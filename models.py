
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import check_random_state

from scipy.special import softmax
from abc import ABCMeta



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

        self.bias = nn.Parameter(torch.zeros((num_users, num_actions)))

    def get_params(self):
        return self.user_embeddings(self.users), self.actions_embeddings(self.actions)
        
    def forward(self, user_ids):
        # Get embeddings for users and actions
        user_embedding = self.user_embeddings(user_ids)
        actions_embedding = self.actions_embeddings
        bias = self.bias[user_ids]
        # Calculate dot product between user and actions embeddings
        scores = user_embedding @ actions_embedding(self.actions).T + bias

        # Apply softmax to get the predicted probability distribution
        return F.softmax(scores, dim=1).unsqueeze(-1)
    
    def to(self, device):
        # Move the module itself
        super().to(device)
        self.actions = self.actions.to(device)
        self.users = self.users.to(device)
        self.bias = self.bias.to(device)
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
    
    def calc_scores(self, user_ids):
        # Ensure user_ids is on the same device as the model
        device = self.user_embeddings.weight.device
        user_ids = user_ids.to(device)

        # Get user embeddings
        user_embedding = self.user_embeddings(user_ids)

        # Ensure self.actions is on the same device
        actions = self.actions.to(device)

        # Get action embeddings
        actions_embedding = self.actions_embeddings(actions)

        # Compute dot product scores
        scores = user_embedding @ actions_embedding.T

        # Return softmaxed scores
        return F.softmax(scores, dim=1).unsqueeze(-1)
        
    def to(self, device):
        # Move the module itself
        super().to(device)
        self.actions = self.actions.to(device)
        self.users = self.users.to(device)
        return self
    
    def seg_emb(self):
        return self.user_embeddings.weight, self.actions_embeddings.weight