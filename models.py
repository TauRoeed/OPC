
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import check_random_state
# from memory_profiler import profile

from scipy.special import softmax
from abc import ABCMeta

"""Regression Model Class for Estimating Mean Reward Functions."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from saito_helpers import check_bandit_feedback_inputs


class NeighborhoodModel(metaclass=ABCMeta):
    def __init__(self, context, actions, action_emb, context_emb, rewards, num_neighbors=5, chunksize=3000, gamma=0.5):
        self.gamma = gamma
        self.num_neighbors = num_neighbors
        self.chunksize = chunksize
        self.fit(action_emb, context_emb, actions, context, rewards)

    def fit(self, action_emb, context_emb, actions, context, rewards):
        self.update(action_emb, context_emb, calc=False)
        self.actions = np.int32(actions)
        self.context = np.int32(context)
        self.reward = np.float32(rewards)
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
        # Preallocate output
        B = test_context.shape[0]
        weighted_sum = np.zeros(B, dtype=np.float16)
        sim_sum = np.zeros(B, dtype=np.float16)

        for chunk_start in range(0, B, self.chunksize):
            chunk_end = min(chunk_start + self.chunksize, B)

            cosine_context = self.context_similarity[np.int32(test_context[chunk_start:chunk_end])][:, self.context]
            cosine_actions = self.action_similarity[np.int32(test_actions[chunk_start:chunk_end])][:, self.actions]
            tot_cosine = self.gamma * cosine_actions + (1 - self.gamma) * cosine_context

            del cosine_context, cosine_actions

            chunk_top_k_idx = np.argpartition(tot_cosine, -self.num_neighbors, axis=1)[:, -self.num_neighbors:]

            # Get similarity and rewards for each example in the chunk
            similarity = tot_cosine[np.arange(chunk_end - chunk_start)[:, None], chunk_top_k_idx]
            r_chunk = self.reward[chunk_top_k_idx]

            weighted_sum[chunk_start:chunk_end] = (similarity * r_chunk).sum(axis=1)
            sim_sum[chunk_start:chunk_end] = similarity.sum(axis=1)

        return weighted_sum / (sim_sum + 1e-8)
    
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



class LinearTransform(nn.Module):
    def __init__(self, embedding_size, embedding_dim):
        super(LinearTransform, self).__init__()
        self.delta = nn.Parameter(torch.zeros((embedding_size, embedding_dim)))

    def forward(self, x, idx=None):
        if idx is None:
            return x + self.delta
        else:   
            return x + self.delta[idx]

    def to(self, device):
        # Move the module itself
        super().to(device)
        self.delta = self.delta.to(device)
        return self


class CFModel(nn.Module):
    def __init__(self, num_users, num_actions, embedding_dim, 
                 initial_user_embeddings=None, initial_actions_embeddings=None, 
                 user_transform=None, action_transform=None):

        super(CFModel, self).__init__()

        self.user_transform = user_transform
        self.action_transform = action_transform

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

        if user_transform is not None:
            for param in  self.user_embeddings.parameters():
                param.requires_grad = False
        if action_transform is not None:
            for param in self.actions_embeddings.parameters():
                param.requires_grad = False


    def get_params(self):
        return self.user_embeddings(self.users), self.actions_embeddings(self.actions)
    

    def forward(self, user_ids):

        user_embedding = self.user_embeddings(user_ids)
        actions_embedding = self.actions_embeddings(self.actions)
        # Apply transform if it exists

        if self.user_transform is not None:
            user_embedding = self.user_transform(user_embedding, user_ids)
        if self.action_transform is not None:
            actions_embedding = self.action_transform(actions_embedding, self.actions)

        scores = user_embedding @ actions_embedding.T

        return F.softmax(scores, dim=1).unsqueeze(-1)
    
    def to(self, device):
        # Move the module itself
        super().to(device)
        self.actions = self.actions.to(device)
        self.users = self.users.to(device)
        
        if self.user_transform is not None:
            self.user_transform = self.user_transform.to(device)
        if self.action_transform is not None:
            self.action_transform = self.action_transform.to(device)

        return self


class LinearCFModel(nn.Module):
    def __init__(self, num_users, num_actions, embedding_dim, 
                 initial_user_embeddings=None, initial_actions_embeddings=None):
        
        super(LinearCFModel, self).__init__()

        self.user_transform = LinearTransform(num_users, embedding_dim)
        self.action_transform = LinearTransform(num_actions, embedding_dim)

        self.cfmodel = CFModel(
                                num_users, 
                                num_actions, 
                                embedding_dim, 
                                initial_user_embeddings, 
                                initial_actions_embeddings,
                                user_transform=self.user_transform,    
                                action_transform=self.action_transform
                               )
        
        for param in self.cfmodel.user_embeddings.parameters():
            param.requires_grad = False

        for param in self.cfmodel.actions_embeddings.parameters():
            param.requires_grad = False

    def forward(self, user_ids):
        return self.cfmodel(user_ids)
        
    def to(self, device):
        # Move the module itself
        super().to(device)
        
        self.cfmodel = self.cfmodel.to(device)
        self.user_transform = self.user_transform.to(device)
        self.action_transform = self.action_transform.to(device)

        return self
    
    def get_params(self):
        emb_x = self.user_transform(self.cfmodel.user_embeddings.weight)
        emb_a = self.action_transform(self.cfmodel.actions_embeddings.weight)
        return emb_x, emb_a



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

# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

@dataclass
class RegressionModel(BaseEstimator):
    """Machine learning model to estimate the reward function (:math:`q(x,a):= \\mathbb{E}[r|x,a]`).

    Note
    -------
    Reward :math:`r` must be either binary or continuous.

    Parameters
    ------------
    base_model: BaseEstimator
        A machine learning model used to estimate the reward function.

    n_actions: int
        Number of actions.

    len_list: int, default=1
        Length of a list of actions in a recommendation/ranking inferface, slate size.
        When Open Bandit Dataset is used, 3 should be set.

    action_context: array-like, shape (n_actions, dim_action_context), default=None
        Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).
        If None, one-hot encoding of the action variable is used as default.

    fitting_method: str, default='normal'
        Method to fit the regression model.
        Must be one of ['normal', 'iw', 'mrdr'] where 'iw' stands for importance weighting and
        'mrdr' stands for more robust doubly robust.

    References
    -----------
    Mehrdad Farajtabar, Yinlam Chow, and Mohammad Ghavamzadeh.
    "More Robust Doubly Robust Off-policy Evaluation.", 2018.

    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    Yusuke Narita, Shota Yasui, and Kohei Yata.
    "Off-policy Bandit and Reinforcement Learning.", 2020.

    """

    base_model: BaseEstimator
    n_actions: int
    len_list: int = 1
    action_context: Optional[np.ndarray] = None
    fitting_method: str = "normal"

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.len_list, "len_list", int, min_val=1)
        if not (
            isinstance(self.fitting_method, str)
            and self.fitting_method in ["normal", "iw", "mrdr"]
        ):
            raise ValueError(
                f"`fitting_method` must be one of 'normal', 'iw', or 'mrdr', but {self.fitting_method} is given"
            )
        if not isinstance(self.base_model, BaseEstimator):
            raise ValueError(
                "`base_model` must be BaseEstimator or a child class of BaseEstimator"
            )

        self.base_model_list = [
            clone(self.base_model) for _ in np.arange(self.len_list)
        ]
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)

    def fit(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the regression model on given logged bandit data.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,)
            Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.
            If None, behavior policy is assumed to be uniform.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a regression model assumes that only a single action is chosen for each data.
            When `len_list` > 1, an array must be given as `position`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.
            When either 'iw' or 'mrdr' is set to `fitting_method`, `action_dist` must be given.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n = context.shape[0]

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
            if not np.allclose(action_dist.sum(axis=1), 1):
                raise ValueError("`action_dist` must be a probability distribution")
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        for pos_ in np.arange(self.len_list):
            idx = position == pos_
            X = self._pre_process_for_reg_model(
                context=context[idx],
                action=action[idx],
                action_context=self.action_context,
            )
            if X.shape[0] == 0:
                raise ValueError(f"No training data at position {pos_}")
            # train the base model according to the given `fitting method`
            if self.fitting_method == "normal":
                self.base_model_list[pos_].fit(X, reward[idx])
            else:
                action_dist_at_pos = action_dist[np.arange(n), action, pos_][idx]
                if self.fitting_method == "iw":
                    sample_weight = action_dist_at_pos / pscore[idx]
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )
                elif self.fitting_method == "mrdr":
                    sample_weight = action_dist_at_pos
                    sample_weight *= 1.0 - pscore[idx]
                    sample_weight /= pscore[idx] ** 2
                    self.base_model_list[pos_].fit(
                        X, reward[idx], sample_weight=sample_weight
                    )

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict the reward function.

        Parameters
        -----------
        context: array-like, shape (n_rounds_of_new_data, dim_context)
            Context vectors of new data.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds_of_new_data, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.

        """
        n = context.shape[0]
        q_hat = np.zeros((n, self.n_actions, self.len_list))
        for action_ in np.arange(self.n_actions):
            for pos_ in np.arange(self.len_list):
                X = self._pre_process_for_reg_model(
                    context=context,
                    action=action_ * np.ones(n, int),
                    action_context=self.action_context,
                )
                q_hat_ = (
                    self.base_model_list[pos_].predict_proba(X)[:, 1]
                    if is_classifier(self.base_model_list[pos_])
                    else self.base_model_list[pos_].predict(X)
                )
                q_hat[np.arange(n), action_, pos_] = q_hat_
        return q_hat

    def fit_predict(
        self,
        context: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        pscore: Optional[np.ndarray] = None,
        position: Optional[np.ndarray] = None,
        action_dist: Optional[np.ndarray] = None,
        n_folds: int = 1,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Fit the regression model on given logged bandit data and estimate the expected rewards on the same data.

        Note
        ------
        When `n_folds` is larger than 1, the cross-fitting procedure is applied.
        See the reference for the details about the cross-fitting technique.

        Parameters
        ----------
        context: array-like, shape (n_rounds, dim_context)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        pscore: array-like, shape (n_rounds,), default=None
            Action choice probabilities (propensity score) of a behavior policy
            in the training set of logged bandit data.
            If None, the the behavior policy is assumed to be uniform random.

        position: array-like, shape (n_rounds,), default=None
            Indices to differentiate positions in a recommendation interface where the actions are presented.
            If None, a regression model assumes that only a single action is chosen for each data.
            When `len_list` > 1, an array must be given as `position`.

        action_dist: array-like, shape (n_rounds, n_actions, len_list), default=None
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.
            When either 'iw' or 'mrdr' is set to `fitting_method`, `action_dist` must be given.

        n_folds: int, default=1
            Number of folds in the cross-fitting procedure.
            When 1 is given, the regression model is trained on the whole logged bandit data.
            Please refer to https://arxiv.org/abs/2002.08536 about the details of the cross-fitting procedure.

        random_state: int, default=None
            `random_state` affects the ordering of the indices, which controls the randomness of each fold.
            See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html for the details.

        Returns
        -----------
        q_hat: array-like, shape (n_rounds, n_actions, len_list)
            Expected rewards of new data estimated by the regression model.

        """
        check_bandit_feedback_inputs(
            context=context,
            action=action,
            reward=reward,
            pscore=pscore,
            position=position,
            action_context=self.action_context,
        )
        n_rounds = context.shape[0]

        check_scalar(n_folds, "n_folds", int, min_val=1)
        check_random_state(random_state)

        if position is None or self.len_list == 1:
            position = np.zeros_like(action)
        else:
            if position.max() >= self.len_list:
                raise ValueError(
                    f"`position` elements must be smaller than `len_list`, but the maximum value is {position.max()} (>= {self.len_list})"
                )
        if self.fitting_method in ["iw", "mrdr"]:
            if not (isinstance(action_dist, np.ndarray) and action_dist.ndim == 3):
                raise ValueError(
                    "when `fitting_method` is either 'iw' or 'mrdr', `action_dist` (a 3-dimensional ndarray) must be given"
                )
            if action_dist.shape != (n_rounds, self.n_actions, self.len_list):
                raise ValueError(
                    f"shape of `action_dist` must be (n_rounds, n_actions, len_list)=({n_rounds, self.n_actions, self.len_list}), but is {action_dist.shape}"
                )
        if pscore is None:
            pscore = np.ones_like(action) / self.n_actions

        if n_folds == 1:
            self.fit(
                context=context,
                action=action,
                reward=reward,
                pscore=pscore,
                position=position,
                action_dist=action_dist,
            )
            return self.predict(context=context)
        else:
            q_hat = np.zeros((n_rounds, self.n_actions, self.len_list))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        kf.get_n_splits(context)
        for train_idx, test_idx in kf.split(context):
            action_dist_tr = (
                action_dist[train_idx] if action_dist is not None else action_dist
            )
            self.fit(
                context=context[train_idx],
                action=action[train_idx],
                reward=reward[train_idx],
                pscore=pscore[train_idx],
                position=position[train_idx],
                action_dist=action_dist_tr,
            )
            q_hat[test_idx, :, :] = self.predict(context=context[test_idx])
        return q_hat

    def _pre_process_for_reg_model(
        self,
        context: np.ndarray,
        action: np.ndarray,
        action_context: np.ndarray,
    ) -> np.ndarray:
        """Preprocess feature vectors to train a regression model.

        Note
        -----
        Please override this method if you want to use another feature enginnering
        for training the regression model.

        Parameters
        -----------
        context: array-like, shape (n_rounds,)
            Context vectors observed for each data in logged bandit data, i.e., :math:`x_i`.

        action: array-like, shape (n_rounds,)
            Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        action_context: array-like, shape shape (n_actions, dim_action_context)
            Context vectors characterizing actions (i.e., a vector representation or an embedding of each action).

        """
        return np.c_[context, action_context[action]]