import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# import gym
# import recogym
# from recogym.envs import env_1_args, RecoEnv1
# from recogym import Configuration
# from recogym.agents import Agent
# from memory_profiler import profile

from sklearn.utils import check_random_state

from sklearn.utils import check_random_state
from models import BPRModel
from training_utils import fit_bpr
from scipy.special import softmax


# import open bandit pipeline (obp)
from estimators import (
    SelfNormalizedInverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR
)

# import debugpy


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


def calc_reward(dataset, policy):
    return np.array([np.sum(dataset['q_x_a'] * policy.squeeze(), axis=1).mean()])


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
    const = 1 / params['ctr']
    q_x_a = (1 / (const + np.exp(-score)))

    original_policy = softmax(our_x @ our_a.T, axis=1)
    best_policy = softmax(emb_x @ emb_a.T, axis=1)

    greedy_policy = np.zeros_like(original_policy)
    greedy_policy[np.arange(params["n_users"]), np.argmax(best_policy, axis=1)] = 1

    pseudo_dataset = dict(q_x_a=q_x_a)

    print(f"Random Item CTR: {q_x_a.mean()}")
    print(f"Optimal greedy CTR: {calc_reward(pseudo_dataset, greedy_policy)[0]}")
    print(f"Optimal Stochastic CTR: {calc_reward(pseudo_dataset, best_policy)[0]}")
    print(f"Our Initial CTR: {calc_reward(pseudo_dataset, original_policy)[0]}")

    user_prior = softmax(np.random.normal(size=(params["n_users"], 1))).squeeze()

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
                emb_dim=params["emb_dim"],
                user_prior=user_prior,
                )


def create_simulation_data_from_pi(dataset: dict, policy: np.ndarray, n_samples, random_state: int = 12345):
    random_ = check_random_state(random_state)

    simulation_data = {'actions':np.zeros(n_samples, dtype=np.int32), 
                       'users': np.zeros(n_samples, dtype=np.int32), 
                       'reward':np.zeros(n_samples),
                       'pscore':np.zeros(n_samples)}

    simulation_data['pi_0'] = policy
    users = random_.choice(np.arange(dataset['n_users']), size=n_samples, p=dataset['user_prior'], replace=True)
    actions = []

    for i in users:
        user_actions = random_.choice(np.arange(dataset['n_actions']), size=1, p=policy[i], replace=True)
        actions.append(np.array(user_actions))

    actions = np.array(actions)

    idx = np.arange(n_samples)
    simulation_data['actions'] = actions.squeeze()
    simulation_data['users'][idx] = users[idx]

    qq = dataset['q_x_a'][users, simulation_data['actions']]
    simulation_data['reward'][idx] = np.squeeze(qq > random_.random(size=qq.shape))
    simulation_data['pscore'][idx] = np.squeeze(policy[users, simulation_data['actions']])
    simulation_data['q_x_a'] = dataset['q_x_a']

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


def calc_reward_criteo(dataset, policy):
    sim = create_simulation_data_from_pi(dataset['env'], policy.squeeze(), 300000)
    return sim['rewards'].mean()
    # return np.array([np.sum(dataset['q_x_a'] * policy.squeeze(), axis=1).mean()])


def eval_policy(model, test_data, original_policy_prob, policy):
    dr = DR()
    dm = DM()
    ipw = IPW()
    sndr = SNDR()

    scores = model.predict(test_data['x_idx'])

    policy = policy[test_data['x_idx']]
    
    actions = test_data['a']
    res = []
    # reward = test_data['q_x_a'][test_data['x_idx'], actions]
    # res.append(reward.mean())
    
    pscore = original_policy_prob[test_data['x_idx'], actions].squeeze()
    pi_e_at_position = policy[test_data['x_idx'], actions].squeeze()

    res.append(dm.estimate_policy_value(policy, scores))
    res.append(dr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))
    res.append(ipw.estimate_policy_value(test_data['r'], test_data['a'], policy, pscore=pscore))
    res.append(sndr.estimate_policy_value(test_data['r'], test_data['a'], policy, scores, pscore=pscore))

    print(get_weights_info(pi_e_at_position, pscore))

    return np.array(res)



def calc_gini(x):
    sorted_x = np.sort(x)
    n = sorted_x.size

    # weights: 1..n
    cum_weights = np.arange(1, n+1, dtype=sorted_x.dtype)

    numerator = np.sum((2 * cum_weights - n - 1) * sorted_x)
    denominator = n * np.sum(sorted_x)

    return numerator / denominator


def calc_ESS(x):
    return x.sum() ** 2 / (x ** 2).sum()


def get_weights_info(policy, original_policy_prob):

    iw = policy.squeeze() / original_policy_prob.squeeze()
    iw = iw.flatten()

    return dict(
                gini=calc_gini(iw),
                ess=calc_ESS(iw),
                max_wi=iw.max(),
                min_wi=iw.min(),
    )


def get_opl_results_dict(reg_results, conv_results):
    reward = conv_results[:, 0]
    return    dict(
                policy_rewards=np.mean(reward),
                # ipw=np.mean(abs(conv_results[: ,3] - reward)),
                # reg_dm=np.mean(abs(reg_results - reward)),
                # conv_dm=np.mean(abs(conv_results[: ,1] - reward)),
                # conv_dr=np.mean(abs(conv_results[: ,2] - reward)),
                # conv_sndr=np.mean(abs(conv_results[: ,4] - reward)),

                ipw=np.mean(conv_results[: ,3]), 
                reg_dm=np.mean(reg_results),
                conv_dm=np.mean(conv_results[: ,1]),
                conv_dr=np.mean(conv_results[: ,2]),
                conv_sndr=np.mean(conv_results[: ,4]),

                ipw_var=np.var(conv_results[: ,3]),
                reg_dm_var=np.var(reg_results),
                conv_dm_var=np.var(conv_results[: ,1]),
                conv_dr_var=np.var(conv_results[: ,2]),
                conv_sndr_var=np.var(conv_results[: ,4]),

                                
                # ipw_p_err=np.mean(abs(conv_results[: ,3] - reward) / reward) * 100,
                # reg_dm_p_err=np.mean(abs(reg_results - reward) / reward) * 100,
                # conv_dm_p_err=np.mean(abs(conv_results[: ,1] - reward) / reward) * 100,
                # conv_dr_p_err=np.mean(abs(conv_results[: ,2] - reward) / reward) * 100,
                # conv_sndr_p_err=np.mean(abs(conv_results[: ,4] - reward) / reward) * 100,
                
                action_diff_to_real=np.mean(conv_results[: ,5]),
                action_delta=np.mean(conv_results[: ,6]),
                context_diff_to_real=np.mean(conv_results[: ,7]),
                context_delta=np.mean(conv_results[: ,8])
                )