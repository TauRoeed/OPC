import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gym
import recogym

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
    env = gym.make('reco-gym-v1',
        n_users=params['n_users'],
        n_items=params['n_actions'],
        random_seed=12345
    )
    
    env.reset()
    random_ = check_random_state(12345)
    emb_a = random_.normal(size=(params["n_actions"], params["emb_dim"]))
    emb_x = random_.normal(size=(params["n_users"], params["emb_dim"]))

    n = params["n_users"]*params["n_actions"]
    pi_0 = np.ones((params["n_users"], params["n_actions"])) / params['n_actions']
    sim_data = create_simulation_data_from_pi(env, pi_0, n)

    bpr_model = BPRModel(
        num_users=params['n_users'],
        num_actions=params['n_actions'],
        embedding_dim=params['emb_dim']
    )


    cf_dataset =  CustomCFDataset(
                                sim_data['users'], 
                                sim_data['actions'], 
                                sim_data['rewards'], 
                                pi_0[sim_data['users']]
                                )
    
    train_loader = DataLoader(cf_dataset, batch_size=100, shuffle=True)
    fit_bpr(bpr_model, train_loader)

    emb_x, emb_a = bpr_model.get_emb()
    emb_x, emb_a = emb_x.detach().cpu().numpy(), emb_a.detach().cpu().numpy()

    noise_a = random_.normal(size=(params["emb_dim"]))
    our_a = (1-params["eps"]) * emb_a + params["eps"] * noise_a

    original_a = our_a.copy()

    noise_x = random_.normal(size=(params["emb_dim"])) 
    our_x = (1-params["eps"]) * emb_x + params["eps"] * noise_x
    original_x = our_x.copy()

    return dict(
        emb_a=emb_a,
        our_a=our_a,
        original_a=original_a,
        emb_x=emb_x,
        our_x=our_x,
        original_x=original_x,
        env=env,
        n_actions=params["n_actions"],
        n_users=params["n_users"],
        emb_dim=emb_a.shape[1]
    )


def create_simulation_data_from_pi(env, policy, n):
    """
    Samples one action per user from the given policy.
    
    Args:
        env: The RecoGym environment (from generate_dataset)
        policy: np.ndarray of shape (n_users, n_actions) – probability distributions over actions for each user
        
    Returns:
        actions: np.ndarray of shape (n_users,) – sampled actions
        rewards: np.ndarray of shape (n_users,) – obtained rewards
        pscore: np.ndarray of shape (n_users,) – policy probability of the sampled action
    """
    n_users, n_actions = policy.shape
    users = np.random.randint(0, n_users, size=n)
    
    actions = np.zeros(n, dtype=int)
    rewards = np.zeros(n, dtype=float)
    pscore = np.zeros(n, dtype=float)

    for i, user_id in enumerate(users):
        action_probs = policy[user_id]
        action = np.random.choice(n_actions, p=action_probs)

        # Step the RecoGym env with this user-action pair
        _, reward, _, _ = env.step((user_id, action))

        actions[i] = action
        rewards[i] = reward
        pscore[i] = action_probs[action]

    return dict(users=users, actions=actions, rewards=rewards, pscore=pscore)


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


def get_opl_results_dict(reg_results, conv_results):
    reward = conv_results[:, 0]
    return    dict(
                policy_rewards=np.mean(reward),
                ipw=np.mean(abs(conv_results[: ,3] - reward)),
                reg_dm=np.mean(abs(reg_results - reward)),
                conv_dm=np.mean(abs(conv_results[: ,1] - reward)),
                conv_dr=np.mean(abs(conv_results[: ,2] - reward)),
                conv_sndr=np.mean(abs(conv_results[: ,4] - reward)),

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


if __name__ == '__main__':
    # data, train, val, test = generate_dataset(10, 10, 20, 8)
    # reg_model = train_model(data, train)

    # convolution = neighborhood_model(np.squeeze(np.eye(10)[train['action'].reshape(-1)]), train['context'], reg_model)
    # convolution.convolve(np.squeeze(np.eye(10)[test['action'].reshape(-1)]), test['context'])
    pass
