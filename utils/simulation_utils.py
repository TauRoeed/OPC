import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
from scipy.special import softmax

from models.estimators import (
    SelfNormalizedInverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR,
)


# ----------------------------
# Dataset helpers
# ----------------------------
class CustomCFDataset(Dataset):
    def __init__(self, user_idx, action_idx, rewards, original_prob, q=0.05):
        self.user_idx = user_idx
        self.action_idx = action_idx
        self.rewards = rewards
        self.original_prob = original_prob
        self.q = q
        self.filter_by_prob()

    def filter_by_prob(self):
        pscore = self.original_prob[self.user_idx, self.action_idx].squeeze()
        qq = np.quantile(pscore, q=[self.q, 1 - self.q])
        mask = (pscore > qq[0]) & (pscore < qq[1])

        self.user_idx = self.user_idx[mask]
        self.action_idx = self.action_idx[mask]
        self.rewards = self.rewards[mask]

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, sample_idx):
        user = torch.tensor(self.user_idx[sample_idx].squeeze())
        action = torch.tensor(self.action_idx[sample_idx].squeeze()).long()
        reward = torch.tensor(self.rewards[sample_idx].squeeze(), dtype=torch.double)
        action_dist = torch.tensor(self.original_prob[user].squeeze())
        return user, action, reward, action_dist


# ----------------------------
# Scalable environment
# ----------------------------
@dataclass
class SyntheticBanditEnv:
    """On-the-fly reward probabilities for (user, action) pairs."""
    emb_x: np.ndarray  # (n_users, d)
    emb_a: np.ndarray  # (n_actions, d)
    ctr: float = 0.0
    temperature: float = 1.0

    def reward_prob(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        x = self.emb_x[users]
        a = self.emb_a[actions]
        logits = (x * a).sum(axis=1) / max(self.temperature, 1e-8)
        logits = logits + float(self.ctr)
        return 1.0 / (1.0 + np.exp(-logits))


# ----------------------------
# Metrics helpers
# ----------------------------
def calc_gini(x: np.ndarray) -> float:
    sorted_x = np.sort(x)
    n = sorted_x.size
    cum_weights = np.arange(1, n + 1, dtype=sorted_x.dtype)
    numerator = np.sum((2 * cum_weights - n - 1) * sorted_x)
    denominator = n * np.sum(sorted_x)
    return float(numerator / denominator)


def calc_ESS(x: np.ndarray) -> float:
    return float(x.sum() ** 2 / (x ** 2).sum())


def get_weights_info(policy, original_policy_prob):
    iw = policy.squeeze() / original_policy_prob.squeeze()
    iw = iw.flatten()
    return dict(
        gini=calc_gini(iw),
        ess=calc_ESS(iw),
        max_wi=float(iw.max()),
        min_wi=float(iw.min()),
    )


# ----------------------------
# Reward computation
# ----------------------------
def calc_reward(dataset: dict, policy):
    """Compute / estimate the value of a policy.

    - If dataset contains 'q_x_a' and policy is a dense matrix, returns exact value.
    - If dataset contains 'env' and policy is a policy object, returns Monte Carlo estimate.
    """
    if isinstance(policy, np.ndarray):
        # expected shape: (n_users, n_actions) or (n_users, n_actions, 1)
        if "q_x_a" not in dataset:
            raise ValueError("calc_reward with a dense policy requires dataset['q_x_a'].")
        pol = policy.squeeze()
        return np.array([np.sum(dataset["q_x_a"] * pol, axis=1).mean()])

    # policy object path
    if "env" not in dataset:
        raise ValueError("calc_reward with a policy object requires dataset['env'].")

    env = dataset["env"]
    n_users = int(dataset["n_users"])
    user_prior = dataset.get("user_prior", None)
    rng = np.random.default_rng(12345)

    n_mc = min(10000, n_users)
    if user_prior is None:
        users = rng.integers(0, n_users, size=n_mc, endpoint=False)
    else:
        users = rng.choice(np.arange(n_users), size=n_mc, replace=True, p=user_prior)

    actions, _ = policy.sample_actions(users)
    p = env.reward_prob(users, actions)
    return np.array([float(p.mean())])


# ----------------------------
# Dataset generation
# ----------------------------
def generate_dataset(params, seed=12345, emb_a=None, emb_x=None, materialize_q_x_a: bool = False,
                     dtype=np.float32, store_original: bool = False, make_user_prior: bool = False):
    random_ = check_random_state(seed)

    # embeddings
    if emb_a is not None:
        emb_a = np.load(emb_a) if isinstance(emb_a, str) else np.asarray(emb_a)
    else:
        emb_a = random_.normal(size=(params["n_actions"], params["emb_dim"])).astype(dtype)

    if emb_x is not None:
        emb_x = np.load(emb_x) if isinstance(emb_x, str) else np.asarray(emb_x)
    else:
        emb_x = random_.normal(size=(params["n_users"], params["emb_dim"])).astype(dtype)

    # noisy "our" embeddings (don’t allocate extra copies unless asked)
    noise_a = random_.normal(size=(emb_a.shape)).astype(dtype)
    our_a = ((1 - params["eps"]) * emb_a + params["eps"] * noise_a).astype(dtype)

    noise_x = random_.normal(size=(emb_x.shape)).astype(dtype)
    our_x = ((1 - params["eps"]) * emb_x + params["eps"] * noise_x).astype(dtype)

    # env always available
    env = SyntheticBanditEnv(emb_x=emb_x, emb_a=emb_a, ctr=float(params.get("ctr", 0.0)))

    # optional q_x_a (dangerous!)
    q_x_a = None
    if materialize_q_x_a:
        score = emb_x @ emb_a.T  # (n_users x n_actions) HUGE
        const = 1.0 / float(params["ctr"])
        q_x_a = (1.0 / (const + np.exp(-score))).astype(dtype)

    # optional user_prior
    user_prior = None
    if make_user_prior:
        logits = random_.normal(size=(params["n_users"],)).astype(dtype)
        logits -= logits.max()
        exp = np.exp(logits)
        user_prior = (exp / exp.sum()).astype(dtype)

    dataset = dict(
        emb_a=emb_a,
        our_a=our_a,
        emb_x=emb_x,
        our_x=our_x,
        n_actions=int(emb_a.shape[0]),
        n_users=int(emb_x.shape[0]),
        emb_dim=int(emb_x.shape[1]),
        env=env,
    )

    if store_original:
        dataset["original_a"] = our_a.copy()
        dataset["original_x"] = our_x.copy()

    if make_user_prior:
        dataset["user_prior"] = user_prior

    if materialize_q_x_a:
        dataset["q_x_a"] = q_x_a

    return dataset

# ----------------------------
# Simulation: dense policy (legacy) and policy object (scalable)
# ----------------------------
def create_simulation_data_from_pi(dataset: dict, policy: np.ndarray, n_samples: int, random_state: int = 12345, chunk_size: int = 100000):
    """Legacy sampler that expects a dense (n_users x n_actions) policy matrix."""
    t0 = time.time()
    random_ = check_random_state(random_state)

    simulation_data = {
        "actions": np.zeros(n_samples, dtype=np.int32),
        "users": np.zeros(n_samples, dtype=np.int32),
        "reward": np.zeros(n_samples, dtype=float),
        "pscore": np.zeros(n_samples, dtype=float),
        "pi_0": policy,
    }

    users = random_.choice(np.arange(dataset["n_users"]), size=n_samples, p=dataset["user_prior"], replace=True)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        u_chunk = users[start:end]

        user_policies = policy[u_chunk]
        cum_p = np.cumsum(user_policies, axis=1)
        r = random_.rand(len(u_chunk), 1)
        actions = (r < cum_p).argmax(axis=1)

        pscore = policy[u_chunk, actions]

        if "q_x_a" in dataset:
            qq = dataset["q_x_a"][u_chunk, actions]
        else:
            qq = dataset["env"].reward_prob(u_chunk, actions)

        rewards = (qq > random_.rand(*qq.shape)).astype(float)

        simulation_data["users"][start:end] = u_chunk
        simulation_data["actions"][start:end] = actions.squeeze()
        simulation_data["reward"][start:end] = rewards.squeeze()
        simulation_data["pscore"][start:end] = pscore.squeeze()

    if "q_x_a" in dataset:
        simulation_data["q_x_a"] = dataset["q_x_a"]

    print(f"Simulation time for {n_samples} samples: {time.time() - t0} seconds")
    return simulation_data


def create_simulation_data_from_policy(dataset: dict, policy, n_samples: int, random_state: int = 12345, chunk_size: int = 100000):
    """Scalable sampler that uses a policy object with sample_actions(users)."""
    t0 = time.time()
    rng = np.random.default_rng(random_state)

    simulation_data = {
        "actions": np.zeros(n_samples, dtype=np.int32),
        "users": np.zeros(n_samples, dtype=np.int32),
        "reward": np.zeros(n_samples, dtype=float),
        "pscore": np.zeros(n_samples, dtype=float),
    }

    n_users = int(dataset["n_users"])
    user_prior = dataset.get("user_prior", None)
    if user_prior is None:
        users = rng.integers(0, n_users, size=n_samples, endpoint=False)
    else:
        users = rng.choice(np.arange(n_users), size=n_samples, p=user_prior, replace=True)

    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        u_chunk = users[start:end]
        a_chunk, p_chunk = policy.sample_actions(u_chunk)

        if "q_x_a" in dataset:
            qq = dataset["q_x_a"][u_chunk, a_chunk]
        else:
            qq = dataset["env"].reward_prob(u_chunk, a_chunk)

        r = (rng.random(size=len(u_chunk)) < qq).astype(float)

        simulation_data["users"][start:end] = u_chunk
        simulation_data["actions"][start:end] = a_chunk
        simulation_data["pscore"][start:end] = p_chunk
        simulation_data["reward"][start:end] = r

    print(f"Simulation time for {n_samples} samples: {time.time() - t0} seconds")
    return simulation_data


def get_train_data(n_actions, train_size, sim_data, idx, emb_x):
    out = dict(
        num_data=train_size,
        num_actions=n_actions,
        x=emb_x[sim_data["users"][idx].flatten()],
        a=sim_data["actions"][idx].flatten(),
        r=sim_data["reward"][idx].flatten(),
        x_idx=sim_data["users"][idx].flatten(),
        pscore=sim_data["pscore"][idx].flatten(),
    )
    if "pi_0" in sim_data:
        out["pi_0"] = sim_data["pi_0"]
    if "q_x_a" in sim_data:
        out["q_x_a"] = sim_data["q_x_a"]
    return out


# ----------------------------
# Evaluation (kept compatible)
# ----------------------------
def eval_policy(model, test_data, original_policy_prob, policy):
    t0 = time.time()

    dr = DR()
    dm = DM()
    ipw = IPW()
    sndr = SNDR()

    scores = model.predict(test_data["x"])
    policy = policy[test_data["x_idx"]]

    actions = test_data["a"]
    pscore = original_policy_prob[test_data["x_idx"], actions].squeeze()
    pi_e_at_position = policy[test_data["x_idx"], actions].squeeze()

    res = []
    res.append(dm.estimate_policy_value(policy, scores))
    res.append(dr.estimate_policy_value(test_data["r"], test_data["a"], policy, scores, pscore=pscore))
    res.append(ipw.estimate_policy_value(test_data["r"], test_data["a"], policy, pscore=pscore))
    res.append(sndr.estimate_policy_value(test_data["r"], test_data["a"], policy, scores, pscore=pscore))

    print(f"Num samples is {len(test_data['r'])}")
    print(get_weights_info(pi_e_at_position, pscore))
    print(f"Eval time: {time.time() - t0} seconds")
    return np.array(res)


def get_opl_results_dict(reg_results, conv_results):
    reward = conv_results[:, 0]
    return dict(
        policy_rewards=float(np.mean(reward)),
        ipw=float(np.mean(conv_results[:, 3])),
        reg_dm=float(np.mean(reg_results)),
        conv_dm=float(np.mean(conv_results[:, 1])),
        conv_dr=float(np.mean(conv_results[:, 2])),
        conv_sndr=float(np.mean(conv_results[:, 4])),
        ipw_var=float(np.var(conv_results[:, 3])),
        reg_dm_var=float(np.var(reg_results)),
        conv_dm_var=float(np.var(conv_results[:, 1])),
        conv_dr_var=float(np.var(conv_results[:, 2])),
        conv_sndr_var=float(np.var(conv_results[:, 4])),
        action_diff_to_real=float(np.mean(conv_results[:, 5])),
        action_delta=float(np.mean(conv_results[:, 6])),
        context_diff_to_real=float(np.mean(conv_results[:, 7])),
        context_delta=float(np.mean(conv_results[:, 8])),
    )