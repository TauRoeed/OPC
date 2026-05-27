import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from scipy.special import softmax

from models.estimators import (
    SelfNormalizedInverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR,
)
from utils.policies import _softmax_rows

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


class CustomCFDatasetPS(Dataset):
    """
    Returns per-sample propensity (pscore) instead of full action distribution.
    """
    def __init__(self, user_idx, action_idx, rewards, pscore):
        self.user_idx = user_idx
        self.action_idx = action_idx
        self.rewards = rewards
        self.pscore = pscore

    def __len__(self):
        return len(self.rewards)

    def __getitem__(self, i):
        user = torch.tensor(int(self.user_idx[i]))
        action = torch.tensor(int(self.action_idx[i]))
        reward = torch.tensor(float(self.rewards[i]), dtype=torch.double)
        pscore = torch.tensor(float(self.pscore[i]), dtype=torch.double)
        return user, action, reward, pscore

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

        return 1.0 / ((1.0 /float(self.ctr)) + np.exp(-logits))


# ----------------------------
# Metrics helpers
# ----------------------------
def calc_gini(x: np.ndarray) -> float:
    sorted_x = np.sort(x)
    n = sorted_x.size
    s = float(np.sum(sorted_x))
    if n == 0 or s == 0.0 or not np.isfinite(s):
        return float("nan")
    cum_weights = np.arange(1, n + 1, dtype=sorted_x.dtype)
    numerator = np.sum((2 * cum_weights - n - 1) * sorted_x)
    denominator = n * s
    return float(numerator / denominator)


def calc_ESS(x: np.ndarray) -> float:
    denom = float(np.sum(x**2))
    if denom == 0.0 or not np.isfinite(denom):
        return float("nan")
    return float(x.sum() ** 2 / denom)


def get_weights_info(policy, original_policy_prob):
    iw = policy.squeeze() / original_policy_prob.squeeze()
    iw = iw.flatten()
    return dict(
        gini=calc_gini(iw),
        ess=calc_ESS(iw),
        max_wi=float(np.nanmax(iw)),
        min_wi=float(np.nanmin(iw)),
    )


def floor_renorm_action_dist(p: np.ndarray, min_prob: float = 1e-15) -> np.ndarray:
    """Floor tiny probs and renormalize rows so float32 softmax tails do not become all-zero."""
    x = np.asarray(p, dtype=np.float64)
    if x.ndim == 2:
        x = np.expand_dims(x, -1)
    x = np.maximum(x, min_prob)
    x /= np.sum(x, axis=1, keepdims=True)
    return x.astype(np.float32)


# -
# --------------------------
# Reward computation
# ----------------------------
def calc_reward(dataset: dict, policy, chunk_size: int = 2048):
    """
    Exact policy value computation without materializing full dense matrix.

    Computes:
        V(pi) = sum_u prior[u] * sum_a pi(a|u) * q(u,a)

    Fully chunked over users for memory safety.
    """

    # -------------------------------
    # Case 1: Dense policy matrix
    # -------------------------------
    if isinstance(policy, np.ndarray):
        if "q_x_a" not in dataset:
            raise ValueError("Dense policy requires dataset['q_x_a'].")

        pol = policy.squeeze()              # (n_users, n_actions)
        q = dataset["q_x_a"]                # (n_users, n_actions)

        val = np.sum(q * pol, axis=1).mean()
        return np.array([float(val)])

    # -------------------------------
    # Case 2: Policy object
    # -------------------------------
    if "env" not in dataset:
        raise ValueError("Policy object requires dataset['env'].")

    env = dataset["env"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    prior = dataset.get("user_prior", np.ones(n_users, dtype=np.float64))
    prior = prior.astype(np.float64)
    prior /= prior.sum()   # normalize once

    total_value = 0.0

    for start in range(0, n_users, chunk_size):
        end = min(start + chunk_size, n_users)
        users = np.arange(start, end, dtype=np.int64)

        # ---- Compute policy probabilities (full softmax block) ----
        logits = policy._logits_block(users)        # (b, A)
        probs = _softmax_rows(logits)               # (b, A)

        # ---- Compute reward probabilities for ALL actions ----
        # Create action grid
        b = end - start
        users_rep = np.repeat(users, n_actions)
        actions_rep = np.tile(np.arange(n_actions), b)

        rewards = env.reward_prob(users_rep, actions_rep)
        rewards = rewards.reshape(b, n_actions)     # (b, A)

        # ---- Expected reward per user ----
        user_values = np.sum(probs * rewards, axis=1)  # (b,)

        total_value += np.sum(user_values * prior[users])

    return float(total_value)


def calc_reward_mc(dataset: dict, policy, n_sim=30):
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
    p = 0.0

    for _ in range(n_sim):
        n_mc = min(10000, n_users)
        if user_prior is None:
            users = rng.integers(0, n_users, size=n_mc, endpoint=False)
        else:
            users = rng.choice(np.arange(n_users), size=n_mc, replace=True, p=user_prior)
        actions, _ = policy.sample_actions(users)
        # Aggregate to scalar expected reward estimate per draw.
        p += float(np.mean(env.reward_prob(users, actions)))

    return np.array([float(p / max(int(n_sim), 1))])



def generate_linear_transform_noise(
    X: np.ndarray,
    *,
    seed: int = 12345,
    sigma: float = 1.0,
    chunk_size: int = 100_000,
) -> np.ndarray:

    """Return one general noise vector per row (item/user)."""
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=False)
    n, d = X.shape
    W = rng.normal(0.0, 1.0, size=(d, d)).astype(np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        Xb = X[s:e]
        b = e - s
        mean = Xb @ W
        out[s:e] = mean + sigma * rng.normal(0.0, 1.0, size=(b, d)).astype(np.float32)
    return out


def generate_random_cluster_template_noise(
    X: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 12345,
    sigma: float = 1.0,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Legacy random-cluster template noise (one vector per row)."""
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=False)
    n, d = X.shape
    centroids = rng.normal(0.0, 1.0, size=(n_clusters, d)).astype(np.float32)
    templates = rng.normal(0.0, 1.0, size=(n_clusters, d)).astype(np.float32)
    c_norm2 = np.sum(centroids * centroids, axis=1)
    out = np.empty_like(X, dtype=np.float32)

    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        Xb = X[s:e]
        b = e - s
        x_norm2 = np.sum(Xb * Xb, axis=1, keepdims=True)
        dist2 = x_norm2 - 2.0 * (Xb @ centroids.T) + c_norm2[None, :]
        cid = np.argmin(dist2, axis=1)
        mean = templates[cid]
        out[s:e] = mean + sigma * rng.normal(0.0, 1.0, size=(b, d)).astype(np.float32)
    return out


def generate_kmeans_cluster_template_noise(
    X: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 12345,
    sigma: float = 1.0,
) -> np.ndarray:
    """KMeans cluster-template noise (one vector per row)."""
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=False)
    n, d = X.shape
    kmeans = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        random_state=int(seed),
        batch_size=min(10_000, max(256, n)),
        n_init=10,
        reassignment_ratio=0.0,
    )
    cluster_ids = kmeans.fit_predict(X).astype(np.int32)
    templates = rng.normal(0.0, 1.0, size=(n_clusters, d)).astype(np.float32)
    mean = templates[cluster_ids]
    return (mean + sigma * rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)).astype(np.float32)


def generate_metadata_projection_noise(
    metadata: np.ndarray,
    *,
    out_dim: int,
    seed: int = 12345,
    sigma: float = 1.0,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """
    Metadata-based noise: project metadata into embedding space and add Gaussian noise.
    Returns one noise vector per row.
    """
    rng = np.random.default_rng(seed)
    M = np.asarray(metadata, dtype=np.float32)
    if M.ndim != 2:
        raise ValueError("metadata must be a 2D array.")
    n, m_dim = M.shape
    if m_dim == 0:
        return np.zeros((n, out_dim), dtype=np.float32)

    Wm = rng.normal(0.0, 1.0, size=(m_dim, out_dim)).astype(np.float32)
    out = np.empty((n, out_dim), dtype=np.float32)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        Mb = M[s:e]
        b = e - s
        mean = Mb @ Wm
        out[s:e] = mean + sigma * rng.normal(0.0, 1.0, size=(b, out_dim)).astype(np.float32)
    return out


def mix_ground_truth_with_noises(
    X_gt: np.ndarray,
    noise_vecs: list[np.ndarray],
    epsilons: list[float],
) -> np.ndarray:
    """
    Compose embeddings:
      gt * (1 - sum(eps)) + sum_i (noise_i * eps_i)
    """
    if len(noise_vecs) != len(epsilons):
        raise ValueError("noise_vecs and epsilons must have same length.")
    eps_sum = float(np.sum(epsilons))
    out = (1.0 - eps_sum) * X_gt.astype(np.float32, copy=False)
    for noise, eps in zip(noise_vecs, epsilons):
        out = out + float(eps) * noise.astype(np.float32, copy=False)
    return out.astype(np.float32, copy=False)


def generate_noised_embeddings(
    X: np.ndarray,
    n_clusters: int,
    eps1: float,
    eps2: float,
    seed: int = 12345,
    chunk_size: int = 100_000,
    sigma1: float = 1.0,
    sigma2: float = 1.0,
    noise_mode: str = "random_centroids",
) -> np.ndarray:
    """
    Backward-compatible wrapper around split noise generators.
    """
    X = X.astype(np.float32, copy=False)
    noise1 = generate_linear_transform_noise(
        X,
        seed=seed,
        sigma=sigma1,
        chunk_size=chunk_size,
    )
    if noise_mode == "random_centroids":
        noise2 = generate_random_cluster_template_noise(
            X,
            n_clusters=n_clusters,
            seed=seed + 73,
            sigma=sigma2,
            chunk_size=chunk_size,
        )
    elif noise_mode == "kmeans_templates":
        noise2 = generate_kmeans_cluster_template_noise(
            X,
            n_clusters=n_clusters,
            seed=seed + 73,
            sigma=sigma2,
        )
    else:
        raise ValueError(f"Unsupported noise_mode='{noise_mode}'.")
    return mix_ground_truth_with_noises(X, [noise1, noise2], [eps1, eps2])


# ----------------------------
# Dataset generation
# ----------------------------
def generate_dataset(params, seed=12345, emb_a=None, emb_x=None, user_prior=None, 
                     metadata_a=None, metadata_x=None, materialize_q_x_a: bool = False, dtype=np.float32, 
                     store_original: bool = False):
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

    # Example: lognormal-ish “activity” distribution
    if user_prior is not None:
        user_prior = np.load(user_prior) if isinstance(user_prior, str) else np.asarray(user_prior)
    else:
        user_prior = random_.exponential(scale=1.0, size=(params["n_users"],)).astype(dtype)
    
    user_prior = user_prior / user_prior.sum()

    # split noise generation + composition:
    # gt * (1 - sum(eps)) + sum_i (noise_i * eps_i)
    noise_mode = params.get("noise_mode", "random_centroids")
    sigma1 = float(params.get("sigma1", 1.0))
    sigma2 = float(params.get("sigma2", 1.0))
    sigma_meta = float(params.get("sigma_meta", 1.0))
    chunk_size = int(params.get("noise_chunk_size", 100_000))

    item_noise_linear = generate_linear_transform_noise(
        emb_a,
        seed=seed,
        sigma=sigma1,
        chunk_size=chunk_size,
    )

    user_noise_linear = generate_linear_transform_noise(
        emb_x,
        seed=seed + 1,
        sigma=sigma1,
        chunk_size=chunk_size,
    )

    if noise_mode == "kmeans_templates":
        item_noise_cluster = generate_kmeans_cluster_template_noise(
            emb_a,
            n_clusters=params["n_clusters"],
            seed=seed + 73,
            sigma=sigma2,
        )

        user_noise_cluster = generate_kmeans_cluster_template_noise(
            emb_x,
            n_clusters=params["n_clusters"],
            seed=seed + 74,
            sigma=sigma2,
        )

    elif noise_mode == "random_centroids":
        item_noise_cluster = generate_random_cluster_template_noise(
            emb_a,
            n_clusters=params["n_clusters"],
            seed=seed + 73,
            sigma=sigma2,
            chunk_size=chunk_size,
        )
        user_noise_cluster = generate_random_cluster_template_noise(
            emb_x,
            n_clusters=params["n_clusters"],
            seed=seed + 74,
            sigma=sigma2,
            chunk_size=chunk_size,
        )
        
    else:
        raise ValueError(f"Unsupported noise_mode='{noise_mode}'.")

    eps1 = float(params["eps1"])
    eps2 = float(params["eps2"])
    item_noises = [item_noise_linear, item_noise_cluster]
    user_noises = [user_noise_linear, user_noise_cluster]
    item_eps = [eps1, eps2]
    user_eps = [eps1, eps2]

    eps_meta = float(params.get("eps_meta", 0.0))
    if eps_meta > 0.0:
        if metadata_a is None:
            metadata_a = params.get("metadata_a", None)
        if metadata_x is None:
            metadata_x = params.get("metadata_x", None)

        meta_a_arr = None
        meta_x_arr = None
        if metadata_a is not None:
            meta_a_arr = np.load(metadata_a) if isinstance(metadata_a, str) else np.asarray(metadata_a)
            if meta_a_arr.shape[0] != emb_a.shape[0]:
                raise ValueError("metadata_a rows must match emb_a rows.")
            item_noises.append(
                generate_metadata_projection_noise(
                    meta_a_arr,
                    out_dim=emb_a.shape[1],
                    seed=seed + 131,
                    sigma=sigma_meta,
                    chunk_size=chunk_size,
                )
            )
            item_eps.append(eps_meta)

        if metadata_x is not None:
            meta_x_arr = np.load(metadata_x) if isinstance(metadata_x, str) else np.asarray(metadata_x)
            if meta_x_arr.shape[0] != emb_x.shape[0]:
                raise ValueError("metadata_x rows must match emb_x rows.")
            user_noises.append(
                generate_metadata_projection_noise(
                    meta_x_arr,
                    out_dim=emb_x.shape[1],
                    seed=seed + 132,
                    sigma=sigma_meta,
                    chunk_size=chunk_size,
                )
            )
            user_eps.append(eps_meta)

        if meta_a_arr is None and meta_x_arr is None:
            raise ValueError("eps_meta > 0 but no metadata_a/metadata_x was provided.")

    our_a = mix_ground_truth_with_noises(emb_a, item_noises, item_eps)
    our_x = mix_ground_truth_with_noises(emb_x, user_noises, user_eps)

    # env always available
    env = SyntheticBanditEnv(emb_x=emb_x, emb_a=emb_a, ctr=float(params.get("ctr", 0.05)))

    # optional q_x_a (dangerous!)
    q_x_a = None
    if materialize_q_x_a:
        score = emb_x @ emb_a.T  # (n_users x n_actions) HUGE
        const = 1.0 / float(params["ctr"])
        q_x_a = (1.0 / (const + np.exp(-score))).astype(dtype)

    dataset = dict(
        emb_a=emb_a.astype(dtype),
        our_a=our_a.astype(dtype),
        emb_x=emb_x.astype(dtype),
        our_x=our_x.astype(dtype),
        n_actions=int(emb_a.shape[0]),
        n_users=int(emb_x.shape[0]),
        emb_dim=int(emb_x.shape[1]),
        env=env,
        user_prior=user_prior,
    )

    dataset["policy_temperature"] = float(params.get("policy_temperature", 1.0))

    if store_original:
        dataset["original_a"] = our_a.copy().astype(dtype)
        dataset["original_x"] = our_x.copy().astype(dtype)

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

    scores = np.asarray(model.predict(test_data["x"]), dtype=np.float32)
    policy_in = np.asarray(policy, dtype=np.float32)
    if policy_in.ndim == 2:
        policy_in = np.expand_dims(policy_in, -1)

    policy_in = floor_renorm_action_dist(policy_in)
    actions = test_data["a"]
    # Prefer logged propensities when present (avoids dense n_users x n_actions pi_b).
    if test_data.get("pscore") is not None:
        pscore = np.asarray(test_data["pscore"], dtype=np.float32).squeeze()
    else:
        if original_policy_prob is None:
            raise ValueError("eval_policy needs test_data['pscore'] or original_policy_prob")
        pscore = original_policy_prob[test_data["x_idx"], actions].squeeze()
    
    print(f"PScore time: {time.time() - t0} seconds")

    pol = policy_in.squeeze(-1) if policy_in.ndim == 3 else policy_in
    # If policy rows already align 1:1 with test rows (e.g. val-only softmax), do not re-index.
    if pol.shape[0] == len(actions):
        policy_rows = pol
    else:
        policy_rows = pol[test_data["x_idx"]]

    # print(f"Policy rows time: {time.time() - t0} seconds")

    local_idx = np.arange(len(actions), dtype=np.int64)
    pi_e_at_position = policy_rows[local_idx, actions].squeeze()

    res = []
    res.append(
        dm.estimate_policy_value(
            policy_in, estimated_rewards_by_reg_model=scores
        )
    )
    # print(f"DM time: {time.time() - t0} seconds")
    res.append(
        dr.estimate_policy_value(
            test_data["r"],
            test_data["a"],
            policy_in,
            estimated_rewards_by_reg_model=scores,
            pscore=pscore,
        )
    )
    # print(f"DR time: {time.time() - t0} seconds")
    res.append(ipw.estimate_policy_value(test_data["r"], test_data["a"], policy_in, pscore=pscore))
    # print(f"IPW time: {time.time() - t0} seconds")
    
    res.append(
        sndr.estimate_policy_value(
            test_data["r"],
            test_data["a"],
            policy_in,
            estimated_rewards_by_reg_model=scores,
            pscore=pscore,
        )
    )
    # print(f"SNDR time: {time.time() - t0} seconds")
    print(f"Num samples is {len(test_data['r'])}")
    print(get_weights_info(pi_e_at_position, pscore))
    # print(f"get_weights_info time: {time.time() - t0} seconds")
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