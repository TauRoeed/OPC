import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import check_random_state
from scipy.special import expit, softmax

from models.estimators import (
    SelfNormalizedInverseProbabilityWeighting as IPW,
    DirectMethod as DM,
    DoublyRobust as DR,
    SelfNormalizedDoublyRobust as SNDR,
)
from utils.policies import _logsumexp_action_chunks
from utils.chunk_progress import iter_user_action_blocks

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
        # Pre-materialize tensors so DataLoader does not call torch.tensor per row
        # (profile: millions of torch.tensor calls dominated wall time with workers=0).
        self.user_idx = torch.as_tensor(np.asarray(user_idx), dtype=torch.long)
        self.action_idx = torch.as_tensor(np.asarray(action_idx), dtype=torch.long)
        self.rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float64)
        self.pscore = torch.as_tensor(np.asarray(pscore), dtype=torch.float64)

    def __len__(self):
        return int(self.rewards.shape[0])

    def __getitem__(self, i):
        return (
            self.user_idx[i],
            self.action_idx[i],
            self.rewards[i],
            self.pscore[i],
        )

    def __getitems__(self, indices):
        """Whole batch in one indexing op (DataLoader calls this instead of per-row
        ``__getitem__``); same values/dtypes as default_collate over rows. Use with
        ``collate_fn=collate_prebatched``."""
        idx = torch.as_tensor(indices, dtype=torch.long)
        return [self.user_idx[idx], self.action_idx[idx], self.rewards[idx], self.pscore[idx]]


def collate_prebatched(batch):
    """DataLoader collate_fn for datasets whose ``__getitems__`` already returns a batch."""
    return batch

# ----------------------------
# Scalable environment
# ----------------------------
@dataclass
class SyntheticBanditEnv:
    """True click model on the clean vectors: q(u, a) = sigmoid(scale * x_u·a_a + offset).

    ``scale`` and ``offset`` come from the world calibration (utils.representation_bias);
    ``ctr`` is the calibration target CTR, kept for reporting (not a ceiling).
    """
    emb_x: np.ndarray  # (n_users, d) clean (centered) vectors
    emb_a: np.ndarray  # (n_actions, d)
    scale: float = 1.0
    offset: float = 0.0
    ctr: float = 0.05

    def reward_prob(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        dots = np.einsum("ij,ij->i", self.emb_x[users], self.emb_a[actions], dtype=np.float64)
        return expit(float(self.scale) * dots + float(self.offset))

    def reward_prob_block(self, users: np.ndarray, a0: int, a1: int) -> np.ndarray:
        """reward_prob for every (user in users) × (action in [a0, a1)); one matmul."""
        dots = (self.emb_x[users] @ self.emb_a[a0:a1].T).astype(np.float64)
        return expit(float(self.scale) * dots + float(self.offset))


def env_reward_block(env, users: np.ndarray, a0: int, a1: int) -> np.ndarray:
    """(len(users), a1 - a0) true reward probs; matmul fast path when the env has one."""
    if hasattr(env, "reward_prob_block"):
        return env.reward_prob_block(users, a0, a1)
    users_rep = np.repeat(users, a1 - a0)
    actions_rep = np.tile(np.arange(a0, a1, dtype=np.int64), len(users))
    return env.reward_prob(users_rep, actions_rep).reshape(len(users), a1 - a0)


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
# Materialize exact env q(u,a) when catalog fits (float32 cells). Same math as
# env.reward_prob; speeds repeated calc_reward without changing values.
EXACT_Q_CACHE_MAX_CELLS = 25_000_000


def ensure_exact_env_q_cache(
    dataset: dict,
    *,
    max_cells: int = EXACT_Q_CACHE_MAX_CELLS,
    user_chunk: int = 2048,
    action_chunk: int = 8192,
) -> bool:
    """Fill ``dataset['q_x_a']`` from ``env.reward_prob`` if catalog is small.

    Returns True when a dense exact q matrix is available afterward.
    """
    existing = dataset.get("q_x_a")
    if existing is not None:
        return True
    env = dataset.get("env")
    if env is None:
        return False
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    if n_users * n_actions > int(max_cells):
        return False

    t0 = time.time()
    q = np.empty((n_users, n_actions), dtype=np.float32)
    for u0 in range(0, n_users, int(user_chunk)):
        u1 = min(u0 + int(user_chunk), n_users)
        users = np.arange(u0, u1, dtype=np.int64)
        for a0 in range(0, n_actions, int(action_chunk)):
            a1 = min(a0 + int(action_chunk), n_actions)
            q[u0:u1, a0:a1] = env_reward_block(env, users, a0, a1)
    dataset["q_x_a"] = q
    print(
        f"[q_x_a] cached exact env rewards {n_users}x{n_actions} "
        f"({q.nbytes / 1e6:.1f} MB) in {time.time() - t0:.2f}s",
        flush=True,
    )
    return True


def calc_reward(dataset: dict, policy, chunk_size: int = 2048):
    """
    Exact policy value computation without materializing full dense matrix.

    Computes:
        V(pi) = sum_u prior[u] * sum_a pi(a|u) * q(u,a)

    Fully chunked over users for memory safety. When ``dataset['q_x_a']`` is
    present (small catalogs via ``ensure_exact_env_q_cache``), uses that matrix
    instead of re-calling ``env.reward_prob`` — identical math.
    """

    # -------------------------------
    # Case 1: Dense policy matrix
    # -------------------------------
    if isinstance(policy, np.ndarray):
        if "q_x_a" not in dataset or dataset["q_x_a"] is None:
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

    pt = max(float(getattr(policy, "temperature", 1.0)), 1e-8)
    device = _exact_reward_device(dataset)
    if device is not None:
        return _exact_value_torch(
            dataset, device, user_emb=policy.user_emb, item_emb=policy.item_emb,
            temperature=pt,
        )

    env = dataset["env"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    prior = _normalized_prior(dataset)
    q_cache = dataset.get("q_x_a")
    action_chunk = int(getattr(policy, "action_chunk", chunk_size))

    total_value = 0.0
    for start in range(0, n_users, int(chunk_size)):
        end = min(start + int(chunk_size), n_users)
        users = np.arange(start, end, dtype=np.int64)
        u = policy.user_emb[users]
        # normalizer depends only on the user block: compute once, not per action block
        log_denom = _logsumexp_action_chunks(u, policy.item_emb, pt, action_chunk)
        user_values = np.zeros(end - start, dtype=np.float64)
        for a0 in range(0, n_actions, action_chunk):
            a1 = min(a0 + action_chunk, n_actions)
            logits = (u @ policy.item_emb[a0:a1].T).astype(np.float64) / pt
            probs = np.exp(logits - log_denom[:, None])
            if q_cache is not None:
                rewards = np.asarray(q_cache[start:end, a0:a1], dtype=np.float64)
            else:
                rewards = env_reward_block(env, users, a0, a1)
            user_values += np.sum(probs * rewards, axis=1)
        total_value += np.sum(user_values * prior[users])

    return float(total_value)


def calc_uniform_reward(
    dataset: dict, *, user_chunk: int = 5000, action_chunk: int = 8192
) -> float:
    """Exact value of the uniform policy: sum_u prior[u] * mean_a q(u, a)."""
    if "env" not in dataset:
        raise ValueError("uniform policy reward needs dataset['env']")
    device = _exact_reward_device(dataset)
    if device is not None:
        return _exact_value_torch(dataset, device)

    env = dataset["env"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    prior = _normalized_prior(dataset)
    total = 0.0
    for start in range(0, n_users, int(user_chunk)):
        end = min(start + int(user_chunk), n_users)
        users = np.arange(start, end, dtype=np.int64)
        user_values = np.zeros(end - start, dtype=np.float64)
        for a0 in range(0, n_actions, int(action_chunk)):
            a1 = min(a0 + int(action_chunk), n_actions)
            user_values += env_reward_block(env, users, a0, a1).sum(axis=1) / float(n_actions)
        total += float(np.sum(user_values * prior[users]))
    return float(total)


# Exact full-catalog values on GPU. OPC_EXACT_REWARD_DEVICE: auto (default: CUDA if
# available) | cpu. Block size in cells (~3 fp32 blocks live, 32MB each); memory is
# released after every call.
EXACT_REWARD_DEVICE_ENV = "OPC_EXACT_REWARD_DEVICE"
EXACT_REWARD_GPU_BLOCK_CELLS = 8 * 1024 * 1024


def _normalized_prior(dataset: dict) -> np.ndarray:
    n_users = int(dataset["n_users"])
    prior = np.asarray(
        dataset.get("user_prior", np.ones(n_users, dtype=np.float64)), dtype=np.float64
    )
    return prior / prior.sum()


def _exact_reward_device(dataset: dict):
    """CUDA device for exact values, or None to use the numpy path."""
    if os.environ.get(EXACT_REWARD_DEVICE_ENV, "auto").strip().lower() == "cpu":
        return None
    if not torch.cuda.is_available():
        return None
    if dataset.get("q_x_a") is None and not isinstance(dataset.get("env"), SyntheticBanditEnv):
        return None
    gpu = int(os.environ.get("OPC_WORKER_GPU", "0"))
    if gpu < 0 or gpu >= torch.cuda.device_count():
        gpu = 0
    return torch.device(f"cuda:{gpu}")


@torch.no_grad()
def _exact_value_torch(
    dataset: dict,
    device,
    *,
    user_emb: np.ndarray | None = None,
    item_emb: np.ndarray | None = None,
    temperature: float = 1.0,
) -> float:
    """sum_u prior[u] * sum_a pi(a|u) q(u,a); softmax policy from embeddings, or uniform if None.

    Same math as the numpy path in fp32 (TF32 matmuls disabled), accumulated in fp64.
    """
    env = dataset["env"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    q_cache = dataset.get("q_x_a")
    rows = max(1, min(n_users, EXACT_REWARD_GPU_BLOCK_CELLS // max(n_actions, 1)))

    def t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=device)

    prev_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        prior = torch.as_tensor(_normalized_prior(dataset), device=device)
        if q_cache is not None:
            q_all = t(q_cache)
        else:
            env_x, env_a_t = t(env.emb_x), t(env.emb_a).T.contiguous()
            env_scale, env_offset = float(env.scale), float(env.offset)
        if user_emb is not None:
            pol_x, pol_a_t = t(user_emb), t(item_emb).T.contiguous()
            pt = max(float(temperature), 1e-8)

        total = torch.zeros((), dtype=torch.float64, device=device)
        for u0 in range(0, n_users, rows):
            u1 = min(u0 + rows, n_users)
            # In-place ops keep ~2 blocks live (same arithmetic as the out-of-place form).
            if q_cache is not None:
                q = q_all[u0:u1]
            else:
                q = (env_x[u0:u1] @ env_a_t).mul_(env_scale).add_(env_offset).sigmoid_()
            if user_emb is None:
                v = q.sum(dim=1, dtype=torch.float64) / float(n_actions)
            else:
                probs = torch.softmax((pol_x[u0:u1] @ pol_a_t).div_(pt), dim=1)
                v = probs.mul_(q).sum(dim=1, dtype=torch.float64)
                del probs
            del q
            total += (v * prior[u0:u1]).sum()
        return float(total)
    finally:
        torch.set_float32_matmul_precision(prev_precision)
        # Return blocks and uploaded embeddings to the driver: parallel workers share the GPU.
        q_all = env_x = env_a_t = pol_x = pol_a_t = prior = q = probs = None
        torch.cuda.empty_cache()


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



# ----------------------------
# Dataset generation
# ----------------------------
_LEGACY_NOISE_KEYS = {
    "eps1", "eps2", "eps_meta", "sigma1", "sigma2", "sigma_meta", "noise_mode", "noise_axis",
    "noise_component", "noise_apply_user", "noise_apply_item", "n_clusters", "policy_temperature",
}


def generate_dataset(params, seed=12345, emb_a=None, emb_x=None, metadata_a=None, metadata_x=None,
                     store_original: bool = True, dtype=np.float32, item_bias=None):
    """Simulated world for one condition (see ``utils.representation_bias.build_world``).

    ``params``: ``bias`` (a level, or 'warp/group/vector' levels; default 'medium'), ``ctr``
    (target CTR of the reference policy), ``best_ctr``, ``centering``, ``logging_spread``,
    ``ctr_reference`` ('logger' | 'uniform'), ``reference_bias``, ``group_source``
    ('cluster' | 'metadata'), ``logging_uniform_mix``, ``strict``, ``pop_strength`` (weight of
    ``item_bias``, BPR's b, in the true score; default 0), ``logger_pop_strength`` (the logger's
    weight; default: the same) and ``logger_greedy_share`` (the logger earns this share of its own
    greedy CTR; default 0.9, 0 / 'off' = the spread temperature). ``n_users``, ``n_actions``,
    ``emb_dim`` are only needed when ``emb_x`` / ``emb_a`` are not given (Gaussian vectors).
    ``store_original`` is kept for callers; the biased snapshot is always stored.
    """
    from utils.representation_bias import DEFAULT_LOGGER_GREEDY_SHARE, WorldConfig, build_world, parse_bias

    legacy = sorted(_LEGACY_NOISE_KEYS & set(params))
    if legacy:
        raise ValueError(
            f"legacy noise parameters {legacy} are no longer supported: the simulator uses "
            "representation-bias levels (params['bias']) and a calibrated logging temperature; "
            "see utils/representation_bias.py"
        )
    random_ = check_random_state(seed)
    if emb_a is not None:
        emb_a = np.load(emb_a) if isinstance(emb_a, str) else np.asarray(emb_a)
    else:
        emb_a = random_.normal(size=(params["n_actions"], params["emb_dim"])).astype(dtype)
    if emb_x is not None:
        emb_x = np.load(emb_x) if isinstance(emb_x, str) else np.asarray(emb_x)
    else:
        emb_x = random_.normal(size=(params["n_users"], params["emb_dim"])).astype(dtype)

    defaults = WorldConfig()
    if isinstance(item_bias, str):
        item_bias = np.load(item_bias)
    config = WorldConfig(
        centering=float(params.get("centering", defaults.centering)),
        pop_strength=float(params.get("pop_strength", defaults.pop_strength)),
        logging_spread=float(params.get("logging_spread", defaults.logging_spread)),
        target_ctr=float(params.get("ctr", defaults.target_ctr)),
        ctr_reference=str(params.get("ctr_reference", defaults.ctr_reference)),
        reference_bias=tuple(parse_bias(params.get("reference_bias", defaults.reference_bias)).values()),
        best_ctr=float(params.get("best_ctr", defaults.best_ctr)),
        group_source=str(params.get("group_source", defaults.group_source)),
        strict=bool(params.get("strict", defaults.strict)),
    )
    logger_pop = params.get("logger_pop_strength")
    return build_world(
        emb_x, emb_a, params.get("bias", "medium"), seed=int(seed), config=config,
        metadata_x=metadata_x, metadata_a=metadata_a,
        logging_uniform_mix=float(params.get("logging_uniform_mix", 0.0)),
        item_bias=item_bias, logger_pop_strength=None if logger_pop is None else float(logger_pop),
        logger_greedy_share=params.get("logger_greedy_share", DEFAULT_LOGGER_GREEDY_SHARE),
    )


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
def _estimator_weight_kwargs(weights) -> dict:
    """lambda_ (clip) or shrink_lambda for the OPE estimators from an importance-weight spec."""
    from utils.importance_weights import parse_weight_spec

    mode, param = parse_weight_spec("none" if weights is None else weights)
    if mode == "dm":
        raise ValueError("weights 'dm' are for DM-only trial selection, not for the post-hoc estimators")
    if mode == "clip":
        return {"lambda_": param}
    if mode == "shrink":
        return {"shrink_lambda": param}
    return {}


def eval_policy(model, test_data, original_policy_prob, policy, weights=None):
    """DM, DR, SNIPW and SNDR estimates on ``test_data``; ``weights`` (none, clip:M, shrink:lambda)
    transforms the importance weights of the three weighted estimators."""
    t0 = time.time()

    wkw = _estimator_weight_kwargs(weights)
    dr = DR(**wkw)
    dm = DM()
    ipw = IPW(**wkw)
    sndr = SNDR(**wkw)

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