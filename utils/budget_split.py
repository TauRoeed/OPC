"""Fixed target-interaction budget: warm-logger (control) rows plus uniform-random (treatment) rows.

For a total budget N and a randomized share rho (docs/cause_baseline.md §4.1):
    N_t = round(rho * N) uniform rows and N_c = N - N_t warm rows, so N_c + N_t = N exactly.
* Warm rows: the first N_c rows of the warm logger's training split for size N, i.e. the rows every other arm
  trains on. That split is a random partition of one simulation, so a prefix is a uniformly random subset. The
  rows keep their exact logger propensities.
* Uniform rows: the first N_t rows of a separate simulation of N rows under pi(a|u) = 1/|A| from the same world
  (users from the same prior, rewards from the same click model), with its own derived seeds; pscore = 1/|A|.
Both parts are nested in rho. Every part records its collection policy, its realised reward sum and its
expected reward sum (sum of the true q(u, a) of the rows), for the cost of exploration.
"""
from __future__ import annotations

import numpy as np

from utils.seeding import derive_seed
from utils.simulation_utils import create_simulation_data_from_policy, get_train_data

WARM_POLICY = "warm_logger"
UNIFORM_POLICY = "uniform"
CAUSE_RHOS = (0.0, 0.01, 0.05, 0.10, 0.15, 0.25)
ROW_KEYS = ("x", "a", "r", "x_idx", "pscore", "q")


class UniformLoggingPolicy:
    """pi(a|u) = 1/|A|: one uniform integer per row from its own generator; pscore is exactly 1/|A|."""

    def __init__(self, n_items: int, rng: np.random.Generator):
        self.n_items = int(n_items)
        self.rng = rng

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(users)
        return self.rng.integers(0, self.n_items, size=n, dtype=np.int64), np.full(n, 1.0 / self.n_items)

    def prob_actions(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.full(len(users), 1.0 / self.n_items)


def budget_counts(n_total: int, rho: float) -> tuple[int, int]:
    """(N_c, N_t) with N_t = round(rho * N) and N_c = N - N_t."""
    rho = float(rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    n_total = int(n_total)
    n_t = int(round(rho * n_total))
    return n_total - n_t, n_t


def uniform_pool_seed(condition_seed: int, train_size: int) -> int:
    return derive_seed(int(condition_seed), "budget_uniform_pool", int(train_size))


def expected_rewards(dataset: dict, users, actions) -> np.ndarray:
    """True q(u, a) of the given rows."""
    users = np.asarray(users, dtype=np.int64)
    actions = np.asarray(actions, dtype=np.int64)
    q = dataset.get("q_x_a")
    if q is not None:
        return np.asarray(q[users, actions], dtype=np.float64)
    return np.asarray(dataset["env"].reward_prob(users, actions), dtype=np.float64)


def simulate_uniform_pool(dataset: dict, n_rows: int, seed: int) -> dict:
    """``n_rows`` logged rows under the uniform logger, in the format of the warm training rows (+ ``q``)."""
    n_actions = int(dataset["n_actions"])
    policy = UniformLoggingPolicy(n_actions, np.random.default_rng(derive_seed(int(seed), "uniform_actions")))
    sim = create_simulation_data_from_policy(dataset, policy, int(n_rows), random_state=int(seed))
    data = get_train_data(n_actions, int(n_rows), sim, np.arange(int(n_rows)), dataset["our_x"])
    data["q"] = expected_rewards(dataset, data["x_idx"], data["a"])
    data["collection_policy"] = UNIFORM_POLICY
    return data


def take_rows(data: dict, start: int, stop: int, policy: str) -> dict:
    out = {k: np.asarray(data[k])[start:stop] for k in ROW_KEYS if k in data}
    out["num_data"] = int(stop - start)
    out["num_actions"] = int(data["num_actions"])
    out["collection_policy"] = policy
    return out


def collection_record(rows: dict, prefix: str) -> dict:
    return {f"{prefix}_rows": int(len(rows["a"])),
            f"{prefix}_reward_sum": float(np.sum(rows["r"])),
            f"{prefix}_expected_reward_sum": float(np.sum(rows["q"]))}


def build_budget_split(dataset: dict, warm_train: dict, uniform_pool: dict, n_total: int, rho: float) -> dict:
    """{"control": warm rows, "treatment": uniform rows, "meta": counts, policies and collection rewards}."""
    n_c, n_t = budget_counts(n_total, rho)
    if len(warm_train["a"]) != int(n_total):
        raise ValueError(f"the warm split has {len(warm_train['a'])} rows, the budget is {n_total}")
    if len(uniform_pool["a"]) < n_t:
        raise ValueError(f"the uniform pool has {len(uniform_pool['a'])} rows, {n_t} are needed")
    warm = dict(warm_train)
    if "q" not in warm:
        warm["q"] = expected_rewards(dataset, warm["x_idx"], warm["a"])
    control = take_rows(warm, 0, n_c, WARM_POLICY)
    treatment = take_rows(uniform_pool, 0, n_t, UNIFORM_POLICY)
    replaced = take_rows(warm, n_c, int(n_total), WARM_POLICY)  # warm rows the other arms see instead
    n_actions = int(dataset["n_actions"])
    meta = {
        "rho": float(rho), "n_total": int(n_total), "n_control": n_c, "n_treatment": n_t,
        "control_policy": WARM_POLICY, "treatment_policy": UNIFORM_POLICY, "uniform_pscore": 1.0 / n_actions,
        **collection_record(control, "control"), **collection_record(treatment, "treatment"),
        **collection_record(replaced, "replaced_warm"),
        "treatment_rows_per_item": n_t / n_actions,
        "treatment_items_covered": int(len(np.unique(treatment["a"]))),
        "treatment_users": int(len(np.unique(treatment["x_idx"]))),
    }
    meta["collection_reward_sum"] = meta["control_reward_sum"] + meta["treatment_reward_sum"]
    meta["collection_expected_reward_sum"] = meta["control_expected_reward_sum"] + meta["treatment_expected_reward_sum"]
    return {"control": control, "treatment": treatment, "meta": meta}
