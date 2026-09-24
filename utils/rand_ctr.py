"""Monte Carlo estimate of rand_CTR (the uniform random policy's CTR) from the simulator.

The world calibration (utils.representation_bias, ``ctr_reference='uniform'``) sets it;
``utils.simulation_utils.calc_uniform_reward`` gives the exact value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from utils.rand_ctr_sample_size import density_regime


def estimate_rand_ctr(
    dataset: dict[str, Any],
    *,
    n_samples: int,
    seed: int = 0,
    chunk_size: int = 100_000,
) -> dict[str, float]:
    """
    Monte Carlo estimate of rho = E[r | x~Unif(users), a~Unif(actions)].

    Uses the simulator env (true reward probabilities).
    """
    env = dataset["env"]
    n_users = int(dataset["n_users"])
    n_actions = int(dataset["n_actions"])
    rng = np.random.default_rng(int(seed))

    n_samples = int(max(1, n_samples))
    sum_r = 0.0
    sum_q = 0.0
    done = 0
    while done < n_samples:
        b = min(chunk_size, n_samples - done)
        users = rng.integers(0, n_users, size=b, dtype=np.int64)
        actions = rng.integers(0, n_actions, size=b, dtype=np.int64)
        q = env.reward_prob(users, actions)
        r = (q > rng.random(b)).astype(np.float64)
        sum_r += float(r.sum())
        sum_q += float(q.sum())
        done += b

    rho_hat = sum_r / n_samples
    q_mean = sum_q / n_samples
    return {
        "rand_ctr": float(rho_hat),
        "rand_q_mean": float(q_mean),
        "n_rand_samples": float(n_samples),
        "density_regime": density_regime(rho_hat),
    }
