"""Estimate rand_CTR and calibrate simulator link parameter ctr from scores."""

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


def mean_oracle_q_under_uniform(
    emb_x: np.ndarray,
    emb_a: np.ndarray,
    *,
    ctr: float,
    n_samples: int = 50_000,
    seed: int = 0,
    temperature: float = 1.0,
) -> float:
    """E[q_oracle(x,a)] for uniform (x,a) under the CTR link."""
    rng = np.random.default_rng(seed)
    n_u, n_a = emb_x.shape[0], emb_a.shape[0]
    users = rng.integers(0, n_u, size=n_samples)
    actions = rng.integers(0, n_a, size=n_samples)
    pt = max(float(temperature), 1e-8)
    logits = (emb_x[users] * emb_a[actions]).sum(axis=1) / pt
    q = 1.0 / ((1.0 / float(ctr)) + np.exp(-logits))
    return float(np.mean(q))


def calibrate_ctr_from_rand(
    emb_x: np.ndarray,
    emb_a: np.ndarray,
    *,
    target_rand_ctr: float,
    ctr_grid: np.ndarray | None = None,
    n_samples: int = 50_000,
    seed: int = 0,
    temperature: float = 1.0,
) -> dict[str, float]:
    """
    Pick ctr so E[q_ctr(x,a)] under uniform (x,a) matches target_rand_ctr.

    Grid search over ctr (monotone in ctr for fixed scores).
    """
    target = float(target_rand_ctr)
    if not (0.0 < target < 1.0):
        raise ValueError(f"target_rand_ctr must be in (0,1), got {target}")
    if ctr_grid is None:
        ctr_grid = np.linspace(0.01, 0.95, 48)

    best_ctr = float(ctr_grid[0])
    best_err = float("inf")
    for c in ctr_grid:
        m = mean_oracle_q_under_uniform(
            emb_x,
            emb_a,
            ctr=float(c),
            n_samples=n_samples,
            seed=seed,
            temperature=temperature,
        )
        err = abs(m - target)
        if err < best_err:
            best_err = err
            best_ctr = float(c)
    return {
        "ctr_calibrated": best_ctr,
        "calibration_error": float(best_err),
        "target_rand_ctr": target,
    }
