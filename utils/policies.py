# policies.py
# Scalable policy objects used across trainers and simulation.
# These policies avoid materializing (n_users x n_actions) matrices.

from __future__ import annotations

import numpy as np


class CandidateSoftmaxDotPolicy:
    """Candidate-softmax policy using dot(user_emb, item_emb) over K candidates.

    - sample_actions(users) returns sampled actions and the (approx) probability of the sampled action.
    - prob_actions(users, actions) approximates p(a|u) by Monte Carlo over candidate sets.
    """

    def __init__(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        n_actions: int,
        K: int = 256,
        temperature: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.user_emb = np.asarray(user_emb)
        self.item_emb = np.asarray(item_emb)
        self.n_actions = int(n_actions)
        self.K = int(K)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng() if rng is None else rng

    def _candidates(self, n: int) -> np.ndarray:
        return self.rng.integers(0, self.n_actions, size=(n, self.K), endpoint=False)

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        n = users.shape[0]
        cand = self._candidates(n)  # (n, K)

        u = self.user_emb[users]  # (n, d)
        a = self.item_emb[cand]   # (n, K, d)

        logits = (u[:, None, :] * a).sum(axis=2) / max(self.temperature, 1e-8)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        cdf = np.cumsum(probs, axis=1)
        r = self.rng.random(size=n)[:, None]
        j = (cdf < r).sum(axis=1)
        j = np.minimum(j, self.K - 1)  
        actions = cand[np.arange(n), j]
        p_chosen = probs[np.arange(n), j]
        return actions.astype(np.int64), p_chosen.astype(np.float64)

    def prob_actions(self, users: np.ndarray, actions: np.ndarray, n_mc: int = 8) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        n = users.shape[0]
        u = self.user_emb[users]

        acc = np.zeros(n, dtype=np.float64)
        for _ in range(int(n_mc)):
            cand = self._candidates(n)
            in_set = (cand == actions[:, None])
            if not in_set.any():
                continue

            a = self.item_emb[cand]
            logits = (u[:, None, :] * a).sum(axis=2) / max(self.temperature, 1e-8)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)

            acc += (probs * in_set).sum(axis=1)

        return acc / max(int(n_mc), 1)


class CandidateRandomSoftmaxPolicy:
    """Scalable replacement for noise_p = softmax(N(0,1)) per user.

    Random logits over K candidates, softmax over K.
    """

    def __init__(
        self,
        n_actions: int,
        K: int = 256,
        rng: np.random.Generator | None = None,
    ):
        self.n_actions = int(n_actions)
        self.K = int(K)
        self.rng = np.random.default_rng() if rng is None else rng

    def _candidates(self, n: int) -> np.ndarray:
        return self.rng.integers(0, self.n_actions, size=(n, self.K), endpoint=False)

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        n = users.shape[0]
        cand = self._candidates(n)

        logits = self.rng.normal(size=(n, self.K))
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)

        cdf = np.cumsum(probs, axis=1)
        r = self.rng.random(size=n)[:, None]
        j = (cdf < r).sum(axis=1)
        j = np.minimum(j, self.K - 1)  
        actions = cand[np.arange(n), j]
        p_chosen = probs[np.arange(n), j]
        return actions.astype(np.int64), p_chosen.astype(np.float64)

    def prob_actions(self, users: np.ndarray, actions: np.ndarray, n_mc: int = 8) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        n = users.shape[0]

        acc = np.zeros(n, dtype=np.float64)
        for _ in range(int(n_mc)):
            cand = self._candidates(n)
            in_set = (cand == actions[:, None])
            if not in_set.any():
                continue

            logits = self.rng.normal(size=(n, self.K))
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / exp.sum(axis=1, keepdims=True)

            acc += (probs * in_set).sum(axis=1)

        return acc / max(int(n_mc), 1)


class MixturePolicy:
    """Mixture over component policies:

        p(a|u) = w0*p0(a|u) + wN*p_noise(a|u) + wO*p_oracle(a|u)

    Components must implement:
      - sample_actions(users) -> (actions, p_chosen)
      - prob_actions(users, actions) -> p(a|u)  (can be approximate)
    """

    def __init__(
        self,
        p0,
        p_noise,
        p_oracle,
        w0: float,
        wN: float,
        wO: float,
        rng: np.random.Generator | None = None,
    ):
        self.p0 = p0
        self.p_noise = p_noise
        self.p_oracle = p_oracle
        self.w0 = float(w0)
        self.wN = float(wN)
        self.wO = float(wO)
        self.rng = np.random.default_rng() if rng is None else rng

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        n = users.shape[0]

        w = np.array([self.w0, self.wN, self.wO], dtype=np.float64)
        w = w / w.sum()
        comp = self.rng.choice(3, size=n, p=w)

        actions = np.empty(n, dtype=np.int64)
        p = np.empty(n, dtype=np.float64)

        m0 = comp == 0
        mN = comp == 1
        mO = comp == 2

        if m0.any():
            a0, p0 = self.p0.sample_actions(users[m0])
            actions[m0] = a0
            p[m0] = self.w0 * p0

        if mN.any():
            aN, pN = self.p_noise.sample_actions(users[mN])
            actions[mN] = aN
            p[mN] = self.wN * pN

        if mO.any():
            aO, pO = self.p_oracle.sample_actions(users[mO])
            actions[mO] = aO
            p[mO] = self.wO * pO

        return actions, p

    def prob_actions(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        p0 = self.p0.prob_actions(users, actions)
        pN = self.p_noise.prob_actions(users, actions)
        pO = self.p_oracle.prob_actions(users, actions)
        return self.w0 * p0 + self.wN * pN + self.wO * pO


def generate_policies(
    num_policies: int,
    pi0_policy,
    oracle_policy,
    n_actions: int,
    use_random: bool = True,
    use_oracle: bool = True,
    jaws: bool = False,
    K_candidates: int = 256,
    seed: int = 12345,
):
    """Large-scale equivalent of the original generate_policies().

    Keeps the same alpha/beta/jaws mixing logic:
        pi_i = (1 - alpha - beta) * pi_0 + alpha * noise_p + beta * pi_oracle

    But returns policy *objects* (MixturePolicy) instead of dense matrices.
    """
    rng = np.random.default_rng(seed)
    noise_policy = CandidateRandomSoftmaxPolicy(n_actions=n_actions, K=K_candidates, rng=rng)

    policies = []
    for _ in range(int(num_policies)):
        alpha = float(rng.uniform(0, 1)) * (1.0 if use_random else 0.0)
        beta = float(rng.uniform(0, 1 - alpha)) * (1.0 if use_oracle else 0.0)

        if jaws:
            p = float(rng.uniform(0, 1))
            if p > 0.5:
                beta = 0.0
            else:
                alpha = 0.0
                beta = float(rng.uniform(0, 1))

        w0 = 1.0 - alpha - beta
        wN = alpha
        wO = beta

        policies.append(
            MixturePolicy(
                p0=pi0_policy,
                p_noise=noise_policy,
                p_oracle=oracle_policy,
                w0=w0,
                wN=wN,
                wO=wO,
                rng=rng,
            )
        )

    return policies
