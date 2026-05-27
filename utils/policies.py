from __future__ import annotations
import numpy as np


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _sample_categorical_rows(probs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0  # prevents rare j==A edge case
    r = rng.random(size=probs.shape[0])[:, None]
    j = (cdf < r).sum(axis=1)
    p_chosen = probs[np.arange(probs.shape[0]), j]
    return j.astype(np.int64), p_chosen.astype(np.float64)


class Policy:
    """
    Exact full-softmax dot-product policy over ALL items:
      logits(u,a) = (user_emb[u] · item_emb[a]) / temperature

    Temperature scales logits before softmax (default 1.0). Full-study runs pass
    ``policy_temperature`` via ``generate_dataset`` into ``dataset["policy_temperature"]``.

    You can set embeddings per run. If embeddings are None, they are generated randomly.

    - sample_actions(users): samples a ~ pi(.|u) and returns exact p(a|u)
    - prob_actions(users, actions): returns exact pi(actions[i]|users[i]) for logged actions

    No candidate sets. No Monte-Carlo. Chunked by users for memory safety.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_emb: np.ndarray | None = None,
        item_emb: np.ndarray | None = None,
        emb_dim: int = 1,
        temperature: float = 1.0,
        user_chunk: int = 1024,
        rng: np.random.Generator | None = None,
    ):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.emb_dim = int(emb_dim)
        self.temperature = float(temperature)
        self.user_chunk = int(user_chunk)
        self.rng = np.random.default_rng() if rng is None else rng

        self.set_embeddings(user_emb=user_emb, item_emb=item_emb)

    def set_embeddings(self, user_emb: np.ndarray | None, item_emb: np.ndarray | None) -> None:
        """
        Set/replace embeddings. If None, generate random embeddings.
        Intended use: call this once per run to swap in base/oracle/noise embeddings.
        """
        if user_emb is None:
            user_emb = self.rng.normal(size=(self.n_users, self.emb_dim)).astype(np.float32)
        else:
            user_emb = np.asarray(user_emb, dtype=np.float32)
            assert user_emb.shape[0] == self.n_users, "user_emb first dim must be n_users"

        if item_emb is None:
            item_emb = self.rng.normal(size=(self.n_items, self.emb_dim)).astype(np.float32)
        else:
            item_emb = np.asarray(item_emb, dtype=np.float32)
            assert item_emb.shape[0] == self.n_items, "item_emb first dim must be n_items"

        assert user_emb.ndim == 2 and item_emb.ndim == 2, "embeddings must be 2D arrays"
        assert user_emb.shape[1] == item_emb.shape[1], "user_emb dim must match item_emb dim"

        self.user_emb = user_emb
        self.item_emb = item_emb

    def _logits_block(self, users_block: np.ndarray) -> np.ndarray:
        u = self.user_emb[users_block]                       # (b, d)
        logits = (u @ self.item_emb.T).astype(np.float64)     # (b, A)
        logits /= max(self.temperature, 1e-8)
        return logits

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        n = users.shape[0]
        actions_out = np.empty(n, dtype=np.int64)
        p_out = np.empty(n, dtype=np.float64)

        for start in range(0, n, self.user_chunk):
            end = min(start + self.user_chunk, n)
            ub = users[start:end]
            logits = self._logits_block(ub)       # (b, A)
            probs = _softmax_rows(logits)         # (b, A)
            a, p = _sample_categorical_rows(probs, self.rng)
            actions_out[start:end] = a
            p_out[start:end] = p

        return actions_out, p_out

    def prob_actions(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Exact pi(actions[i] | users[i]) under full softmax over ALL items.
        Deterministic: no randomness.
        """
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        assert users.shape == actions.shape, "users/actions must be same shape"
        n = users.shape[0]
        out = np.empty(n, dtype=np.float64)

        for start in range(0, n, self.user_chunk):
            end = min(start + self.user_chunk, n)
            ub = users[start:end]
            ab = actions[start:end]
            logits = self._logits_block(ub)  # (b, A)

            # stable logsumexp per row
            m = logits.max(axis=1, keepdims=True)
            lse = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))).squeeze(1)  # (b,)

            logit_a = logits[np.arange(end - start), ab]  # (b,)
            out[start:end] = np.exp(logit_a - lse)

        return out


class MixturePolicy:
    """
    Exact stable mixture:
      pi_mix(a|u) = w0*pi0(a|u) + wN*piN(a|u) + wO*piO(a|u)

    Sampling chooses a component then samples from it.
    prob_actions is exact weighted sum.
    """

    def __init__(
        self,
        p0: Policy,
        pN: Policy,
        pO: Policy,
        w0: float,
        wN: float,
        wO: float,
        rng: np.random.Generator | None = None,
    ):
        w = np.array([w0, wN, wO], dtype=np.float64)
        assert np.all(w >= 0), "weights must be nonnegative"
        s = w.sum()
        assert s > 0, "weights must sum to > 0"
        w /= s
        self.w0, self.wN, self.wO = float(w[0]), float(w[1]), float(w[2])
        self.p0, self.pN, self.pO = p0, pN, pO
        self.rng = np.random.default_rng() if rng is None else rng

    def sample_actions(self, users: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        users = np.asarray(users, dtype=np.int64)
        n = users.shape[0]
        comp = self.rng.choice(3, size=n, p=[self.w0, self.wN, self.wO])

        actions = np.empty(n, dtype=np.int64)

        m0 = comp == 0
        mN = comp == 1
        mO = comp == 2

        if m0.any():
            actions[m0], _ = self.p0.sample_actions(users[m0])
        if mN.any():
            actions[mN], _ = self.pN.sample_actions(users[mN])
        if mO.any():
            actions[mO], _ = self.pO.sample_actions(users[mO])

        # exact mixture propensity for chosen actions
        p = self.prob_actions(users, actions)
        return actions, p

    def prob_actions(self, users: np.ndarray, actions: np.ndarray) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        p0 = self.p0.prob_actions(users, actions)
        pN = self.pN.prob_actions(users, actions)
        pO = self.pO.prob_actions(users, actions)
        return self.w0 * p0 + self.wN * pN + self.wO * pO


def generate_policies(
    num_policies: int,
    base_policy: Policy,
    oracle_policy: Policy,
    noise_policy: Policy,
    use_random: bool = True,
    use_oracle: bool = True,
    jaws: bool = False,
    seed: int = 12345,
) -> list[MixturePolicy]:
    """
    Same alpha/beta/jaws logic as your original generate_policies(),
    but returns MixturePolicy objects with exact propensities.

    noise_policy can be re-initialized per run (fixed within run).
    """
    rng = np.random.default_rng(seed)
    policies: list[MixturePolicy] = []

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

        policies.append(MixturePolicy(base_policy, noise_policy, oracle_policy, w0, wN, wO, rng=rng))

    return policies
