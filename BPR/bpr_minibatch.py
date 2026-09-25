"""Mini-batch BPR (BPR v2): optional item bias, uniform or popularity-proportional negatives,
per-interaction or per-user sampling, and early stopping on one held-out liked item per user.

    score(u, i) = b_i + x_u · a_i                                 (b_i only with item_bias)
    loss(u,i,j) = −ln σ(score(u, i) − score(u, j))
                  + (λ/2) (‖x_u‖² + ‖a_i‖² + ‖a_j‖²) + (λ_b/2) (b_i² + b_j²)

for triples (u, i, j): item i liked by user u, item j not. Each batch sums the per-triple
gradients of every touched row and takes one Adagrad step per row, so a popular item drawn
many times in one batch does not take one huge step. Numpy only: the same seed gives the same
vectors on any machine, with or without a GPU.

Early stopping holds out one random liked item for each user with at least
``val_min_positives`` liked items, tracks recall@k after every epoch (one pass over the
training interactions), keeps the best epoch, and by default refits on all interactions for
that many epochs.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

NEGATIVES = ("uniform", "popularity")
SAMPLING = ("interaction", "user")


@dataclass
class BPRConfig:
    factors: int = 32
    item_bias: bool = True
    negatives: str = "uniform"            # "uniform" or "popularity" (∝ n_j ** negative_gamma)
    negative_gamma: float = 0.75
    sampling: str = "interaction"         # liked items: every interaction equally likely, or every user
    learning_rate: float = 0.1            # Adagrad step size (best on ml and kuairand held-out recall)
    regularization: float = 1e-4          # λ on the vectors, per sampled occurrence
    bias_regularization: float = 1e-4     # λ_b on the item biases
    batch_size: int = 8192
    init_scale: float = 0.01
    adagrad_init: float = 0.1             # initial Adagrad accumulator
    early_stopping: bool = True
    max_epochs: int = 100
    patience: int = 5                     # epochs without a min_rel_improvement gain before stopping
    min_rel_improvement: float = 1e-3
    val_min_positives: int = 3
    val_users: int = 5000
    eval_k: int = 20
    refit: bool = True                    # after early stopping, retrain on all interactions
    epochs: int = 20                      # fixed epochs when early_stopping is off
    random_state: int = 0

    def __post_init__(self):
        if self.negatives not in NEGATIVES:
            raise ValueError(f"negatives must be one of {NEGATIVES}, got {self.negatives!r}")
        if self.sampling not in SAMPLING:
            raise ValueError(f"sampling must be one of {SAMPLING}, got {self.sampling!r}")
        for name in ("factors", "batch_size", "max_epochs", "patience", "val_users", "eval_k", "epochs"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.learning_rate <= 0 or self.adagrad_init <= 0:
            raise ValueError("learning_rate and adagrad_init must be > 0")

    @classmethod
    def from_dict(cls, params: dict) -> "BPRConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown BPR settings {unknown}; known: {sorted(known)}")
        return cls(**params)


# --------------------------------------------------------------------------- data helpers
def _binary_csr(X) -> csr_matrix:
    X = csr_matrix(X, dtype=np.float32, copy=True)
    X.sum_duplicates()
    X.sort_indices()
    X.eliminate_zeros()
    X.data[:] = 1.0
    return X


def data_fingerprint(X) -> str:
    """Short hash of the interaction structure (shape and liked-item indices)."""
    X = _binary_csr(X)
    h = hashlib.sha1()
    h.update(str(X.shape).encode())
    h.update(np.ascontiguousarray(X.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(X.indices, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def holdout_split(X, *, min_positives: int, seed: int):
    """Hold out one random liked item per user with >= min_positives liked items.

    Returns (X_train, users, items): the held-out pair (users[k], items[k]) is not in X_train.
    """
    X = _binary_csr(X)
    rng = np.random.default_rng(seed)
    deg = np.diff(X.indptr).astype(np.int64)
    users = np.flatnonzero(deg >= int(min_positives)).astype(np.int64)
    pick = X.indptr[users].astype(np.int64) + (rng.random(len(users)) * deg[users]).astype(np.int64)
    items = X.indices[pick].astype(np.int64)
    keep = np.ones(X.nnz, dtype=bool)
    keep[pick] = False
    rows = np.repeat(np.arange(X.shape[0], dtype=np.int64), deg)
    X_train = csr_matrix((X.data[keep], (rows[keep], X.indices[keep])), shape=X.shape)
    X_train.sort_indices()
    return X_train, users, items


def validation_split(X, config: BPRConfig):
    """The early-stopping split for a config: (X_train, users, items), at most val_users users."""
    X_train, vu, vi = holdout_split(X, min_positives=config.val_min_positives,
                                    seed=int(np.random.SeedSequence([int(config.random_state), 3]).generate_state(1)[0]))
    if len(vu) > config.val_users:
        pick = np.sort(np.random.default_rng(np.random.SeedSequence([int(config.random_state), 4]))
                       .choice(len(vu), size=config.val_users, replace=False))
        vu, vi = vu[pick], vi[pick]
    return X_train, vu, vi


def evaluate_ranking(U, V, b, X_train, users, items, k: int = 20) -> dict:
    """recall@k (= hit rate, one held-out item per user) and NDCG@k, ranking only items the
    user did not like in training."""
    users = np.asarray(users, dtype=np.int64)
    items = np.asarray(items, dtype=np.int64)
    n_items = V.shape[0]
    rows_per_block = max(1, min(1024, 4_000_000 // max(n_items, 1)))
    X_train = X_train.tocsr()
    hits = ndcg = 0.0
    for s in range(0, len(users), rows_per_block):
        uu, ii = users[s : s + rows_per_block], items[s : s + rows_per_block]
        S = U[uu] @ V.T
        if b is not None:
            S += b[None, :]
        sub = X_train[uu]
        S[np.repeat(np.arange(len(uu)), np.diff(sub.indptr)), sub.indices] = -np.inf
        target = S[np.arange(len(uu)), ii]
        rank = (S > target[:, None]).sum(axis=1)
        hit = rank < k
        hits += float(hit.sum())
        ndcg += float((hit / np.log2(rank + 2.0)).sum())
    n = max(len(users), 1)
    return {"recall": hits / n, "ndcg": ndcg / n}


def _segment_sum(idx: np.ndarray, vals: np.ndarray):
    """Unique indices and the sum of vals per index (deterministic order)."""
    order = np.argsort(idx, kind="stable")
    s_idx = idx[order]
    starts = np.flatnonzero(np.r_[True, s_idx[1:] != s_idx[:-1]])
    return s_idx[starts], np.add.reduceat(vals[order], starts, axis=0)


class TripleSampler:
    """Draws (user, liked item, negative item) triples from a binary CSR matrix."""

    def __init__(self, X, config: BPRConfig, rng: np.random.Generator):
        X = _binary_csr(X)
        self.config, self.rng = config, rng
        self.n_users, self.n_items = X.shape
        self.indptr = X.indptr.astype(np.int64)
        self.indices = X.indices.astype(np.int64)
        self.deg = np.diff(self.indptr)
        self.row_of = np.repeat(np.arange(self.n_users, dtype=np.int64), self.deg)
        self.keys = self.row_of * self.n_items + self.indices  # sorted: rows, then items
        self.active = np.flatnonzero(self.deg > 0)
        self.item_counts = np.bincount(self.indices, minlength=self.n_items)
        self.weights = self.cdf = None
        if config.negatives == "popularity":
            self.weights = self.item_counts.astype(np.float64) ** float(config.negative_gamma)
            self.cdf = np.cumsum(self.weights / self.weights.sum())
            self.cdf[-1] = 1.0
        self.dropped = 0  # triples dropped because the user likes every item

    def positives(self, n: int):
        if self.config.sampling == "interaction":
            idx = self.rng.integers(0, len(self.indices), size=n)
            return self.row_of[idx], self.indices[idx]
        u = self.active[self.rng.integers(0, len(self.active), size=n)]
        offset = (self.rng.random(n) * self.deg[u]).astype(np.int64)
        return u, self.indices[self.indptr[u] + offset]

    def _draw(self, n: int) -> np.ndarray:
        if self.cdf is None:
            return self.rng.integers(0, self.n_items, size=n)
        return np.minimum(np.searchsorted(self.cdf, self.rng.random(n), side="right"), self.n_items - 1)

    def is_liked(self, u: np.ndarray, j: np.ndarray) -> np.ndarray:
        keys = u * self.n_items + j
        order = np.argsort(keys, kind="stable")  # sorted queries: far fewer cache misses
        sorted_keys = keys[order]
        pos = np.minimum(np.searchsorted(self.keys, sorted_keys), len(self.keys) - 1)
        liked = np.empty(len(keys), dtype=bool)
        liked[order] = self.keys[pos] == sorted_keys
        return liked

    def _draw_unliked(self, user: int):
        """One negative from the items the user does not like (exact; for heavy users)."""
        allowed = np.ones(self.n_items, dtype=bool)
        allowed[self.indices[self.indptr[user] : self.indptr[user + 1]]] = False
        if self.weights is None:
            cand = np.flatnonzero(allowed)
            if len(cand) == 0:
                return 0, False
            return int(cand[self.rng.integers(len(cand))]), True
        w = np.where(allowed, self.weights, 0.0)
        total = w.sum()
        if total <= 0:
            return 0, False
        cdf = np.cumsum(w / total)
        return int(min(np.searchsorted(cdf, self.rng.random(), side="right"), self.n_items - 1)), True

    def negatives(self, u: np.ndarray):
        """Negatives for users u, redrawn while liked; returns (j, keep mask).

        Only the redrawn entries are re-checked. After 100 rounds the few left (users who
        like almost everything) are drawn from their unliked items directly, which is the
        same distribution rejection would reach.
        """
        j = self._draw(len(u))
        todo = np.flatnonzero(self.is_liked(u, j))
        for _ in range(100):
            if len(todo) == 0:
                break
            j[todo] = self._draw(len(todo))
            todo = todo[self.is_liked(u[todo], j[todo])]
        keep = np.ones(len(u), dtype=bool)
        for pos in todo:
            j[pos], keep[pos] = self._draw_unliked(int(u[pos]))
        self.dropped += int((~keep).sum())
        return j, keep


# --------------------------------------------------------------------------- model
class MiniBatchBPR:
    def __init__(self, config: BPRConfig | None = None, **overrides):
        self.config = config or BPRConfig(**overrides)
        self.user_factors = self.item_factors = self.item_bias = None
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.epochs_trained = 0
        self.n_val_users = 0
        self.dropped_negatives = 0
        self.seconds = {}

    # parameters -------------------------------------------------------------
    def _init_params(self, n_users: int, n_items: int):
        c = self.config
        rng = np.random.default_rng(np.random.SeedSequence([int(c.random_state), 1]))
        self.U = (c.init_scale * rng.standard_normal((n_users, c.factors))).astype(np.float32)
        self.V = (c.init_scale * rng.standard_normal((n_items, c.factors))).astype(np.float32)
        self.b = np.zeros(n_items, dtype=np.float32) if c.item_bias else None
        self.GU = np.full_like(self.U, c.adagrad_init)
        self.GV = np.full_like(self.V, c.adagrad_init)
        self.Gb = np.full(n_items, c.adagrad_init, dtype=np.float32) if c.item_bias else None

    def _adagrad(self, P, G, rows, grad):
        G[rows] += grad * grad
        P[rows] -= self.config.learning_rate * grad / np.sqrt(G[rows])

    def batch_gradients(self, u, i, j):
        """Per-triple gradients of the loss (for the update and for gradient checks)."""
        c = self.config
        xu, ai, aj = self.U[u], self.V[i], self.V[j]
        diff = ai - aj
        x = np.einsum("ij,ij->i", xu, diff)
        if self.b is not None:
            x = x + self.b[i] - self.b[j]
        g = expit(-x)  # −d loss / d x; keeps the parameters' float type
        lam = c.regularization
        grads = {
            "u": -g[:, None] * diff + lam * xu,
            "i": -g[:, None] * xu + lam * ai,
            "j": g[:, None] * xu + lam * aj,
        }
        if self.b is not None:
            lb = c.bias_regularization
            grads["bi"] = -g + lb * self.b[i]
            grads["bj"] = g + lb * self.b[j]
        return grads, float(np.mean(np.logaddexp(0.0, -x)))

    def batch_loss(self, u, i, j) -> float:
        """Total loss of a batch of triples, with the same per-occurrence penalty as the update."""
        c = self.config
        xu, ai, aj = self.U[u].astype(np.float64), self.V[i].astype(np.float64), self.V[j].astype(np.float64)
        x = np.einsum("ij,ij->i", xu, ai - aj)
        pen = c.regularization / 2 * (np.sum(xu**2) + np.sum(ai**2) + np.sum(aj**2))
        if self.b is not None:
            bi, bj = self.b[i].astype(np.float64), self.b[j].astype(np.float64)
            x = x + bi - bj
            pen += c.bias_regularization / 2 * (np.sum(bi**2) + np.sum(bj**2))
        return float(np.sum(np.logaddexp(0.0, -x)) + pen)

    def _step(self, u, i, j) -> float:
        grads, loss = self.batch_gradients(u, i, j)
        rows, grad = _segment_sum(u, grads["u"])
        self._adagrad(self.U, self.GU, rows, grad)
        items = np.concatenate([i, j])
        rows, grad = _segment_sum(items, np.concatenate([grads["i"], grads["j"]]))
        self._adagrad(self.V, self.GV, rows, grad)
        if self.b is not None:
            rows, grad = _segment_sum(items, np.concatenate([grads["bi"], grads["bj"]]))
            self._adagrad(self.b, self.Gb, rows, grad)
        return loss

    def _train(self, X, n_epochs: int, *, eval_set=None, log=None):
        """Train from fresh parameters; with eval_set, stop early and keep the best epoch."""
        c = self.config
        n_users, n_items = X.shape
        self._init_params(n_users, n_items)
        sampler = TripleSampler(X, c, np.random.default_rng(np.random.SeedSequence([int(c.random_state), 2])))
        n_batches = math.ceil(X.nnz / c.batch_size)
        best, best_metric, stale, history = None, -1.0, 0, []
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            losses = []
            for _ in range(n_batches):
                u, i = sampler.positives(c.batch_size)
                j, keep = sampler.negatives(u)
                if not keep.all():
                    u, i, j = u[keep], i[keep], j[keep]
                losses.append(self._step(u, i, j))
            row = {"epoch": epoch, "triples": epoch * n_batches * c.batch_size, "loss": float(np.mean(losses)),
                   "train_seconds": time.time() - t0}
            if eval_set is not None:
                X_train, vu, vi = eval_set
                t1 = time.time()
                row.update(evaluate_ranking(self.U, self.V, self.b, X_train, vu, vi, c.eval_k))
                row["eval_seconds"] = time.time() - t1
                if row["recall"] > best_metric * (1.0 + c.min_rel_improvement):
                    best_metric, stale = row["recall"], 0
                    best = (epoch, self.U.copy(), self.V.copy(), None if self.b is None else self.b.copy())
                else:
                    stale += 1
            history.append(row)
            if log:
                extra = f" recall@{c.eval_k} {row['recall']:.4f} ndcg@{c.eval_k} {row['ndcg']:.4f}" if eval_set is not None else ""
                log(f"epoch {epoch:3d} loss {row['loss']:.4f}{extra} ({row['train_seconds']:.1f}s)")
            if eval_set is not None and stale >= c.patience:
                break
        self.dropped_negatives += sampler.dropped
        return history, best

    def fit(self, X, log=print) -> "MiniBatchBPR":
        c = self.config
        X = _binary_csr(X)
        t0 = time.time()
        X_train = vu = vi = None
        if c.early_stopping:
            X_train, vu, vi = validation_split(X, c)
            if len(vu) == 0 and log:
                log(f"no user has {c.val_min_positives}+ liked items to hold out: training {c.epochs} fixed epochs instead")
        if c.early_stopping and len(vu) > 0:
            self.n_val_users = int(len(vu))
            self.history, best = self._train(X_train, c.max_epochs, eval_set=(X_train, vu, vi), log=log)
            self.best_epoch = int(best[0])
            self.seconds["search"] = time.time() - t0
            if c.refit:
                if log:
                    log(f"refit on all {X.nnz:,} interactions for {self.best_epoch} epochs")
                t1 = time.time()
                self._train(X, self.best_epoch, log=log)
                self.seconds["refit"] = time.time() - t1
            else:
                _, self.U, self.V, self.b = best
            self.epochs_trained = self.best_epoch
        else:
            self.history, _ = self._train(X, c.epochs, log=log)
            self.epochs_trained = c.epochs
            self.seconds["train"] = time.time() - t0
        self.user_factors, self.item_factors, self.item_bias = self.U, self.V, self.b
        return self

    def summary(self) -> dict:
        """JSON-able description of the run (for {ds}_bpr_meta.json)."""
        best = next((h for h in self.history if h["epoch"] == self.best_epoch), None)
        return {
            "trainer": "minibatch_adagrad",
            "settings": asdict(self.config),
            "epochs_trained": int(self.epochs_trained),
            "best_epoch": self.best_epoch,
            "validation": None if best is None else {
                "users": self.n_val_users, "k": self.config.eval_k,
                "recall": best["recall"], "ndcg": best["ndcg"],
            },
            "history": self.history,
            "dropped_negatives": int(self.dropped_negatives),
            "seconds": {k: round(v, 1) for k, v in self.seconds.items()},
        }

    def save_embeddings(self, user_path: str, item_path: str, bias_path: str | None = None):
        np.save(user_path, self.user_factors.astype(np.float32))
        np.save(item_path, self.item_factors.astype(np.float32))
        if bias_path is not None and self.item_bias is not None:
            np.save(bias_path, self.item_bias.astype(np.float32))
