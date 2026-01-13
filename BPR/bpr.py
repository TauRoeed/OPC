import numpy as np
from numpy.random import default_rng
from scipy.sparse import csr_matrix


import numpy as np
from numpy.random import default_rng
from scipy.sparse import csr_matrix
from tqdm.auto import trange


def topk_similar_items(
    item_factors: np.ndarray,   # shape [n_items, k]
    query_col: int,
    k: int = 10,
    exclude_self: bool = True,
):
    """
    Returns: (top_cols, top_sims)
    """
    V = item_factors.astype(np.float32)

    # cosine similarity: (V @ vq) / (||V|| * ||vq||)
    vq = V[query_col]
    V_norm = np.linalg.norm(V, axis=1) + 1e-12
    vq_norm = np.linalg.norm(vq) + 1e-12

    sims = (V @ vq) / (V_norm * vq_norm)

    if exclude_self:
        sims[query_col] = -np.inf

    top = np.argpartition(sims, -k)[-k:]
    top = top[np.argsort(sims[top])[::-1]]
    return top, sims[top]


def show_item_neighbors(
    model,
    item_id,
    item2idx,
    idx2item,
    item_info=None,     # pd.DataFrame indexed by item_id
    k: int = 10,
    fields=None,        # list of columns to show from item_info
):
    """
    model must have: model.item_factors (or model.V)
    """
    # find column index of query item
    q_col = item2idx[item_id]

    top_cols, top_sims = topk_similar_items(model.item_factors, q_col, k=k)

    neighbors = idx2item[top_cols]

    print(f"\nQuery item_id: {item_id}")
    
    if item_info is not None:
        if fields is None:
            fields = [c for c in item_info.columns][:5]  # default: first few columns
        print("Query metadata:")
        print(item_info.loc[[item_id], fields])

    print(f"\nTop {k} nearest neighbors:")
    for rank, (nid, sim) in enumerate(zip(neighbors, top_sims), 1):
        line = f"{rank:2d}. {nid}   sim={float(sim):.4f}"
        if item_info is not None:
            meta = item_info.loc[nid]
            # show selected fields in one line if possible
            extras = []
            for f in (fields or []):
                if f in meta.index:
                    extras.append(f"{f}={meta[f]}")
            if extras:
                line += "   | " + " , ".join(extras)
        print(line)

    return neighbors, top_sims


class BayesianPersonalizedRanking:
    """
    BPR-MF (Matrix Factorization with Bayesian Personalized Ranking loss)

    Two training modes:
      - mode="per_user": one update per (non-empty) user per epoch (like your class)
      - mode="samples": many sampled (u,i,j) triples per epoch (usually better)

    Works with any CSR user×item implicit matrix.
    """

    def __init__(
        self,
        factors=64,
        learning_rate=0.05,
        regularization=1e-4,
        epochs=20,
        random_state=42,
        mode="samples",                 # "samples" or "per_user"
        samples_per_epoch=200_000,      # used only if mode="samples"
    ):
        self.factors = int(factors)
        self.lr = float(learning_rate)
        self.reg = float(regularization)
        self.epochs = int(epochs)
        self.rng = default_rng(random_state)

        assert mode in ("samples", "per_user")
        self.mode = mode
        self.samples_per_epoch = int(samples_per_epoch)

        self.user_factors = None
        self.item_factors = None

    # ---------- core utilities ----------

    @staticmethod
    def _softplus_neg_x(x: float) -> float:
        # stable softplus(-x) = log(1+exp(-x))
        return float(np.logaddexp(0.0, -x))

    def _init_factors(self, num_users: int, num_items: int):
        self.user_factors = (0.01 * self.rng.standard_normal((num_users, self.factors))).astype(np.float32)
        self.item_factors = (0.01 * self.rng.standard_normal((num_items, self.factors))).astype(np.float32)

    def _sample_negative(self, num_items: int, pos_items: np.ndarray, pos_set: set | None):
        # pos_set avoids O(k) membership on array
        while True:
            j = int(self.rng.integers(num_items))
            if (pos_set is not None and j not in pos_set) or (pos_set is None and j not in pos_items):
                return j

    # ---------- training ----------

    def fit(self, user_items: csr_matrix):
        if not isinstance(user_items, csr_matrix):
            raise TypeError("user_items must be a scipy.sparse.csr_matrix")

        X = user_items.tocsr()
        num_users, num_items = X.shape
        self._init_factors(num_users, num_items)

        indptr, indices = X.indptr, X.indices

        for epoch in trange(self.epochs, desc=f"BPR ({self.mode})"):
            loss = 0.0
            updates = 0

            if self.mode == "per_user":
                # like your class: one update per non-empty user
                for u in range(num_users):
                    start, end = indptr[u], indptr[u + 1]
                    pos = indices[start:end]
                    if pos.size == 0:
                        continue

                    i = int(pos[self.rng.integers(pos.size)])
                    pos_set = set(pos.tolist())  # faster negative membership
                    j = self._sample_negative(num_items, pos, pos_set)

                    u_vec = self.user_factors[u]
                    i_vec = self.item_factors[i]
                    j_vec = self.item_factors[j]

                    x = float(np.dot(u_vec, i_vec - j_vec))
                    s = 1.0 / (1.0 + np.exp(x))  # sigmoid(-x)

                    grad_u = s * (j_vec - i_vec) + self.reg * u_vec
                    grad_i = s * (-u_vec) + self.reg * i_vec
                    grad_j = s * (u_vec) + self.reg * j_vec

                    self.user_factors[u] -= self.lr * grad_u
                    self.item_factors[i] -= self.lr * grad_i
                    self.item_factors[j] -= self.lr * grad_j

                    loss += self._softplus_neg_x(x)
                    updates += 1

            else:
                # better: many sampled (u,i,j) triples per epoch
                for _ in range(self.samples_per_epoch):
                    u = int(self.rng.integers(num_users))
                    start, end = indptr[u], indptr[u + 1]
                    if start == end:
                        continue

                    pos = indices[start:end]
                    i = int(pos[self.rng.integers(pos.size)])

                    # For typical Myket/ML1M, pos size is modest → set() is fine
                    pos_set = set(pos.tolist())
                    j = self._sample_negative(num_items, pos, pos_set)

                    u_vec = self.user_factors[u]
                    i_vec = self.item_factors[i]
                    j_vec = self.item_factors[j]

                    x = float(np.dot(u_vec, i_vec - j_vec))
                    s = 1.0 / (1.0 + np.exp(x))  # sigmoid(-x)

                    grad_u = s * (j_vec - i_vec) + self.reg * u_vec
                    grad_i = s * (-u_vec) + self.reg * i_vec
                    grad_j = s * (u_vec) + self.reg * j_vec

                    self.user_factors[u] -= self.lr * grad_u
                    self.item_factors[i] -= self.lr * grad_i
                    self.item_factors[j] -= self.lr * grad_j

                    loss += self._softplus_neg_x(x)
                    updates += 1

            # lightweight feedback each epoch
            avg = loss / max(updates, 1)
            # tqdm prints are nice but this keeps it simple:
            # (You can also use set_postfix, but trange handle isn't stored here.)
            print(f"epoch {epoch+1}/{self.epochs}  updates={updates}  avg_loss={avg:.4f}")

        return self

    # ---------- inference ----------

    def recommend(self, user_items: csr_matrix, userid: int, N: int = 10, filter_seen: bool = True):
        if self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Call fit() before recommend().")

        scores = (self.item_factors @ self.user_factors[userid]).astype(np.float32)

        if filter_seen:
            seen = user_items[userid].indices
            scores = scores.copy()
            scores[seen] = -np.inf

        top = np.argpartition(scores, -N)[-N:]
        top = top[np.argsort(scores[top])[::-1]]
        return top, scores[top]

    def score(self, userid: int, itemid: int) -> float:
        return float(np.dot(self.user_factors[userid], self.item_factors[itemid]))
    