"""Reward models of docs/opc_adaptive_estimator_study.md §5.4, fit on the logs only.

- D1 ``lowrank``: logit q̂(x, a) = c + u·x + v·a + s · xᵀ(I + PQᵀ)a, with P, Q ∈ R^{K×r}.
- D2 ``hetero``: logit q̂(x, a) = exp(v_αᵀx) (s · xᵀ(I + PQᵀ)a + γ) + v_βᵀx + v·a + c.

x and a are the source vectors (``our_x``, ``our_a``), as the existing logistic model (D0) uses. Fit by full-batch
L-BFGS on the Bernoulli NLL of the logged rows, with an L2 penalty on (u, v, P, Q, v_α, v_β). Q starts as a seeded
orthonormal matrix and P at 0, so the fit starts at the bilinear source model. ``BilinearRewardLookup`` serves q̂ rows
with the interface the trainer's lookups have (training batches, item blocks, pair predictions).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

REWARD_KINDS = ("lowrank", "hetero")
RANK = 8
L2 = 1e-4


class BilinearRewardModel(nn.Module):
    def __init__(self, dim: int, kind: str = "lowrank", rank: int = RANK, seed: int = 0, dtype=torch.float64):
        super().__init__()
        if kind not in REWARD_KINDS:
            raise ValueError(f"kind must be one of {REWARD_KINDS}, got {kind!r}")
        self.kind, self.rank = kind, min(int(rank), int(dim))
        rank = self.rank
        g = torch.Generator().manual_seed(int(seed))
        q0, _ = torch.linalg.qr(torch.randn(dim, rank, generator=g, dtype=dtype))
        self.c = nn.Parameter(torch.zeros((), dtype=dtype))
        self.s = nn.Parameter(torch.ones((), dtype=dtype))
        if kind == "lowrank":  # D2's user main effect is v_β
            self.u = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.v = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.P = nn.Parameter(torch.zeros(dim, rank, dtype=dtype))
        self.Q = nn.Parameter(q0.contiguous().clone())  # QR returns column-major; L-BFGS views flat grads
        if kind == "hetero":
            self.v_alpha = nn.Parameter(torch.zeros(dim, dtype=dtype))
            self.v_beta = nn.Parameter(torch.zeros(dim, dtype=dtype))
            self.gamma = nn.Parameter(torch.zeros((), dtype=dtype))

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def _inter(self, x, a, pairs: bool):
        """xᵀ(I + PQᵀ)a for the user × item grid (pairs=False) or row pairs (pairs=True)."""
        if pairs:
            return (x * a).sum(-1) + ((x @ self.P) * (a @ self.Q)).sum(-1)
        return x @ a.T + (x @ self.P) @ (a @ self.Q).T

    def logits(self, x, a, pairs: bool = False):
        inter = self.s * self._inter(x, a, pairs)
        va = a @ self.v
        if self.kind == "lowrank":
            ux = x @ self.u
            return self.c + (ux + va if pairs else ux[:, None] + va[None, :]) + inter
        scale = torch.exp(x @ self.v_alpha)
        base = x @ self.v_beta + self.c
        if pairs:
            return scale * (inter + self.gamma) + base + va
        return scale[:, None] * (inter + self.gamma) + base[:, None] + va[None, :]

    def penalty(self):
        terms = [self.v, self.P, self.Q]  # ||P||² + ||Q||²: the usual penalty of a factored PQᵀ
        terms += [self.u] if self.kind == "lowrank" else [self.v_alpha, self.v_beta]
        return sum((t ** 2).sum() for t in terms)


def fit_bilinear_reward(x_rows, a_rows, r, *, kind: str, seed: int, device, rank: int = RANK, l2: float = L2,
                        max_iter: int = 400) -> BilinearRewardModel:
    """Fit on logged rows (their user and item vectors and clicks): the mean Bernoulli NLL + l2 · penalty, full-batch
    L-BFGS (strong Wolfe). The intercept and the scale start at the one-dimensional logistic fit on xᵀa."""
    dev = torch.device(device)
    X = torch.as_tensor(np.asarray(x_rows), dtype=torch.float64, device=dev)
    A = torch.as_tensor(np.asarray(a_rows), dtype=torch.float64, device=dev)
    y = torch.as_tensor(np.asarray(r), dtype=torch.float64, device=dev)
    m = BilinearRewardModel(X.shape[1], kind, rank, seed).to(dev)
    bce = nn.BCEWithLogitsLoss()
    # warm start: c, s of logit = c + s xᵀa
    with torch.no_grad():
        z = (X * A).sum(-1)
    cs = torch.zeros(2, dtype=torch.float64, device=dev, requires_grad=True)
    opt0 = torch.optim.LBFGS([cs], max_iter=100, line_search_fn="strong_wolfe")

    def closure0():
        opt0.zero_grad()
        loss = bce(cs[0] + cs[1] * z, y)
        loss.backward()
        return loss

    opt0.step(closure0)
    with torch.no_grad():
        m.c.copy_(cs[0].detach())
        m.s.copy_(cs[1].detach())
    opt = torch.optim.LBFGS(list(m.parameters()), max_iter=int(max_iter), line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12, history_size=50)

    def closure():
        opt.zero_grad()
        loss = bce(m.logits(X, A, pairs=True), y) + float(l2) * m.penalty()
        loss.backward()
        return loss

    opt.step(closure)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class _PairPredictor:
    """The ``regression_model`` face of the lookup: ``predict_pairs(user_vectors, actions)``."""

    def __init__(self, lookup):
        self.lookup = lookup
        self.n_actions = lookup.n_actions
        self.len_list = 1

    def predict_pairs(self, ctx, actions):
        lk = self.lookup
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(ctx), dtype=torch.float64, device=lk.device)
            a = lk.A[torch.as_tensor(np.asarray(actions, dtype=np.int64), device=lk.device)]
            return torch.sigmoid(lk.model.logits(x, a, pairs=True)).float().cpu().numpy()


class BilinearRewardLookup:
    """q̂ rows of a fitted ``BilinearRewardModel`` for user indices: the trainer's lookup interface (``__getitem__``
    for training batches, ``qhat_block_fn`` / ``qhat_block_numpy`` / ``qhat_rows_numpy`` for selection, and
    ``regression_model.predict_pairs`` with ``user_context`` rows)."""

    q_hat_all = None

    def __init__(self, model: BilinearRewardModel, user_vectors, item_vectors, device, action_chunk: int = 8192):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.user_context = np.asarray(user_vectors)
        self.X = torch.as_tensor(self.user_context, dtype=torch.float64, device=self.device)
        self.A = torch.as_tensor(np.asarray(item_vectors), dtype=torch.float64, device=self.device)
        self.n_actions = int(self.A.shape[0])
        self.action_chunk = int(action_chunk)
        self.regression_model = _PairPredictor(self)

    @torch.no_grad()
    def _rows(self, users, a0=0, a1=None, device=None):
        idx = torch.as_tensor(np.asarray(users.cpu() if isinstance(users, torch.Tensor) else users,
                                         dtype=np.int64).reshape(-1), device=self.device)
        a1 = self.n_actions if a1 is None else a1
        out = torch.sigmoid(self.model.logits(self.X[idx], self.A[a0:a1])).float()
        return out if device is None else out.to(device)

    def __getitem__(self, users):
        return self._rows(users)

    def qhat_block_fn(self, device):
        return lambda u, a0, a1: self._rows(u, a0, a1, torch.device(device))

    def qhat_block_numpy(self, users, a0, a1):
        return self._rows(users, a0, a1).cpu().numpy()

    def qhat_rows_numpy(self, users):
        return self._rows(users).cpu().numpy()


def crossfit_bilinear(dataset: dict, rows: dict, user_fold: np.ndarray, *, kind: str, seed: int, device,
                      folds: int = 5, rank: int = RANK) -> tuple[list, BilinearRewardLookup]:
    """The fold models (fold k fit on the rows of the other folds) and the full-data model of ``rows`` (a
    ``get_train_data`` dict), as lookups: the cross-fitting of D0 (``run_full_study._crossfit_bundles``)."""
    X, A = np.asarray(dataset["our_x"]), np.asarray(dataset["our_a"])
    users = np.asarray(rows["x_idx"], dtype=np.int64)
    acts = np.asarray(rows["a"], dtype=np.int64)
    r = np.asarray(rows["r"], dtype=np.float64)
    fold_of_row = np.asarray(user_fold)[users]
    fold_lookups = []
    for k in range(int(folds)):
        keep = fold_of_row != k
        m = fit_bilinear_reward(X[users[keep]], A[acts[keep]], r[keep], kind=kind, seed=seed * 31 + k, device=device, rank=rank)
        fold_lookups.append(BilinearRewardLookup(m, X, A, device))
    full = BilinearRewardLookup(fit_bilinear_reward(X[users], A[acts], r, kind=kind, seed=seed * 31 + folds,
                                                    device=device, rank=rank), X, A, device)
    return fold_lookups, full
