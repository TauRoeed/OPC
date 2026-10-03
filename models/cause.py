"""CausE (Bonner & Vasile, RecSys '18) in PyTorch: the released objective of criteo-research/CausE @ 957e556.

The model is the SP2V logit ``z(i, k) = alpha * <U_i, P_k> + b_i + b_k + b`` over an item table whose rows are:
    prod: control row a and treatment row a + n_items for item a (2 * n_items rows; untied biases);
    avg:  item rows 0..n_items-1 and one pooled treatment row (every randomized row is mapped to it).
Per minibatch the loss is the released one (docs/cause_baseline.md §2.4):
    mean CE + l2_pen * (½|U|² + ½|P|² + ½|b_users|² + ½|b_items|²) + cf_pen * tie
    prod tie: mean_B |P_k - sg(P_r(k))|_1,  r(k) = k + offset if k < offset else k     (one-way, frequency-weighted)
    avg tie:  mean_B |P_k/|P_k| - sg(P_pool)/|P_pool||_1                              (TF l2_normalize)
``tie='symmetric'`` drops the stop-gradient (the paper's eq. 18 reading). Optimizers: ``sgd`` (released: plain,
constant lr) and ``momentum_decay`` (the paper's description as the audit implemented it: TF MomentumOptimizer
with lr0 * max(0, 1 - step / total_steps)). Every update is dense, as in the released graph: its L2 term is
built even when l2_pen = 0, so TF densifies every embedding gradient (verified against TF).
One deliberate deviation (docs/cause_baseline.md §2.9): in CausE-avg the pooled row's tie with itself is exactly 0
here, while TF's two normalisation kernels round differently and give it a ±1 subgradient.
tests/test_cause_tf_reference.py checks this module against the unmodified TensorFlow graph.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CAUSE_VARIANTS = ("prod", "avg")
CAUSE_OPTIMIZERS = ("sgd", "momentum_decay")
CAUSE_TIES = ("one_way", "symmetric")
# prediction name -> (trained variant, item rows used to predict)
CAUSE_PREDICTIONS = {"prod_c": ("prod", "control"), "prod_t": ("prod", "treatment"), "avg": ("avg", "control")}
ALPHA_INIT = 1e-8  # models.py: alpha = get_variable(..., constant_initializer(0.00000001))
TF_NORMALIZE_EPS = 1e-12  # tf.nn.l2_normalize: x * rsqrt(max(sum(x**2), epsilon))


def _tf_l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x * torch.rsqrt(torch.clamp((x * x).sum(dim=dim, keepdim=True), min=TF_NORMALIZE_EPS))


def xavier_uniform(shape: tuple[int, int], generator: torch.Generator | None = None) -> torch.Tensor:
    """tf.contrib.layers.xavier_initializer on a 2-D variable: U(-l, l), l = sqrt(6 / (rows + cols))."""
    limit = math.sqrt(6.0 / float(shape[0] + shape[1]))
    return (torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * limit


class CausELayout:
    """Item-table rows for one variant on a catalog of ``n_items`` actions (OPC's layout, no free id)."""

    def __init__(self, variant: str, n_items: int):
        if variant not in CAUSE_VARIANTS:
            raise ValueError(f"variant must be one of {CAUSE_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.n_items = int(n_items)

    @property
    def n_rows(self) -> int:
        return 2 * self.n_items if self.variant == "prod" else self.n_items + 1

    @property
    def tie_offset(self) -> int | None:
        return self.n_items if self.variant == "prod" else None

    @property
    def pooled_row(self) -> int | None:
        return self.n_items if self.variant == "avg" else None

    def train_rows(self, actions: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """Row of each training interaction: control rows for S_c, treatment (prod) or pooled (avg) rows for S_t."""
        actions = np.asarray(actions, dtype=np.int64)
        treatment = np.asarray(treatment, dtype=bool)
        if self.variant == "prod":
            return np.where(treatment, actions + self.n_items, actions)
        return np.where(treatment, self.n_items, actions)

    def prediction_rows(self, side: str) -> np.ndarray:
        """Rows that score the catalog: ``control`` (prod-C, avg) or ``treatment`` (prod-T)."""
        base = np.arange(self.n_items, dtype=np.int64)
        if side == "control":
            return base
        if side == "treatment" and self.variant == "prod":
            return base + self.n_items
        raise ValueError(f"{self.variant} has no {side!r} prediction rows")


class CausEModel(nn.Module):
    """SP2V/CausE parameters: users U, b_users; item table P, b_items; global bias b; scale alpha."""

    def __init__(
        self,
        n_users: int,
        n_rows: int,
        dim: int,
        *,
        variant: str,
        tie_offset: int | None = None,
        pooled_row: int | None = None,
        alpha_init: float = ALPHA_INIT,
        generator: torch.Generator | None = None,
        emulate_tf_pooled_rounding: bool = False,
    ):
        super().__init__()
        # diagnostic only (docs/cause_baseline.md §8.1): normalise the pooled vector by a different formula, so the
        # pooled rows' self-difference rounds to ±1 ulp and abs() passes ±1 subgradients, like TF's two kernels
        self.emulate_tf_pooled_rounding = bool(emulate_tf_pooled_rounding)
        if variant not in CAUSE_VARIANTS:
            raise ValueError(f"variant must be one of {CAUSE_VARIANTS}, got {variant!r}")
        if variant == "prod" and tie_offset is None:
            raise ValueError("prod needs tie_offset (rows k < offset are tied to k + offset)")
        if variant == "avg" and pooled_row is None:
            raise ValueError("avg needs pooled_row (the single treatment row)")
        self.variant = variant
        self.tie_offset = None if tie_offset is None else int(tie_offset)
        self.pooled_row = None if pooled_row is None else int(pooled_row)
        self.dim = int(dim)
        self.user_emb = nn.Parameter(xavier_uniform((int(n_users), self.dim), generator))
        self.item_emb = nn.Parameter(xavier_uniform((int(n_rows), self.dim), generator))
        self.user_bias = nn.Parameter(torch.zeros(int(n_users)))
        self.item_bias = nn.Parameter(torch.zeros(int(n_rows)))
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    @classmethod
    def for_layout(cls, layout: CausELayout, n_users: int, dim: int, **kw) -> "CausEModel":
        return cls(n_users, layout.n_rows, dim, variant=layout.variant, tie_offset=layout.tie_offset,
                   pooled_row=layout.pooled_row, **kw)

    def logits(self, users: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        # F.embedding: the same lookup as indexing, with a faster deterministic backward
        emb = (F.embedding(users, self.user_emb) * F.embedding(rows, self.item_emb)).sum(dim=-1)
        b_u = F.embedding(users, self.user_bias[:, None])[:, 0]
        b_i = F.embedding(rows, self.item_bias[:, None])[:, 0]
        return self.alpha * emb + b_u + b_i + self.global_bias

    def tie_targets(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.where(rows < self.tie_offset, rows + self.tie_offset, rows)

    def tie(self, rows: torch.Tensor, *, symmetric: bool = False) -> torch.Tensor:
        """The released L1 discrepancy term (before cf_pen), a mean over the batch rows."""
        p = F.embedding(rows, self.item_emb)
        if self.variant == "prod":
            t = F.embedding(self.tie_targets(rows), self.item_emb)
            t = t if symmetric else t.detach()
            return (p - t).abs().sum(dim=-1).mean()
        c = self.item_emb[self.pooled_row]
        c = c if symmetric else c.detach()
        if self.emulate_tf_pooled_rounding:
            c_hat = c / torch.sqrt(torch.clamp((c * c).sum(), min=TF_NORMALIZE_EPS))
        else:
            c_hat = _tf_l2_normalize(c, dim=0)
        return (_tf_l2_normalize(p, dim=1) - c_hat).abs().sum(dim=-1).mean()

    def l2(self) -> torch.Tensor:
        """tf.nn.l2_loss(U) + l2_loss(P) + l2_loss(b_items) + l2_loss(b_users) (global bias, alpha unpenalized)."""
        return 0.5 * (self.user_emb.pow(2).sum() + self.item_emb.pow(2).sum()
                      + self.item_bias.pow(2).sum() + self.user_bias.pow(2).sum())

    def loss(self, users, rows, labels, *, l2_pen: float = 0.0, cf_pen: float = 1.0, symmetric: bool = False):
        """(total, mean cross-entropy) for one minibatch."""
        ce = F.binary_cross_entropy_with_logits(self.logits(users, rows), labels)
        total = ce
        if l2_pen:
            total = total + float(l2_pen) * self.l2()
        if cf_pen:
            total = total + float(cf_pen) * self.tie(rows, symmetric=symmetric)
        return total, ce

    @torch.no_grad()
    def policy_vectors(self, rows: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """(user, item) vectors whose dot product is alpha<U_u, P_row(j)> + b_row(j): the per-user ranking and
        softmax of the logits (the user and global biases are constant over items)."""
        rows = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.item_emb.device)
        ones = torch.ones(self.user_emb.shape[0], 1, device=self.user_emb.device)
        user = torch.cat([self.alpha * self.user_emb, ones], dim=1)
        item = torch.cat([self.item_emb[rows], self.item_bias[rows][:, None]], dim=1)
        return user.float().cpu().numpy(), item.float().cpu().numpy()


class CausEOptimizer:
    """``sgd``: var -= lr * grad (tf.train.GradientDescentOptimizer). ``momentum_decay``: TF MomentumOptimizer
    (accum = m * accum + grad; var -= lr_t * accum) with lr_t = lr0 * max(0, 1 - step / total_steps)."""

    def __init__(self, model: CausEModel, *, kind: str, lr: float, total_steps: int = 1, momentum: float = 0.9):
        if kind not in CAUSE_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {CAUSE_OPTIMIZERS}, got {kind!r}")
        self.model = model
        self.kind = kind
        self.lr = float(lr)
        self.total_steps = max(1, int(total_steps))
        self.momentum = float(momentum)
        self.step_count = 0
        self.buffers = {n: torch.zeros_like(p) for n, p in model.named_parameters()} if kind != "sgd" else {}

    def current_lr(self) -> float:
        if self.kind == "sgd":
            return self.lr
        return self.lr * max(0.0, 1.0 - float(self.step_count) / float(self.total_steps))

    @torch.no_grad()
    def step(self) -> None:
        lr_t = self.current_lr()
        named = [(n, p) for n, p in self.model.named_parameters() if p.grad is not None]
        params = [p for _, p in named]
        grads = [p.grad for _, p in named]
        if self.kind == "sgd":
            torch._foreach_add_(params, grads, alpha=-lr_t)
        else:
            bufs = [self.buffers[n] for n, _ in named]
            torch._foreach_mul_(bufs, self.momentum)
            torch._foreach_add_(bufs, grads)
            torch._foreach_add_(params, bufs, alpha=-lr_t)
        self.step_count += 1


def train_steps(
    model: CausEModel,
    batches,
    *,
    optimizer: str = "sgd",
    lr: float = 1.0,
    l2_pen: float = 0.0,
    cf_pen: float = 1.0,
    symmetric: bool = False,
    momentum: float = 0.9,
    total_steps: int | None = None,
) -> list[tuple[float, float]]:
    """Train on an explicit sequence of (users, rows, labels) tensor batches; returns (loss, ce) per step.

    The loss is evaluated at the parameters before the update, as TF's ``sess.run([apply_grads, loss])`` does."""
    batches = list(batches)
    opt = CausEOptimizer(model, kind=optimizer, lr=lr, total_steps=total_steps or len(batches), momentum=momentum)
    history = []
    for users, rows, labels in batches:
        model.zero_grad(set_to_none=False)
        total, ce = model.loss(users, rows, labels, l2_pen=l2_pen, cf_pen=cf_pen, symmetric=symmetric)
        total.backward()
        opt.step()
        history.append((float(total.detach()), float(ce.detach())))
    return history


def epoch_batches(n_rows: int, batch_size: int, epochs: int, *, seed: int, reshuffle: bool = False):
    """Index batches: one seeded permutation replayed every epoch (the released tf.data shuffle->cache order),
    or a fresh permutation per epoch (``reshuffle``). The last batch of an epoch may be partial, as in TF."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_rows))
    for _ in range(int(epochs)):
        if reshuffle:
            order = rng.permutation(int(n_rows))
        for s in range(0, int(n_rows), int(batch_size)):
            yield order[s:s + int(batch_size)]


def n_steps(n_rows: int, batch_size: int, epochs: int) -> int:
    return int(epochs) * int(math.ceil(int(n_rows) / int(batch_size)))


def fit_cause(
    model: CausEModel,
    users: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int = 512,
    optimizer: str = "momentum_decay",
    lr: float = 0.1,
    l2_pen: float = 0.0,
    cf_pen: float = 1.0,
    symmetric: bool = False,
    momentum: float = 0.9,
    seed: int = 0,
    reshuffle: bool = False,
    device: torch.device | str = "cpu",
) -> dict:
    """Train on (user, item-row, label) interactions. Returns training diagnostics."""
    device = torch.device(device)
    model.to(device)
    users_t = torch.as_tensor(np.asarray(users, dtype=np.int64), device=device)
    rows_t = torch.as_tensor(np.asarray(rows, dtype=np.int64), device=device)
    labels_t = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=device)
    total = n_steps(len(users_t), batch_size, epochs)
    opt = CausEOptimizer(model, kind=optimizer, lr=lr, total_steps=total, momentum=momentum)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    grads = [p.grad for p in model.parameters()]
    last_loss = last_ce = float("nan")
    finite = True
    n = len(users_t)
    rng = np.random.default_rng(int(seed))
    order = torch.as_tensor(rng.permutation(n), device=device)  # the epoch_batches order, uploaded once
    for _ in range(int(epochs)):
        if reshuffle:
            order = torch.as_tensor(rng.permutation(n), device=device)
        u_ep, r_ep, y_ep = users_t[order], rows_t[order], labels_t[order]
        for s in range(0, n, int(batch_size)):
            u, r, y = u_ep[s:s + batch_size], r_ep[s:s + batch_size], y_ep[s:s + batch_size]
            torch._foreach_zero_(grads)
            loss, ce = model.loss(u, r, y, l2_pen=l2_pen, cf_pen=cf_pen, symmetric=symmetric)
            loss.backward()
            opt.step()
            last_loss, last_ce = loss.detach(), ce.detach()
    if total:
        last_loss, last_ce = float(last_loss), float(last_ce)
        finite = bool(np.isfinite(last_loss)) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return {"steps": int(total), "final_batch_loss": last_loss, "final_batch_ce": last_ce, "finite": finite,
            "alpha": float(model.alpha.detach())}


@torch.no_grad()
def predict_logits(model: CausEModel, users: np.ndarray, rows: np.ndarray, *, chunk: int = 65536) -> np.ndarray:
    device = model.item_emb.device
    out = []
    users = np.asarray(users, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    for s in range(0, len(users), chunk):
        u = torch.as_tensor(users[s:s + chunk], device=device)
        r = torch.as_tensor(rows[s:s + chunk], device=device)
        out.append(model.logits(u, r).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def prediction_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """NLL (mean CE, as tf sigmoid_cross_entropy_with_logits), MSE of sigmoid(logit) and AUC."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    nll = float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))) if len(z) else float("nan")
    p = 1.0 / (1.0 + np.exp(-z))
    mse = float(np.mean((p - y) ** 2)) if len(z) else float("nan")
    auc = float("nan")
    if len(z) and 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y.astype(int), z))
    return {"nll": nll, "mse": mse, "auc": auc}
