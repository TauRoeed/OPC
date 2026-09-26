"""What each OPC objective variant optimizes, checked against literal full-data computations.

OPC trains with the log trick: the weight w_i = pi(a_i|x_i) / pi_b(a_i|x_i) enters as a detached
coefficient times grad log pi(a_i|x_i). For a weight transform g (none, clip:M, shrink:lam) the
minibatch ascent direction of every SNDR-family variant is then

    (1/b) sum_{i in B} [grad DM_i + g(w_i) (r_i - q_i) / N * grad log pi_i]
  = (1/b) sum_{i in B} [grad DM_i + grad H(w_i) (r_i - q_i) / N],     H(w) = int_0^w g(t)/t dt,

with N held fixed during backprop: 1 ('dr'), the full-data mean weight at the start of the epoch
('sndr --sn-scope global') or the batch's own mean weight (legacy 'sndr', which also keeps 1/|B| in
place of 1/b). The literal full-data SNDR ratio V = mean DM + sum g(w)(r - q) / sum g(w) has

    grad V = mean grad DM + (1/(n c)) sum g'(w_i) grad w_i [(r_i - q_i) - R],
    c = mean g(w),  R = sum g(w)(r - q) / sum g(w),

so 'global' at the refresh point is the DR direction with the correction divided by c: it lacks the
term -(R/c) grad c (no gradient through the denominator), uses g/w in place of g' (the log-trick
convention, shared by all three variants), and uses a stale c after the epoch's first step.
"""

import math

import pytest
import torch
from torch.utils.data import DataLoader

from models.custom_losses import (
    CRMPolicyLoss,
    DMPolicyLoss,
    DRPolicyLoss,
    IPWPolicyLoss,
    KLCRMPolicyLoss,
    KLPolicyLoss,
    NaiveRewardPolicyLoss,
    SNDRPolicyLoss,
)
from training import training_utils
from training.training_utils import full_data_mean_weight, minibatch_loss
from utils.importance_weights import parse_weight_spec
from utils.simulation_utils import CustomCFDatasetPS, collate_prebatched

SPECS = ["none", "clip:2", "shrink:4"]  # small parameters so that many weights pass them


def _world(seed=0, n=1000, n_users=80, n_actions=12, d=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_users, d, generator=g, dtype=torch.float64)
    theta0 = torch.randn(d, n_actions, generator=g, dtype=torch.float64)
    logger = torch.softmax(X @ theta0, dim=1)
    theta = theta0 + 0.6 * torch.randn(d, n_actions, generator=g, dtype=torch.float64)  # near, not at, the logger
    q_hat = 0.05 + 0.3 * torch.rand(n_users, n_actions, generator=g, dtype=torch.float64)
    # a reward model that is off by 0.1 on average, so the self-normalized residual R is not ~0
    q_true = (q_hat + 0.1 + 0.15 * torch.randn(n_users, n_actions, generator=g, dtype=torch.float64)).clamp(0.01, 0.9)
    users = torch.randint(0, n_users, (n,), generator=g)
    actions = torch.multinomial(logger[users], 1, generator=g).squeeze(1)
    return dict(X=X, theta=theta, q_hat=q_hat, users=users, actions=actions,
                pscore=logger[users, actions], rewards=torch.bernoulli(q_true[users, actions], generator=g))


class _Softmax(torch.nn.Module):
    """pi(.|u) = softmax(X_u theta): one parameter shared by all rows, like the trained CFModel."""

    def __init__(self, X, theta):
        super().__init__()
        self.register_buffer("X", X)
        self.theta = torch.nn.Parameter(theta.clone())

    def forward(self, users):
        return torch.softmax(self.X[users] @ self.theta, dim=1)


def _g(w, spec):
    mode, p = parse_weight_spec(spec)
    if mode == "clip":
        return torch.clamp(w, max=p)
    if mode == "shrink":
        return p * w / (w * w + p)
    return w


def _H(w, spec):
    """int_0^w g(t)/t dt: the weight whose exact gradient the log trick follows."""
    mode, p = parse_weight_spec(spec)
    if mode == "clip":
        return torch.where(w <= p, w, p * (1.0 + torch.log(w / p)))
    if mode == "shrink":
        return math.sqrt(p) * torch.atan(w / math.sqrt(p))
    return w


def _pieces(world, theta, users, actions, pscore, rewards):
    """Per-row DM value, raw weight and residual, with every gradient path attached (no detach)."""
    pi = torch.softmax(world["X"][users] @ theta, dim=1)
    w = pi[torch.arange(len(users)), actions] / pscore
    return (world["q_hat"][users] * pi).sum(1), w, rewards - world["q_hat"][users, actions]


def _literal(world, theta):
    return _pieces(world, theta, world["users"], world["actions"], world["pscore"], world["rewards"])


def _grad(fn, world):
    theta = world["theta"].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(fn(*_literal(world, theta)), theta)
    return grad


def _loader(world, batch_size, seed=0):
    ds = CustomCFDatasetPS(world["users"].numpy(), world["actions"].numpy(), world["rewards"].numpy(),
                           world["pscore"].numpy())
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_prebatched,
                      generator=torch.Generator().manual_seed(seed))


def _epoch_direction(world, criterion, batch_size, *, reduce=minibatch_loss):
    """One epoch of the trainer's real DataLoader at fixed parameters: the summed minibatch ascent
    directions, times b / n (per-row units, comparable with the gradient of a full-data mean)."""
    loader = _loader(world, batch_size)
    model = _Softmax(world["X"], world["theta"])
    total, sizes = torch.zeros_like(world["theta"]), []
    for users, actions, rewards, pscore in loader:
        loss = reduce(criterion, pscore, world["q_hat"][users], model(users), rewards, actions, loader.batch_size)
        (grad,) = torch.autograd.grad(loss, model.theta)
        total -= grad
        sizes.append(int(users.shape[0]))
    return total * batch_size / len(loader.dataset), sizes


def _plain_mean(criterion, pscore, scores, policy, rewards, actions, nominal):
    return criterion(pscore, scores, policy, rewards, actions)  # the loop before the short-batch fix


def _close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)


def _far(a, b, rel=1e-3):
    assert float((a - b).norm()) > rel * float(b.norm()), (float((a - b).norm()), float(b.norm()))


@pytest.fixture(scope="module")
def world():
    w = _world()
    _, iw, _ = _literal(w, w["theta"])
    assert float((iw > 2).double().mean()) > 0.05 and float(iw.max()) > 10  # every transform is active
    return w


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("log_trick", [True, False])
def test_dr_follows_the_gradient_of_a_per_row_objective(world, spec, log_trick):
    loss = DRPolicyLoss(use_log_trick=log_trick, weights=spec)
    got, sizes = _epoch_direction(world, loss, 384)
    assert sizes == [384, 384, 232]  # the real DataLoader, with its short final batch
    weight = _H if log_trick else _g
    _close(got, _grad(lambda dm, w, res: dm.mean() + (weight(w, spec) * res).mean(), world))
    if spec == "none":  # raw weights: the gradient of the DR estimate itself
        _close(got, _grad(lambda dm, w, res: dm.mean() + (w * res).mean(), world))
    elif log_trick:  # not the gradient of the transformed DR estimate DM + g(w)(r - q)
        _far(got, _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).mean(), world))


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("log_trick", [True, False])
def test_global_sndr_is_a_stop_gradient_surrogate_not_exact_sndr(world, spec, log_trick):
    loss = SNDRPolicyLoss(use_log_trick=log_trick, weights=spec, normalization="global")
    loss.set_global_normalizer(full_data_mean_weight(_Softmax(world["X"], world["theta"]),
                                                     _loader(world, 384).dataset, loss, "cpu"))
    dm, w, res = _literal(world, world["theta"])
    c = float(_g(w, spec).mean())
    R = float((_g(w, spec) * res).sum() / _g(w, spec).sum())
    assert loss.global_normalizer == pytest.approx(c, rel=1e-12)  # the trainer's value, at these parameters
    got, _ = _epoch_direction(world, loss, 384)
    # what it follows: DR's direction with the correction divided by the fixed number c
    weight = _H if log_trick else _g
    _close(got, _grad(lambda dm, w, res: dm.mean() + (weight(w, spec) * res).mean() / c, world))
    # the literal full-data SNDR ratio, differentiated through numerator and denominator
    exact = _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).sum() / _g(w, spec).sum(), world)
    _far(got, exact)
    # the gap is the denominator's gradient -(R/c) grad c, plus g/w in place of g' under the log trick
    held = _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).mean() / c, world)
    _close(exact - held, -(R / c) * _grad(lambda dm, w, res: _g(w, spec).mean(), world))
    if spec == "none" or not log_trick:
        _close(got, held)


def test_global_sndr_uses_a_stale_normalizer_within_the_epoch(world, monkeypatch):
    """The normalizer is computed once per epoch; after each optimizer step the policy's full-data mean
    weight moves while the loss keeps dividing by the epoch-start value."""
    device = "cuda" if torch.cuda.is_available() else "cpu"  # train() expects the model on the GPU when there is one
    model = _Softmax(world["X"], world["theta"]).to(device)
    loss = SNDRPolicyLoss(use_log_trick=True, weights="shrink:4", normalization="global")
    loader = _loader(world, 128)
    set_values, in_use, current = [], [], []
    real_set = loss.set_global_normalizer
    monkeypatch.setattr(loss, "set_global_normalizer", lambda v: set_values.append(v) or real_set(v))
    real_step = torch.optim.Adam.step

    def step(self, *a, **k):
        out = real_step(self, *a, **k)
        in_use.append(loss.global_normalizer)
        current.append(full_data_mean_weight(model, loader.dataset, loss, device))
        return out

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    training_utils.train(model, loader, world["q_hat"].to(device), criterion=loss, num_epochs=2, lr=0.05, device=device)
    steps = len(loader)  # 8 per epoch: 7 of 128 rows and 1 of 104
    assert len(set_values) == 2 and len(in_use) == 2 * steps
    assert in_use[:steps] == [set_values[0]] * steps and in_use[steps:] == [set_values[1]] * steps
    assert set_values[1] == pytest.approx(current[steps - 1], rel=1e-12)  # refreshed at the epoch boundary
    drift = [abs(c - n) / n for c, n in zip(current, in_use)]
    assert min(drift[: steps - 1]) > 1e-4  # stale after every step before the refresh


@pytest.mark.parametrize("spec", SPECS)
def test_legacy_sndr_is_a_per_batch_stop_gradient_surrogate(world, spec):
    n = len(world["users"])
    one = SNDRPolicyLoss(use_log_trick=True, weights=spec, normalization="batch")
    got, sizes = _epoch_direction(world, one, n)
    assert sizes == [n]
    dm, w, res = _literal(world, world["theta"])
    c = float(_g(w, spec).mean())
    # one batch of all rows: the same direction as 'global' at its refresh point, not the ratio's gradient
    _close(got, _grad(lambda dm, w, res: dm.mean() + (_H(w, spec) * res).mean() / c, world))
    _far(got, _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).sum() / _g(w, spec).sum(), world))
    # without the log trick the batch denominator carries its gradient: exact SNDR of the batch
    direct = SNDRPolicyLoss(use_log_trick=False, weights=spec, normalization="batch")
    _close(_epoch_direction(world, direct, n)[0],
           _grad(lambda dm, w, res: dm.mean() + (_g(w, spec) * res).sum() / _g(w, spec).sum(), world))


@pytest.mark.parametrize("spec", SPECS)
def test_legacy_sndr_keeps_one_equal_weight_mean_per_batch(world, spec):
    """Legacy SNDR is a ratio per minibatch: each batch, the short one included, contributes the direction
    of its own stop-gradient ratio once, so the epoch direction depends on the partition (the batch size)."""
    n, batch = len(world["users"]), 384
    loss = SNDRPolicyLoss(use_log_trick=True, weights=spec, normalization="batch")
    assert not loss.per_example_additive
    got, sizes = _epoch_direction(world, loss, batch)
    assert sizes == [384, 384, 232]
    _close(got, _epoch_direction(world, loss, batch, reduce=_plain_mean)[0])  # unchanged by the fix
    theta, total = world["theta"].clone().requires_grad_(True), torch.zeros_like(world["theta"])
    for users, actions, rewards, pscore in _loader(world, batch):  # the same permutation
        dm, w, res = _pieces(world, theta, users, actions, pscore, rewards)
        c_batch = float(_g(w, spec).mean().detach())
        (grad,) = torch.autograd.grad(dm.mean() + (_H(w, spec) * res).mean() / c_batch, theta)
        total += grad
    _close(got, total * batch / n)
    _far(got, _epoch_direction(world, loss, n)[0])  # not the one-batch (full-data) direction


ADDITIVE = {
    "dr": lambda: DRPolicyLoss(weights="shrink:4"),
    "dr_direct": lambda: DRPolicyLoss(weights="clip:2", use_log_trick=False),
    "sndr_global": lambda: SNDRPolicyLoss(weights="shrink:4", normalization="global"),
    "kl_global": lambda: KLPolicyLoss(gamma=0.1, weights="shrink:4", normalization="global"),
    "dm": lambda: DMPolicyLoss(use_log_trick=False),
    "naive": lambda: NaiveRewardPolicyLoss(use_log_trick=False),
    "ipw": lambda: IPWPolicyLoss(weights="clip:2"),
}


@pytest.mark.parametrize("name", sorted(ADDITIVE))
@pytest.mark.parametrize("n,batch,sizes", [(1000, 384, [384, 384, 232]), (1001, 1000, [1000, 1]),
                                           (1000, 2048, [1000]), (1000, 250, [250] * 4)])
def test_short_final_batch_rows_carry_the_same_weight(name, n, batch, sizes):
    world = _world(seed=3, n=n)
    loss = ADDITIVE[name]()
    assert loss.per_example_additive
    if loss.needs_global_normalizer:
        loss.set_global_normalizer(1.7)  # any fixed number: the objective is additive given it
    full, one = _epoch_direction(world, loss, n)
    assert one == [n]
    got, seen = _epoch_direction(world, loss, batch)
    assert seen == sizes
    _close(got, full)  # every row weighs 1/b, the final short batch included
    if sizes[-1] != batch and len(sizes) > 1:  # the plain per-batch mean upweights the short batch's rows
        _far(_epoch_direction(world, loss, batch, reduce=_plain_mean)[0], full)


def test_per_batch_losses_are_flagged():
    for loss in (SNDRPolicyLoss(normalization="batch"), KLPolicyLoss(normalization="batch"), CRMPolicyLoss(),
                 KLCRMPolicyLoss()):
        assert not loss.per_example_additive
    for loss in (SNDRPolicyLoss(normalization="global"), SNDRPolicyLoss(normalization="none"), DRPolicyLoss(),
                 KLPolicyLoss(normalization="none"), DMPolicyLoss(), NaiveRewardPolicyLoss(), IPWPolicyLoss()):
        assert loss.per_example_additive


def test_the_training_loop_weights_rows_equally(world, monkeypatch):
    """run_train_loop itself (DataLoader, minibatch_loss, backward, step) at fixed parameters: an optimizer
    that only sums the gradients, and no gradient clipping."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, batch = len(world["users"]), 384

    class Accumulate(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})
            self.total = None

        def step(self, closure=None):
            g = self.param_groups[0]["params"][0].grad.detach().clone()
            self.total = g if self.total is None else self.total + g

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: torch.zeros(()))

    def loop_direction(loss):
        model = _Softmax(world["X"], world["theta"]).to(device)
        opt = Accumulate(model.parameters())
        training_utils.run_train_loop(model, _loader(world, batch), opt, world["q_hat"].to(device), loss, device=device)
        return -opt.total.cpu() * batch / n

    dr = DRPolicyLoss(weights="shrink:4")
    _close(loop_direction(dr), _epoch_direction(world, dr, n)[0])  # = the full-data gradient
    legacy = SNDRPolicyLoss(weights="shrink:4", normalization="batch")
    _close(loop_direction(legacy), _epoch_direction(world, legacy, batch, reduce=_plain_mean)[0])
