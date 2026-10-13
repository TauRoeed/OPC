"""The OPC objectives of docs/opc_adaptive_estimator_study.md §5 (models/adaptive_losses.py): the existing loss when no
extension is set, the enumerated support terms, the CRM terms, the adaptive λ and the multiplier, gradients against
finite differences."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.adaptive_losses import A3_GRID, AdaptiveOPCLoss, LoggerRows, a2_lambda
from models.custom_losses import DRPolicyLoss
from models.shared_objectives import AnchoredPolicyLoss

D64 = torch.float64


def _batch(n=64, n_items=30, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ux = torch.randn(40, k, generator=g, dtype=D64)
    ia = torch.randn(n_items, k, generator=g, dtype=D64)
    T = 0.7
    logger = LoggerRows(ux.numpy(), ia.numpy(), T, "cpu")
    users = torch.randint(0, 40, (n,), generator=g)
    mu = torch.softmax((ux[users] @ ia.T) / T, dim=1)
    acts = torch.multinomial(mu, 1, generator=g)[:, 0]
    pscore = mu[torch.arange(n), acts]
    scores = torch.rand(n, n_items, generator=g, dtype=D64) * 0.3
    rewards = (torch.rand(n, generator=g, dtype=D64) < 0.2).to(D64)
    theta = (0.3 * torch.randn(k, k, generator=g, dtype=D64)).requires_grad_(True)

    def policy(th):
        return torch.softmax((ux[users] @ (torch.eye(k, dtype=D64) + th) @ ia.T) / T, dim=1)

    return SimpleNamespace(ux=ux, ia=ia, T=T, logger=logger, users=users, mu=mu, acts=acts, pscore=pscore,
                           scores=scores, rewards=rewards, theta=theta, policy=policy)


def _logger64(b):
    """The logger rows in float64 (the tests compare float64 sums)."""
    return lambda users: torch.softmax((b.ux[users.long()] @ b.ia.T) / b.T, dim=1)


def test_without_extensions_it_is_the_existing_dr_loss_with_its_anchor():
    b = _batch()
    for spec in ("harmonic:0.1", "none"):
        new = AdaptiveOPCLoss(weights=spec)
        old = DRPolicyLoss(use_log_trick=False, weights=spec)
        p = b.policy(b.theta)
        v_new = new(b.pscore, b.scores, p, b.rewards, b.acts)
        v_old = old(b.pscore, b.scores, p, b.rewards, b.acts)
        assert float(v_new) == pytest.approx(float(v_old), rel=1e-12)
        g_new = torch.autograd.grad(v_new, b.theta)[0]
        g_old = torch.autograd.grad(old(b.pscore, b.scores, b.policy(b.theta), b.rewards, b.acts), b.theta)[0]
        assert torch.allclose(g_new, g_old, rtol=1e-10, atol=1e-14)
    pen = lambda: (b.theta ** 2).sum()
    new = AdaptiveOPCLoss(weights="harmonic:0.1", anchor_lambda=0.01, anchor_penalty=pen)
    old = AnchoredPolicyLoss(DRPolicyLoss(use_log_trick=False, weights="harmonic:0.1"), 0.01, pen)
    p = b.policy(b.theta)
    assert float(new(b.pscore, b.scores, p, b.rewards, b.acts)) == pytest.approx(
        float(old(b.pscore, b.scores, p, b.rewards, b.acts)), rel=1e-12)


def test_support_terms_are_the_enumerated_statistics():
    b = _batch()
    p = b.policy(b.theta)
    base = float(AdaptiveOPCLoss(weights="none")(b.pscore, b.scores, p, b.rewards, b.acts))
    s2 = float((p ** 2 / b.mu).sum(1).mean())
    kl = float((p * (torch.log(p) - torch.log(b.mu))).sum(1).mean())
    for support, extra in ((("s2", 0.003), 0.003 * (s2 - 1)), (("kl", 0.01), 0.01 * kl)):
        crit = AdaptiveOPCLoss(weights="none", logger=b.logger, support=support)
        crit.logger = _logger64(b)
        crit.batch_users = b.users
        assert float(crit(b.pscore, b.scores, p, b.rewards, b.acts)) == pytest.approx(base + extra, rel=1e-9)
    crit = AdaptiveOPCLoss(weights="none", logger=b.logger, support=("s2_hard", 2.0))
    crit.logger, crit.batch_users, crit.nu = _logger64(b), b.users, 0.4
    viol = s2 / 2.0 - 1.0  # the normalized violation, ρ = 0.1
    expect = base + 0.4 * viol + 0.05 * max(viol, 0.0) ** 2
    assert float(crit(b.pscore, b.scores, p, b.rewards, b.acts)) == pytest.approx(expect, rel=1e-9)


def test_crm_terms_and_the_delta_contribution():
    b = _batch()
    p = b.policy(b.theta)
    plain = AdaptiveOPCLoss(weights="none")
    f = plain.contributions(b.pscore, b.scores, p, b.rewards, b.acts)
    crit = AdaptiveOPCLoss(weights="none", crm=(2.0, 25_000, "abs"))
    expect = -float(f.mean()) + 2.0 * math.sqrt(float(f.var()) / 25_000 + 1e-12)
    assert float(crit(b.pscore, b.scores, p, b.rewards, b.acts)) == pytest.approx(expect, rel=1e-10)
    # delta: the source is the logger, so its contribution is DR with π = μ (w ≡ 1)
    f0 = plain.contributions(b.pscore, b.scores, b.mu, b.rewards, b.acts)
    crit = AdaptiveOPCLoss(weights="none", logger=b.logger, crm=(1.0, 25_000, "delta"))
    crit.logger, crit.batch_users = _logger64(b), b.users
    expect = -float(f.mean()) + math.sqrt(float((f - f0).var()) / 25_000 + 1e-12)
    assert float(crit(b.pscore, b.scores, p, b.rewards, b.acts)) == pytest.approx(expect, rel=1e-9)
    assert not crit.per_example_additive and plain.per_example_additive


def test_the_gradient_of_every_term_matches_finite_differences():
    b = _batch(n=48)
    pen = lambda th=None: ((b.theta if th is None else th) ** 2).sum()
    crit = AdaptiveOPCLoss(weights="harmonic:0.05", logger=b.logger, support=("s2", 0.002), crm=(1.0, 1000, "delta"),
                           anchor_lambda=0.01, anchor_penalty=pen)
    crit.logger, crit.batch_users = _logger64(b), b.users

    def loss(th):
        crit.anchor_penalty = lambda: (th ** 2).sum()
        return crit(b.pscore, b.scores, b.policy(th), b.rewards, b.acts)

    th = b.theta.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(loss(th), th)[0]
    eps = 1e-6
    for i, j in ((0, 0), (1, 3), (4, 2)):
        tp, tm = th.detach().clone(), th.detach().clone()
        tp[i, j] += eps
        tm[i, j] -= eps
        fd = (float(loss(tp)) - float(loss(tm))) / (2 * eps)
        assert float(g[i, j]) == pytest.approx(fd, rel=1e-5, abs=1e-9)


class _Model(torch.nn.Module):
    """A tiny softmax policy over the catalog for the epoch hooks."""

    def __init__(self, b):
        super().__init__()
        self.b = b
        self.theta = torch.nn.Parameter(b.theta.detach().clone())
        self.actions = torch.arange(b.ia.shape[0])

    def forward(self, users):
        k = self.b.ux.shape[1]
        return torch.softmax((self.b.ux[users.long()] @ (torch.eye(k, dtype=D64) + self.theta) @ self.b.ia.T)
                             / self.b.T, dim=1)


class _Rows:
    def __init__(self, b):
        self.user_idx, self.action_idx, self.pscore, self.rewards = b.users, b.acts, b.pscore, b.rewards

    def __len__(self):
        return len(self.user_idx)


class _Scores:  # q̂ rows by user: the batch's rows are per logged row, so map each user to its first row
    def __init__(self, b):
        self.m = torch.zeros(40, b.scores.shape[1], dtype=D64)
        for i, u in enumerate(b.users.tolist()):
            self.m[u] = b.scores[i]

    def __getitem__(self, u):
        return self.m[u]


def test_adaptive_lambda_and_the_multiplier_follow_their_rules():
    b = _batch(n=80)
    model, rows, sc = _Model(b), _Rows(b), _Scores(b)
    with torch.no_grad():
        p_all = model(b.users)
    s2 = float((p_all ** 2 / b.mu).sum(1).mean())
    a2 = AdaptiveOPCLoss(weights="a2", logger=b.logger)
    a2.logger = _logger64(b)
    a2.epoch_start(model, rows, sc, "cpu")
    assert a2.lam == pytest.approx(a2_lambda(s2, 80), rel=1e-9)
    a3 = AdaptiveOPCLoss(weights="a3", logger=b.logger)
    a3.logger = _logger64(b)
    a3.epoch_start(model, rows, sc, "cpu")
    q = sc[b.users]
    f = {lam: AdaptiveOPCLoss(weights="none" if lam == 0 else f"harmonic:{lam}").contributions(
        b.pscore, q, p_all, b.rewards, b.acts) for lam in A3_GRID}
    mse = {lam: max(float((f[lam] - f[0.0]).mean()) ** 2 - float((f[lam] - f[0.0]).var()) / 80, 0.0)
           + float(f[lam].var()) / 80 for lam in A3_GRID}
    assert a3.lam == min(mse, key=mse.get)
    hard = AdaptiveOPCLoss(weights="none", logger=b.logger, support=("s2_hard", 1.5))
    hard.logger = _logger64(b)
    hard.epoch_end(model, rows, sc, "cpu")
    assert hard.nu == pytest.approx(max(0.0, 0.1 * (s2 / 1.5 - 1.0)), rel=1e-9)
