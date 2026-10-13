"""The OPC objectives of docs/opc_adaptive_estimator_study.md §5 (families A–C), as one minibatch criterion.

Per logged row i of a minibatch, with the training reward model q̂ (the cross-fitted lookup) and the current policy π:

    f_i = Σ_j π(j|x_i) q̂(x_i, j) + g(w_i) (r_i − q̂(x_i, a_i)),     w_i = π(a_i|x_i) / μ_i,

the per-row DR contribution of the existing OPC arms (``dr_sndr_surrogate``: the direct gradient, no
self-normalization). The criterion is the negative of

    mean_B(f) − λ_anchor R(θ) − [support term] − [CRM term]

- **weights** g: a fixed spec (``none``, ``harmonic:λ``) or adaptive. ``a2``: λ = min(1, sqrt(2 log(1/δ) /
  (3 n Ŝ2))), Ŝ2 the enumerated second moment of π/μ over the training rows' users (Metelli, Russo and Restelli
  2021, Theorem 5.1, with I₂ = S2). ``a3``: the λ of a grid minimizing max(0, mean(d)² − var(d)/n) + var(f_λ)/n
  over the training rows, d = f_λ − f_raw. Both are set at the start of each epoch from the current θ
  (``epoch_start``) and held fixed during the epoch: no gradient flows through λ.
- **support**: ``("s2", β)``: β (S2_B − 1); ``("s2_hard", C)``: the augmented Lagrangian on the normalized violation
  g = S2/C − 1, ν g_B + (ρ/2) max(0, g_B)², ν ← max(0, ν + ρ g) at each epoch's end over the training users
  (``epoch_end``), ρ = 0.1;
  ``("kl", β)``: β KL_B(π‖μ). S2_B and KL_B are means over the minibatch's users of Σ_a π²/μ and Σ_a π log(π/μ),
  enumerated over the catalog with the known logger μ.
- **crm**: ``(γ, N, "abs")``: γ sqrt(var_B(f) / N + 1e-12), the standard error of the full-data estimate with the
  minibatch's per-row variance; ``(γ, N, "delta")``: the same for f − f₀, f₀ the source policy's contribution (the
  source is the logger, so w₀ ≡ 1 and f₀ = Σ_j μ q̂ + (r − q̂)).

With weights fixed, no support and no CRM term, the criterion equals the existing ``DRPolicyLoss`` (with its anchor
through ``AnchoredPolicyLoss``); tests/test_adaptive_losses.py checks it.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from models.custom_losses import dr_sndr_surrogate
from utils.importance_weights import parse_weight_spec

A2_DELTA = 0.05
A3_GRID = (0.0, 0.005, 0.01, 0.03, 0.1, 0.2)
HARD_RHO = 0.1  # value units on the normalized violation S2/C − 1 (docs/opc_adaptive_estimator_study.md §5.2a:
# ρ = 1 on the raw S2 − C dwarfs the value and stalls Adam)


def a2_lambda(s2: float, n: int, delta: float = A2_DELTA) -> float:
    return float(min(1.0, math.sqrt(2.0 * math.log(1.0 / delta) / (3.0 * max(float(s2), 1.0) * int(n)))))


class LoggerRows:
    """μ(·|x) of the known logger, for given users: softmax(x · a / T) over the catalog (the source vectors)."""

    def __init__(self, user_vectors, item_vectors, temperature: float, device):
        self.x = torch.as_tensor(np.asarray(user_vectors), dtype=torch.float32, device=device)
        self.a = torch.as_tensor(np.asarray(item_vectors), dtype=torch.float32, device=device)
        self.T = float(temperature)

    def __call__(self, users) -> torch.Tensor:
        return torch.softmax((self.x[users.long()] @ self.a.T) / self.T, dim=1)


class AdaptiveOPCLoss(nn.Module):
    needs_qhat = True

    def __init__(self, *, weights="harmonic:0.1", logger: LoggerRows | None = None, support=None, crm=None,
                 anchor_lambda: float = 0.0, anchor_penalty=None, n_train: int | None = None, log_eps: float = 1e-10):
        super().__init__()
        self.weights = str(weights)
        self.adaptive = self.weights in ("a2", "a3")
        self.lam = 0.1 if self.adaptive else None  # the adaptive λ before the first epoch_start
        if not self.adaptive:
            self.mode, self.param = parse_weight_spec(self.weights)
        self.logger = logger
        self.support = support
        self.crm = crm
        self.anchor_lambda, self.anchor_penalty = float(anchor_lambda), anchor_penalty
        self.n_train = n_train
        self.log_eps = log_eps
        self.nu = 0.0  # the hard constraint's multiplier
        self.batch_users = None
        self.history = []  # per epoch: λ, S2, ν
        if (support is not None or (crm is not None and crm[2] == "delta") or self.adaptive) and logger is None:
            raise ValueError("the support terms, delta-CRM and the adaptive λ need the known logger")
        if support is not None and support[0] not in ("s2", "s2_hard", "kl"):
            raise ValueError(f"support must be ('s2', β), ('s2_hard', C) or ('kl', β), got {support!r}")
        if crm is not None and (crm[2] not in ("abs", "delta") or not crm[0] >= 0):
            raise ValueError(f"crm must be (γ ≥ 0, N, 'abs' | 'delta'), got {crm!r}")

    # the training loop sets the minibatch's users when this is True (training_utils.run_train_loop)
    @property
    def wants_users(self) -> bool:
        return self.support is not None or (self.crm is not None and self.crm[2] == "delta")

    @property
    def per_example_additive(self) -> bool:  # the CRM penalty is a per-minibatch statistic
        return self.crm is None

    def _spec(self):
        if self.adaptive:
            return ("none", math.inf) if self.lam <= 0 else ("harmonic", float(self.lam))
        return self.mode, self.param

    def contributions(self, pscore, scores, policy_prob, rewards, actions, spec=None) -> torch.Tensor:
        mode, param = spec or self._spec()
        return dr_sndr_surrogate(scores, policy_prob, actions, rewards, pscore, use_iw=True, use_log_trick=False,
                                 log_eps=self.log_eps, iw_mode=mode, iw_param=param, normalizer="none")

    def forward(self, pscore, scores, policy_prob, rewards, actions):
        f = self.contributions(pscore, scores, policy_prob, rewards, actions)
        loss = -f.mean()
        if self.anchor_lambda > 0.0:
            loss = loss + self.anchor_lambda * self.anchor_penalty()
        mu = None
        if self.wants_users:
            if self.batch_users is None:
                raise RuntimeError("the support / delta terms need the minibatch's users (run_train_loop sets them)")
            mu = self.logger(self.batch_users)
        if self.support is not None:
            kind, par = self.support
            if kind == "kl":
                kl = (policy_prob * (torch.log(policy_prob.clamp(min=self.log_eps)) - torch.log(mu.clamp(min=1e-30))))
                loss = loss + float(par) * kl.sum(dim=1).mean()
            else:
                s2 = (policy_prob * policy_prob / mu.clamp(min=1e-30)).sum(dim=1).mean()
                if kind == "s2":
                    loss = loss + float(par) * (s2 - 1.0)
                else:
                    viol = s2 / float(par) - 1.0
                    loss = loss + self.nu * viol + 0.5 * HARD_RHO * torch.clamp(viol, min=0.0) ** 2
        if self.crm is not None:
            gamma, n_full, kind = self.crm
            d = f
            if kind == "delta":
                idx = torch.arange(actions.shape[0], device=f.device)
                q_a = scores[idx, actions]
                f0 = (mu * scores).sum(dim=1) + (rewards - q_a)
                d = f - f0.to(f.dtype)
            var = d.var(unbiased=True) if d.numel() > 1 else d.new_zeros(())
            loss = loss + float(gamma) * torch.sqrt(var / float(n_full) + 1e-12)
        return loss

    # ------------------------------------------------------------------------------------------- epoch hooks
    @torch.no_grad()
    def _full_pass(self, model, dataset, scores_all, device, cells: int = 32 * 1024 * 1024):
        """Over every training row, in order: per-row S2(x_i) (enumerated) and, for A3, the per-row contributions
        at each grid λ."""
        was = model.training
        model.eval()
        try:
            n = len(dataset)
            n_items = int(model.actions.numel())
            step = max(1, int(cells) // max(1, n_items))
            s2, fs = [], {lam: [] for lam in A3_GRID} if self.weights == "a3" else {}
            for s in range(0, n, step):
                users = dataset.user_idx[s:s + step].to(device)
                acts = dataset.action_idx[s:s + step].to(device)
                ps = dataset.pscore[s:s + step].to(device)
                rw = dataset.rewards[s:s + step].to(device)
                prob = model(users)
                prob = prob.squeeze(-1) if prob.dim() == 3 else prob
                mu = self.logger(users)
                s2.append((prob.double() ** 2 / mu.double().clamp(min=1e-300)).sum(dim=1))
                if fs:
                    q = scores_all[users.long()]
                    q = (q.squeeze(-1) if q.dim() == 3 else q).to(prob.device)
                    for lam in A3_GRID:
                        spec = ("none", math.inf) if lam == 0 else ("harmonic", lam)
                        fs[lam].append(self.contributions(ps, q, prob, rw, acts, spec=spec).double())
            return torch.cat(s2), {lam: torch.cat(v) for lam, v in fs.items()}
        finally:
            model.train(was)

    def epoch_start(self, model, dataset, scores_all, device):
        if not self.adaptive:
            return
        s2_rows, fs = self._full_pass(model, dataset, scores_all, device)
        n = int(s2_rows.numel())
        rec = {"S2_train": float(s2_rows.mean())}
        if self.weights == "a2":
            self.lam = a2_lambda(float(s2_rows.mean()), n)
        else:
            f0, best = fs[0.0], None
            for lam in A3_GRID:
                d = fs[lam] - f0
                mse = max(float(d.mean()) ** 2 - float(d.var()) / n, 0.0) + float(fs[lam].var()) / n
                rec[f"mse_{lam:g}"] = mse
                if best is None or mse < best[0]:
                    best = (mse, lam)
            self.lam = float(best[1])
        rec["lambda"] = float(self.lam)
        self.history.append(rec)

    def epoch_end(self, model, dataset, scores_all, device):
        if self.support is None or self.support[0] != "s2_hard":
            return
        s2_rows, _ = self._full_pass_s2_only(model, dataset, device)
        s2 = float(s2_rows.mean())
        self.nu = max(0.0, self.nu + HARD_RHO * (s2 / float(self.support[1]) - 1.0))
        self.history.append({"S2_train": s2, "nu": self.nu})

    @torch.no_grad()
    def _full_pass_s2_only(self, model, dataset, device, cells: int = 32 * 1024 * 1024):
        was = model.training
        model.eval()
        try:
            n_items = int(model.actions.numel())
            step = max(1, int(cells) // max(1, n_items))
            out = []
            for s in range(0, len(dataset), step):
                users = dataset.user_idx[s:s + step].to(device)
                prob = model(users)
                prob = prob.squeeze(-1) if prob.dim() == 3 else prob
                mu = self.logger(users)
                out.append((prob.double() ** 2 / mu.double().clamp(min=1e-300)).sum(dim=1))
            return torch.cat(out), None
        finally:
            model.train(was)
