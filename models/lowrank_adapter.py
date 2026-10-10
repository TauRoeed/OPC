"""The matched rank-4 adapter of docs/structured_scenario_shift_study.md §4.

M_θ = I + U Vᵀ (U, V ∈ ℝ^{K×r}; D of U D Vᵀ absorbed into U: the same set of rank ≤ r matrices), applied on the item
side: a'_i = a_i + U (Vᵀ a_i), so ŝ_θ(u, i) = x_uᵀ M_θ a_i; the user side is the identity. U starts at 0 (the source
policy) and V at a seeded orthonormal V_0, the same for every arm and trial of a world. The policy is ``CFModel``'s
softmax(s ŝ / T) with the learned logit scale s = exp(30 θ_s).

Heads: ``policy`` (OPC), ``click`` (the ordinary likelihood: σ(s ŝ / T + c)), ``calib`` (the calibration-aware
likelihood: σ(exp(w_αᵀ x) (s ŝ / T + γ) + w_βᵀ x + c), the nuisance starting at 0). The gated family (§7) adds a
user gate: ŝ_θ = xᵀ a + g_θ(x) xᵀ U Vᵀ a, g_θ(x) = σ(v_θᵀ x + d_θ).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.models import CFModel
from utils.seeding import derive_seed

ADAPTER_RANK = 4


def initial_V(dim: int, rank: int, seed: int) -> np.ndarray:
    """V_0: a seeded K × r matrix with orthonormal columns (Q of a Gaussian draw, positive R diagonal)."""
    q, r = np.linalg.qr(np.random.default_rng(derive_seed(int(seed), "lowrank_adapter", "init")).normal(size=(dim, rank)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


class IdentityTransform(nn.Module):
    """The user side of the adapter: x unchanged (a transform, so ``CFModel`` freezes the user vectors)."""

    def forward(self, x: torch.Tensor, idx=None):
        return x


class LowRankCorrection(nn.Module):
    """a' = a + U (Vᵀ a) = (I + U Vᵀ) a (row vectors: a + (a V) Uᵀ); exactly a while U = 0."""

    def __init__(self, dim: int, rank: int, V0):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.as_tensor(np.asarray(V0, dtype=np.float32)).clone())

    def matrix(self) -> torch.Tensor:
        return self.U @ self.V.T

    def forward(self, x: torch.Tensor, idx=None):
        return x + (x @ self.V) @ self.U.T


class SharedLowRankModel(CFModel):
    """The rank-r adapter policy (``CFModel`` with ``IdentityTransform`` / ``LowRankCorrection`` and a learned logit
    scale) and its click heads; ``gated`` adds the user gate of the gated family."""

    MODES = ("policy", "click", "calib")

    def __init__(self, num_users, num_actions, embedding_dim, *, initial_user_embeddings, initial_actions_embeddings,
                 temperature, seed: int, mode: str = "policy", rank: int = ADAPTER_RANK, logit_scale: float = 1.0,
                 click_intercept: float = 0.0, gated: bool = False):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        super().__init__(num_users, num_actions, embedding_dim, initial_user_embeddings=initial_user_embeddings,
                         initial_actions_embeddings=initial_actions_embeddings, user_transform=IdentityTransform(),
                         action_transform=LowRankCorrection(embedding_dim, rank, initial_V(embedding_dim, rank, seed)),
                         temperature=temperature, logit_scale=logit_scale, learn_logit_scale=True)
        self.mode, self.rank, self.gated = mode, int(rank), bool(gated)
        if gated:
            self.gate_w = nn.Parameter(torch.zeros(embedding_dim))
            self.gate_b = nn.Parameter(torch.zeros(()))
        if mode in ("click", "calib"):
            self.click_intercept = nn.Parameter(torch.tensor(float(click_intercept), dtype=torch.float32))
        else:
            self.register_parameter("click_intercept", None)
        if mode == "calib":
            self.w_alpha = nn.Parameter(torch.zeros(embedding_dim))
            self.w_beta = nn.Parameter(torch.zeros(embedding_dim))
            self.gamma = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------------------------------- scores and logits
    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x @ self.gate_w + self.gate_b)

    def score(self, user_ids, item_ids=None) -> torch.Tensor:
        """ŝ_θ(u, i) for the users × (all or the given) items, before the logit scale and the temperature."""
        x = self.user_embeddings(user_ids)
        a = self.actions_embeddings(self.actions if item_ids is None else item_ids)
        t = self.action_transform
        if not self.gated:
            return x @ t(a).T
        return x @ a.T + self._gate(x)[:, None] * ((x @ t.U) @ (a @ t.V).T)

    def policy_logits(self, user_ids):
        if not self.gated:
            return super().policy_logits(user_ids)
        scale = self._scale()
        logits = self.score(user_ids)
        if scale is not None:
            logits = logits * scale
        return logits / max(self.temperature, 1e-8)

    def calib_logits(self, user_ids, logits: torch.Tensor) -> torch.Tensor:
        """exp(w_αᵀ x) (logits + γ) + w_βᵀ x + c for policy logits s ŝ / T of these users."""
        x = self.user_embeddings(user_ids)
        return torch.exp(x @ self.w_alpha)[:, None] * (logits + self.gamma) + (x @ self.w_beta)[:, None] + self.click_intercept

    def forward(self, user_ids):
        if self.mode == "policy":
            return super().forward(user_ids)
        logits = self.policy_logits(user_ids)
        if self.mode == "click":
            return (logits + self.click_intercept).unsqueeze(-1)
        return self.calib_logits(user_ids, logits).unsqueeze(-1)

    def get_params(self):
        """Exported vectors that score s ŝ_θ by dot product. The gated family has no exact two-vector export (the gate
        is per user times a rank-r term): users [s x, s g(x) (xᵀ U)] and items [a, Vᵀ a] do score it exactly."""
        if not self.gated:
            return super().get_params()
        was = self.training
        self.eval()
        try:
            x, a = self.user_embeddings.weight, self.actions_embeddings.weight
            t = self.action_transform
            ex = torch.cat([x, self._gate(x)[:, None] * (x @ t.U)], dim=1)
            ea = torch.cat([a, a @ t.V], dim=1)
            scale = self._scale()
            return (ex if scale is None else ex * scale), ea
        finally:
            self.train(was)

    def pair_click_logits(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """The click logits of the (user, item) pairs (numpy; the likelihood arms' native selection)."""
        with torch.no_grad():
            dev = next(self.parameters()).device
            u = torch.as_tensor(np.asarray(users, np.int64), device=dev)
            i = torch.as_tensor(np.asarray(items, np.int64), device=dev)
            out = []
            for s in range(0, len(u), 8192):
                uu, ii = u[s:s + 8192], i[s:s + 8192]
                x = self.user_embeddings(uu)
                a = self.actions_embeddings(ii)
                t = self.action_transform
                sc = (x * t(a)).sum(dim=1) if not self.gated else (
                    (x * a).sum(dim=1) + self._gate(x) * ((x @ t.U) * (a @ t.V)).sum(dim=1))
                scale = self._scale()
                lg = (sc if scale is None else sc * scale) / max(self.temperature, 1e-8)
                if self.mode == "calib":
                    lg = (torch.exp(x @ self.w_alpha) * (lg + self.gamma) + x @ self.w_beta + self.click_intercept)
                elif self.mode == "click":
                    lg = lg + self.click_intercept
                out.append(lg.double().cpu())
            return torch.cat(out).numpy()

    def correction_norms(self) -> dict:
        t = self.action_transform
        out = {"norm_UV": float(t.matrix().detach().norm()), "norm_U": float(t.U.detach().norm()),
               "norm_V": float(t.V.detach().norm())}
        if self.gated:
            out["norm_gate_w"] = float(self.gate_w.detach().norm())
        if self.mode == "calib":
            out.update({"norm_w_alpha": float(self.w_alpha.detach().norm()), "norm_w_beta": float(self.w_beta.detach().norm()),
                        "calib_gamma": float(self.gamma.detach())})
        return out


class LowRankAnchor(nn.Module):
    """R(θ) = E_i ‖U Vᵀ a_i‖² / E_i ‖a_i‖² (uniform over the catalog): the scale-normalized item displacement, the
    low-rank analogue of ``SourceAnchor`` (the user side is the identity). With S = E[a aᵀ]: tr(U Vᵀ S V Uᵀ) / E‖a‖²."""

    def __init__(self, item_vectors):
        super().__init__()
        v = np.asarray(item_vectors, dtype=np.float64)
        self.register_buffer("S_a", torch.as_tensor(v.T @ v / v.shape[0], dtype=torch.float32))
        self.register_buffer("m2_a", torch.tensor(float((v ** 2).sum(axis=1).mean()), dtype=torch.float32))

    def forward(self, model: SharedLowRankModel) -> torch.Tensor:
        t = model.action_transform
        W = t.V.T @ self.S_a @ t.V  # r × r
        return torch.trace(t.U @ W @ t.U.T) / self.m2_a
