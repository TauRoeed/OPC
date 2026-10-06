"""One global correction model, one regularizer, three training objectives (docs/shared_objective_study.md §1-§3).

The model is OPC's policy (``CFModel`` with ``GlobalLinearCorrection`` on both sides and the learned logit scale
s = exp(30 θ_s)): u' = (I + D_u) x + b_u, a' = (I + D_a) a + b_a, g(u, j) = u'·a'_j, π(j|u) = softmax_j(s g / T).
The likelihood arms read the same logits as a click model, q(u, j) = σ(s g / T + c), with one intercept c that no
ranking or policy depends on. Every objective adds λ R(θ), R the relative mean squared displacement of the user and item
representations from the source (``SourceAnchor``).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import LOGIT_SCALE_SPEED, CFModel, GlobalLinearCorrection

SHARED_OBJECTIVES = ("likelihood", "iw_likelihood", "opc")
LAMBDA_GRID = (0.0, 0.001, 0.01, 0.1, 1.0)  # §2: the same candidate strengths for every objective
IW_CLIP = 10.0  # the one pre-specified robustness variant of the weighted likelihood (§3)
HEAD_MIN_SLOPE = 1e-3  # the click head's scale is a positive slope (§3)


class SharedCorrectionModel(CFModel):
    """OPC's policy model (``CFModel`` with a ``GlobalLinearCorrection`` per side and a learned logit scale). In
    ``policy`` mode it is ``CFModel`` exactly (the softmax policy over all items); in ``click`` mode ``forward``
    returns the click logits s g / T + c instead, c being ``click_intercept``."""

    MODES = ("policy", "click")

    def __init__(self, num_users, num_actions, embedding_dim, *, initial_user_embeddings, initial_actions_embeddings,
                 temperature, mode: str = "policy", logit_scale: float = 1.0, click_intercept: float = 0.0):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        super().__init__(num_users, num_actions, embedding_dim, initial_user_embeddings=initial_user_embeddings,
                         initial_actions_embeddings=initial_actions_embeddings,
                         user_transform=GlobalLinearCorrection(embedding_dim),
                         action_transform=GlobalLinearCorrection(embedding_dim),
                         temperature=temperature, logit_scale=logit_scale, learn_logit_scale=True)
        self.mode = mode
        if mode == "click":
            self.click_intercept = nn.Parameter(torch.tensor(float(click_intercept), dtype=torch.float32))
        else:
            self.register_parameter("click_intercept", None)

    def forward(self, user_ids):
        if self.mode == "policy":
            return super().forward(user_ids)
        return (self.policy_logits(user_ids) + self.click_intercept).unsqueeze(-1)

    def correction_norms(self) -> dict:
        """Frobenius / Euclidean norms of the four correction parameters."""
        u, a = self.user_transform, self.action_transform
        return {"norm_D_u": float(u.delta.detach().norm()), "norm_b_u": float(u.bias.detach().norm()),
                "norm_D_a": float(a.delta.detach().norm()), "norm_b_a": float(a.bias.detach().norm())}


class SourceAnchor(nn.Module):
    """R(θ) = E_u ‖u' − x‖² / E_u ‖x‖²  +  E_j ‖a' − a‖² / E_j ‖a‖², uniform over the catalog's users and items.

    With the source's second moment S = E[x xᵀ] and mean μ: E ‖D x + b‖² = tr(D S Dᵀ) + 2 bᵀ D μ + ‖b‖² (D acts on
    column vectors, as ``GlobalLinearCorrection``'s ``F.linear`` does on rows). Source quantities only."""

    def __init__(self, user_vectors, item_vectors):
        super().__init__()
        for name, v in (("x", user_vectors), ("a", item_vectors)):
            v = np.asarray(v, dtype=np.float64)
            self.register_buffer(f"S_{name}", torch.as_tensor(v.T @ v / v.shape[0], dtype=torch.float32))
            self.register_buffer(f"mu_{name}", torch.as_tensor(v.mean(axis=0), dtype=torch.float32))
            self.register_buffer(f"m2_{name}", torch.tensor(float((v ** 2).sum(axis=1).mean()), dtype=torch.float32))

    @staticmethod
    def _displacement(transform: GlobalLinearCorrection, S: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        D, b = transform.delta, transform.bias
        return torch.trace(D @ S @ D.T) + 2.0 * torch.dot(b, D @ mu) + torch.dot(b, b)

    def forward(self, model: CFModel) -> torch.Tensor:
        return (self._displacement(model.user_transform, self.S_x, self.mu_x) / self.m2_x
                + self._displacement(model.action_transform, self.S_a, self.mu_a) / self.m2_a)


def iw_weights(pscore, n_actions: int, clip: float | None = None):
    """w = μ(a|x) / p with μ uniform over the catalog: 1 / (P p), clipped at ``clip`` when given (numpy or torch)."""
    w = 1.0 / (float(n_actions) * pscore)
    if clip is None:
        return w
    return w.clamp(max=float(clip)) if isinstance(w, torch.Tensor) else np.minimum(w, float(clip))


class LikelihoodLoss(nn.Module):
    """(1/n) Σ_t w_t ℓ(r_t, σ(z_t)) + λ R(θ): z the model's click logits at the logged items, ℓ the Bernoulli
    negative log-likelihood, w = 1 (``weighting='none'``) or μ/p toward the uniform action distribution
    (``'uniform'``, raw or clipped at ``clip``). A mean of per-row terms (no self-normalization); the logged
    propensities enter only through w, the reward model not at all."""

    needs_qhat = False
    per_example_additive = True
    WEIGHTINGS = ("none", "uniform")

    def __init__(self, n_actions: int, *, weighting: str = "none", clip: float | None = None, lam: float = 0.0,
                 penalty=None):
        super().__init__()
        if weighting not in self.WEIGHTINGS:
            raise ValueError(f"weighting must be one of {self.WEIGHTINGS}, got {weighting!r}")
        if weighting == "none" and clip is not None:
            raise ValueError("clipping applies to the weighted likelihood only")
        if float(lam) > 0.0 and penalty is None:
            raise ValueError("lam > 0 needs the penalty R(θ)")
        self.n_actions, self.weighting, self.clip = int(n_actions), weighting, clip
        self.lam, self.penalty = float(lam), penalty

    def row_weights(self, pscore: torch.Tensor) -> torch.Tensor | None:
        if self.weighting == "none":
            return None
        return iw_weights(pscore.to(torch.float64), self.n_actions, self.clip)

    def forward(self, pscore, scores, logits, rewards, actions):
        _ = scores
        rows = torch.arange(actions.shape[0], device=logits.device)
        z = logits[rows, actions.long()]
        nll = F.binary_cross_entropy_with_logits(z, rewards.to(z.dtype), reduction="none")
        w = self.row_weights(pscore)
        loss = (nll * w.to(z.dtype)).mean() if w is not None else nll.mean()
        if self.lam > 0.0:
            loss = loss + self.lam * self.penalty()
        return loss


class AnchoredPolicyLoss(nn.Module):
    """A per-row policy loss plus λ R(θ) (λ > 0; with λ = 0 the trainer uses the base loss itself, unchanged)."""

    def __init__(self, base: nn.Module, lam: float, penalty):
        super().__init__()
        if not float(lam) > 0.0:
            raise ValueError("use the base loss itself when lam = 0")
        if not getattr(base, "per_example_additive", False) or getattr(base, "normalization", "none") != "none":
            raise ValueError("the source anchor is added to per-row losses without a normalizer (dr) only")
        self.base, self.lam, self.penalty = base, float(lam), penalty
        self.needs_qhat = getattr(base, "needs_qhat", True)
        self.per_example_additive = True

    def forward(self, pscore, scores, policy_prob, rewards, actions):
        return self.base(pscore, scores, policy_prob, rewards, actions) + self.lam * self.penalty()


def fit_click_head(g0, clicks, weights=None) -> tuple[float, float]:
    """(θ_s, c) of the click head q = σ(s g0 + c), s = exp(30 θ_s), minimizing the (weighted) Bernoulli NLL with the
    correction maps at the identity: a one-feature logistic regression without penalty. ``g0``: the logger's logits
    ⟨x, a⟩ / T at the logged pairs. The slope is a scale, floored at ``HEAD_MIN_SLOPE``."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(g0, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(clicks, dtype=np.float64).round().astype(int)
    fit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000, tol=1e-10)
    fit.fit(X, y, sample_weight=None if weights is None else np.asarray(weights, dtype=np.float64))
    slope = max(float(fit.coef_[0, 0]), HEAD_MIN_SLOPE)
    return float(np.log(slope) / LOGIT_SCALE_SPEED), float(fit.intercept_[0])


def weighted_nll(z, clicks, weights=None) -> float:
    """(1/n) Σ w ℓ(r, σ(z)) in float64 (validation NLL; ``weights`` None for the plain NLL)."""
    z = np.asarray(z, dtype=np.float64)
    r = np.asarray(clicks, dtype=np.float64)
    nll = np.logaddexp(0.0, z) - r * z  # −r log σ(z) − (1 − r) log(1 − σ(z))
    return float(np.mean(nll if weights is None else np.asarray(weights, dtype=np.float64) * nll))


def weight_stats(pscore, n_actions: int, clip: float = IW_CLIP) -> dict:
    """The uniform-reference weights w = 1 / (P p) of logged rows: mean, variance, maximum, quantiles, effective sample
    size (Σw)² / Σw² (and as a share of n), and how much clipping at ``clip`` changes."""
    w = iw_weights(np.asarray(pscore, dtype=np.float64), n_actions)
    n = len(w)
    out = {"w_mean": float(w.mean()), "w_var": float(w.var()), "w_max": float(w.max())}
    for q in (0.5, 0.9, 0.99, 0.999):
        out[f"w_q{q * 100:g}"] = float(np.quantile(w, q))
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    wc = np.minimum(w, clip)
    ess_c = float(wc.sum() ** 2 / (wc ** 2).sum())
    out.update({"w_ess": ess, "w_ess_share": ess / n, "w_clip_share": float((w > clip).mean()),
                "w_clip_mass_share": float((w - wc).sum() / w.sum()), "wclip_mean": float(wc.mean()),
                "wclip_ess": ess_c, "wclip_ess_share": ess_c / n})
    return out
