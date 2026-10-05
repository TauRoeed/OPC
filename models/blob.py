"""BLOB's bandit layer (Sakhi, Bonner, Rohde & Vasile, KDD 2020) in PyTorch: the released TensorFlow graph of
criteo-research/blob @ e15cb38, ``models/models_organic_bandit.py`` (both variational families).

BLOB treats a user embedding ω̂_u (K) and an organic item embedding matrix Ψ (P × K) as given, and fits a Bayesian
logistic click model for a recommended action a (docs/blob_controlled_integration.md §1):

    logit(u, a) = s+(w_a) Ψ_loc[a]·ω̂_u + s+(w_b) Ψ_cov[a] ζ Lᵀ ω̂_u + w_c + κ_a,     L Lᵀ = Ψ_covᵀ Ψ_cov / P

with priors w_a ~ N(wa_m, wa_s²), w_b ~ N(wb_m, wa_s²) (sic: the released graph uses wa_s), w_c ~ N(wc_m, wc_s²),
κ ~ N(0, kappa_s² I), ζ ~ N(0, I) (K × K), and a factorized Gaussian posterior on all of them. The loss is the negative
ELBO, mean Bernoulli NLL under the local reparameterization trick + KL / N, minimized by TensorFlow 1's Adam.

Two variational families, as released:
- ``mnq`` (paper: BLOB-MNQ): ζ's posterior has two diagonal factors (K and K stds). The released noise term
  uses the PRIOR stds (``tf.exp(2*zeta_0_std1)``), not the posterior's, so ζ's posterior stds enter only the KL;
  reproduced here.
- ``nq`` (paper: BLOB-NQ): one std per element of ζ (K² stds), used in the noise term.

Other released details reproduced:
- with ``norm`` the graph divides Ψ's columns by their L2 norms *in place*, through ``Psi_cov = Psi_loc``, so both the
  prior mean and the covariance factor use the normalized Ψ (``alias_loc=True``; ``False`` gives the paper's
  unnormalized prior mean);
- the point prediction is ω̂ β̂ᵀ + κ̂ with β̂ = s+(μ_wa) Ψ_loc + s+(μ_wb) Ψ_cov μ_ζ Lᵀ, and it omits w_c (a constant,
  so rankings are unaffected);
- the optimizer is TensorFlow 1's ``AdamOptimizer(1e-3)`` (``TFAdam`` below: ε is added to sqrt(v) unscaled).

tests/test_blob_tf_reference.py checks this module step by step against the unmodified TensorFlow graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

BLOB_FAMILIES = ("mnq", "nq")


@dataclass(frozen=True)
class BlobPriors:
    """The released priors (simulate_abtest_with_bandit.py: wa_m -1, wb_m -6, wc_m -4.5, wa_s 1, wb_s 1, wc_s 10,
    kappa_s 0.01; s_zeta 1 in the graph). ``wb_s`` is kept for the record: the graph uses ``wa_s`` for w_b."""
    wa_m: float = -1.0
    wb_m: float = -6.0
    wc_m: float = -4.5
    wa_s: float = 1.0
    wb_s: float = 1.0
    wc_s: float = 10.0
    kappa_s: float = 0.01
    s_zeta: float = 1.0


def prepare_psi(psi: np.ndarray, *, norm: bool = True, alias_loc: bool = True):
    """(Ψ_loc, Ψ_cov, L) as the released graph builds them: Ψ_cov = Ψ with columns divided by their L2 norms when
    ``norm``; Ψ_loc is the same array (``alias_loc``, the released in-place division) or the original Ψ; L =
    chol(Ψ_covᵀ Ψ_cov / P)."""
    psi = np.array(psi, dtype=np.float32, copy=True)
    psi_cov = psi / np.linalg.norm(psi, axis=0, keepdims=True) if norm else psi.copy()
    psi_loc = psi_cov if (norm and alias_loc) else psi
    cov_k = psi_cov.T.dot(psi_cov) / psi.shape[0]
    chol = np.linalg.cholesky(cov_k).astype(np.float32)
    return psi_loc.astype(np.float32), psi_cov.astype(np.float32), chol


def _kl_multivariate(means: torch.Tensor, logstds: torch.Tensor, means_0: torch.Tensor, stds_0: torch.Tensor) -> torch.Tensor:
    """The released ``KL_multivariate``: factorized posterior vs a factorized prior, 0.5 (dets + trace term); the
    count term is the size of the first axis, as in the graph."""
    dets = 2.0 * torch.sum(torch.log(stds_0) - logstds) - float(means.shape[0])
    norm_trace = torch.sum(((means - means_0) ** 2 + torch.exp(logstds) ** 2) / stds_0 ** 2)
    return 0.5 * (dets + norm_trace)


class BlobBandit(nn.Module):
    """One BLOB bandit layer over fixed (ω̂, Ψ). ``family``: ``mnq`` or ``nq``. The variational parameters start where
    the released graph starts them (at the prior: means at the prior means, log-stds at the prior log-stds, ζ at
    0)."""

    def __init__(self, psi: np.ndarray, *, family: str = "mnq", priors: BlobPriors = BlobPriors(), norm: bool = True,
                 alias_loc: bool = True, l_scale: float = 1.0):
        """``l_scale``: L becomes l_scale · chol(Ψ̃ᵀΨ̃ / P). √(P / P₀) gives chol(Ψ̃ᵀΨ̃ / P₀), the catalog-normalized
        parameterization of docs/blob_prior_calibration.md; 1 is the release."""
        super().__init__()
        if family not in BLOB_FAMILIES:
            raise ValueError(f"family must be one of {BLOB_FAMILIES}, got {family!r}")
        self.family = family
        self.priors = priors
        psi_loc, psi_cov, chol = prepare_psi(psi, norm=norm, alias_loc=alias_loc)
        P, K = psi_cov.shape
        self.P, self.K = int(P), int(K)
        self.l_scale = float(l_scale)
        self.register_buffer("psi_loc", torch.as_tensor(psi_loc))
        self.register_buffer("psi_cov", torch.as_tensor(psi_cov))
        self.register_buffer("chol", torch.as_tensor(chol if l_scale == 1.0 else chol * np.float32(l_scale)))
        pr = priors
        f32 = dict(dtype=torch.float32)
        # priors (constants of the graph)
        self.register_buffer("zeta_0_std1", pr.s_zeta * torch.ones(K, 1, **f32))
        self.register_buffer("zeta_0_std2", torch.ones(K, 1, **f32))
        self.register_buffer("bias_0_mean", torch.tensor([pr.wc_m], **f32))
        self.register_buffer("bias_0_std", torch.tensor([pr.wc_s], **f32))
        self.register_buffer("wa_0_mean", torch.tensor([pr.wa_m], **f32))
        self.register_buffer("wa_0_std", torch.tensor([pr.wa_s], **f32))
        self.register_buffer("wb_0_mean", torch.tensor([pr.wb_m], **f32))
        self.register_buffer("wb_0_std", torch.tensor([pr.wa_s], **f32))  # the released graph: wb_0_std = [wa_s]
        self.register_buffer("kappa_0_mean", torch.zeros(P, 1, **f32))
        self.register_buffer("kappa_0_std", pr.kappa_s * torch.ones(P, 1, **f32))
        # variational posterior
        if family == "mnq":
            self.zeta_means = nn.Parameter(torch.zeros(K, K, **f32))
            self.zeta_logstd1 = nn.Parameter(torch.log(self.zeta_0_std1.clone()))
            self.zeta_logstd2 = nn.Parameter(torch.log(self.zeta_0_std2.clone()))
        else:
            self.zeta_means = nn.Parameter(torch.zeros(K * K, 1, **f32))
            self.zeta_logstd = nn.Parameter(torch.log(pr.s_zeta * torch.ones(K * K, 1, **f32)))
        self.bias_means = nn.Parameter(self.bias_0_mean.clone())
        self.bias_logstd = nn.Parameter(torch.log(self.bias_0_std.clone()))
        self.wa_means = nn.Parameter(self.wa_0_mean.clone())
        self.wa_logstd = nn.Parameter(torch.log(self.wa_0_std.clone()))
        self.wb_means = nn.Parameter(self.wb_0_mean.clone())
        self.wb_logstd = nn.Parameter(torch.log(self.wb_0_std.clone()))
        self.kappa_means = nn.Parameter(self.kappa_0_mean.clone())
        self.kappa_logstd = nn.Parameter(torch.log(self.kappa_0_std.clone()))

    # ----------------------------------------------------------------------------------------------- training
    def noisy_logits(self, x: torch.Tensor, a: torch.Tensor, noise: dict | None = None) -> torch.Tensor:
        """The released graph's ``predictions`` for user embeddings ``x`` (B × K) and actions ``a`` (B): one sample
        of the local reparameterization. ``noise``: the four standard-normal draws (``wa``, ``wb``, ``band``,
        ``bias``, each B × 1); drawn here when not given."""
        B, K = x.shape[0], self.K
        if noise is None:
            noise = {k: torch.randn(B, 1, device=x.device) for k in ("wa", "wb", "band", "bias")}
        l_omega = x @ self.chol
        psi_a_loc = self.psi_loc[a]
        psi_a = self.psi_cov[a]
        kappa_a = self.kappa_means[a]
        kappa_logstd_a = self.kappa_logstd[a]
        pred_org = F.softplus(self.wa_means + torch.exp(self.wa_logstd) * noise["wa"]) * \
            torch.sum(psi_a_loc * x, dim=1, keepdim=True)
        wb = F.softplus(self.wb_means + torch.exp(self.wb_logstd) * noise["wb"])
        if self.family == "mnq":
            # the released R1, R2 use the prior stds (tf.exp(2*zeta_0_std1)), not the posterior's
            r1 = (psi_a ** 2) @ torch.exp(2 * self.zeta_0_std1)
            r2 = (l_omega ** 2) @ torch.exp(2 * self.zeta_0_std2)
            mean_band = torch.sum((psi_a @ self.zeta_means) * l_omega, dim=1, keepdim=True)
            pred_band = wb * (mean_band + torch.sqrt(r1 * r2) * noise["band"])
        else:
            r = (l_omega.reshape(-1, 1, K) * psi_a.reshape(-1, K, 1)).reshape(-1, K * K)
            r_cov = (r ** 2) @ torch.exp(2 * self.zeta_logstd)
            pred_band = wb * (r @ self.zeta_means + torch.sqrt(r_cov) * noise["band"])
        pred_bias = self.bias_means + kappa_a + \
            torch.sqrt(torch.exp(2 * self.bias_logstd) + torch.exp(2 * kappa_logstd_a)) * noise["bias"]
        return pred_org + pred_band + pred_bias

    def kl(self) -> torch.Tensor:
        """KL(Q | P) summed over ζ, w_c, w_a, w_b and κ, as the released graph computes it."""
        K = self.K
        if self.family == "mnq":
            zeta_logstd = (self.zeta_logstd1.reshape(K, 1) + self.zeta_logstd2.reshape(1, K)).reshape(K * K)
            zeta_0_std = (self.zeta_0_std1.reshape(K, 1) * self.zeta_0_std2.reshape(1, K)).reshape(K * K)
            kl_zeta = _kl_multivariate(self.zeta_means.reshape(K * K), zeta_logstd,
                                       torch.zeros(K * K, device=zeta_logstd.device), zeta_0_std)
        else:
            kl_zeta = _kl_multivariate(self.zeta_means, self.zeta_logstd, torch.zeros_like(self.zeta_means),
                                       self.priors.s_zeta * torch.ones_like(self.zeta_means))
        return (_kl_multivariate(self.bias_means, self.bias_logstd, self.bias_0_mean, self.bias_0_std)
                + _kl_multivariate(self.wa_means, self.wa_logstd, self.wa_0_mean, self.wa_0_std)
                + _kl_multivariate(self.wb_means, self.wb_logstd, self.wb_0_mean, self.wb_0_std)
                + kl_zeta
                + _kl_multivariate(self.kappa_means, self.kappa_logstd, self.kappa_0_mean, self.kappa_0_std))

    def neg_elbo(self, x, a, y, n_total: int, noise: dict | None = None):
        """(neg_ELBO, neg_log_prob, kl) of one minibatch: mean Bernoulli NLL + KL / N, N the dataset size."""
        logits = self.noisy_logits(x, a, noise)
        nll = F.binary_cross_entropy_with_logits(logits, y.reshape(-1, 1).to(logits.dtype), reduction="mean")
        kl = self.kl()
        return nll + kl / float(n_total), nll, kl

    # --------------------------------------------------------------------------------------------- prediction
    @torch.no_grad()
    def point_beta(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(β̂ (P × K), κ̂ (P)): the released point estimate."""
        zeta = self.zeta_means.reshape(self.K, self.K)
        beta = F.softplus(self.wa_means) * self.psi_loc + F.softplus(self.wb_means) * (self.psi_cov @ zeta @ self.chol.T)
        return beta, self.kappa_means[:, 0].clone()

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """The released ``bandit_prediction``: ω̂ β̂ᵀ + κ̂ (no w_c), every action."""
        beta, kappa = self.point_beta()
        return x @ beta.T + kappa


class TFAdam:
    """TensorFlow 1's ``AdamOptimizer`` (β1 0.9, β2 0.999, ε 1e-8): lr_t = lr √(1 − β2ᵗ) / (1 − β1ᵗ),
    θ ← θ − lr_t m / (√v + ε). It differs from torch.optim.Adam only in where ε enters."""

    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = [p for p in params]
        self.lr, self.beta1, self.beta2, self.eps = float(lr), float(beta1), float(beta2), float(eps)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            p.sub_(lr_t * m / (torch.sqrt(v) + self.eps))


def tf_param_order(model: BlobBandit) -> list:
    """The parameters in the released graph's variable creation order (Adam keeps per-variable state, so the order
    only matters for readability)."""
    if model.family == "mnq":
        zeta = [model.zeta_means, model.zeta_logstd1, model.zeta_logstd2]
    else:
        zeta = [model.zeta_means, model.zeta_logstd]
    return zeta + [model.bias_means, model.bias_logstd, model.wa_means, model.wa_logstd, model.wb_means,
                   model.wb_logstd, model.kappa_means, model.kappa_logstd]


def fit_blob(model: BlobBandit, x: np.ndarray, a: np.ndarray, y: np.ndarray, *, epochs: int, batch_size: int = 1024,
             lr: float = 1e-3, seed: int = 0, device: torch.device | str | None = None) -> dict:
    """Train with the released loop: every epoch a shuffled pass in minibatches of ``batch_size`` (the last one
    short), one TF-Adam step per minibatch on the noisy negative ELBO with fresh noise. The shuffles and the noise
    come from ``seed``. Returns {'steps', 'finite', 'last_neg_elbo'}; a non-finite loss stops training."""
    device = torch.device(device) if device is not None else next(model.buffers()).device
    model.to(device)
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    at = torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    n = int(xt.shape[0])
    opt = TFAdam(tf_param_order(model), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_gen = torch.Generator(device=device).manual_seed(int(seed) + 1)
    steps, finite, last = 0, True, float("nan")
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, int(batch_size)):
            idx = order[s:s + int(batch_size)]
            b = int(idx.shape[0])
            noise = {k: torch.randn(b, 1, generator=noise_gen, device=device) for k in ("wa", "wb", "band", "bias")}
            loss, _nll, _kl = model.neg_elbo(xt[idx], at[idx], yt[idx], n, noise)
            if not torch.isfinite(loss):
                finite = False
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            last = float(loss.detach())
        if not finite:
            break
    return {"steps": steps, "finite": finite, "last_neg_elbo": last}


class BlobBanditBatch(nn.Module):
    """T independent BLOB bandit layers over the same (ω̂, Ψ), trained together: trial t has its own priors,
    learning rate and noise, and sees the same minibatches. Trial t equals a ``BlobBandit`` run with the same priors,
    batches and noise (tests/test_blob.py)."""

    def __init__(self, psi: np.ndarray, priors: list[BlobPriors], *, family: str = "mnq", norm: bool = True,
                 alias_loc: bool = True, l_scales: list[float] | None = None):
        """``l_scales``: one L scale per trial (``BlobBandit``'s ``l_scale``); None is the release for every trial."""
        super().__init__()
        if family not in BLOB_FAMILIES:
            raise ValueError(f"family must be one of {BLOB_FAMILIES}, got {family!r}")
        self.family = family
        psi_loc, psi_cov, chol = prepare_psi(psi, norm=norm, alias_loc=alias_loc)
        P, K = psi_cov.shape
        self.P, self.K, self.T = int(P), int(K), len(priors)
        self.register_buffer("psi_loc", torch.as_tensor(psi_loc))
        self.register_buffer("psi_cov", torch.as_tensor(psi_cov))
        self.register_buffer("chol", torch.as_tensor(chol))
        scales = np.ones(self.T, dtype=np.float32) if l_scales is None else np.asarray(l_scales, dtype=np.float32)
        if scales.shape != (self.T,):
            raise ValueError(f"l_scales must have one value per trial ({self.T}), got shape {scales.shape}")
        self.scaled = bool((scales != 1.0).any())
        self.register_buffer("l_scale", torch.as_tensor(scales))
        col = lambda name: torch.tensor([[getattr(p, name)] for p in priors], dtype=torch.float32)  # T x 1
        self.register_buffer("s_zeta", col("s_zeta"))
        self.register_buffer("bias_0_mean", col("wc_m"))
        self.register_buffer("bias_0_std", col("wc_s"))
        self.register_buffer("wa_0_mean", col("wa_m"))
        self.register_buffer("wa_0_std", col("wa_s"))
        self.register_buffer("wb_0_mean", col("wb_m"))
        self.register_buffer("wb_0_std", col("wa_s"))  # the released graph: wb_0_std = [wa_s]
        self.register_buffer("kappa_0_std", col("kappa_s")[:, :, None].expand(self.T, P, 1).clone())
        T = self.T
        if family == "mnq":
            self.zeta_means = nn.Parameter(torch.zeros(T, K, K))
            self.zeta_logstd1 = nn.Parameter(torch.log(self.s_zeta[:, :, None].expand(T, K, 1).clone()))
            self.zeta_logstd2 = nn.Parameter(torch.zeros(T, K, 1))
        else:
            self.zeta_means = nn.Parameter(torch.zeros(T, K * K, 1))
            self.zeta_logstd = nn.Parameter(torch.log(self.s_zeta[:, :, None].expand(T, K * K, 1).clone()))
        self.bias_means = nn.Parameter(self.bias_0_mean.clone())
        self.bias_logstd = nn.Parameter(torch.log(self.bias_0_std.clone()))
        self.wa_means = nn.Parameter(self.wa_0_mean.clone())
        self.wa_logstd = nn.Parameter(torch.log(self.wa_0_std.clone()))
        self.wb_means = nn.Parameter(self.wb_0_mean.clone())
        self.wb_logstd = nn.Parameter(torch.log(self.wb_0_std.clone()))
        self.kappa_means = nn.Parameter(torch.zeros(T, P, 1))
        self.kappa_logstd = nn.Parameter(torch.log(self.kappa_0_std.clone()))

    def params_in_order(self) -> list:
        zeta = [self.zeta_means, self.zeta_logstd1, self.zeta_logstd2] if self.family == "mnq" else \
            [self.zeta_means, self.zeta_logstd]
        return zeta + [self.bias_means, self.bias_logstd, self.wa_means, self.wa_logstd, self.wb_means,
                       self.wb_logstd, self.kappa_means, self.kappa_logstd]

    def noisy_logits(self, x: torch.Tensor, a: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """T × B logits; ``noise`` is T × B × 4 (wa, wb, band, bias)."""
        K = self.K
        l_omega = x @ self.chol
        psi_a_loc, psi_a = self.psi_loc[a], self.psi_cov[a]
        org = torch.sum(psi_a_loc * x, dim=1)[None, :]
        pred_org = F.softplus(self.wa_means + torch.exp(self.wa_logstd) * noise[..., 0]) * org
        wb = F.softplus(self.wb_means + torch.exp(self.wb_logstd) * noise[..., 1])
        if self.family == "mnq":
            # the released R1, R2 use the prior stds: exp(2 * s_zeta) and exp(2 * 1), not the posterior's
            r1 = (psi_a ** 2).sum(1)[None, :] * torch.exp(2 * self.s_zeta)  # T x B
            r2 = ((l_omega ** 2).sum(1) * float(np.exp(2.0)))[None, :]
            mean_band = torch.einsum("bk,tkj,bj->tb", psi_a, self.zeta_means, l_omega)
            pred_band = wb * (mean_band + torch.sqrt(r1 * r2) * noise[..., 2])
        else:
            r = (l_omega.reshape(-1, 1, K) * psi_a.reshape(-1, K, 1)).reshape(-1, K * K)
            mean_band = torch.einsum("bq,tq->tb", r, self.zeta_means[:, :, 0])
            r_cov = torch.einsum("bq,tq->tb", r ** 2, torch.exp(2 * self.zeta_logstd[:, :, 0]))
            pred_band = wb * (mean_band + torch.sqrt(r_cov) * noise[..., 2])
        if self.scaled:  # L → c_t L scales the band term's mean and its noise std by c_t
            pred_band = pred_band * self.l_scale[:, None]
        kappa_a = self.kappa_means[:, a, 0]
        kappa_logstd_a = self.kappa_logstd[:, a, 0]
        pred_bias = self.bias_means + kappa_a + \
            torch.sqrt(torch.exp(2 * self.bias_logstd) + torch.exp(2 * kappa_logstd_a)) * noise[..., 3]
        return pred_org + pred_band + pred_bias

    def kl(self) -> torch.Tensor:
        """Per-trial KL (T)."""
        K, P = self.K, self.P

        def kl(means, logstds, means_0, stds_0, count):
            dets = 2.0 * torch.sum(torch.log(stds_0) - logstds, dim=tuple(range(1, means.dim()))) - float(count)
            tr = torch.sum(((means - means_0) ** 2 + torch.exp(logstds) ** 2) / stds_0 ** 2, dim=tuple(range(1, means.dim())))
            return 0.5 * (dets + tr)

        T = self.T
        if self.family == "mnq":
            zl = (self.zeta_logstd1.reshape(T, K, 1) + self.zeta_logstd2.reshape(T, 1, K)).reshape(T, K * K)
            z0 = (self.s_zeta.reshape(T, 1, 1) * torch.ones(T, K, K, device=zl.device)).reshape(T, K * K)
            kl_zeta = kl(self.zeta_means.reshape(T, K * K), zl, torch.zeros_like(zl), z0, K * K)
        else:
            z0 = self.s_zeta.reshape(T, 1, 1).expand_as(self.zeta_means)
            kl_zeta = kl(self.zeta_means, self.zeta_logstd, torch.zeros_like(self.zeta_means), z0, K * K)
        return (kl(self.bias_means, self.bias_logstd, self.bias_0_mean, self.bias_0_std, 1)
                + kl(self.wa_means, self.wa_logstd, self.wa_0_mean, self.wa_0_std, 1)
                + kl(self.wb_means, self.wb_logstd, self.wb_0_mean, self.wb_0_std, 1)
                + kl_zeta
                + kl(self.kappa_means, self.kappa_logstd, torch.zeros_like(self.kappa_means), self.kappa_0_std, P))

    def neg_elbo(self, x, a, y, n_total: int, noise: torch.Tensor) -> torch.Tensor:
        """Per-trial negative ELBO (T)."""
        logits = self.noisy_logits(x, a, noise)
        nll = F.binary_cross_entropy_with_logits(logits, y[None, :].expand_as(logits).to(logits.dtype),
                                                 reduction="none").mean(dim=1)
        return nll + self.kl() / float(n_total)

    @torch.no_grad()
    def point(self, t: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """(β̂, κ̂, μ_wc) of trial t."""
        zeta = self.zeta_means[t].reshape(self.K, self.K)
        chol = self.chol * self.l_scale[t] if self.scaled else self.chol
        beta = F.softplus(self.wa_means[t]) * self.psi_loc + \
            F.softplus(self.wb_means[t]) * (self.psi_cov @ zeta @ chol.T)
        return beta, self.kappa_means[t, :, 0].clone(), float(self.bias_means[t, 0])

    @torch.no_grad()
    def finite(self) -> torch.Tensor:
        ok = torch.ones(self.T, dtype=torch.bool, device=self.wa_means.device)
        for p in self.params_in_order():
            ok &= torch.isfinite(p.reshape(self.T, -1)).all(dim=1)
        return ok


class TFAdamBatch(TFAdam):
    """TF-Adam with a learning rate per trial (the leading axis of every parameter)."""

    def __init__(self, params, lrs, **kw):
        super().__init__(params, lr=1.0, **kw)
        self.lrs = torch.as_tensor(np.asarray(lrs, dtype=np.float32))
        # each parameter's learning rates, on its device and broadcastable over it (copied once, not every step)
        self._lrs = [self.lrs.to(p.device).reshape((-1,) + (1,) * (p.dim() - 1)) for p in self.params]

    @torch.no_grad()
    def step(self):
        self.t += 1
        scale = float(np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t))
        for p, m, v, lrs in zip(self.params, self.m, self.v, self._lrs):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            lr = lrs * scale
            p.sub_(lr * m / (torch.sqrt(v) + self.eps))


def fit_blob_batch(model: BlobBanditBatch, x: np.ndarray, a: np.ndarray, y: np.ndarray, *, epochs: int, lrs,
                   batch_size: int = 1024, order_seed: int = 0, noise_seed: int = 1,
                   device: torch.device | str | None = None, noise_index=None) -> dict:
    """Train T trials together with the released loop (``fit_blob``): one shared shuffled batch order per epoch
    (from ``order_seed``), independent noise per trial. A trial whose loss turns non-finite is frozen at its last
    finite parameters (its gradient is zeroed). ``noise_index`` (T): trials with the same index share one noise
    stream (paired variants of one configuration); None gives every trial its own. Returns {'steps', 'finite' (T,)}."""
    device = torch.device(device) if device is not None else model.psi_loc.device
    model.to(device)
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    at = torch.as_tensor(np.asarray(a, dtype=np.int64), device=device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
    n, T = int(xt.shape[0]), model.T
    params = model.params_in_order()
    opt = TFAdamBatch(params, lrs)
    gen = torch.Generator(device="cpu").manual_seed(int(order_seed))
    noise_gen = torch.Generator(device=device).manual_seed(int(noise_seed))
    alive = torch.ones(T, dtype=torch.bool, device=device)
    if noise_index is not None:
        stream = torch.as_tensor(np.asarray(noise_index, dtype=np.int64), device=device)
        if stream.shape != (T,):
            raise ValueError(f"noise_index must have one entry per trial ({T})")
        n_streams = int(stream.max()) + 1
    steps = 0
    for _ in range(int(epochs)):
        order = torch.randperm(n, generator=gen).to(device)
        for s in range(0, n, int(batch_size)):
            idx = order[s:s + int(batch_size)]
            if noise_index is None:
                noise = torch.randn(T, int(idx.shape[0]), 4, generator=noise_gen, device=device)
            else:
                noise = torch.randn(n_streams, int(idx.shape[0]), 4, generator=noise_gen, device=device)[stream]
            loss = model.neg_elbo(xt[idx], at[idx], yt[idx], n, noise)
            alive &= torch.isfinite(loss)
            opt.zero_grad()
            torch.where(alive, loss, torch.zeros_like(loss)).sum().backward()
            with torch.no_grad():  # a frozen trial's gradient is 0; multiplying by the mask needs no host sync
                keep = alive.to(torch.float32)
                for p in params:
                    if p.grad is not None:
                        p.grad.nan_to_num_(0.0).mul_(keep.reshape((-1,) + (1,) * (p.grad.dim() - 1)))
            opt.step()
            steps += 1
    return {"steps": steps, "finite": (alive & model.finite()).cpu().numpy()}
