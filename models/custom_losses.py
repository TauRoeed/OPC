import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/code")

import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch.nn as nn

from utils.importance_weights import parse_weight_spec

PROPENSITY_MODES = ("logged", "uniform")


def _align_policy_scores(scores, policy_prob):
    """(batch, n_actions) tensors; squeeze trailing singleton dims."""
    if policy_prob.dim() == 3 and policy_prob.shape[-1] == 1:
        policy_prob = policy_prob.squeeze(-1)
    if scores.dim() == 3 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    return scores, policy_prob


def uses_importance_weighting(propensity_mode: str) -> bool:
    """``uniform`` = no IW (iw=1); ``logged`` = iw = pi_e / pi_b."""
    return str(propensity_mode).lower() != "uniform"


def batch_mc_kl(pi_e_at_action, pi_b_at_action, log_eps=1e-10):
    """Batch MC estimate of E[KL(pi_b||pi_e)] from logged a ~ pi_b.

    Uses mean over batch of (log pi_b - log pi_e) at logged actions only.
    No softmax across batch, no full action sum.
    """
    pi_b = pi_b_at_action.detach().clamp(min=log_eps)
    pi_e = pi_e_at_action.clamp(min=log_eps)
    return (torch.log(pi_b) - torch.log(pi_e)).mean()


def importance_weights(pi_e_at_action, pscore, use_iw: bool, log_eps=1e-10):
    if use_iw:
        return pi_e_at_action / pscore.clamp(min=log_eps)
    return torch.ones_like(pi_e_at_action)


def clipped_importance_weights(pi_e_at_action, pscore, clip_m, use_iw: bool, log_eps=1e-10):
    """IPS weights clipped at M (Swaminathan & Joachims, CRM / Eq. 2)."""
    iw = importance_weights(pi_e_at_action, pscore, use_iw, log_eps)
    if not use_iw:
        return iw
    return torch.clamp(iw, max=float(clip_m))


def shrink_importance_weights(pi_e_at_action, pscore, shrink_lambda, use_iw: bool, log_eps=1e-10):
    """Su et al. DR shrinkage: ŵ = (λ w) / (λ + w²)."""
    iw = importance_weights(pi_e_at_action, pscore, use_iw, log_eps)
    if not use_iw:
        return iw
    lam = float(shrink_lambda)
    return (lam * iw) / (lam + iw * iw)


def transform_importance_weights(
    pi_e_at_action,
    pscore,
    *,
    use_iw: bool,
    iw_mode: str = "clip",
    clip_m: float = 10.0,
    shrink_lambda: float = 10.0,
    log_eps: float = 1e-10,
):
    """Apply raw / clip / Su-shrink transform to IPS weights."""
    mode = str(iw_mode).lower()
    if mode == "dm":
        raise ValueError("weights 'dm' (all zero) are for DM-only selection; train DM-only with the 'dm' loss")
    if mode in ("raw", "none"):
        return importance_weights(pi_e_at_action, pscore, use_iw, log_eps)
    if mode == "shrink":
        return shrink_importance_weights(
            pi_e_at_action, pscore, shrink_lambda, use_iw, log_eps
        )
    # default: clip
    return clipped_importance_weights(
        pi_e_at_action, pscore, clip_m, use_iw, log_eps
    )


def crm_per_sample_u(rewards, iw_clipped):
    """Per-row u_hi = delta_i * clip_iw with delta = -reward in [-1, 0]."""
    return -rewards * iw_clipped


def crm_variance_penalty(u, crm_lambda, eps=1e-10):
    """lambda * sqrt(Var(u) / n) from CRM principle (Eq. 5)."""
    n = u.shape[0]
    if n <= 1:
        return u.new_zeros(())
    var = u.var(unbiased=True)
    return float(crm_lambda) * torch.sqrt(var / n + eps)


def crm_surrogate(
    pi_e_at_action,
    rewards,
    pscore,
    *,
    clip_m: float,
    crm_lambda: float,
    use_iw: bool,
    use_log_trick: bool,
    log_eps: float = 1e-10,
    iw_mode: str = "clip",
    shrink_lambda: float = 10.0,
):
    """CRM objective: clipped/shrunk IPS risk + variance penalty (Eq. 5).

    IPS uses a REINFORCE surrogate with detached transformed weights so training
    still gets gradients when every ratio hits the clip ceiling.
    """
    iw = transform_importance_weights(
        pi_e_at_action,
        pscore,
        use_iw=use_iw,
        iw_mode=iw_mode,
        clip_m=clip_m,
        shrink_lambda=shrink_lambda,
        log_eps=log_eps,
    )
    iw_pg = iw.detach()
    grad_term = policy_grad_surrogate(pi_e_at_action, use_log_trick, log_eps)
    ips_term = -(rewards * iw_pg * grad_term).mean()

    iw_var = iw.detach() if use_log_trick else iw
    u = crm_per_sample_u(rewards, iw_var)
    return ips_term + crm_variance_penalty(u, crm_lambda)


def policy_grad_surrogate(pi_at_action, use_log_trick=True, log_eps=1e-10):
    """Policy-gradient factor at the logged action (IPW path)."""
    pi = pi_at_action.squeeze().clamp(min=log_eps)
    if use_log_trick:
        return torch.log(pi)
    return torch.ones_like(pi)


def grad_importance_weights(iw, use_log_trick: bool):
    return iw.detach() if use_log_trick else iw


def dm_reward(scores, policy_prob):
    """Pathwise DM value: sum_a q_hat(x,a) * pi_e(a|x)."""
    return (scores * policy_prob).sum(dim=1)


DR_NORMALIZATIONS = ("none", "global", "batch")


def dr_correction(iw, rewards, q_at_action, normalizer="batch"):
    """Per-row DR correction w_i (r_i - q_i) / c.

    ``normalizer``: ``'batch'`` (c = the minibatch mean of ``iw``: legacy SNDR, whose objective
    changes with the batch size; c carries a gradient only when ``iw`` does, i.e. not under the log
    trick), ``None`` or ``'none'`` (c = 1: plain DR, a per-example additive objective), or a number
    (c held fixed: ``--sn-scope global`` passes the full-data mean weight computed at the start of the
    epoch, so no gradient flows through c and c is stale after the epoch's first step).
    """
    if isinstance(normalizer, str):
        if normalizer == "batch":
            return iw * (rewards - q_at_action) / iw.mean()
        if normalizer != "none":
            raise ValueError(f"normalizer must be 'batch', 'none', None or a number, got {normalizer!r}")
        normalizer = None
    if normalizer is None:
        return iw * (rewards - q_at_action)
    return iw * (rewards - q_at_action) / float(normalizer)


def sndr_r_hat(iw, rewards, q_at_action, dm):
    """Per-row SNDR value estimate (evaluation / logging)."""
    return dm + dr_correction(iw, rewards, q_at_action)


def dr_sndr_surrogate(
    scores,
    policy_prob,
    actions,
    rewards,
    pscore,
    *,
    use_iw: bool,
    use_log_trick: bool,
    log_eps: float = 1e-10,
    iw_mode: str = "none",
    iw_param: float = float("inf"),
    normalizer="batch",
):
    """SNDR policy surrogate: correction + DM, each scaled by log pi when requested.

    ``iw_mode`` / ``iw_param``: weight transform (none, clip at M, Su shrinkage with lambda).
    ``normalizer``: how the correction is scaled (``dr_correction``): per minibatch (default, the
    older form), not at all (DR), or by a given full-data mean weight.

    Log trick:
      - detach pi in IW and DM coefficients
      - multiply both terms by log pi (all actions for DM, logged action for correction)
    Direct:
      - attached pi throughout (pathwise DM + IW correction)
    """
    n = actions.shape[0]
    idx = torch.arange(n, device=policy_prob.device)
    q_factual = scores[idx, actions].squeeze()
    pi_a = policy_prob[idx, actions].squeeze()
    log_p = torch.log(policy_prob.clamp(min=log_eps))

    wkw = dict(use_iw=use_iw, iw_mode=iw_mode, clip_m=iw_param, shrink_lambda=iw_param, log_eps=log_eps)
    if use_log_trick:
        pi_coef = policy_prob.detach()
        iw = transform_importance_weights(pi_a.detach(), pscore, **wkw).detach()
        correction = dr_correction(iw, rewards, q_factual, normalizer)
        corr_term = correction * log_p[idx, actions]
        dm_term = (scores * pi_coef * log_p).sum(dim=1)
    else:
        iw = transform_importance_weights(pi_a, pscore, **wkw)
        corr_term = dr_correction(iw, rewards, q_factual, normalizer)
        dm_term = dm_reward(scores, policy_prob)

    return corr_term + dm_term


def dr_sndr_loss(
    scores,
    policy_prob,
    actions,
    rewards,
    pscore,
    *,
    use_iw: bool,
    use_log_trick: bool,
    log_eps: float = 1e-10,
    iw_mode: str = "none",
    iw_param: float = float("inf"),
    normalizer="batch",
):
    """Minimize negative SNDR surrogate (ascend policy value)."""
    return -dr_sndr_surrogate(
        scores,
        policy_prob,
        actions,
        rewards,
        pscore,
        use_iw=use_iw,
        use_log_trick=use_log_trick,
        log_eps=log_eps,
        iw_mode=iw_mode,
        iw_param=iw_param,
        normalizer=normalizer,
    ).mean()


class _BanditPolicyLossBase(nn.Module):
    needs_qhat = True
    # The loss is a mean of per-row terms (no per-minibatch statistic), so the trainer can weight a
    # short minibatch by its rows (training_utils.minibatch_loss). Per-batch losses override this.
    per_example_additive = True

    def __init__(self, log_eps=1e-10, use_log_trick=True, propensity_mode="logged", weights="none", normalization="batch"):
        super().__init__()
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)
        # scale of the DR correction in the SNDR-family losses: 'batch' (per minibatch, older),
        # 'none' (plain DR) or 'global' (full-data mean weight, set by the trainer every epoch)
        if normalization not in DR_NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {DR_NORMALIZATIONS}, got {normalization!r}")
        self.normalization = normalization
        self.global_normalizer = None
        # importance-weight transform for the IW / SNDR terms (utils.importance_weights spec)
        self.iw_mode, self.iw_param = parse_weight_spec(weights)
        self.propensity_mode = str(propensity_mode).lower()
        if self.propensity_mode not in PROPENSITY_MODES:
            raise ValueError(
                f"propensity_mode must be one of {PROPENSITY_MODES}, got {propensity_mode!r}"
            )

    def _use_iw(self) -> bool:
        return uses_importance_weighting(self.propensity_mode)

    def _prepare_iw(self, pi_e_at_position, pscore):
        iw = transform_importance_weights(
            pi_e_at_position, pscore, use_iw=self._use_iw(), iw_mode=self.iw_mode,
            clip_m=self.iw_param, shrink_lambda=self.iw_param, log_eps=self.log_eps,
        )
        iw_val = iw.detach()
        iw_grad = grad_importance_weights(iw, self.use_log_trick)
        return iw_val, iw_grad

    @property
    def needs_global_normalizer(self) -> bool:
        return self.normalization == "global"

    def set_global_normalizer(self, value: float) -> None:
        """The full-data mean transformed weight under the current policy (``--sn-scope global``)."""
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"global normalizer must be a positive number, got {value!r}")
        self.global_normalizer = float(value)

    def _normalizer(self):
        if self.normalization == "global":
            if self.global_normalizer is None:
                raise RuntimeError("normalization='global' needs set_global_normalizer() before the first batch")
            return self.global_normalizer
        return None if self.normalization == "none" else "batch"

    def _logged_action_prob(self, policy_prob, actions):
        n = actions.shape[0]
        idx = torch.arange(n, device=policy_prob.device)
        return policy_prob[idx, actions].squeeze()

    def _dr_sndr_loss(self, pscore, scores, policy_prob, rewards, actions):
        return dr_sndr_loss(
            scores,
            policy_prob,
            actions,
            rewards,
            pscore,
            use_iw=self._use_iw(),
            use_log_trick=self.use_log_trick,
            log_eps=self.log_eps,
            iw_mode=self.iw_mode,
            iw_param=self.iw_param,
            normalizer=self._normalizer(),
        )


class IPWPolicyLoss(_BanditPolicyLossBase):
    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        _, iw_grad = self._prepare_iw(pi_e_at_position, pscore)
        grad_term = policy_grad_surrogate(
            pi_e_at_position, self.use_log_trick, self.log_eps
        )

        reinforce_grad = iw_grad * original_policy_rewards * grad_term
        return (-reinforce_grad).mean()


class NaiveRewardPolicyLoss(_BanditPolicyLossBase):
    """Pure naive reward objective: no IW, DM, SNDR, KL, or CRM.

    Pathwise (default):  L = -mean(r * pi_theta(a|x))
    Log-trick:           L = -mean(r * log pi_theta(a|x))
    """

    needs_qhat = False

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        _ = pscore, scores
        if policy_prob.dim() == 3 and policy_prob.shape[-1] == 1:
            policy_prob = policy_prob.squeeze(-1)
        pi_e = self._logged_action_prob(policy_prob, original_policy_actions)
        if self.use_log_trick:
            return -(
                original_policy_rewards * torch.log(pi_e.clamp(min=self.log_eps))
            ).mean()
        return -(original_policy_rewards * pi_e).mean()


class SNDRPolicyLoss(_BanditPolicyLossBase):
    """DM(q_hat) + importance-weighted correction; ``normalization`` sets how the correction is
    scaled: per minibatch ('batch', legacy SNDR), by a fixed full-data mean weight refreshed every
    epoch ('global') or not at all ('none', see ``DRPolicyLoss``). Under the log trick the weights,
    and so both normalizers, carry no gradient: none of the three is the gradient of the SNDR ratio
    (docs/training_losses.md, section 3.4)."""

    @property
    def per_example_additive(self) -> bool:
        return self.normalization != "batch"  # legacy SNDR is a ratio per minibatch

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        return self._dr_sndr_loss(
            pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
        )


class DRPolicyLoss(SNDRPolicyLoss):
    """Doubly robust policy objective without self-normalization: DM(q_hat) + w (r - q_hat), a mean of
    per-row terms, so its minibatch gradients add up to the full-data gradient for any batch size. With
    raw weights under the log trick the gradient is that of the DR estimate; with a weight transform g
    it is the gradient of DM + H(w) (r - q_hat), H(w) = integral of g(t)/t from 0 to w, not of the
    transformed estimate DM + g(w) (r - q_hat) (docs/training_losses.md, section 3.4)."""

    def __init__(self, log_eps=1e-10, use_log_trick=True, propensity_mode="logged", weights="none"):
        super().__init__(log_eps=log_eps, use_log_trick=use_log_trick, propensity_mode=propensity_mode,
                         weights=weights, normalization="none")


def dm_surrogate(scores, policy_prob, *, use_log_trick: bool, log_eps: float = 1e-10):
    """Direct-method surrogate per row: sum_a pi(a|x) q_hat(x, a) (the SNDR surrogate's DM term).

    Exact over all actions, so the log-trick form (pi detached, times log pi) has the same
    gradient as the pathwise one."""
    if use_log_trick:
        return (scores * policy_prob.detach() * torch.log(policy_prob.clamp(min=log_eps))).sum(dim=1)
    return dm_reward(scores, policy_prob)


class DMPolicyLoss(_BanditPolicyLossBase):
    """Direct method (the no-propensity baseline with a reward model): maximize the policy's value
    under q_hat on the logged contexts. Logged actions, rewards and propensities are not used."""

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        _ = pscore, original_policy_rewards, original_policy_actions
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        return -dm_surrogate(scores, policy_prob, use_log_trick=self.use_log_trick, log_eps=self.log_eps).mean()


class KLPolicyLoss(_BanditPolicyLossBase):
    """SNDR PG + batch MC KL toward logging policy (logged actions only)."""

    @property
    def per_example_additive(self) -> bool:
        return self.normalization != "batch"

    def __init__(self, gamma=0.05, log_eps=1e-10, use_log_trick=True, propensity_mode="logged", weights="none",
                 normalization="batch"):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
            weights=weights,
            normalization=normalization,
        )
        self.gamma = gamma

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)

        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        dr_loss = self._dr_sndr_loss(
            pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
        )
        kl = batch_mc_kl(pi_e_at_position, pscore, self.log_eps)
        return dr_loss + self.gamma * kl


class CRMPolicyLoss(_BanditPolicyLossBase):
    """Counterfactual Risk Minimization: clipped/shrunk IPS + variance penalty (Eq. 5)."""

    per_example_additive = False  # the variance penalty is a per-minibatch statistic

    def __init__(
        self,
        clip_m: float = 10.0,
        crm_lambda: float = 1.0,
        log_eps=1e-10,
        use_log_trick=True,
        propensity_mode="logged",
        iw_mode: str = "clip",
        shrink_lambda: float = 10.0,
    ):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
        self.clip_m = float(clip_m)
        self.crm_lambda = float(crm_lambda)
        self.iw_mode = str(iw_mode).lower()
        self.shrink_lambda = float(shrink_lambda)

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)
        return crm_surrogate(
            pi_e_at_position,
            original_policy_rewards,
            pscore,
            clip_m=self.clip_m,
            crm_lambda=self.crm_lambda,
            use_iw=self._use_iw(),
            use_log_trick=self.use_log_trick,
            log_eps=self.log_eps,
            iw_mode=self.iw_mode,
            shrink_lambda=self.shrink_lambda,
        )


class KLCRMPolicyLoss(_BanditPolicyLossBase):
    """Unified training loss: SNDR log-trick + KL + CRM variance penalty.

    L = -SNDR_surrogate + gamma * KL(pi_b || pi_e) + crm_lambda * sqrt(Var(u)/n)

    where u_i = -r_i * ŵ_i and ŵ is clip or Su-shrink of iw. Both ``gamma`` and
    ``crm_lambda`` are Optuna-tuned. CRM variance keeps gradients through
    transformed IW so ``crm_lambda`` affects updates under the log-trick SNDR path
    (unlike standalone CRM, which detaches Var(u)).
    """

    per_example_additive = False  # legacy SNDR and the variance penalty are per-minibatch statistics

    def __init__(
        self,
        gamma: float = 0.05,
        clip_m: float = 10.0,
        crm_lambda: float = 1.0,
        log_eps=1e-10,
        use_log_trick=True,
        propensity_mode="logged",
        iw_mode: str = "clip",
        shrink_lambda: float = 10.0,
    ):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
        self.gamma = float(gamma)
        self.clip_m = float(clip_m)
        self.crm_lambda = float(crm_lambda)
        self.iw_mode = str(iw_mode).lower()
        self.shrink_lambda = float(shrink_lambda)

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)

        dr_loss = self._dr_sndr_loss(
            pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
        )
        kl = batch_mc_kl(pi_e_at_position, pscore, self.log_eps)

        iw = transform_importance_weights(
            pi_e_at_position,
            pscore,
            use_iw=self._use_iw(),
            iw_mode=self.iw_mode,
            clip_m=self.clip_m,
            shrink_lambda=self.shrink_lambda,
            log_eps=self.log_eps,
        )
        u = crm_per_sample_u(original_policy_rewards, iw)
        return dr_loss + self.gamma * kl + crm_variance_penalty(u, self.crm_lambda)
