import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/code")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch.nn as nn

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
):
    """CRM objective: clipped IPS risk + variance penalty (Eq. 5).

    IPS uses a REINFORCE surrogate with detached clipped weights so training
    still gets gradients when every ratio hits the clip ceiling.
    """
    iw = clipped_importance_weights(
        pi_e_at_action, pscore, clip_m, use_iw, log_eps
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


def dr_correction(iw, rewards, q_at_action):
    """Per-row SNDR correction: w_i * (r_i - q_i) / mean(w)."""
    return iw * (rewards - q_at_action) / iw.mean()


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
):
    """SNDR policy surrogate: correction + DM, each scaled by log pi when requested.

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

    if use_log_trick:
        pi_coef = policy_prob.detach()
        iw = importance_weights(pi_a.detach(), pscore, use_iw, log_eps).detach()
        correction = dr_correction(iw, rewards, q_factual)
        corr_term = correction * log_p[idx, actions]
        dm_term = (scores * pi_coef * log_p).sum(dim=1)
    else:
        iw = importance_weights(pi_a, pscore, use_iw, log_eps)
        corr_term = dr_correction(iw, rewards, q_factual)
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
    ).mean()


class _BanditPolicyLossBase(nn.Module):
    def __init__(self, log_eps=1e-10, use_log_trick=True, propensity_mode="logged"):
        super().__init__()
        self.log_eps = log_eps
        self.use_log_trick = bool(use_log_trick)
        self.propensity_mode = str(propensity_mode).lower()
        if self.propensity_mode not in PROPENSITY_MODES:
            raise ValueError(
                f"propensity_mode must be one of {PROPENSITY_MODES}, got {propensity_mode!r}"
            )

    def _use_iw(self) -> bool:
        return uses_importance_weighting(self.propensity_mode)

    def _prepare_iw(self, pi_e_at_position, pscore):
        iw = importance_weights(
            pi_e_at_position, pscore, self._use_iw(), self.log_eps
        )
        iw_val = iw.detach()
        iw_grad = grad_importance_weights(iw, self.use_log_trick)
        return iw_val, iw_grad

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


class SNDRPolicyLoss(_BanditPolicyLossBase):
    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        n = original_policy_actions.shape[0]
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        return self._dr_sndr_loss(
            pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
        )


class KLPolicyLoss(_BanditPolicyLossBase):
    """SNDR PG + batch MC KL toward logging policy (logged actions only)."""

    def __init__(self, gamma=0.05, log_eps=1e-10, use_log_trick=True, propensity_mode="logged"):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
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
    """Counterfactual Risk Minimization: clipped IPS + variance penalty (Eq. 5)."""

    def __init__(
        self,
        clip_m: float = 10.0,
        crm_lambda: float = 1.0,
        log_eps=1e-10,
        use_log_trick=True,
        propensity_mode="logged",
    ):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
        self.clip_m = float(clip_m)
        self.crm_lambda = float(crm_lambda)

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
        )


class KLCRMPolicyLoss(_BanditPolicyLossBase):
    """Unified training loss: SNDR log-trick + KL + CRM variance penalty.

    L = -SNDR_surrogate + gamma * KL(pi_b || pi_e) + crm_lambda * sqrt(Var(u)/n)

    where u_i = -r_i * clip(iw_i, M). Both ``gamma`` and ``crm_lambda`` are Optuna-tuned.
    CRM variance keeps gradients through clipped IW so ``crm_lambda`` affects updates
    under the log-trick SNDR path (unlike standalone CRM, which detaches Var(u)).
    """

    def __init__(
        self,
        gamma: float = 0.05,
        clip_m: float = 10.0,
        crm_lambda: float = 1.0,
        log_eps=1e-10,
        use_log_trick=True,
        propensity_mode="logged",
    ):
        super().__init__(
            log_eps=log_eps,
            use_log_trick=use_log_trick,
            propensity_mode=propensity_mode,
        )
        self.gamma = float(gamma)
        self.clip_m = float(clip_m)
        self.crm_lambda = float(crm_lambda)

    def forward(self, pscore, scores, policy_prob, original_policy_rewards, original_policy_actions):
        scores, policy_prob = _align_policy_scores(scores, policy_prob)
        pi_e_at_position = self._logged_action_prob(policy_prob, original_policy_actions)

        dr_loss = self._dr_sndr_loss(
            pscore, scores, policy_prob, original_policy_rewards, original_policy_actions
        )
        kl = batch_mc_kl(pi_e_at_position, pscore, self.log_eps)

        iw = clipped_importance_weights(
            pi_e_at_position,
            pscore,
            self.clip_m,
            self._use_iw(),
            self.log_eps,
        )
        u = crm_per_sample_u(original_policy_rewards, iw)
        return dr_loss + self.gamma * kl + crm_variance_penalty(u, self.crm_lambda)
