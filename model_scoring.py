import numpy as np
import scipy
import torch
from scipy.special import softmax
from sklearn.utils import check_random_state

import simulation_utils

random_state = 12345
random_ = check_random_state(random_state)


def get_scores_dict(
    dr_naive_mean,
    dr_naive_ci,
    dr_naive_se,
    dr_boot_mean,
    dr_boot_std,
    dr_boot_ci,
    dr_uni_mean,
    dr_uni_std,
    dr_uni_ci,
    ipw_uni_mean,
    ipw_uni_std,
    ipw_uni_ci,
    ipw_boot_mean,
    ipw_boot_std,
    ipw_boot_ci,
    cv_stats_uniform,
    cv_stats_exp,
    loss_uniform,
    loss_exp,
    score_naive_minus_cv_uniform,
    score_naive_minus_cv_exp,
):
    """
    Build a clean, unified scores dictionary for a model's evaluation.

    Inputs are raw scalars from score_model_modular.
    """
    return {
        # ---- Naive DR ----
        "dr_naive_mean": dr_naive_mean,
        "dr_naive_ci_low": dr_naive_ci[0],
        "dr_naive_ci_high": dr_naive_ci[1],
        "dr_naive_se": dr_naive_se,

        # ---- Bootstrap DR ----
        "dr_boot_mean": dr_boot_mean,
        "dr_boot_std": dr_boot_std,
        "dr_boot_ci_low": dr_boot_ci[0],
        "dr_boot_ci_high": dr_boot_ci[1],

        "dr_uni_mean": dr_uni_mean,
        "dr_uni_std": dr_uni_std,
        "dr_uni_ci_low": dr_uni_ci[0],
        "dr_uni_ci_high": dr_uni_ci[1],

        # ---- Uniform IPW ----
        "ipw_uni_mean": ipw_uni_mean,
        "ipw_uni_std": ipw_uni_std,
        "ipw_uni_ci_low": ipw_uni_ci[0],
        "ipw_uni_ci_high": ipw_uni_ci[1],

        # ---- Bootstrap IPW ----
        "ipw_boot_mean": ipw_boot_mean,
        "ipw_boot_std": ipw_boot_std,
        "ipw_boot_ci_low": ipw_boot_ci[0],
        "ipw_boot_ci_high": ipw_boot_ci[1],

        # ---- CV Stats (Uniform Weights) ----
        "cv_rmse_uniform": cv_stats_uniform["rmse"],
        "cv_signed_rmse_uniform": cv_stats_uniform["signed_rmse"],
        "cv_bias_uniform": cv_stats_uniform["bias"],
        "cv_bias_lb_signed_uniform": cv_stats_uniform["bias_lb_signed"],

        # ---- CV Stats (Exp Weights) ----
        "cv_rmse_exp": cv_stats_exp["rmse"],
        "cv_signed_rmse_exp": cv_stats_exp["signed_rmse"],
        "cv_bias_exp": cv_stats_exp["bias"],
        "cv_bias_lb_signed_exp": cv_stats_exp["bias_lb_signed"],

        # ---- Conservative Losses ----
        "loss_uniform": loss_uniform,
        "loss_exp": loss_exp,

        # ---- Final Scores ----
        "score_naive_minus_cv_uniform": score_naive_minus_cv_uniform,
        "score_naive_minus_cv_exp": score_naive_minus_cv_exp,
    }


def dros_shrinkage(iw: np.ndarray, lam: float):
    """Doubly Robust with optimistic shrinkage."""
    return (lam * iw) / (iw**2 + lam)


def dr_shrinkage_rewards(
    pscore: np.ndarray,
    scores: np.ndarray,
    policy_prob: np.ndarray,
    original_policy_rewards: np.ndarray,
    users: np.ndarray,
    original_policy_actions: np.ndarray,
    lam: float = 3.0,
) -> np.ndarray:
    """
    DR with optimistic shrinkage (DRos-style), per-round contributions.
    """
    pi_e_at_position = policy_prob[users, original_policy_actions].squeeze()
    iw = pi_e_at_position / pscore
    iw = dros_shrinkage(iw, lam=lam)

    q_hat_factual = scores[users, original_policy_actions].squeeze()
    dm_reward = (scores * policy_prob)[users].sum(axis=1)

    r_hat = dm_reward + iw * (original_policy_rewards - q_hat_factual)
    return r_hat


def ipw_rewards(
    pscore: np.ndarray,
    policy_prob: np.ndarray,
    original_policy_rewards: np.ndarray,
    users: np.ndarray,
    original_policy_actions: np.ndarray,
) -> np.ndarray:
    """
    Plain IPW per-round contributions.
    """
    pi_e_at_position = policy_prob[users, original_policy_actions].squeeze()
    iw = pi_e_at_position / pscore
    r_hat = iw * original_policy_rewards
    return r_hat



def uniform_bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 500,
    m = 0.7,
    random_state: int | None = None,
):
    """
    Exponential (Bayesian) bootstrap for the mean of per-sample values.

    Returns:
        mean_est: float, bootstrap mean of the mean
        std_est: float, bootstrap std of the mean
        ci: tuple (low, high), 95% percentile CI
        all_samples: np.ndarray of length n_bootstrap
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    n_samples = int(n*m)
    boot_samples = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        boot_sample = values[rng.choice(n, size=n_samples, replace=True)]
        boot_samples[b] = np.mean(boot_sample)

    mean_est = boot_samples.mean()
    std_est = boot_samples.std(ddof=1)
    low, high = np.percentile(boot_samples, [2.5, 97.5])

    return mean_est, std_est, (low, high), boot_samples


def exp_bootstrap_mean(
    values: np.ndarray,
    n_bootstrap: int = 500,
    random_state: int | None = None,
):
    """
    Exponential (Bayesian) bootstrap for the mean of per-sample values.

    Returns:
        mean_est: float, bootstrap mean of the mean
        std_est: float, bootstrap std of the mean
        ci: tuple (low, high), 95% percentile CI
        all_samples: np.ndarray of length n_bootstrap
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    boot_samples = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        weights = rng.exponential(scale=1.0, size=n)
        weights = weights / weights.sum()
        boot_samples[b] = np.sum(weights * values)

    mean_est = boot_samples.mean()
    std_est = boot_samples.std(ddof=1)
    low, high = np.percentile(boot_samples, [2.5, 97.5])

    return mean_est, std_est, (low, high), boot_samples


def _single_split_diff(
    u_sub: np.ndarray,
    e_sub: np.ndarray,
    est_size: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """
    One random split of u_sub/e_sub into estimator/unbiased subsets,
    returning the signed difference of weighted means: unb_est - est_est.
    """
    m = len(u_sub)
    assert len(e_sub) == m
    assert len(weights) == m
    assert 1 <= est_size < m

    perm = rng.permutation(m)
    est_idx = perm[:est_size]
    unb_idx = perm[est_size:]

    w_est = weights[est_idx]
    w_unb = weights[unb_idx]

    est_est = np.sum(w_est * e_sub[est_idx]) / np.sum(w_est)
    unb_est = np.sum(w_unb * u_sub[unb_idx]) / np.sum(w_unb)

    return float(unb_est - est_est)  # <-- keep sign


def cv_error_between_estimators(
    unbiased_vec: np.ndarray,
    estimator_vec: np.ndarray,
    n_outer: int = 15,
    n_inner: int = 100,
    m_exponent: float = 0.7,
    use_exponential: bool = False,
    alpha: float = 0.05,          # one-sided level for lower bound
    random_state: int | None = None,
):
    """
    Two-level m-out-of-n CV-style error between 'unbiased' and 'estimator' vectors,
    keeping sign and giving a conservative one-sided lower bound on the signed error.

    Returns:
        {
          'rmse': float,                # sqrt(E[d^2])
          'signed_rmse': float,         # sign(mean(d)) * rmse
          'bias': float,                # mean(d)
          'bias_lb_signed': float,      # one-sided lower bound on bias in direction of sign
        }
    where d = (unbiased_est - estimator_est) across resampling repetitions.
    """
    rng = np.random.default_rng(random_state)
    n = len(unbiased_vec)
    m = max(10, int(n ** m_exponent))

    all_diffs = []  # collect signed diffs across *all* outer+inner reps

    for _ in range(n_outer):
        # m-out-of-n subsampling with replacement
        idx_sub = rng.choice(n, size=m, replace=True)
        u_sub = unbiased_vec[idx_sub]
        e_sub = estimator_vec[idx_sub]

        # variance-ratio split between estimator/unbiased subsets
        ratio = np.var(e_sub) / (np.var(u_sub) + np.var(e_sub) + 1e-6)
        if ratio == 0 or ratio == 1 or np.isnan(ratio):
            ratio = 0.5
        ratio = max(1.0 / m, min((m - 1) / m, ratio))
        est_size = max(1, int(m * ratio))

        # ---- precompute weights for all inner repetitions ----
        if use_exponential:
            # shape (n_inner, m)
            weight_mat = rng.exponential(scale=1.0, size=(n_inner, m))
            # optional normalization (not strictly necessary for diff, but nice):
            weight_mat = weight_mat / weight_mat.sum(axis=1, keepdims=True)
        else:
            weight_mat = np.ones((n_inner, m), dtype=float)

        # inner reps: get signed diffs
        for i in range(n_inner):
            diff = _single_split_diff(
                u_sub=u_sub,
                e_sub=e_sub,
                est_size=est_size,
                weights=weight_mat[i],
                rng=rng,
            )
            all_diffs.append(diff)

    diffs = np.asarray(all_diffs, dtype=float)
    # basic moments
    rmse = float(np.sqrt(np.mean(diffs**2)))
    bias = float(diffs.mean())
    sign = 1.0 if bias >= 0 else -1.0

    # one-sided lower bound on bias in direction of sign
    # if sign > 0, we care about "how positive is bias at least?"
    # if sign < 0, we care about "how negative is bias at least?"
    if sign > 0:
        # e.g. 5%-quantile of diffs; clamp at 0 so we don't flip sign
        q = np.percentile(diffs, 100 * alpha)
        bias_lb = max(0.0, q)
    else:
        # for negative bias, take (1-alpha)-quantile (most negative side),
        # clamp at 0 from above
        q = np.percentile(diffs, 100 * (1 - alpha))
        bias_lb = min(0.0, q)

    bias_lb_signed = float(bias_lb)  # already has sign

    signed_rmse = float(sign * rmse)

    return {
        "rmse": rmse,
        "signed_rmse": signed_rmse,
        "bias": bias,
        "bias_lb_signed": bias_lb_signed,
    }


def score_model_modular(
    val_dataset: dict,
    scores_all: torch.Tensor,
    policy_prob: np.ndarray,
    lam_dr: float = 3.0,
    n_bootstrap: int = 500,
    random_state: int | None = None,
):
    """
    Modular scoring utility for a single model.
    Produces DR/IPW vectors, Bootstrap, CV statistics,
    and returns a clean scores_dict + scores_array + weights_info.
    """

    # ----------- Extract data -----------
    pscore = val_dataset["pscore"]
    users = val_dataset["x_idx"]
    reward = val_dataset["r"]
    actions = val_dataset["a"]

    scores = scores_all.detach().cpu().numpy().squeeze()
    prob = policy_prob[users, actions].squeeze()
    weights_info = simulation_utils.get_weights_info(prob, pscore)
    # print(f"Validation weights_info: {weights_info}")

    # ----------- Per-round contributions -----------
    dr_vec = dr_shrinkage_rewards(
        pscore=pscore,
        scores=scores,
        policy_prob=policy_prob,
        original_policy_rewards=reward,
        users=users,
        original_policy_actions=actions,
        lam=lam_dr,
    )

    ipw_vec = ipw_rewards(
        pscore=pscore,
        policy_prob=policy_prob,
        original_policy_rewards=reward,
        users=users,
        original_policy_actions=actions,
    )

    # ----------- Naive DR -----------
    dr_naive_mean = dr_vec.mean()
    dr_naive_se = dr_vec.std(ddof=1) / np.sqrt(len(dr_vec))
    t_crit = scipy.stats.t.ppf(0.975, len(dr_vec) - 1)

    dr_naive_ci = (
        dr_naive_mean - t_crit * dr_naive_se,
        dr_naive_mean + t_crit * dr_naive_se,
    )

    # ----------- Bootstrap DR -----------
    dr_boot_mean, dr_boot_std, dr_boot_ci, _ = exp_bootstrap_mean(
        dr_vec, n_bootstrap=n_bootstrap, random_state=random_state
    )

    # ----------- Bootstrap IPW -----------
    ipw_boot_mean, ipw_boot_std, ipw_boot_ci, _ = exp_bootstrap_mean(
        ipw_vec,
        n_bootstrap=n_bootstrap,
        random_state=None if random_state is None else random_state + 7,
    )
    
    dr_uni_mean, dr_uni_std, dr_uni_ci, _ = uniform_bootstrap_mean(
        dr_vec,
        n_bootstrap=n_bootstrap,
        random_state=None if random_state is None else random_state + 3,
    )
    
    ipw_uni_mean, ipw_uni_std, ipw_uni_ci, _ = uniform_bootstrap_mean(
        ipw_vec,
        n_bootstrap=n_bootstrap,
        random_state=None if random_state is None else random_state + 5,
    )

    # ----------- CV: DR vs IPW (uniform weights) -----------
    cv_stats_uniform = cv_error_between_estimators(
        unbiased_vec=dr_vec,
        estimator_vec=ipw_vec,
        n_outer=15,
        n_inner=100,
        m_exponent=0.7,
        use_exponential=False,
        alpha=0.05,
        random_state=random_state,
    )

    # ----------- CV: DR vs IPW (exponential weights) -----------
    cv_stats_exp = cv_error_between_estimators(
        unbiased_vec=dr_vec,
        estimator_vec=ipw_vec,
        n_outer=15,
        n_inner=100,
        m_exponent=0.7,
        use_exponential=True,
        alpha=0.05,
        random_state=None if random_state is None else random_state + 11,
    )

    # ----------- Conservative loss (never underestimate) -----------
    loss_uniform = max(
        cv_stats_uniform["rmse"], abs(cv_stats_uniform["bias_lb_signed"])
    )

    loss_exp = max(cv_stats_exp["rmse"], abs(cv_stats_exp["bias_lb_signed"]))

    score_naive_minus_cv_uniform = dr_naive_mean - loss_uniform
    score_naive_minus_cv_exp = dr_naive_mean - loss_exp

    # ----------- Build scores dict using helper -----------
    scores_dict = get_scores_dict(
        dr_naive_mean=dr_naive_mean,
        dr_naive_ci=dr_naive_ci,
        dr_naive_se=dr_naive_se,
        dr_boot_mean=dr_boot_mean,
        dr_boot_std=dr_boot_std,
        dr_boot_ci=dr_boot_ci,
        dr_uni_mean=dr_uni_mean,
        dr_uni_std=dr_uni_std,
        dr_uni_ci=dr_uni_ci,
        ipw_uni_mean=ipw_uni_mean,
        ipw_uni_std=ipw_uni_std,
        ipw_uni_ci=ipw_uni_ci,
        ipw_boot_mean=ipw_boot_mean,
        ipw_boot_std=ipw_boot_std,
        ipw_boot_ci=ipw_boot_ci,
        cv_stats_uniform=cv_stats_uniform,
        cv_stats_exp=cv_stats_exp,
        loss_uniform=loss_uniform,
        loss_exp=loss_exp,
        score_naive_minus_cv_uniform=score_naive_minus_cv_uniform,
        score_naive_minus_cv_exp=score_naive_minus_cv_exp,
    )

    # ----------- Flat array for optimization -----------
    scores_array = np.array(
        [
            dr_naive_mean,
            dr_boot_mean,
            ipw_boot_mean,
            cv_stats_uniform["rmse"],
            cv_stats_exp["rmse"],
            score_naive_minus_cv_uniform,
            score_naive_minus_cv_exp,
        ],
        dtype=float,
    )

    # # ----------- ESS safeguard -----------
    # if weights_info["ess"] < len(reward) * 0.01:
    #     print("Warning: Low ESS — returning -inf for main score.")
    #     scores_dict["dr_naive_mean"] = -np.inf
    #     scores_array[0] = -np.inf

    return scores_dict, scores_array, weights_info
