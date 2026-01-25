# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

from typing import Optional

import numpy as np
from numpy import log
from numpy import sqrt
from numpy import var
from scipy import stats
from sklearn.utils import check_scalar

# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Useful Tools."""
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar
import torch


def estimate_bias_in_ope(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
) -> float:
    """Helper to estimate a bias in OPE.

    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Rewards observed for each data in logged bandit data, i.e., :math:`r_t`.

    iw: array-like, shape (n_rounds,)
        Importance weight for each data in logged bandit data, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.

    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` is a hyperparameter value.

    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_i` and action :math:`a_i`.

    Returns
    ----------
    estimated_bias: float
        Estimated the bias in OPE.
        This is based on the direct bias estimation stated on page 17 of Su et al.(2020).

    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """
    if q_hat is None:
        q_hat = np.zeros(reward.shape[0])
    estimated_bias_arr = (iw - iw_hat) * (reward - q_hat)
    estimated_bias = np.abs(estimated_bias_arr.mean())

    return estimated_bias


def estimate_high_probability_upper_bound_bias(
    reward: np.ndarray,
    iw: np.ndarray,
    iw_hat: np.ndarray,
    q_hat: Optional[np.ndarray] = None,
    delta: float = 0.05,
) -> float:
    """Helper to estimate a high probability upper bound of bias in OPE.

    Parameters
    ----------
    reward: array-like, shape (n_rounds,)
        Rewards observed for each data in logged bandit data, i.e., :math:`r_t`.

    iw: array-like, shape (n_rounds,)
        Importance weight for each data in logged bandit data, i.e., :math:`w(x,a)=\\pi_e(a|x)/ \\pi_b(a|x)`.

    iw_hat: array-like, shape (n_rounds,)
        Importance weight (IW) modified by a hyparpareter. How IW is modified depends on the estimator as follows.
            - clipping: :math:`\\hat{w}(x,a) := \\min \\{ \\lambda, w(x,a) \\}`
            - switching: :math:`\\hat{w}(x,a) := w(x,a) \\cdot \\mathbb{I} \\{ w(x,a) < \\lambda \\}`
            - shrinkage: :math:`\\hat{w}(x,a) := (\\lambda w(x,a)) / (\\lambda + w^2(x,a))`
        where :math:`\\lambda` and :math:`\\lambda` are hyperparameters.

    q_hat: array-like, shape (n_rounds,), default=None
        Estimated expected reward given context :math:`x_i` and action :math:`a_i`.

    delta: float, default=0.05
        A confidence delta to construct a high probability upper bound based on Bernstein inequality.

    Returns
    ----------
    bias_upper_bound: float
        Estimated (high probability) upper bound of the bias.
        This upper bound is based on the direct bias estimation
        stated on page 17 of Su et al.(2020).

    References
    ----------
    Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudik.
    "Doubly Robust Off-Policy Evaluation with Shrinkage.", 2020.

    """
    check_scalar(delta, "delta", (int, float), min_val=0.0, max_val=1.0)

    estimated_bias = estimate_bias_in_ope(
        reward=reward,
        iw=iw,
        iw_hat=iw_hat,
        q_hat=q_hat,
    )
    n = reward.shape[0]
    bias_upper_bound = estimated_bias
    bias_upper_bound += sqrt((2 * (iw**2).mean() * log(2 / delta)) / n)
    bias_upper_bound += (2 * iw.max() * log(2 / delta)) / (3 * n)

    return bias_upper_bound


def check_confidence_interval_arguments(
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Optional[ValueError]:
    """Check confidence interval arguments.

    Parameters
    ----------
    alpha: float, default=0.05
        Significance level.

    n_bootstrap_samples: int, default=10000
        Number of resampling performed in bootstrap sampling.

    random_state: int, default=None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_random_state(random_state)
    check_scalar(alpha, "alpha", float, min_val=0.0, max_val=1.0)
    check_scalar(n_bootstrap_samples, "n_bootstrap_samples", int, min_val=1)


def estimate_confidence_interval_by_bootstrap(
    samples: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap_samples: int = 10000,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Estimate confidence interval using bootstrap.

    Parameters
    ----------
    samples: array-like
        Empirical observed samples to be used to estimate cumulative distribution function.

    alpha: float, default=0.05
        Significance level.

    n_bootstrap_samples: int, default=10000
        Number of resampling performed in bootstrap sampling.

    random_state: int, default=None
        Controls the random seed in bootstrap sampling.

    Returns
    ----------
    estimated_confidence_interval: Dict[str, float]
        Dictionary storing the estimated mean and upper-lower confidence bounds.

    """
    check_confidence_interval_arguments(
        alpha=alpha, n_bootstrap_samples=n_bootstrap_samples, random_state=random_state
    )

    boot_samples = list()
    random_ = check_random_state(random_state)
    for _ in np.arange(n_bootstrap_samples):
        boot_samples.append(np.mean(random_.choice(samples, size=samples.shape[0])))
    lower_bound = np.percentile(boot_samples, 100 * (alpha / 2))
    upper_bound = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))
    return {
        "mean": np.mean(boot_samples),
        f"{100 * (1. - alpha)}% CI (lower)": lower_bound,
        f"{100 * (1. - alpha)}% CI (upper)": upper_bound,
    }


def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on an array.

    Parameters
    -------------
    array: object
        Input object to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {type(array)}"
        )
    if array.ndim != expected_dim:
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {array.ndim}D array"
        )


def check_tensor(
    tensor: torch.tensor,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on a tensor.

    Parameters
    -------------
    array: object
        Input object to check.

    name: str
        Name of the input array.

    expected_dim: int, default=1
        Expected dimension of the input array.

    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"`{name}` must be {expected_dim}D tensor, but got {type(tensor)}"
        )
    if tensor.ndim != expected_dim:
        raise ValueError(
            f"`{name}` must be {expected_dim}D tensor, but got {tensor.ndim}D tensor"
        )


def check_bandit_feedback_inputs(
    context: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    expected_reward: Optional[np.ndarray] = None,
    position: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    action_context: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for bandit learning or simulation.

    Parameters
    -----------
    context: array-like, shape (n_rounds, dim_context)
        Context vectors observed for each data, i.e., :math:`x_i`.

    action: array-like, shape (n_rounds,)
        Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

    reward: array-like, shape (n_rounds,)
        Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

    expected_reward: array-like, shape (n_rounds, n_actions), default=None
        Expected reward of each data, i.e., :math:`\\mathbb{E}[r_i|x_i,a_i]`.

    position: array-like, shape (n_rounds,), default=None
        Indices to differentiate positions in a recommendation interface where the actions are presented.

    pscore: array-like, shape (n_rounds,)
        Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

    action_context: array-like, shape (n_actions, dim_action_context)
        Context vectors characterizing each action.

    """
    check_array(array=context, name="context", expected_dim=2)
    check_array(array=action, name="action", expected_dim=1)
    check_array(array=reward, name="reward", expected_dim=1)
    if expected_reward is not None:
        check_array(array=expected_reward, name="expected_reward", expected_dim=2)
        if not (
            context.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == expected_reward.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == expected_reward.shape[0]`"
                ", but found it False"
            )
        if not (
            np.issubdtype(action.dtype, np.integer)
            and action.min() >= 0
            and action.max() < expected_reward.shape[1]
        ):
            raise ValueError(
                "`action` elements must be integers in the range of [0, `expected_reward.shape[1]`)"
            )
    else:
        if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
            raise ValueError("`action` elements must be non-negative integers")
    if pscore is not None:
        check_array(array=pscore, name="pscore", expected_dim=1)
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == pscore.shape[0]`"
                ", but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("`pscore` must be positive")

    if position is not None:
        check_array(array=position, name="position", expected_dim=1)
        if not (
            context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]
        ):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0] == position.shape[0]`"
                ", but found it False"
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("`position` elements must be non-negative integers")
    else:
        if not (context.shape[0] == action.shape[0] == reward.shape[0]):
            raise ValueError(
                "Expected `context.shape[0] == action.shape[0] == reward.shape[0]`"
                ", but found it False"
            )
    if action_context is not None:
        check_array(array=action_context, name="action_context", expected_dim=2)
        if not (
            np.issubdtype(action.dtype, np.integer)
            and action.min() >= 0
            and action.max() < action_context.shape[0]
        ):
            raise ValueError(
                "`action` elements must be integers in the range of [0, `action_context.shape[0]`)"
            )
    else:
        if not (np.issubdtype(action.dtype, np.integer) and action.min() >= 0):
            raise ValueError("`action` elements must be non-negative integers")


def check_ope_inputs(
    action_dist: np.ndarray,
    position: Optional[np.ndarray] = None,
    action: Optional[np.ndarray] = None,
    reward: Optional[np.ndarray] = None,
    pscore: Optional[np.ndarray] = None,
    estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
    estimated_importance_weights: Optional[np.ndarray] = None,
) -> Optional[ValueError]:
    """Check inputs for ope.

    Parameters
    -----------
    action_dist: array-like, shape (n_rounds, n_actions, len_list)
        Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

    position: array-like, shape (n_rounds,), default=None
        Indices to differentiate positions in a recommendation interface where the actions are presented.

    action: array-like, shape (n_rounds,), default=None
        Actions sampled by the logging/behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

    reward: array-like, shape (n_rounds,), default=None
        Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

    pscore: array-like, shape (n_rounds,)
        Action choice probabilities of the logging/behavior policy (propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

    estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
        Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.

    estimated_importance_weights: array-like, shape (n_rounds,), default=None
        Importance weights estimated via supervised classification, i.e., :math:`\\hat{w}(x_t, a_t)`.

    """
    # action_dist
    check_array(array=action_dist, name="action_dist", expected_dim=3)
    if not np.allclose(action_dist.sum(axis=1), 1):
        raise ValueError("`action_dist` must be a probability distribution")

    # position
    if position is not None:
        check_array(array=position, name="position", expected_dim=1)
        if not (position.shape[0] == action_dist.shape[0]):
            raise ValueError(
                "Expected `position.shape[0] == action_dist.shape[0]`, but found it False"
            )
        if not (np.issubdtype(position.dtype, np.integer) and position.min() >= 0):
            raise ValueError("`position` elements must be non-negative integers")
        if position.max() >= action_dist.shape[2]:
            raise ValueError(
                "`position` elements must be smaller than `action_dist.shape[2]`"
            )
    elif action_dist.shape[2] > 1:
        raise ValueError(
            "`position` elements must be given when `action_dist.shape[2] > 1`"
        )

    # estimated_rewards_by_reg_model
    if estimated_rewards_by_reg_model is not None:
        if estimated_rewards_by_reg_model.shape != action_dist.shape:
            raise ValueError(
                "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
            )

    if estimated_importance_weights is not None:
        if not (action.shape[0] == estimated_importance_weights.shape[0]):
            raise ValueError(
                "Expected `action.shape[0] == estimated_importance_weights.shape[0]`, but found it False"
            )
        if np.any(estimated_importance_weights < 0):
            raise ValueError("estimated_importance_weights must be non-negative")

    # action, reward
    if action is not None or reward is not None:
        check_array(array=action, name="action", expected_dim=1)
        check_array(array=reward, name="reward", expected_dim=1)
        if not (action.shape[0] == reward.shape[0]):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0]`, but found it False"
            )
        if not (
            np.issubdtype(action.dtype, np.integer)
            and action.min() >= 0
            and action.max() < action_dist.shape[1]
        ):
            raise ValueError(
                "`action` elements must be integers in the range of [0, `action_dist.shape[1]`)"
            )

    # pscore
    if pscore is not None:
        if pscore.ndim != 1:
            raise ValueError("pscore must be 1-dimensional")
        if not (action.shape[0] == reward.shape[0] == pscore.shape[0]):
            raise ValueError(
                "Expected `action.shape[0] == reward.shape[0] == pscore.shape[0]`, but found it False"
            )
        if np.any(pscore <= 0):
            raise ValueError("`pscore` must be positive")


def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator
