"""The vectorized reward-model q_hat must equal sklearn's own per-pair predictions.

The fast path assumes q_hat(x, a) = expit(x @ w_x + a @ w_a + b), i.e. the default
concat features ``[x, action_context[a]]`` fed to a binary LogisticRegression. If the
features or base model change, these tests fail instead of results going silently wrong.
"""

import numpy as np
import pytest
import torch
from sklearn.base import is_classifier
from sklearn.linear_model import LogisticRegression

from models.models import RegressionModel, _ConstantBinaryProbaClassifier
from training.trainer_trials import RegressionScoresLookup

N_USERS, N_ACTIONS, DIM = 120, 300, 8


def _fitted_model(rewards=None, seed=0):
    rng = np.random.default_rng(seed)
    user_x = rng.standard_normal((N_USERS, DIM)).astype(np.float32)
    action_x = rng.standard_normal((N_ACTIONS, DIM)).astype(np.float32)
    n = 4000
    users = rng.integers(0, N_USERS, n)
    actions = rng.integers(0, N_ACTIONS, n)
    if rewards is None:
        logits = (user_x[users] * action_x[actions]).sum(axis=1)
        rewards = (rng.random(n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    model = RegressionModel(
        n_actions=N_ACTIONS,
        len_list=1,
        action_context=action_x,
        base_model=LogisticRegression(random_state=0),
    )
    model.fit(user_x[users], actions, np.broadcast_to(rewards, (n,)).astype(float))
    return model, user_x


def _per_pair_reference(model, context, a0, a1):
    """q_hat via sklearn predict_proba on the model's own feature function."""
    n = context.shape[0]
    return np.stack(
        [model.predict_pairs(context, np.full(n, a, dtype=int)) for a in range(a0, a1)],
        axis=1,
    )


def test_block_matches_sklearn_per_pair():
    model, user_x = _fitted_model()
    assert model.linear_qhat_parts(0) is not None, "fast path should apply to LogisticRegression"
    for a0, a1 in [(0, N_ACTIONS), (17, 133)]:
        fast = model.predict_user_action_block(user_x, a0, a1)[:, :, 0]
        np.testing.assert_allclose(fast, _per_pair_reference(model, user_x, a0, a1), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_single_label_fallback_predicts_probability(label):
    model, user_x = _fitted_model(rewards=np.array(label))
    base = model.base_model_list[0]
    assert isinstance(base, _ConstantBinaryProbaClassifier)
    assert is_classifier(base), "sklearn must see the fallback as a classifier (predict_proba)"
    expected = np.clip(label, 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(model.predict_user_action_block(user_x[:5], 0, N_ACTIONS), expected)
    np.testing.assert_allclose(_per_pair_reference(model, user_x[:5], 0, 10), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gpu_batch_rows_match_cpu():
    model, user_x = _fitted_model()
    lookup = RegressionScoresLookup(model, user_x, torch.device("cuda:0"))
    assert lookup._linear_gpu is not None
    idx = np.array([3, 7, 7, 0, 119, 42])
    gpu_rows = lookup[torch.as_tensor(idx, device="cuda:0")].cpu().numpy()
    np.testing.assert_allclose(gpu_rows, lookup.qhat_rows_numpy(idx), rtol=1e-6, atol=1e-12)
