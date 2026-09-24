"""Post-hoc estimates must feed the reward model its own user vectors.

The reward model is fit on (or, for the oracle, defined on) one fixed copy of the user
vectors, ``bundle["user_context"]``. A trial's learned embeddings define the *policy*
being evaluated; they must never be used as reward-model inputs.
"""

import numpy as np
import pytest

from models.estimators import DirectMethod
from training.trainer_trials import (
    IndexToContextModelWrapper,
    _build_regression_logged_split,
    fit_shared_regression_bundle,
    get_trial_results,
)
from utils.simulation_utils import generate_dataset

N_USERS, N_ITEMS, DIM = 80, 300, 6  # enough items for a 5% logger next to a 30% best item


def _dataset():
    rng = np.random.default_rng(0)
    return generate_dataset(
        params={"bias": "high", "ctr": 0.05}, seed=0,
        emb_x=rng.standard_normal((N_USERS, DIM)).astype(np.float32),
        emb_a=rng.standard_normal((N_ITEMS, DIM)).astype(np.float32),
    )


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("reward_model", ["oracle", "regression"])
def test_posthoc_dm_uses_reward_model_vectors(reward_model):
    ds = _dataset()
    split = _build_regression_logged_split(
        ds, ds["our_x"], ds["our_a"], 500, 400, 0, split_seed=3, regression_size=3000
    )
    bundle = fit_shared_regression_bundle(ds, split["reg_data"], reward_model=reward_model)
    context = bundle["user_context"]
    assert not context.flags.writeable, "the reward model's vectors must be read-only"

    # A trained policy: its embeddings differ from both the biased and the clean vectors.
    rng = np.random.default_rng(1)
    learned_x = ds["our_x"] + 0.7 * rng.standard_normal(ds["our_x"].shape).astype(np.float32)
    learned_a = ds["our_a"] + 0.7 * rng.standard_normal(ds["our_a"].shape).astype(np.float32)

    model = bundle["regression_model"]
    res = get_trial_results(
        learned_x, learned_a, ds["emb_x"], ds["emb_a"], ds["original_x"], ds["original_a"],
        ds, split["val_data"], None, IndexToContextModelWrapper(model, ds["our_x"]), model,
        DirectMethod(), reward_context=context,
    )

    users = np.asarray(split["val_data"]["x_idx"], dtype=np.int64)
    pi = _softmax(learned_x[users].astype(np.float64) @ learned_a.astype(np.float64).T / ds["policy_temperature"])
    q_hat = model.predict_user_action_block(context[users], 0, N_ITEMS)[:, :, 0]
    expected_dm = float(np.mean((pi * q_hat).sum(axis=1)))

    np.testing.assert_allclose(res["reg_dm"], expected_dm, rtol=1e-5)
    np.testing.assert_allclose(res["conv_dm"], expected_dm, rtol=1e-5)

    if reward_model == "oracle":
        # The oracle's inputs are the clean vectors, so DM equals the true value on val users.
        q_true = ds["env"].reward_prob_block(users, 0, N_ITEMS)
        np.testing.assert_allclose(expected_dm, float(np.mean((pi * q_true).sum(axis=1))), rtol=1e-5)
