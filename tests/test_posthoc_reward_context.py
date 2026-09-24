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

N_USERS, N_ITEMS, DIM, CTR, TEMP = 80, 50, 6, 0.05, 1.0


def _dataset():
    rng = np.random.default_rng(0)
    params = {
        "n_users": N_USERS, "n_actions": N_ITEMS, "emb_dim": DIM, "n_clusters": 8,
        "eps1": 0.3, "eps2": 0.3, "eps_meta": 0.0, "sigma1": 1.0, "sigma2": 1.0, "sigma_meta": 1.0,
        "noise_mode": "kmeans_templates", "noise_axis": "combined", "noise_component": "combined",
        "noise_apply_user": True, "noise_apply_item": True,
        "ctr": CTR, "policy_temperature": TEMP, "logging_uniform_mix": 0.0,
    }
    return generate_dataset(
        params=params, seed=0, store_original=True,
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

    # A trained policy: its embeddings differ from both the noisy and the clean vectors.
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
    pi = _softmax(learned_x[users].astype(np.float64) @ learned_a.astype(np.float64).T / TEMP)
    q_hat = model.predict_user_action_block(context[users], 0, N_ITEMS)[:, :, 0]
    expected_dm = float(np.mean((pi * q_hat).sum(axis=1)))

    np.testing.assert_allclose(res["reg_dm"], expected_dm, rtol=1e-5)
    np.testing.assert_allclose(res["conv_dm"], expected_dm, rtol=1e-5)

    if reward_model == "oracle":
        # The oracle's inputs are the clean vectors, so DM equals the true value on val users.
        q_true = 1.0 / (1.0 / CTR + np.exp(-(ds["emb_x"][users].astype(np.float64) @ ds["emb_a"].astype(np.float64).T)))
        np.testing.assert_allclose(expected_dm, float(np.mean((pi * q_true).sum(axis=1))), rtol=1e-5)
