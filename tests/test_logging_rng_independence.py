"""Logged actions must be drawn independently of the user draw (bug present from 69fffab to 2026-10-03).

``create_simulation_data_from_policy`` draws users from ``default_rng(random_state)``. The logger used to be
seeded with the same integer, so each row's action reused its user's uniform: ~95% of users always got the
same action (ml, 25k rows). The test compares how often a user's consecutive logged actions repeat with the
collision probability sum_a pi(a|u)^2 that independent draws imply.
"""
import numpy as np
import pytest

from training.trainer_trials import _simulate_from_embedding_policy
from utils.policies import Policy
from utils.simulation_utils import SyntheticBanditEnv, create_simulation_data_from_policy


def _world(n_users=300, n_items=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_users, d)).astype(np.float32)
    a = rng.normal(size=(n_items, d)).astype(np.float32)
    prior = rng.exponential(size=n_users)
    prior /= prior.sum()
    env = SyntheticBanditEnv(emb_x=x, emb_a=a, scale=0.5, offset=-2.0)
    return {"n_users": n_users, "n_actions": n_items, "our_x": x, "our_a": a, "user_prior": prior, "env": env,
            "policy_temperature": 1.5, "emb_dim": d}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def test_logged_actions_repeat_only_as_often_as_independent_draws_would():
    ds = _world()
    sim = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], 30_000, random_state=25_225_124)
    probs = _softmax(ds["our_x"].astype(np.float64) @ ds["our_a"].astype(np.float64).T / ds["policy_temperature"])
    collision = (probs ** 2).sum(axis=1)
    users, actions = sim["users"], sim["actions"]
    repeats, expected = [], []
    for u in np.unique(users):
        acts = actions[users == u]
        if len(acts) >= 2:
            repeats.extend((acts[1:] == acts[:-1]).astype(float))
            expected.extend([collision[u]] * (len(acts) - 1))
    repeats, expected = np.asarray(repeats), np.asarray(expected)
    se = np.sqrt(np.sum(expected * (1 - expected))) / len(expected)
    assert abs(repeats.mean() - expected.mean()) < 5 * se, (repeats.mean(), expected.mean())
    assert abs(np.corrcoef(users, actions)[0, 1]) < 0.05


def test_logged_pscores_are_the_logger_probabilities():
    ds = _world(seed=1)
    sim = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], 5_000, random_state=11)
    probs = _softmax(ds["our_x"].astype(np.float64) @ ds["our_a"].astype(np.float64).T / ds["policy_temperature"])
    assert np.allclose(sim["pscore"], probs[sim["users"], sim["actions"]], rtol=1e-6)


def test_simulation_is_deterministic_per_random_state():
    ds = _world(seed=2)
    s1 = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], 2_000, random_state=5)
    s2 = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], 2_000, random_state=5)
    for k in ("users", "actions", "reward", "pscore"):
        assert np.array_equal(s1[k], s2[k])


def test_a_policy_sharing_the_simulation_seed_is_rejected():
    ds = _world(seed=3)
    policy = Policy(n_users=ds["n_users"], n_items=ds["n_actions"], user_emb=ds["our_x"], item_emb=ds["our_a"],
                    emb_dim=ds["emb_dim"], temperature=1.5, rng=np.random.default_rng(7))
    with pytest.raises(ValueError, match="same state"):
        create_simulation_data_from_policy(ds, policy, 100, random_state=7)
