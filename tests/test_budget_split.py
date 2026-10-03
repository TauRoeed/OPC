"""Fixed-budget warm/uniform split for the CausE comparison (utils/budget_split.py)."""
import numpy as np
import pytest
from scipy import stats

from training.trainer_trials import _simulate_from_embedding_policy
from utils.budget_split import (CAUSE_RHOS, UNIFORM_POLICY, WARM_POLICY, budget_counts, build_budget_split,
                                expected_rewards, simulate_uniform_pool, uniform_pool_seed)
from utils.simulation_utils import SyntheticBanditEnv, calc_uniform_reward, get_train_data


def _world(n_users=300, n_items=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_users, d)).astype(np.float32)
    a = rng.normal(size=(n_items, d)).astype(np.float32)
    prior = rng.exponential(size=n_users)
    prior /= prior.sum()
    env = SyntheticBanditEnv(emb_x=x, emb_a=a, scale=0.5, offset=-1.5)
    return {"n_users": n_users, "n_actions": n_items, "our_x": x, "our_a": a, "user_prior": prior, "env": env,
            "policy_temperature": 1.5, "emb_dim": d}


def _warm(ds, n, seed=123):
    sim = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], n, random_state=seed)
    idx = np.random.default_rng(seed).permutation(n)  # like the random reg/train/val partition
    return get_train_data(ds["n_actions"], n, sim, idx, ds["our_x"])


@pytest.mark.parametrize("n", [5_000, 25_000, 100_000, 12_345])
def test_counts_add_up_to_the_budget(n):
    for rho in CAUSE_RHOS + (1.0,):
        n_c, n_t = budget_counts(n, rho)
        assert n_c + n_t == n and n_t == round(rho * n) and n_c >= 0


def test_counts_at_25k_and_invalid_shares():
    assert [budget_counts(25_000, r)[1] for r in CAUSE_RHOS] == [0, 250, 1250, 2500, 3750, 6250]
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            budget_counts(100, bad)


def test_uniform_pool_actions_are_uniform_and_independent_of_users():
    ds = _world()
    pool = simulate_uniform_pool(ds, 80_000, seed=17)
    counts = np.bincount(pool["a"], minlength=ds["n_actions"])
    assert stats.chisquare(counts).pvalue > 1e-4
    assert np.all(pool["pscore"] == 1.0 / ds["n_actions"])
    users, actions = pool["x_idx"], pool["a"]
    order = np.argsort(users, kind="stable")
    same_user = users[order][1:] == users[order][:-1]
    repeat = (actions[order][1:] == actions[order][:-1])[same_user]
    se = np.sqrt((1 / 40) * (1 - 1 / 40) / repeat.size)
    assert abs(repeat.mean() - 1 / 40) < 5 * se  # a user's consecutive actions repeat at the uniform rate
    assert abs(np.corrcoef(users, actions)[0, 1]) < 0.03


def test_uniform_pool_users_follow_the_world_prior():
    ds = _world(seed=1)
    pool = simulate_uniform_pool(ds, 60_000, seed=5)
    prior = ds["user_prior"]
    vals = prior[pool["x_idx"]]
    expected = float(np.sum(prior ** 2))  # E[prior(u)] for u ~ prior
    se = np.sqrt(float(np.sum(prior ** 3)) - expected ** 2) / np.sqrt(len(vals))
    assert abs(vals.mean() - expected) < 5 * se


def test_expected_rewards_are_the_true_click_probabilities():
    ds = _world(seed=2)
    pool = simulate_uniform_pool(ds, 50_000, seed=9)
    assert np.allclose(pool["q"], ds["env"].reward_prob(pool["x_idx"], pool["a"]))
    v_uniform = calc_uniform_reward(ds)
    se = pool["q"].std() / np.sqrt(len(pool["q"]))
    assert abs(pool["q"].mean() - v_uniform) < 5 * se
    se_r = np.sqrt(v_uniform * (1 - v_uniform) / len(pool["r"]))
    assert abs(pool["r"].mean() - pool["q"].mean()) < 5 * se_r


def test_split_is_an_exact_nested_prefix_with_no_extra_rows():
    ds = _world(seed=3)
    n = 4_000
    warm = _warm(ds, n)
    pool = simulate_uniform_pool(ds, n, seed=uniform_pool_seed(100, n))
    previous = None
    for rho in CAUSE_RHOS:
        s = build_budget_split(ds, warm, pool, n, rho)
        c, t, m = s["control"], s["treatment"], s["meta"]
        assert len(c["a"]) + len(t["a"]) == n == m["n_control"] + m["n_treatment"]
        assert np.array_equal(c["a"], warm["a"][: m["n_control"]]) and np.array_equal(c["x_idx"], warm["x_idx"][: m["n_control"]])
        assert np.array_equal(c["pscore"], warm["pscore"][: m["n_control"]])  # exact logger propensities kept
        assert np.array_equal(t["a"], pool["a"][: m["n_treatment"]]) and np.all(t["pscore"] == 1 / ds["n_actions"])
        assert c["collection_policy"] == WARM_POLICY and t["collection_policy"] == UNIFORM_POLICY
        assert m["control_reward_sum"] == pytest.approx(warm["r"][: m["n_control"]].sum())
        assert m["replaced_warm_rows"] == m["n_treatment"]
        assert m["collection_reward_sum"] == pytest.approx(m["control_reward_sum"] + m["treatment_reward_sum"])
        assert m["control_expected_reward_sum"] == pytest.approx(
            expected_rewards(ds, warm["x_idx"][: m["n_control"]], warm["a"][: m["n_control"]]).sum())
        if previous is not None:  # nested: a larger rho randomizes a superset and keeps a prefix of the warm rows
            assert np.array_equal(t["a"][: len(previous["treatment"]["a"])], previous["treatment"]["a"])
            assert np.array_equal(c["a"], previous["control"]["a"][: len(c["a"])])
        previous = s


def test_split_rejects_a_wrong_budget():
    ds = _world(seed=4)
    warm = _warm(ds, 1_000)
    pool = simulate_uniform_pool(ds, 100, seed=1)
    with pytest.raises(ValueError, match="warm split"):
        build_budget_split(ds, warm, pool, 2_000, 0.05)
    with pytest.raises(ValueError, match="uniform pool"):
        build_budget_split(ds, warm, pool, 1_000, 0.25)


def test_pool_is_deterministic_and_seeded_per_condition_and_size():
    ds = _world(seed=5)
    a = simulate_uniform_pool(ds, 2_000, seed=uniform_pool_seed(100, 25_000))
    b = simulate_uniform_pool(ds, 2_000, seed=uniform_pool_seed(100, 25_000))
    for k in ("x_idx", "a", "r"):
        assert np.array_equal(a[k], b[k])
    assert len({uniform_pool_seed(100, 25_000), uniform_pool_seed(101, 25_000), uniform_pool_seed(100, 100_000)}) == 3
