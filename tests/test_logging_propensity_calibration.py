"""The logged data must be draws from the logger whose probabilities they record.

From 69fffab (2026-09-24) to dbc401b (2026-10-04) every logged row's action reused the uniform that drew its
user (the logger's generator was seeded like the simulation's): a user's action was nearly fixed, while the
stored pscore was the logger's softmax probability. The older tests only checked marginal properties (the
pscore equals pi0(a|u) for the logged pair; the mean reward matches V(pi0)), which the coupled sampler also
satisfied. The tests here check the conditional and joint distributions themselves, on the production path
``_simulate_from_embedding_policy``; every one of them fails on the code before the fix
(docs/simulator_fix_opc_revalidation_20261004.md, Phase 1).

The smallest case (``test_minimal_case_two_users_two_actions``): two users with equal prior and a logger
that picks either of two actions with probability 1/2 for both. With the shared stream, user 0 is drawn when
U < 1/2 and action 0 when U < 1/2, so user 0 always got action 0 and user 1 always action 1, each row
recording pscore 0.5.
"""
import numpy as np
import pytest
from scipy import stats

from training.trainer_trials import _simulate_from_embedding_policy
from utils.simulation_utils import SyntheticBanditEnv, calc_reward, generate_dataset

P_MIN = 1e-6  # deterministic seeds: a correct sampler clears this by orders of magnitude, the coupled one fails it


def _world(n_users=300, n_items=40, d=6, seed=0, temperature=1.5, uniform_mix=0.0, prior=None, user_vectors=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_users, d)).astype(np.float32) if user_vectors is None else user_vectors.astype(np.float32)
    a = rng.normal(size=(n_items, d)).astype(np.float32)
    if prior is None:
        prior = rng.exponential(size=n_users)
    prior = np.asarray(prior, dtype=np.float64) / np.sum(prior)
    env = SyntheticBanditEnv(emb_x=rng.normal(size=(n_users, d)).astype(np.float32), emb_a=a, scale=0.5, offset=-1.5)
    return {"n_users": n_users, "n_actions": n_items, "our_x": x, "our_a": a, "user_prior": prior, "env": env,
            "policy_temperature": float(temperature), "logging_uniform_mix": float(uniform_mix), "emb_dim": d}


def _pi0(ds, factor=1.0):
    """The logger's exact probabilities (float64), or its logits times ``factor`` (0: the uniform policy)."""
    n = ds["n_actions"]
    if factor == 0.0:
        return np.full((ds["n_users"], n), 1.0 / n)
    z = factor * (ds["our_x"].astype(np.float64) @ ds["our_a"].astype(np.float64).T) / ds["policy_temperature"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    alpha = ds.get("logging_uniform_mix", 0.0) if factor == 1.0 else 0.0
    return (1.0 - alpha) * p + alpha / n


def _q(ds):
    return ds["env"].reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"]).astype(np.float64)


def _log(ds, n, random_state=25_225_124):
    sim = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], n, random_state=random_state)
    return (np.asarray(sim["users"], np.int64), np.asarray(sim["actions"], np.int64),
            np.asarray(sim["reward"], np.float64), np.asarray(sim["pscore"], np.float64))


def _gof_pvalue(counts, expected):
    """Chi-square goodness of fit, pooling the cells expected below 5 into one."""
    counts, expected = np.asarray(counts, float), np.asarray(expected, float)
    small = expected < 5
    if small.any():
        counts = np.append(counts[~small], counts[small].sum())
        expected = np.append(expected[~small], expected[small].sum())
        if expected[-1] == 0:
            counts, expected = counts[:-1], expected[:-1]
    if len(counts) < 2:
        return 1.0
    return float(stats.chisquare(counts, expected * counts.sum() / expected.sum()).pvalue)


# --------------------------------------------------------------------------- 1. fixed-context calibration
@pytest.mark.parametrize("temperature", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("uniform_mix", [0.0, 0.3])
def test_each_fixed_user_gets_actions_with_the_recorded_probabilities(temperature, uniform_mix):
    """Four fixed users (equal prior), ~10,000 draws each: P_emp(a | u) against the recorded pi0(a | u), from a
    sharp (T = 0.3) to a flat (T = 3) logger, with and without a uniform mixture."""
    ds = _world(n_users=4, n_items=12, seed=7, temperature=temperature, uniform_mix=uniform_mix, prior=np.ones(4))
    users, actions, _r, pscore = _log(ds, 40_000)
    pi0 = _pi0(ds)
    np.testing.assert_allclose(pscore, pi0[users, actions], rtol=1e-12, atol=0)
    for u in range(4):
        acts = actions[users == u]
        counts = np.bincount(acts, minlength=ds["n_actions"])
        assert _gof_pvalue(counts, len(acts) * pi0[u]) > P_MIN, (u, counts, np.round(len(acts) * pi0[u]))
        # the recorded propensity of the most logged action is its actual frequency
        top = int(np.argmax(pi0[u]))
        se = np.sqrt(pi0[u, top] * (1 - pi0[u, top]) / len(acts))
        assert abs(np.mean(acts == top) - pi0[u, top]) < 5 * se + 1e-12


# --------------------------------------------------------------------------- 2. independent randomness
def test_actions_are_independent_of_the_user_when_the_logger_is_the_same_for_all():
    """Every user has the same vectors, so pi0(. | u) is one distribution and the action must not depend on
    which user was drawn (the coupled sampler mapped each user's slice of [0, 1) to its own actions)."""
    rng = np.random.default_rng(3)
    ds = _world(n_users=20, n_items=8, seed=3, temperature=4.0,
                user_vectors=np.repeat(0.2 * rng.normal(size=(1, 6)), 20, 0))  # small logits: a spread logger
    assert _pi0(ds)[0].max() < 0.5  # (with a near-deterministic logger the test would have no power)
    users, actions, _r, _p = _log(ds, 40_000)
    table = np.zeros((20, 8))
    np.add.at(table, (users, actions), 1)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    assert stats.chi2_contingency(table).pvalue > P_MIN
    assert abs(stats.spearmanr(users, actions).statistic) < 0.03


def test_users_follow_the_prior_and_rewards_follow_q_given_the_logged_pair():
    ds = _world(n_users=30, n_items=10, seed=4, temperature=1.0)
    users, actions, rewards, _p = _log(ds, 60_000)
    assert _gof_pvalue(np.bincount(users, minlength=30), 60_000 * ds["user_prior"]) > P_MIN
    q = _q(ds)
    # within every logged (user, action) cell the click rate is q(u, a): a reward reusing the action's
    # uniform would bias these cells even when the overall mean is right
    cell = users * ds["n_actions"] + actions
    n_cell = np.bincount(cell, minlength=30 * 10)
    clicks = np.bincount(cell, weights=rewards, minlength=30 * 10)
    qc = q.reshape(-1)
    keep = n_cell * qc * (1 - qc) >= 5
    z = (clicks[keep] - n_cell[keep] * qc[keep]) / np.sqrt(n_cell[keep] * qc[keep] * (1 - qc[keep]))
    assert stats.chi2.sf(np.sum(z ** 2), keep.sum()) > P_MIN
    resid = rewards - q[users, actions]
    assert abs(np.corrcoef(resid, actions)[0, 1]) < 0.03 and abs(np.corrcoef(resid, users)[0, 1]) < 0.03


# --------------------------------------------------------------------------- 3. the joint distribution
def test_the_joint_of_users_and_actions_is_prior_times_pscore():
    """P_emp(u, a) against prior(u) * pi0(a | u) over every cell, and the importance-weight identities:
    E[pi_e / pscore] = 1 and E[pi_e / pscore * r] = V(pi_e) for targets near and far from the logger."""
    ds = _world(seed=1)
    n = 200_000
    users, actions, rewards, pscore = _log(ds, n)
    pi0 = _pi0(ds)
    joint = ds["user_prior"][:, None] * pi0
    counts = np.zeros_like(joint)
    np.add.at(counts, (users, actions), 1)
    assert _gof_pvalue(counts.reshape(-1), n * joint.reshape(-1)) > P_MIN
    q = _q(ds)
    for factor in (2.0, 0.5, 0.0):  # sharper logger, flatter logger, uniform
        pe = _pi0(ds, factor)
        w = pe[users, actions] / pscore
        assert abs(w.mean() - 1.0) < 5 * w.std() / np.sqrt(n), (factor, w.mean())
        v = float(ds["user_prior"] @ (pe * q).sum(axis=1))
        wr = w * rewards
        assert abs(wr.mean() - v) < 5 * wr.std() / np.sqrt(n), (factor, wr.mean(), v)


# --------------------------------------------------------------------------- 4. propensity accounting
@pytest.mark.parametrize("uniform_mix", [0.0, 0.3])
def test_stored_pscore_is_the_generating_policys_probability(uniform_mix):
    """The stored pscore is the logger's probability (formula), and it is the frequency at which each fixed
    user actually receives the logged action (realized)."""
    ds = _world(n_users=6, n_items=10, seed=5, temperature=0.7, uniform_mix=uniform_mix, prior=np.ones(6))
    users, actions, _r, pscore = _log(ds, 60_000)
    pi0 = _pi0(ds)
    np.testing.assert_allclose(pscore, pi0[users, actions], rtol=1e-12, atol=0)
    assert np.all(pscore >= uniform_mix / ds["n_actions"])
    for u in range(6):
        rows = users == u
        n_u = rows.sum()
        for a in np.unique(actions[rows]):
            p = pscore[rows & (actions == a)][0]  # what each of these rows recorded
            freq = np.mean(actions[rows] == a)
            assert abs(freq - p) < 5 * np.sqrt(p * (1 - p) / n_u) + 1e-12, (u, a, freq, p)


def test_uniform_rows_store_exactly_one_over_the_catalog():
    """A uniform logger (the mixture at weight 1, or constant logits): every pscore is exactly 1/|A|, and every
    user receives every action equally often (the coupled sampler gave each user a fixed slice of the catalog)."""
    n_items = 37
    for ds in (_world(n_users=40, n_items=n_items, seed=6, uniform_mix=1.0),  # the mixture at weight 1
               _world(n_users=40, n_items=n_items, seed=6, user_vectors=np.zeros((40, 6)))):  # constant logits
        users, actions, _r, pscore = _log(ds, 60_000)
        assert np.all(pscore == 1.0 / n_items)
        assert _gof_pvalue(np.bincount(actions, minlength=n_items), np.full(n_items, 60_000 / n_items)) > P_MIN
        table = np.zeros((40, n_items))
        np.add.at(table, (users, actions), 1)
        table = table[table.sum(axis=1) > 0]
        assert stats.chi2_contingency(table).pvalue > P_MIN


def test_softmax_rows_store_the_exact_softmax_in_float64():
    ds = _world(seed=8, temperature=0.25)  # sharp: many small probabilities
    users, actions, _r, pscore = _log(ds, 20_000)
    x, a = ds["our_x"].astype(np.float64), ds["our_a"].astype(np.float64)
    for i in range(0, 20_000, 997):  # row by row, independently of the vectorized reference above
        z = x[users[i]] @ a.T / ds["policy_temperature"]
        p = np.exp(z - z.max())
        assert pscore[i] == pytest.approx(p[actions[i]] / p.sum(), rel=1e-12)


# --------------------------------------------------------------------------- 5. logger sharpness
@pytest.fixture(scope="module")
def sharpness_worlds():
    from test_world import _embeddings

    X, A = _embeddings()
    return {s: generate_dataset({"bias": "medium", "ctr": 0.05, "logger_greedy_share": s}, seed=0, emb_x=X, emb_a=A)
            for s in (0.6, 0.8, 0.95)}


def test_a_sharper_logger_is_more_concentrated_in_the_population_and_in_the_logs(sharpness_worlds):
    exact, logged = [], []
    for share, ds in sharpness_worlds.items():
        pi0 = _pi0(ds)
        prior = ds["user_prior"].astype(np.float64) / ds["user_prior"].sum()
        entropy = float(prior @ -(pi0 * np.log(pi0)).sum(axis=1))
        collision = (pi0 ** 2).sum(axis=1)
        ess_uniform = 1.0 / float(prior @ ((1.0 / ds["n_actions"]) ** 2 / pi0).sum(axis=1))  # uniform target
        value = float(np.asarray(calc_reward(ds, _logger_of(ds))).reshape(-1)[0])
        exact.append((entropy, float(prior @ collision), ess_uniform, value))
        users, actions, rewards, _p = _log(ds, 30_000, random_state=11)
        repeats, expected = _repeats(users, actions, collision)
        se = np.sqrt(np.sum(expected * (1 - expected))) / len(expected)
        assert abs(repeats.mean() - expected.mean()) < 5 * se, (share, repeats.mean(), expected.mean())
        assert abs(rewards.mean() - value) < 5 * np.sqrt(value * (1 - value) / len(rewards))
        freq = np.bincount(actions, minlength=ds["n_actions"]) / len(actions)
        logged.append((rewards.mean(), float(-(freq[freq > 0] * np.log(freq[freq > 0])).sum())))
    entropies, collisions, ess, values = zip(*exact)
    assert list(entropies) == sorted(entropies, reverse=True) and list(ess) == sorted(ess, reverse=True)
    assert list(collisions) == sorted(collisions) and list(values) == sorted(values)
    mean_r, logged_entropy = zip(*logged)
    assert list(mean_r) == sorted(mean_r) and list(logged_entropy) == sorted(logged_entropy, reverse=True)


def _logger_of(ds):
    from test_world import _logger

    return _logger(ds)


def _repeats(users, actions, collision):
    """Whether each user's consecutive logged actions repeat, and the probability sum_a pi0(a|u)^2 they do."""
    order = np.argsort(users, kind="stable")
    u, a = users[order], actions[order]
    same_user = u[1:] == u[:-1]
    return (a[1:] == a[:-1])[same_user].astype(float), collision[u[1:][same_user]]


# --------------------------------------------------------------------------- 6. the smallest failing case
def test_minimal_case_two_users_two_actions():
    """Two users, equal prior, pi0 = (1/2, 1/2) for both: each user must receive both actions about equally.
    With the coupled streams user 0 always received action 0 and user 1 always action 1."""
    ds = _world(n_users=2, n_items=2, seed=0, temperature=1.0, prior=np.ones(2), user_vectors=np.zeros((2, 6)))
    users, actions, _r, pscore = _log(ds, 2_000)
    assert np.all(pscore == 0.5)
    for u in (0, 1):
        share = np.mean(actions[users == u] == 0)
        assert 0.4 < share < 0.6, (u, share)
