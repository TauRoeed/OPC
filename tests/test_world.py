"""The simulated world (utils/representation_bias.py): calibration targets are met on the
full population, the truth is fixed across bias configurations, draws depend only on the
seed, and every path that evaluates the click model agrees.

The toy catalog is smaller than ``WorldConfig.calib_rows``, so the bias levels are
calibrated on every user and item and the signal-kept targets hold exactly.
"""

import json

import numpy as np
import pytest
import torch

import utils.representation_bias as rb
from training.trainer_trials import AnalyticRewardModel, fit_shared_regression_bundle
from utils.policies import Policy
from utils.representation_bias import (
    LEVEL_SIGNAL_KEPT,
    SignalKept,
    WorldCalibrationError,
    WorldConfig,
    bias_label,
    calibrate_world,
    parse_bias,
    resolve_bias_configs,
)
from utils.simulation_utils import (
    calc_reward,
    calc_uniform_reward,
    create_simulation_data_from_policy,
    generate_dataset,
)

N_USERS, N_ITEMS, DIM = 2500, 1800, 16
CONFIGS = ("none", "low", "medium", "high", "high/none/none", "none/high/none", "none/none/high")


def _embeddings(seed=0):
    # BPR-like: a shared mean direction (popularity) plus item vectors with uneven norms.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=DIM) + 0.6 * rng.normal(size=(N_USERS, DIM))
    A = rng.normal(size=DIM) + rng.gamma(2.0, 0.4, size=(N_ITEMS, 1)) * rng.normal(size=(N_ITEMS, DIM))
    return X.astype(np.float32), A.astype(np.float32)


@pytest.fixture(scope="module")
def emb():
    return _embeddings()


@pytest.fixture(scope="module")
def worlds(emb):
    X, A = emb
    return {b: generate_dataset({"bias": b, "ctr": 0.05}, seed=0, emb_x=X, emb_a=A) for b in CONFIGS}


def _brute_signal_kept(bx, ba, X, A):
    S = bx.astype(np.float64) @ ba.astype(np.float64).T
    R = X.astype(np.float64) @ A.astype(np.float64).T
    S -= S.mean(axis=1, keepdims=True)
    R -= R.mean(axis=1, keepdims=True)
    return float(np.mean((S * R).sum(axis=1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(R, axis=1))))


def _logger(ds):
    return Policy(
        n_users=ds["n_users"], n_items=ds["n_actions"], user_emb=ds["our_x"], item_emb=ds["our_a"],
        emb_dim=ds["emb_dim"], temperature=ds["policy_temperature"], rng=np.random.default_rng(0),
    )


# ------------------------------------------------------------------ configs and labels
def test_parse_bias_and_labels():
    assert parse_bias("medium") == {"warp": "medium", "group": "medium", "vector": "medium"}
    assert parse_bias("high/none/low") == {"warp": "high", "group": "none", "vector": "low"}
    assert parse_bias({"group": "low"}) == {"warp": "none", "group": "low", "vector": "none"}
    assert bias_label("low/low/low") == "low"
    assert bias_label("high/none/low") == "w-high.g-none.v-low"
    for spec in ("high/none/low", "none", ("low", "medium", "high")):
        assert parse_bias(bias_label(spec)) == parse_bias(spec)
    assert resolve_bias_configs(["low", "low/low/low", "high/none/low"]) == ["low", "w-high.g-none.v-low"]
    for bad in ("extreme", "catastrophic", "high/low", "w-high.q-low", {"noise": "low"}):
        with pytest.raises(ValueError):
            parse_bias(bad)


def test_invalid_config_rejected(emb):
    X, A = emb
    for kw in ({"centering": 1.5}, {"logging_spread": 0.0}, {"target_ctr": 0.4},
               {"ctr_reference": "oracle"}, {"group_source": "metadata_only"}):
        with pytest.raises(ValueError):
            calibrate_world(X, A, seed=0, config=WorldConfig(**kw))


def test_legacy_noise_params_rejected(emb):
    X, A = emb
    for legacy in ({"eps1": 0.1}, {"noise_mode": "kmeans_templates"}, {"policy_temperature": 2.0}):
        with pytest.raises(ValueError, match="legacy"):
            generate_dataset({"bias": "low", "ctr": 0.05, **legacy}, seed=0, emb_x=X, emb_a=A)


# ------------------------------------------------------------------ calibration targets
def test_signal_kept_shortcut_matches_brute_force():
    rng = np.random.default_rng(5)
    X, A = rng.normal(size=(300, 12)) + 0.5, rng.normal(size=(500, 12)) - 0.3
    for scale in (0.0, 0.3, 3.0):
        bx = X + scale * rng.normal(size=X.shape)
        ba = A @ (np.eye(12) + scale * rng.normal(size=(12, 12)) / 4)
        assert SignalKept(X, A)(bx, ba) == pytest.approx(_brute_signal_kept(bx, ba, X, A), abs=1e-12)


def test_bias_levels_keep_their_signal(worlds):
    for level in ("none", "low", "medium", "high"):
        ds = worlds[level]
        brute = _brute_signal_kept(ds["our_x"], ds["our_a"], ds["emb_x"], ds["emb_a"])
        assert ds["world"]["signal_kept"] == pytest.approx(LEVEL_SIGNAL_KEPT[level], abs=0.01)
        assert brute == pytest.approx(ds["world"]["signal_kept"], abs=1e-4)  # float32 rounding of our_*
    # one type at level L keeps the per-type share kappa_L, between the joint target and 1
    kappa = worlds["high"]["world"]["per_type_signal_kept"]["high"]
    assert LEVEL_SIGNAL_KEPT["high"] < kappa < 1
    for single in ("high/none/none", "none/high/none", "none/none/high"):
        assert worlds[single]["world"]["signal_kept"] == pytest.approx(kappa, abs=0.02)


def test_levels_are_nested(worlds):
    kept = [worlds[lvl]["world"]["signal_kept"] for lvl in ("none", "low", "medium", "high")]
    assert kept == sorted(kept, reverse=True) and kept[0] == pytest.approx(1.0)
    table = worlds["low"]["world"]["eps_table"]
    for kind in rb.BIAS_TYPES:
        eps = [table[kind][lvl] for lvl in ("none", "low", "medium", "high")]
        assert eps[0] == 0.0 and eps == sorted(eps) and eps[-1] <= 1.0


def test_logging_spread_on_all_users(worlds):
    ds = worlds["none"]
    scores = ds["emb_x"].astype(np.float64) @ ds["emb_a"].astype(np.float64).T
    eff = rb._effective_items(scores, ds["policy_temperature"])
    assert eff / ds["n_actions"] == pytest.approx(0.5, abs=0.02)


def test_click_model_targets_on_full_population(worlds):
    ds = worlds["medium"]
    w, env = ds["world"], ds["env"]
    q = env.reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"])
    prior = ds["user_prior"].astype(np.float64) / ds["user_prior"].sum()
    assert float(prior @ q.max(axis=1)) == pytest.approx(0.30, abs=0.01)  # best item per user
    assert calc_reward(ds, _logger(ds)) == pytest.approx(0.05, abs=0.002)  # reference logger
    assert calc_uniform_reward(ds) == pytest.approx(w["uniform_ctr"], abs=0.002)
    assert w["reference_ctr"] == pytest.approx(0.05, abs=1e-9)  # calibration sample, exact
    assert w["best_item_ctr"] == pytest.approx(0.30, abs=1e-9)
    assert 0.0 < q.min() and q.max() < 1.0


def test_uniform_reference_targets_random_policy(emb):
    X, A = emb
    ds = generate_dataset({"bias": "medium", "ctr": 0.08, "ctr_reference": "uniform"}, seed=0, emb_x=X, emb_a=A)
    assert calc_uniform_reward(ds) == pytest.approx(0.08, abs=0.003)
    assert ds["world"]["uniform_ctr"] == pytest.approx(0.08, abs=1e-9)


# ------------------------------------------------------------------ fixed truth, seeds
def test_truth_is_fixed_across_bias_configs(worlds):
    ref = worlds["none"]
    np.testing.assert_array_equal(ref["our_x"], ref["emb_x"])  # no bias = the clean vectors
    np.testing.assert_array_equal(ref["our_a"], ref["emb_a"])
    for b, ds in worlds.items():
        for k in ("emb_x", "emb_a", "user_prior"):
            np.testing.assert_array_equal(ds[k], ref[k])
        assert (ds["env"].scale, ds["env"].offset) == (ref["env"].scale, ref["env"].offset)
        assert ds["policy_temperature"] == ref["policy_temperature"]
        if b != "none":
            assert not np.array_equal(ds["our_x"], ref["our_x"])
            assert not np.array_equal(ds["our_a"], ref["our_a"])


def test_centering_is_off_by_default_and_removes_share_of_mean(emb, worlds):
    X, A = emb
    mean = X.astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(worlds["none"]["emb_x"].mean(axis=0), mean, atol=1e-5)
    on = generate_dataset({"bias": "none", "ctr": 0.05, "centering": 0.8}, seed=0, emb_x=X, emb_a=A)
    np.testing.assert_allclose(on["emb_x"].mean(axis=0), 0.2 * mean, atol=1e-5)


def test_same_seed_same_world_other_seed_differs(emb):
    X, A = emb
    rb._CALIBRATION_CACHE.clear()
    a = generate_dataset({"bias": "high/low/medium", "ctr": 0.05}, seed=3, emb_x=X, emb_a=A)
    rb._CALIBRATION_CACHE.clear()
    b = generate_dataset({"bias": "high/low/medium", "ctr": 0.05}, seed=3, emb_x=X, emb_a=A)
    c = generate_dataset({"bias": "high/low/medium", "ctr": 0.05}, seed=4, emb_x=X, emb_a=A)
    for k in ("our_x", "our_a", "user_prior"):
        np.testing.assert_array_equal(a[k], b[k])
        assert not np.array_equal(a[k], c[k])
    assert json.dumps(a["world"], sort_keys=True) == json.dumps(b["world"], sort_keys=True)


def test_block_size_changes_the_world_only_by_rounding(emb, monkeypatch):
    X, A = emb
    rb._CALIBRATION_CACHE.clear()
    a = generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=X, emb_a=A)
    rb._CALIBRATION_CACHE.clear()
    monkeypatch.setattr(rb, "BLOCK_CELLS", 5_000)  # a few users per block
    b = generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=X, emb_a=A)
    rb._CALIBRATION_CACHE.clear()
    # BLAS may round blocks of different heights differently (1 ulp); nothing else changes
    for key in ("alpha", "b", "logging_temperature", "reference_ctr", "uniform_ctr", "logging_ctr", "signal_kept"):
        assert a["world"][key] == pytest.approx(b["world"][key], rel=1e-12)
    assert a["world"]["eps_table"] == b["world"]["eps_table"]
    np.testing.assert_array_equal(a["our_x"], b["our_x"])


def test_world_record_is_json(worlds):
    w = json.loads(json.dumps(worlds["medium"]["world"]))
    for key in ("eps_table", "per_type_signal_kept", "logging_temperature", "alpha", "b", "scale",
                "offset", "reference_ctr", "best_item_ctr", "uniform_ctr", "signal_kept", "logging_ctr",
                "groups", "config"):
        assert key in w
    assert w["config"]["reference_bias"] == ["medium", "medium", "medium"]


# ------------------------------------------------------------------ groups
def test_metadata_groups_and_fallback(emb, capsys):
    X, A = emb
    rng = np.random.default_rng(1)
    meta_a = np.eye(5)[rng.integers(0, 5, size=N_ITEMS)]  # item categories
    cfg = WorldConfig(group_source="metadata")
    cal = calibrate_world(X, A, seed=0, config=cfg, metadata_a=meta_a)
    assert cal["groups"]["items"] == "metadata" and cal["groups"]["users"] == "cluster"
    assert "no usable users metadata" in capsys.readouterr().out
    groups = cal["sides"]["items"].groups
    for g in np.unique(groups):  # every metadata group is one category
        assert len(np.unique(meta_a[groups == g].argmax(axis=1))) == 1
    rb._CALIBRATION_CACHE.clear()


# ------------------------------------------------------------------ infeasible targets
def test_unreachable_best_item_raises(emb):
    X, A = emb
    # a near-greedy clean reference logger already picks each user's best item
    cfg = dict(logging_spread=1.02 / N_ITEMS, reference_bias=("none", "none", "none"))
    with pytest.raises(WorldCalibrationError, match="unreachable"):
        calibrate_world(X, A, seed=0, config=WorldConfig(**cfg))
    cal = calibrate_world(X, A, seed=0, config=WorldConfig(strict=False, **cfg))
    assert cal["best_item_ctr"] < 0.30
    rb._CALIBRATION_CACHE.clear()


# ------------------------------------------------------------------ click-model paths agree
def test_click_model_paths_agree(worlds, monkeypatch):
    ds = worlds["medium"]
    env = ds["env"]
    users = np.arange(200)
    block = env.reward_prob_block(users, 0, ds["n_actions"])
    pairs = env.reward_prob(np.repeat(users, ds["n_actions"]), np.tile(np.arange(ds["n_actions"]), len(users)))
    np.testing.assert_allclose(pairs.reshape(block.shape), block, atol=1e-6)

    oracle = fit_shared_regression_bundle(ds, {}, reward_model="oracle")
    q_oracle = oracle["regression_model"].predict_user_action_block(oracle["user_context"][users], 0, ds["n_actions"])
    np.testing.assert_allclose(q_oracle[:, :, 0], block, atol=1e-6)
    logging_score = fit_shared_regression_bundle(ds, {}, reward_model="logging_score")["regression_model"]
    expected = AnalyticRewardModel.from_env(env, ds["our_a"], kind="x")._link(ds["our_x"][users] @ ds["our_a"].T)
    np.testing.assert_array_equal(logging_score.predict_user_action_block(ds["our_x"][users], 0, ds["n_actions"])[:, :, 0], expected)

    monkeypatch.setenv("OPC_EXACT_REWARD_DEVICE", "cpu")
    cpu_unif, cpu_log = calc_uniform_reward(ds), calc_reward(ds, _logger(ds))
    if torch.cuda.is_available():
        monkeypatch.setenv("OPC_EXACT_REWARD_DEVICE", "auto")
        assert calc_uniform_reward(ds) == pytest.approx(cpu_unif, abs=1e-7)
        assert calc_reward(ds, _logger(ds)) == pytest.approx(cpu_log, abs=1e-7)


def test_logged_rewards_match_exact_logging_value(worlds):
    ds = worlds["medium"]
    logger = _logger(ds)
    n = 400_000
    sim = create_simulation_data_from_policy(ds, logger, n, random_state=7)
    exact = calc_reward(ds, logger)
    z = (sim["reward"].mean() - exact) / np.sqrt(exact * (1 - exact) / n)
    assert abs(z) < 4, (sim["reward"].mean(), exact, z)
    assert ds["world"]["logging_ctr"] == pytest.approx(exact, rel=0.05)


def test_worlds_do_not_share_state_through_the_cache(emb):
    X, A = emb
    a = generate_dataset({"bias": "low", "ctr": 0.05}, seed=0, emb_x=X, emb_a=A)
    clean, eps_low = a["emb_x"].copy(), a["world"]["eps_table"]["warp"]["low"]
    a["emb_x"] *= 0.0  # in-place edits of one condition must not reach the next
    a["user_prior"][:] = 0.0
    a["world"]["eps_table"]["warp"]["low"] = -1.0
    b = generate_dataset({"bias": "low", "ctr": 0.05}, seed=0, emb_x=X, emb_a=A)
    np.testing.assert_array_equal(b["emb_x"], clean)
    assert b["user_prior"].sum() > 0 and b["world"]["eps_table"]["warp"]["low"] == eps_low
