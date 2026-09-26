"""The sharpened logger (utils/representation_bias.py, ``logger_greedy_share``, default 0.9): per
condition the logger's temperature is lowered from the spread temperature until it earns that
share of its own greedy CTR. The truth, the vectors and the click model do not change; ``off``
keeps the spread logger (the logger before 2026-09-26), and the study pipeline logs from and
starts its policies at the sharpened logger."""

import argparse

import numpy as np
import pandas as pd
import pytest
from test_world import _embeddings, _logger

from training.trainer_trials import _simulate_from_embedding_policy
from utils.representation_bias import (
    DEFAULT_LOGGER_GREEDY_SHARE,
    WorldCalibrationError,
    add_world_arguments,
    describe_world,
    parse_logger_greedy_share,
    sharpen_logger,
    world_options_from_args,
    world_run_key_suffix,
)
from utils.simulation_utils import SyntheticBanditEnv, calc_reward, generate_dataset


@pytest.fixture(scope="module")
def emb():
    return _embeddings()


def _world(emb, bias="medium", **params):
    X, A = emb
    return generate_dataset({"bias": bias, "ctr": 0.05, **params}, seed=0, emb_x=X, emb_a=A)


def _full_population_share(ds):
    """The logger's CTR over its greedy CTR on every user, prior-weighted."""
    prior = ds["user_prior"].astype(np.float64) / ds["user_prior"].sum()
    q = ds["env"].reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"])
    scores = ds["our_x"].astype(np.float64) @ ds["our_a"].astype(np.float64).T
    greedy = float(prior @ q[np.arange(len(q)), scores.argmax(axis=1)])
    return calc_reward(ds, _logger(ds)) / greedy


def test_default_and_parsing():
    assert DEFAULT_LOGGER_GREEDY_SHARE == 0.9
    for off in ("off", "OFF", "none", "0", 0, 0.0, None):
        assert parse_logger_greedy_share(off) == 0.0
    assert parse_logger_greedy_share("0.95") == 0.95
    for bad in (1.0, "1", 1.5, -0.1, "nan", "inf", "sharp"):
        with pytest.raises(ValueError):
            parse_logger_greedy_share(bad)


@pytest.mark.parametrize("share", [0.6, 0.8, 0.9, 0.95])
def test_logger_earns_the_share_of_its_greedy_ctr(emb, share):
    ds = _world(emb, logger_greedy_share=share)
    w = ds["world"]
    assert w["logger_greedy_share"] == share
    assert w["logger_share_achieved"] == pytest.approx(share, abs=1e-9)  # calibration users, exact
    assert w["logger_softmax_ctr"] == pytest.approx(share * w["logger_greedy_ctr"], rel=1e-9)
    assert ds["policy_temperature"] == w["logging_temperature"]
    assert w["logging_temperature"] == pytest.approx(w["spread_temperature"] / w["logger_sharpness"], rel=1e-12)
    # the calibration users are a 1,000-user sample of the prior: the full population is close
    assert _full_population_share(ds) == pytest.approx(share, abs=0.02)
    assert w["logging_ctr"] == pytest.approx(w["logger_softmax_ctr"], rel=0.03)  # sampled items agree


def test_a_higher_share_is_a_sharper_logger(emb):
    worlds = [_world(emb, logger_greedy_share=s)["world"] for s in (0.6, 0.8, 0.9, 0.95)]
    temps = [w["logging_temperature"] for w in worlds]
    items = [w["logger_effective_items"] for w in worlds]
    assert temps == sorted(temps, reverse=True) and items == sorted(items, reverse=True)
    assert all(w["logger_sharpness"] > 1.0 for w in worlds)  # the spread logger earns far less
    assert items[-1] < 0.1 * _world(emb, logger_greedy_share="off")["world"]["logger_effective_items"]


def test_a_share_below_the_spread_logger_flattens_it(emb):
    off = _world(emb, logger_greedy_share="off")["world"]
    own = off["spread_logger_ctr"] / off["logger_greedy_ctr"]
    w = _world(emb, logger_greedy_share=own / 2)["world"]
    assert w["logger_sharpness"] < 1.0
    assert w["logger_share_achieved"] == pytest.approx(own / 2, abs=1e-9)


@pytest.mark.parametrize("bias", ["none", "low", "high", "high/none/none", "none/high/none", "none/none/high"])
def test_every_bias_config_gets_its_own_sharpened_logger(emb, bias):
    ds, ref = _world(emb, bias), _world(emb)
    assert ds["world"]["logger_share_achieved"] == pytest.approx(0.9, abs=1e-9)
    assert ds["world"]["spread_temperature"] == ref["world"]["spread_temperature"]  # the truth is shared
    if bias == "none":  # no bias: the logger's own ranking is the true one
        assert ds["world"]["logger_greedy_ctr"] == pytest.approx(ds["world"]["best_item_ctr"], abs=0.02)


def test_only_the_logger_temperature_changes(emb):
    off, on = _world(emb, logger_greedy_share="off"), _world(emb)
    for k in ("our_x", "our_a", "emb_x", "emb_a", "user_prior"):
        np.testing.assert_array_equal(off[k], on[k])
    assert (off["env"].scale, off["env"].offset) == (on["env"].scale, on["env"].offset)
    for k in ("alpha", "b", "reference_ctr", "best_item_ctr", "uniform_ctr", "eps_table", "signal_kept",
              "spread_temperature", "logger_greedy_ctr", "spread_logger_ctr", "clean_logger_effective_items"):
        assert off["world"][k] == on["world"][k], k
    assert off["policy_temperature"] == off["world"]["spread_temperature"] == off["world"]["logging_temperature"]
    assert off["world"]["logger_sharpness"] == 1.0 and off["world"]["logger_greedy_share"] == 0.0
    assert off["world"]["logger_softmax_ctr"] == off["world"]["spread_logger_ctr"]
    assert on["policy_temperature"] < off["policy_temperature"]
    assert on["world"]["logging_ctr"] > 2 * off["world"]["logging_ctr"]


def test_uniform_mix_goes_on_top_of_the_sharpened_softmax(emb):
    on, mixed = _world(emb), _world(emb, logging_uniform_mix=0.3)
    assert mixed["policy_temperature"] == on["policy_temperature"]  # the share is the softmax part's
    w = mixed["world"]
    assert w["logging_ctr"] == pytest.approx(0.7 * on["world"]["logging_ctr"] + 0.3 * w["uniform_ctr"], rel=1e-12)


def test_logged_data_come_from_the_sharpened_logger(emb):
    ds = _world(emb)
    sim = _simulate_from_embedding_policy(ds, ds["our_x"], ds["our_a"], 20_000, random_state=3)
    users, actions = np.asarray(sim["users"]), np.asarray(sim["actions"])
    lg = ds["our_x"][users].astype(np.float64) @ ds["our_a"].astype(np.float64).T / ds["policy_temperature"]
    lg -= lg.max(axis=1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(sim["pscore"], p[np.arange(len(users)), actions], rtol=1e-4)
    exact = calc_reward(ds, _logger(ds))
    assert abs(sim["reward"].mean() - exact) < 4 * np.sqrt(exact * (1 - exact) / len(users))


def test_sharpen_logger_matches_a_dense_computation():
    rng = np.random.default_rng(5)
    cx, ca = rng.normal(size=(40, 6)), rng.normal(size=(90, 6))  # the truth
    bx, ba = cx + 0.5 * rng.normal(size=cx.shape), ca + 0.5 * rng.normal(size=ca.shape)  # the logger's view
    env = SyntheticBanditEnv(emb_x=cx.astype(np.float32), emb_a=ca.astype(np.float32), scale=0.8, offset=-2.0)
    users = rng.integers(0, 40, size=300)
    out = sharpen_logger(bx, ba, users, 1.7, env, 0.85)
    s = bx[users] @ ba.T / 1.7
    q = env.reward_prob_block(users, 0, 90)
    greedy = q[np.arange(len(users)), s.argmax(axis=1)].mean()
    lg = s * out["factor"]
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    assert (p * q).sum(axis=1).mean() / greedy == pytest.approx(0.85, abs=1e-9)
    assert out["temperature"] == pytest.approx(1.7 / out["factor"], rel=1e-15)
    assert out["greedy_ctr"] == pytest.approx(greedy, rel=1e-7)  # click probabilities kept in float32
    off = sharpen_logger(bx, ba, users, 1.7, env, 0.0)
    assert off["factor"] == 1.0 and off["softmax_ctr"] == off["spread_ctr"]


def test_an_unreachable_share_raises():
    # two tied items: greedy takes the better one, every softmax splits between them
    bx, ba = np.ones((1, 1)), np.ones((2, 1))
    env = SyntheticBanditEnv(emb_x=np.ones((1, 1), dtype=np.float32), emb_a=np.array([[3.0], [0.0]], dtype=np.float32),
                             scale=1.0, offset=-1.0)
    with pytest.raises(WorldCalibrationError, match="unreachable: even logits x1e"):
        sharpen_logger(bx, ba, np.zeros(4, dtype=np.int64), 1.0, env, 0.95)
    # scores unrelated to the truth: even the uniform policy earns most of the greedy CTR
    rng = np.random.default_rng(5)
    env = SyntheticBanditEnv(emb_x=rng.normal(size=(40, 6)).astype(np.float32),
                             emb_a=rng.normal(size=(90, 6)).astype(np.float32), scale=0.8, offset=-2.0)
    with pytest.raises(WorldCalibrationError, match="x1e-06|x1e-6"):
        sharpen_logger(rng.normal(size=(40, 6)), rng.normal(size=(90, 6)), rng.integers(0, 40, 300), 1.7, env, 0.5)


def test_cli_run_key_and_description(emb):
    p = argparse.ArgumentParser()
    add_world_arguments(p)
    default = world_options_from_args(p.parse_args([]))
    assert default["logger_greedy_share"] == 0.9 and world_run_key_suffix(default) == ""
    off = world_options_from_args(p.parse_args(["--logger-greedy-share", "off"]))
    assert off["logger_greedy_share"] == 0.0 and world_run_key_suffix(off) == "__lgs=0"
    sharp = world_options_from_args(p.parse_args(["--logger-greedy-share", "0.95"]))
    assert world_run_key_suffix(sharp) == "__lgs=0.95"
    with pytest.raises(SystemExit):
        p.parse_args(["--logger-greedy-share", "1.2"])
    assert "90% of its greedy CTR" in describe_world(_world(emb)["world"])
    assert "sharpening off" in describe_world(_world(emb, logger_greedy_share="off")["world"])


def test_study_logs_from_and_starts_at_the_sharpened_logger(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=1,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000)
    initial = {}
    for share in (0.9, 0.0):
        run_dir = tmp_path / f"lgs{share}"
        run_dir.mkdir()
        opc, nop, *_, meta = _run_condition(**kw, run_dir=run_dir, world_options={"logger_greedy_share": share})
        w = meta["world"]
        assert w["logger_greedy_share"] == share
        summary = _finalize_summary_df(opc, nop, meta)
        assert set(summary["method"]) == {"opc", "no_propensity"} and (summary["logger_greedy_share"] == share).all()
        assert (summary["logger_sharpness"] > 1.0).all() == (share > 0)
        trials = pd.read_csv(run_dir / "trials_long.csv")
        initial[share] = trials["initial_reward"].unique()
        assert len(initial[share]) == 1  # both methods start at the same logger
    assert initial[0.9][0] > 2 * initial[0.0][0]  # the sharpened logger is the better starting point
