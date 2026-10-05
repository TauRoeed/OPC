"""training/policy_diagnostics.py: the pick diagnostics against dense numpy, and the selected-policy files of the OPC,
CausE-cap and BLOB arms (--save-policies) reproduce the arms' own greedy values."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.policy_diagnostics import (BIN_NAMES, PI0_BINS, POLICY_SUFFIX, greedy_pick_diagnostics,
                                         load_selected_policy, logger_reference, pairwise_table, save_selected_policy)
from utils.simulation_utils import _normalized_prior, calc_greedy_reward


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import build_condition_world
    from utils.seeding import seed_everything

    root = tmp_path_factory.mktemp("policy_diag")
    _toy_embeddings(root)
    seed_everything(0)
    dataset, *_ = build_condition_world("toy", Path(root), "medium", 0.05, 0, world_options={})
    return dataset


def _dense(dataset):
    from training.trainer_trials import _policy_temperature

    env = dataset["env"]
    q = 1 / (1 + np.exp(-(env.scale * np.asarray(env.emb_x, np.float64) @ np.asarray(env.emb_a, np.float64).T
                          + env.offset)))
    z = np.asarray(dataset["our_x"], np.float64) @ np.asarray(dataset["our_a"], np.float64).T / _policy_temperature(dataset)
    pi0 = np.exp(z - z.max(1, keepdims=True))
    return q, pi0 / pi0.sum(1, keepdims=True)


def test_greedy_diagnostics_match_dense_numpy(toy):
    rng = np.random.default_rng(0)
    K = toy["our_x"].shape[1]
    ux = rng.standard_normal((toy["n_users"], K + 1)).astype(np.float32)
    ia = rng.standard_normal((toy["n_actions"], K + 1)).astype(np.float32)
    offset = rng.standard_normal(toy["n_users"]).astype(np.float32)
    d = greedy_pick_diagnostics(toy, ux, ia, offset=offset)
    assert d["value_greedy"] == pytest.approx(calc_greedy_reward(toy, ux, ia), abs=1e-9)
    q, pi0 = _dense(toy)
    prior = _normalized_prior(toy)
    P = toy["n_actions"]
    f = ux.astype(np.float64) @ ia.T.astype(np.float64) + offset[:, None]
    picks = f.argmax(1)
    rows = np.arange(toy["n_users"])
    p = 1 / (1 + np.exp(-f))
    assert d["optimism_at_pick"] == pytest.approx(prior @ (p[rows, picks] - q[rows, picks]), abs=1e-5)
    ce = np.logaddexp(0, f) - q * f
    assert d["logged_ce"] == pytest.approx(prior @ (pi0 * ce).sum(1), rel=1e-4)
    b = np.digitize(pi0 * P, PI0_BINS, right=True)  # torch.bucketize(right=False): edges[i-1] < x <= edges[i]
    for k, name in enumerate(BIN_NAMES):
        m = b == k
        pairs = prior @ m.sum(1)
        assert d[f"pairs_{name}"] == pytest.approx(pairs / P, abs=1e-6)
        assert d[f"logged_{name}"] == pytest.approx(prior @ (pi0 * m).sum(1), abs=1e-6)
        if pairs > 0:
            assert d[f"mae_{name}"] == pytest.approx(prior @ (np.abs(p - q) * m).sum(1) / pairs, rel=1e-4)
    assert sum(d[f"pairs_{n}"] for n in BIN_NAMES) == pytest.approx(1.0)
    assert sum(d[f"logged_{n}"] for n in BIN_NAMES) == pytest.approx(1.0, abs=1e-6)
    below = pi0[rows, picks] * P < 1
    assert d["pick_below_uniform"] == pytest.approx(prior[below].sum(), abs=1e-9)
    z = np.asarray(toy["our_x"], np.float32) @ np.asarray(toy["our_a"], np.float32).T
    rank = (z > z[rows, picks][:, None]).sum(1)
    assert d["pick_rank_ge100"] == pytest.approx(prior[rank >= 100].sum(), abs=1e-9)
    order = np.argsort(rank, kind="stable")
    assert d["pick_rank_median"] == rank[order][np.searchsorted(np.cumsum(prior[order]), 0.5)]
    top1 = rank == 0
    assert d["agree_logger_top1"] == pytest.approx(prior[top1].sum(), abs=1e-9)
    assert d["q_at_picks_not_top1"] == pytest.approx(prior[~top1] @ q[rows, picks][~top1] / prior[~top1].sum(), rel=1e-6)


def test_the_logger_agrees_with_itself_and_pairs_are_consistent(toy):
    ref = logger_reference(toy)
    d = greedy_pick_diagnostics(toy, toy["our_x"], toy["our_a"], ref=ref, return_picks=True)
    assert d["agree_logger_top1"] == pytest.approx(1.0) and d["in_logger_top10"] == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    other = greedy_pick_diagnostics(toy, toy["our_x"] + rng.standard_normal(toy["our_x"].shape).astype(np.float32),
                                    toy["our_a"], ref=ref, return_picks=True)
    t = pairwise_table(_normalized_prior(toy), {"a_logger": d, "b_other": other}, toy["n_actions"])
    r = t.iloc[0]
    assert r["value_diff"] == pytest.approx(d["value_greedy"] - other["value_greedy"], abs=1e-12)
    assert r["value_diff"] == pytest.approx(r["value_diff_a_rarer"] + r["value_diff_b_rarer"], abs=1e-12)
    assert r["agree"] + r["users_a_rarer"] + r["users_b_rarer"] <= 1.0 + 1e-12


def test_save_and_load_roundtrip(tmp_path):
    ux, ia = np.ones((3, 2), np.float32), np.zeros((4, 2), np.float32)
    save_selected_policy(tmp_path / "x.npz", ux, ia, offset=0.5, arm="opc", value_greedy=0.25)
    z = load_selected_policy(tmp_path / "x.npz")
    np.testing.assert_array_equal(z["ux"], ux)
    assert float(z["offset"]) == 0.5 and z["arm"] == "opc" and z["value_greedy"] == 0.25


def test_every_arm_saves_its_selected_policy(tmp_path):
    from test_reproducibility import _toy_embeddings
    from training.run_full_study import build_condition_world, _run_condition
    from utils.seeding import seed_everything

    _toy_embeddings(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cause = {"family": "cap", "rhos": [0.0], "dim": 8, "batch_size": 128, "n_trials": 3, "epochs": [3], "temper": True,
             "bias_inits": ["base_rate"]}
    blob = {"families": ["nq"], "n_trials": 3, "epochs": [2], "batch_size": 128, "pick_diagnostics": True}
    out = _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000],
                         n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                         policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=run_dir, slim=True,
                         shared_regression_size=2000, methods=("opc", "cause", "blob"), sampler="random",
                         return_extra=True, cause_options=cause, blob_options=blob, save_policies=True)
    opc_df, extra = out[0], out[5]
    seed_everything(0)
    dataset, *_ = build_condition_world("toy", tmp_path, "medium", 0.05, 0, world_options={})
    expect = {"opc_n1000_r0": float(opc_df.loc[1000, "policy_rewards_greedy"]),
              "causecap_c_r000_n1000": float(extra["causecap_c_r000"][0].loc[1000, "policy_rewards_greedy"]),
              "causecap_t_r000_n1000": float(extra["causecap_t_r000"][0].loc[1000, "policy_rewards_greedy"]),
              "blob_nq_n1000": float(extra["blob_nq"][0].loc[1000, "policy_rewards_greedy"])}
    for name, value in expect.items():
        pol = load_selected_policy(run_dir / f"{name}{POLICY_SUFFIX}")
        d = greedy_pick_diagnostics(dataset, pol["ux"], pol["ia"], offset=pol["offset"])
        assert d["value_greedy"] == pytest.approx(value, abs=1e-9), name
        assert pol["value_greedy"] == pytest.approx(value, abs=1e-12), name
    trials = extra["blob_nq"][1]
    assert trials["diag_value_greedy"].to_numpy() == pytest.approx(trials["value_greedy"].to_numpy(), abs=1e-9)
    assert np.isfinite(trials["diag_mae_lt0.1"]).all() or (trials["diag_pairs_lt0.1"] == 0).any()
