"""The greedy (exploitation) value logged next to every policy's softmax value: the true CTR of
recommending each user the policy's top item (calc_greedy_reward), on the GPU and the CPU, per
trial (actual_reward_greedy) and for the selected policy (policy_rewards_greedy)."""

import numpy as np
import pandas as pd
import pytest
import torch
from test_world import _embeddings

from utils.simulation_utils import calc_greedy_reward, generate_dataset


@pytest.fixture(scope="module")
def ds():
    X, A = _embeddings()
    return generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=X, emb_a=A)


def _dense_greedy(ds, ux, ia):
    prior = ds["user_prior"].astype(np.float64) / ds["user_prior"].sum()
    q = ds["env"].reward_prob_block(np.arange(ds["n_users"]), 0, ds["n_actions"])
    best = (ux.astype(np.float32) @ ia.astype(np.float32).T).argmax(axis=1)
    return float(prior @ q[np.arange(len(q)), best])


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_greedy_value_matches_a_dense_computation(ds, device, monkeypatch):
    if device == "auto" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    monkeypatch.setenv("OPC_EXACT_REWARD_DEVICE", device)
    rng = np.random.default_rng(3)
    for ux, ia in ((ds["our_x"], ds["our_a"]), (ds["our_x"] + 0.3 * rng.normal(size=ds["our_x"].shape), ds["our_a"])):
        assert calc_greedy_reward(ds, ux, ia) == pytest.approx(_dense_greedy(ds, ux, ia), rel=1e-6)
    # the temperature (a positive scale on the scores) never moves the argmax
    assert calc_greedy_reward(ds, 7.0 * ds["our_x"], ds["our_a"]) == calc_greedy_reward(ds, ds["our_x"], ds["our_a"])
    # the logger's greedy value on every user is close to the calibration users' estimate
    assert calc_greedy_reward(ds, ds["our_x"], ds["our_a"]) == pytest.approx(ds["world"]["logger_greedy_ctr"], abs=0.02)


def test_study_logs_the_greedy_value(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import ALL_STUDY_METHODS, _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    opc, nop, _, _, meta, extra = _run_condition(
        dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=3,
        batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
        policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, run_dir=run_dir, methods=ALL_STUDY_METHODS,
        return_extra=True)
    trials = pd.read_csv(run_dir / "trials_long.csv")
    assert trials["actual_reward_greedy"].notna().all()
    summary = _finalize_summary_df(opc, nop, meta, extra=extra)
    assert summary["policy_rewards_greedy"].notna().all()
    logger = summary[summary.train_size == 0]["policy_rewards_greedy"]
    assert logger.nunique() == 1  # every arm starts at the same logger
    # tempering never changes the ranking: every tempered-logger trial has the logger's greedy value
    tempered = trials[trials.method == "tempered_logger"]["actual_reward_greedy"]
    np.testing.assert_allclose(tempered, logger.iloc[0], rtol=1e-12)
    # the selected policy's greedy value is its trial's
    sel = trials[trials.is_best_in_run.astype(bool)].set_index("method")["actual_reward_greedy"]
    got = summary[summary.train_size == 1000].set_index("method")["policy_rewards_greedy"]
    np.testing.assert_allclose(got.loc[sel.index], sel, rtol=1e-12)
