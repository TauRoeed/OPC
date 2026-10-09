"""The new shared OPC arms of docs/opc_gradient_regime_study.md §6-§8: raw-DR OPC, oracle-q training and the
matched-steps replays with larger batches; their configuration keys; the batch sampler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from test_shared_objectives import N, _trainer, toy  # noqa: F401 (the module's toy world fixture)
from training.training_utils import MatchedStepsBatchSampler

PARAMS = ["param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay", "param_anchor_lambda"]


def test_the_matched_steps_sampler_draws_full_batches_without_replacement_within_passes():
    s = MatchedStepsBatchSampler(n=10, batch_rows=4, steps_per_epoch=5, seed=3)
    epoch1, epoch2 = [list(b) for b in s], [list(b) for b in s]
    assert len(epoch1) == len(epoch2) == 5 and all(len(b) == 4 for b in epoch1 + epoch2)
    for i in range(0, 4, 2):  # two batches per permutation of 10 rows: disjoint within a pass
        assert not set(epoch1[i]) & set(epoch1[i + 1])
    again = [list(b) for b in MatchedStepsBatchSampler(n=10, batch_rows=4, steps_per_epoch=5, seed=3)]
    assert again == epoch1  # reproducible from its seed
    full = [list(b) for b in MatchedStepsBatchSampler(n=7, batch_rows=100, steps_per_epoch=3, seed=0)]
    assert full == [list(range(7))] * 3  # the full batch every step


def test_raw_dr_opc_is_paired_with_shared_opc_and_trains_raw_weights(toy, tmp_path):  # noqa: F811
    _, ds = toy
    h_summary, h = _trainer(ds, tmp_path, "shared_opc", seed_label="shared", shared_objective="opc")
    r_summary, r = _trainer(ds, tmp_path, "shared_opc_raw", seed_label="shared", shared_objective="opc",
                            train_weights="none")
    pd.testing.assert_frame_equal(h[PARAMS], r[PARAMS])  # trial k: the same configuration
    assert not np.allclose(h["actual_reward"], r["actual_reward"])  # a different objective
    assert h_summary.loc[N, "arm_train_weights"] == "harmonic:0.1" and r_summary.loc[N, "arm_train_weights"] == "none"
    for col in ("train_rows_sha1", "val_rows_sha1", "train_click_sum"):  # the same rows
        assert h_summary.loc[N, col] == r_summary.loc[N, col]


def test_oracle_q_training_reads_the_true_click_model(toy, tmp_path):  # noqa: F811
    from training.trainer_trials import _scores_lookup_from_bundle, fit_shared_regression_bundle

    _, ds = toy
    lookup = _scores_lookup_from_bundle(fit_shared_regression_bundle(ds, None, reward_model="oracle"), torch.device("cpu"))
    users = np.array([0, 5, 17])
    np.testing.assert_allclose(lookup[torch.as_tensor(users)].numpy(),
                               ds["env"].reward_prob_block(users, 0, int(ds["n_actions"])), rtol=1e-5, atol=1e-7)
    base_summary, base = _trainer(ds, tmp_path, "shared_opc", seed_label="shared", shared_objective="opc")
    oq_summary, oq = _trainer(ds, tmp_path, "shared_opc_oq", seed_label="shared", shared_objective="opc",
                              train_reward="oracle")
    pd.testing.assert_frame_equal(base[PARAMS], oq[PARAMS])
    assert not np.allclose(base["actual_reward"], oq["actual_reward"])
    assert oq_summary.loc[N, "arm_train_reward"] == "oracle"
    with pytest.raises(ValueError, match="shared OPC"):
        _trainer(ds, tmp_path, "x", train_reward="oracle")


@pytest.mark.parametrize("train_batch", [8192, "full"])
def test_the_matched_steps_replay_keeps_the_trials_steps(toy, tmp_path, train_batch):  # noqa: F811
    _, ds = toy
    summary, t = _trainer(ds, tmp_path, f"shared_opc_b{train_batch}", seed_label="shared", shared_objective="opc",
                          train_batch=train_batch)
    _, base = _trainer(ds, tmp_path, "shared_opc", seed_label="shared", shared_objective="opc")
    pd.testing.assert_frame_equal(base[PARAMS], t[PARAMS])
    steps = t["param_num_epochs"] * np.ceil(N / t["param_batch_size"])
    np.testing.assert_array_equal(t["train_steps"].to_numpy(), steps.to_numpy())
    assert (t["train_batch_rows"] == min(N, 8192)).all()  # N = 1000: both are the full batch here
    assert summary.loc[N, "arm_train_batch"] == str(train_batch)


def test_the_new_arms_have_their_own_configuration_keys():
    from training.run_state import arm_config, config_key

    cfg = {"n_trials": 20, "train_weights": "harmonic:0.1", "shared_options": {"lambdas": [0.0, 0.001, 0.01, 0.1, 1.0]}}
    keys = {m: config_key(arm_config(m, cfg)) for m in ("shared_opc", "shared_opc_raw", "shared_opc_oq",
                                                          "shared_opc_b8192", "shared_opc_bfull", "shared_opc_raw_oq")}
    assert len(set(keys.values())) == len(keys)
    base = arm_config("shared_opc", cfg)
    assert "train_reward" not in base["shared"] and base["train_weights"] == "harmonic:0.1"  # unchanged
    assert arm_config("shared_opc_raw", cfg)["train_weights"] == "none"
    assert arm_config("shared_opc_bfull", cfg)["shared"]["train_batch"] == "full"
