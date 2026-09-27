"""Oracle repair bound (training/oracle_repair.py): the learner's own policy class trained on the true click
probabilities, scored by the study's exact value functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from models.models import GlobalLinearCorrection
from training.oracle_repair import (
    ORACLE_CLASSES,
    build_oracle_model,
    exact_values,
    fit_oracle,
    logger_values,
    main,
    oracle_repair,
    true_q_rows,
)
from training.run_full_study import build_condition_world
from utils.seeding import seed_everything

FAST = dict(steps=300, fit_users=2000, batch_users=512)


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    tmp = tmp_path_factory.mktemp("oracle")
    _toy_embeddings(tmp)
    out = {}
    for bias in ("none", "high/none/none", "none/none/high"):
        seed_everything(0)
        out[bias] = build_condition_world("toy", tmp, bias, 0.05, 0)[0]
    return tmp, out


def test_true_q_rows_is_the_simulator_click_model(worlds):
    _, w = worlds
    d = w["high/none/none"]
    users = torch.arange(0, 40)
    items = torch.as_tensor(np.asarray(d["env"].emb_a), dtype=torch.float32)
    got = true_q_rows(d, users, items).numpy()
    want = d["env"].reward_prob_block(np.arange(40), 0, d["n_actions"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-7)


def test_oracle_class_is_the_learners_and_starts_at_the_logger(worlds):
    _, w = worlds
    d = w["high/none/none"]
    dim = int(d["emb_dim"])
    log = logger_values(d)
    for cls in ORACLE_CLASSES:
        model = build_oracle_model(d, cls)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n = sum(p.numel() for p in trainable)
        if cls == "scale":
            assert model.user_transform is None and n == 1
        else:
            assert isinstance(model.user_transform, GlobalLinearCorrection) and isinstance(model.action_transform, GlobalLinearCorrection)
            assert n == 2 * (dim * dim + dim) + (cls == "linear+scale")
        assert not model.user_embeddings.weight.requires_grad and not model.actions_embeddings.weight.requires_grad
        assert model.temperature == pytest.approx(float(d["policy_temperature"]))
        start = exact_values(d, model)  # before any step: exactly the logger
        assert start["value"] == pytest.approx(log["value"], abs=1e-9) and start["greedy"] == pytest.approx(log["greedy"], abs=1e-12)


def test_logger_value_is_the_studys_initial_reward(worlds):
    from training.run_full_study import _run_condition

    tmp, w = worlds
    run_dir = tmp / "study"
    run_dir.mkdir()
    _run_condition(dataset_name="toy", emb_dir=tmp, bias="high/none/none", ctr=0.05, seed=0, train_sizes=[1000], n_trials=1,
                   batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
                   policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",), run_dir=run_dir)
    initial = pd.read_csv(run_dir / "trials_long.csv")["initial_reward"].iloc[0]
    # the same exact-value function; the trainer allows TF32 matmuls on the GPU, so float32 rounding differs slightly
    assert logger_values(w["high/none/none"])["value"] == pytest.approx(float(initial), rel=1e-6)


@pytest.mark.parametrize("bias", ["high/none/none", "none/none/high"])
def test_oracle_never_loses_and_sharpening_keeps_the_ranking(worlds, bias):
    _, w = worlds
    d = w[bias]
    res = oracle_repair(d, lrs=(1e-2,), **FAST)
    assert res["logger_greedy"] <= res["logger_ceiling"] + 1e-12
    for cls in ORACLE_CLASSES:
        assert res[f"oracle_{cls}_value"] >= res["logger_value"] - 1e-6  # starts at the logger, only ascends
        assert res[f"oracle_{cls}_greedy"] <= res["logger_ceiling"] + 1e-12
    assert res["oracle_scale_greedy"] == pytest.approx(res["logger_greedy"], abs=1e-12)  # sharpening alone: same ranking
    assert res["oracle_linear+scale_logit_scale"] > 0


def test_warp_is_repairable_on_the_toy(worlds):
    """A warp is one shared linear map; the linear class can undo it, so the oracle recovers most of the
    ranking loss (the toy has 8 dimensions, so the fit is quick)."""
    _, w = worlds
    res = oracle_repair(w["high/none/none"], classes=("linear",), lrs=(1e-2,), steps=1500, fit_users=4000, batch_users=1024)
    loss = res["logger_ceiling"] - res["logger_greedy"]
    assert loss > 0.005
    assert (res["oracle_linear_greedy"] - res["logger_greedy"]) / loss > 0.5


def test_no_bias_ranking_is_already_the_ceiling(worlds):
    _, w = worlds
    log = logger_values(w["none"])
    assert log["greedy"] == pytest.approx(log["ceiling"], rel=1e-6)


def test_deterministic(worlds):
    _, w = worlds
    d = w["none/none/high"]
    a, ta = fit_oracle(d, "linear+scale", lr=1e-2, **FAST)
    b, tb = fit_oracle(d, "linear+scale", lr=1e-2, **FAST)
    assert ta == tb and exact_values(d, a) == exact_values(d, b)


def test_cli_writes_rows_and_skips_done(worlds, tmp_path):
    tmp, _ = worlds
    argv = ["--datasets", "toy", "--bias-configs", "high/none/none", "--seeds", "0", "--classes", "linear", "scale",
            "--lrs", "0.01", "--steps", "100", "--fit-users", "1000", "--batch-users", "256", "--emb-dir", str(tmp),
            "--out", str(tmp_path / "o")]
    main(argv)
    rows = pd.read_csv(tmp_path / "o" / "oracle_repair.csv")
    assert len(rows) == 1 and rows["stage"].iloc[0] == "development"
    for col in ("logger_value", "logger_greedy", "logger_ceiling", "oracle_linear_value", "oracle_linear_greedy",
                "oracle_scale_value", "oracle_linear_flat", "oracle_linear_lr", "temperature", "bias_label"):
        assert col in rows.columns
    main(argv)  # already done: nothing appended
    assert len(pd.read_csv(tmp_path / "o" / "oracle_repair.csv")) == 1
    assert (tmp_path / "o" / "oracle_settings.json").exists()
