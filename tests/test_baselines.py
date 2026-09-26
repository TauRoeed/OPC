"""The opt-in baselines and the logit scale: the DM loss and DM-only selection (weights 'dm'),
the CFModel logit scale (fixed, learned, exported), the tempered-logger trials, and the study
arms (--methods dm tempered_logger, --learn-logit-scale). Defaults leave the older code paths
exactly as they were."""

import argparse
import math

import numpy as np
import pandas as pd
import pytest
import torch

from models.custom_losses import DMPolicyLoss, SNDRPolicyLoss, dm_surrogate, transform_importance_weights
from models.models import LOGIT_SCALE_SPEED, CFModel, make_policy_transform
from utils.importance_weights import parse_weight_spec, transform_weights, weight_spec_label
from utils.policies import Policy
from utils.simulation_utils import _estimator_weight_kwargs


def _batch(seed=0, n=64, n_actions=30):
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(n, n_actions, generator=g)
    logits = torch.randn(n, n_actions, generator=g, requires_grad=True)
    actions = torch.randint(0, n_actions, (n,), generator=g)
    rewards = torch.bernoulli(torch.full((n,), 0.2), generator=g)
    pscore = torch.rand(n, generator=g) * 0.1 + 0.01
    return scores, logits, actions, rewards, pscore


# ------------------------------------------------------------------ DM loss and 'dm' weights
@pytest.mark.parametrize("log_trick", [False, True])
def test_dm_loss_is_the_dm_term_and_ignores_the_logged_data(log_trick):
    scores, logits, actions, rewards, pscore = _batch()
    prob = torch.softmax(logits, dim=1)
    loss = DMPolicyLoss(use_log_trick=log_trick)(pscore, scores, prob, rewards, actions)
    other = DMPolicyLoss(use_log_trick=log_trick)(pscore * 3, scores, prob, 1 - rewards, (actions + 1) % 30)
    assert torch.equal(loss, other)  # logged actions, rewards and propensities do not enter
    if not log_trick:
        torch.testing.assert_close(loss, -(scores * prob).sum(dim=1).mean())
    (g,) = torch.autograd.grad(loss, logits)
    (g_path,) = torch.autograd.grad(-(scores * torch.softmax(logits, dim=1)).sum(dim=1).mean(), logits)
    torch.testing.assert_close(g, g_path)  # exact over all actions: the log trick has the same gradient


def test_dm_term_is_the_sndr_dm_part():
    scores, logits, actions, rewards, pscore = _batch(1)
    prob = torch.softmax(logits, dim=1)
    # SNDR = DM + correction; with rewards equal to q_hat at the logged action, the correction is 0
    q_at = scores[torch.arange(len(actions)), actions]
    sndr = SNDRPolicyLoss(use_log_trick=False)(pscore, scores, prob, q_at, actions)
    dm = DMPolicyLoss(use_log_trick=False)(pscore, scores, prob, q_at, actions)
    torch.testing.assert_close(sndr, dm)
    torch.testing.assert_close(dm_surrogate(scores, prob, use_log_trick=False), (scores * prob).sum(dim=1))


def test_dm_weight_spec():
    assert parse_weight_spec("dm") == ("dm", 0.0) and parse_weight_spec(("dm", 5)) == ("dm", 0.0)
    assert weight_spec_label("DM") == "dm"
    w = np.array([0.1, 1.0, 50.0])
    np.testing.assert_array_equal(transform_weights(w, "dm"), np.zeros(3))
    with pytest.raises(ValueError):
        parse_weight_spec("dm:3")
    with pytest.raises(ValueError, match="DM-only selection"):  # not a training transform
        transform_importance_weights(torch.ones(3), torch.ones(3), use_iw=True, iw_mode="dm")
    with pytest.raises(ValueError, match="post-hoc"):
        _estimator_weight_kwargs("dm")


# ------------------------------------------------------------------ CFModel logit scale
def _cf(pop=False, **kw):
    rng = np.random.default_rng(0)
    U = torch.tensor(rng.normal(size=(12, 5)), dtype=torch.float32)
    V = torch.tensor(rng.normal(size=(9, 5)), dtype=torch.float32)
    extra = {"item_popularity": rng.normal(size=9).astype(np.float32), "pop_weight": 0.7} if pop else {}
    m = CFModel(12, 9, 5, initial_user_embeddings=U.clone(), initial_actions_embeddings=V.clone(),
                user_transform=make_policy_transform("linear", 5), action_transform=make_policy_transform("linear", 5),
                temperature=0.6, **extra, **kw)
    return m, U, V, extra


@pytest.mark.parametrize("pop", [False, True])
def test_default_scale_is_the_older_model_exactly(pop):
    m, U, V, extra = _cf(pop)
    assert m.log_logit_scale is None and m.logit_scale == 1.0 and m._scale() is None
    assert "log_logit_scale" not in dict(m.named_parameters())
    with torch.no_grad():
        logits = U @ V.T + (0.7 * torch.as_tensor(extra["item_popularity"]) if pop else 0.0)
        assert torch.equal(m(torch.arange(12))[:, :, 0], torch.softmax(logits / 0.6, dim=1))


@pytest.mark.parametrize("pop", [False, True])
@pytest.mark.parametrize("learn", [False, True])
def test_scale_sharpens_and_is_exported_into_the_user_vectors(pop, learn):
    m, U, V, extra = _cf(pop, logit_scale=2.5, learn_logit_scale=learn)
    assert m.logit_scale == pytest.approx(2.5, rel=1e-6)
    with torch.no_grad():
        logits = U @ V.T + (0.7 * torch.as_tensor(extra["item_popularity"]) if pop else 0.0)
        prob = m(torch.arange(12))[:, :, 0]
        torch.testing.assert_close(prob, torch.softmax(2.5 * logits / 0.6, dim=1), rtol=1e-5, atol=1e-7)
        ex, ea = m.get_params()
    # the exported vectors score s * (u.a + w.b): the simulator's Policy reproduces the model
    pol = Policy(n_users=12, n_items=9, user_emb=ex.numpy(), item_emb=ea.numpy(), emb_dim=ex.shape[1], temperature=0.6)
    np.testing.assert_allclose(pol._probs_block(np.arange(12)), prob.numpy(), rtol=1e-5, atol=1e-7)
    if pop:  # the item side keeps w.b: the learned popularity weight is still read off the items
        np.testing.assert_allclose(ea[:, -1].numpy(), 0.7 * extra["item_popularity"], rtol=1e-6)
    clone = m.clone()
    assert clone.logit_scale == pytest.approx(m.logit_scale) and (clone.log_logit_scale is not None) == learn


def test_learned_scale_moves_fast_and_starts_where_asked():
    m, U, V, _ = _cf(logit_scale=1.0, learn_logit_scale=True)
    assert m.log_logit_scale.item() == 0.0 and LOGIT_SCALE_SPEED > 1
    target = torch.zeros(12, 9)
    target[torch.arange(12), (U @ V.T).argmax(dim=1)] = 1.0  # a greedy target: sharpen
    opt = torch.optim.Adam([m.log_logit_scale], lr=1e-3)
    for _ in range(40):
        opt.zero_grad()
        (-(m(torch.arange(12))[:, :, 0] * target).sum(dim=1).mean()).backward()
        opt.step()
    # 40 Adam steps at lr 1e-3 move log s by up to 40 * 1e-3 * speed
    assert 1.5 < m.logit_scale < math.exp(40 * 1e-3 * LOGIT_SCALE_SPEED) * 1.01
    with pytest.raises(ValueError):
        _cf(logit_scale=0.0)


# ------------------------------------------------------------------ trainer: DM selection, tempered logger
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _dataset_paths
    from utils.simulation_utils import generate_dataset

    root = tmp_path_factory.mktemp("toy")
    _toy_embeddings(root)
    up, ip, _, _ = _dataset_paths(root, "toy")
    return root, generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))


def _trainer(ds, tmp_path, **kw):
    """(summary, trials) of one trainer call; the trials table is the logged trials_long."""
    from training.trainer_trials import regression_trainer_trial

    logs = {"trials": tmp_path / "trials_long.csv", "runs": tmp_path / "runs_long.csv"}
    for f in logs.values():
        f.unlink(missing_ok=True)
    args = dict(train_sizes=[1000], dataset=ds, batch_size=None, val_size=1000, val_min=1000, n_trials=3, slim=True,
                shared_regression_size=2000, search_use_log_trick=False, use_log_trick_fixed=False, log_paths=logs)
    summary, _ = regression_trainer_trial(**{**args, **kw})
    return summary, pd.read_csv(logs["trials"])


def test_dm_selection_scores_q_hat_alone(toy, tmp_path):
    _, ds = toy
    summary, trials = _trainer(ds, tmp_path, method_label="dm", policy_loss_types=("dm",), select_estimator="dm",
                               log_select_weights=("dm", "clip:10"))
    assert (trials["ess"] == 0.0).all()  # every weight is 0
    # the selection score is the DM value: the 'dm' variant logged by the tuning hook is the same number
    np.testing.assert_allclose(trials["r_hat"], trials["sel_r_hat[dm]"], rtol=1e-6)
    assert not np.allclose(trials["r_hat"], trials["sel_r_hat[clip:10]"])
    with pytest.raises(ValueError, match="select_estimator"):
        _trainer(ds, tmp_path, select_estimator="ips")
    with pytest.raises(ValueError, match="logged"):
        _trainer(ds, tmp_path, select_estimator="dm", propensity_mode="uniform")


def test_tempered_logger_trials_are_the_logger_times_a_scale(toy, tmp_path):
    from training.trainer_trials import TEMPER_SCALE_RANGE, _policy_reward_from_embeddings

    _, ds = toy
    summary, trials = _trainer(ds, tmp_path, method_label="tempered_logger", temper_only=True, n_trials=4)
    assert trials["param_logit_scale"].between(*TEMPER_SCALE_RANGE).all()
    assert trials["param_lr"].isna().all() and (trials["param_num_epochs"] == -1).all()  # nothing trained
    for s, r in zip(trials["param_logit_scale"], trials["actual_reward"]):
        assert r == pytest.approx(_policy_reward_from_embeddings(ds, ds["our_x"] * s, ds["our_a"]), rel=1e-5)
    best = trials.loc[trials["is_best_in_run"], "param_logit_scale"].item()
    assert summary.loc[1000, "logit_scale"] == pytest.approx(best)


def test_learned_scale_reaches_the_trainer(toy, tmp_path):
    _, ds = toy
    _, fixed = _trainer(ds, tmp_path, n_trials=2)
    _, learned = _trainer(ds, tmp_path, n_trials=2, learn_logit_scale=True)
    assert (fixed["logit_scale"] == 1.0).all()
    assert (learned["logit_scale"] != 1.0).all()


# ------------------------------------------------------------------ the study arms
def test_study_runs_the_baseline_arms(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import (
        ALL_STUDY_METHODS,
        VALID_STUDY_METHODS,
        _finalize_summary_df,
        _normalize_study_methods,
        _run_condition,
        _summary_has_methods,
    )

    assert VALID_STUDY_METHODS == ("opc", "no_propensity")  # the default arms are unchanged
    assert set(ALL_STUDY_METHODS) == {"opc", "no_propensity", "dm", "tempered_logger"}
    with pytest.raises(ValueError):
        _normalize_study_methods(["opc", "oracle"])
    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000)
    run_dir = tmp_path / "all"
    run_dir.mkdir()
    out = _run_condition(**kw, run_dir=run_dir, methods=ALL_STUDY_METHODS, learn_logit_scale=True, return_extra=True)
    *_, meta, extra = out
    assert set(extra) == {"dm", "tempered_logger"} and meta["learn_logit_scale"] is True
    summary = _finalize_summary_df(out[0], out[1], meta, extra=extra)
    assert set(summary["method"]) == set(ALL_STUDY_METHODS) and summary["learn_logit_scale"].all()
    trials = pd.read_csv(run_dir / "trials_long.csv")
    assert set(trials["method"]) == set(ALL_STUDY_METHODS)
    assert trials.groupby("method")["initial_reward"].nunique().eq(1).all()  # every arm starts at the same logger
    assert trials["initial_reward"].nunique() == 1
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    assert _summary_has_methods(run_dir / "summary_metrics.csv", ["dm", "opc"])
    assert not _summary_has_methods(run_dir / "summary_metrics.csv", ["opc", "other"])
    # the default call keeps its 5-item return and runs only the default arms
    run_dir = tmp_path / "default"
    run_dir.mkdir()
    out = _run_condition(**kw, run_dir=run_dir)
    assert len(out) == 5 and set(pd.read_csv(run_dir / "trials_long.csv")["method"]) == {"opc", "no_propensity"}


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_accepts_the_baselines_and_the_scale(module, monkeypatch, capsys):
    import sys

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    text = " ".join(capsys.readouterr().out.split())
    assert "tempered_logger" in text and "--learn-logit-scale" in text


def test_every_arm_logs_the_dr_selection_variants(toy, tmp_path):
    """The 2 x 2 design after the fact: no-prop trials (naive training, naive selection) also get the
    DR selection scores with the logged propensities, equal to what OPC's scoring gives them."""
    from training.trainer_trials import _selection_score_variants

    _, ds = toy
    _, trials = _trainer(ds, tmp_path, method_label="no_propensity", policy_loss_types=("naive",),
                         propensity_mode="uniform", log_select_weights=("clip:10", "dm"))
    assert trials[["sel_r_hat[clip:10]", "sel_ci_low[clip:10]", "sel_r_hat[dm]"]].notna().all().all()
    assert (trials["ess"] == trials["ess"].iloc[0]).all()  # its own selection stays naive (no weights)
    assert not np.allclose(trials["r_hat"], trials["sel_r_hat[clip:10]"])
