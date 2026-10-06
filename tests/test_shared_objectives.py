"""The shared correction model and the three objectives (docs/shared_objective_study.md): equivalence with OPC's model
and objective, the regularizer and the losses against direct computations, the weights, the click head, the paired
search, data identity and the selection paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from models.custom_losses import DRPolicyLoss
from models.models import LOGIT_SCALE_SPEED, CFModel, GlobalLinearCorrection
from models.shared_objectives import (
    LAMBDA_GRID,
    AnchoredPolicyLoss,
    LikelihoodLoss,
    SharedCorrectionModel,
    SourceAnchor,
    fit_click_head,
    iw_weights,
    weight_stats,
    weighted_nll,
)
from test_reproducibility import _toy_embeddings

N, K, USERS, ITEMS = 1000, 8, 400, 600  # the toy world of tests/test_reproducibility.py


def _vectors(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((30, 6)).astype(np.float32), rng.standard_normal((40, 6)).astype(np.float32)


def _set_params(model, seed=1, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for t in (model.user_transform, model.action_transform):
            t.delta.copy_(scale * torch.randn(t.delta.shape, generator=g))
            t.bias.copy_(scale * torch.randn(t.bias.shape, generator=g))
        model.log_logit_scale.fill_(0.04)


def _pair(mode="policy", temperature=0.7):
    x, a = _vectors()
    common = dict(initial_user_embeddings=torch.as_tensor(x), initial_actions_embeddings=torch.as_tensor(a),
                  temperature=temperature)
    opc = CFModel(30, 40, 6, user_transform=GlobalLinearCorrection(6), action_transform=GlobalLinearCorrection(6),
                  learn_logit_scale=True, **common)
    shared = SharedCorrectionModel(30, 40, 6, mode=mode, click_intercept=-1.5, **common)
    for m in (opc, shared):
        _set_params(m)
    return opc, shared, x, a


# ------------------------------------------------------------------ the model: identical parameters, identical policy
def test_identical_parameters_give_identical_logits_rankings_and_policies():
    opc, shared, _x, _a = _pair()
    users = torch.arange(30)
    assert [n for n, _ in opc.named_parameters()] == [n for n, _ in shared.named_parameters()]  # same order, no extra
    assert torch.equal(opc.policy_logits(users), shared.policy_logits(users))
    assert torch.equal(opc(users), shared(users))  # the softmax policy
    ox, oa = opc.get_params()
    sx, sa = shared.get_params()
    assert torch.equal(ox, sx) and torch.equal(oa, sa)  # the exported vectors: rankings and the evaluated policy
    assert torch.equal((ox @ oa.T).argmax(1), (sx @ sa.T).argmax(1))


def test_the_model_starts_at_the_logger_and_reads_its_logits_as_click_logits():
    x, a = _vectors()
    model = SharedCorrectionModel(30, 40, 6, initial_user_embeddings=torch.as_tensor(x),
                                  initial_actions_embeddings=torch.as_tensor(a), temperature=0.7, mode="click",
                                  logit_scale=0.25, click_intercept=-2.0)
    users = torch.arange(30)
    expected = 0.25 * torch.as_tensor(x) @ torch.as_tensor(a).T / 0.7 - 2.0
    torch.testing.assert_close(model(users).squeeze(-1), expected, rtol=1e-5, atol=1e-5)
    assert model.logit_scale == pytest.approx(0.25, rel=1e-6)
    opc, shared, *_ = _pair(mode="click")
    torch.testing.assert_close(shared(users).squeeze(-1), opc.policy_logits(users) - 1.5)


def test_the_opc_objective_has_identical_gradients_on_both_models():
    opc, shared, *_ = _pair()
    g = torch.Generator().manual_seed(3)
    users = torch.randint(0, 30, (64,), generator=g)
    actions = torch.randint(0, 40, (64,), generator=g)
    rewards = torch.bernoulli(torch.full((64,), 0.2), generator=g).double()
    pscore = (torch.rand(64, generator=g) * 0.2 + 0.01).double()
    scores = torch.rand(64, 40, generator=g)
    loss = DRPolicyLoss(use_log_trick=False, weights="harmonic:0.1")
    grads = []
    for m in (opc, shared):
        m.zero_grad()
        value = loss(pscore, scores, m(users).squeeze(-1), rewards, actions)
        value.backward()
        grads.append((value.detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    assert torch.equal(grads[0][0], grads[1][0])
    assert grads[0][1].keys() == grads[1][1].keys() and len(grads[0][1]) == 5
    for name in grads[0][1]:
        assert torch.equal(grads[0][1][name], grads[1][1][name]), name


# ------------------------------------------------------------------ the regularizer
def test_the_source_anchor_is_the_relative_displacement_of_the_vectors():
    _opc, shared, x, a = _pair()
    anchor = SourceAnchor(x, a)
    with torch.no_grad():
        u2 = shared.user_transform(torch.as_tensor(x)).double().numpy()
        a2 = shared.action_transform(torch.as_tensor(a)).double().numpy()
    x64, a64 = x.astype(np.float64), a.astype(np.float64)
    direct = (((u2 - x64) ** 2).sum(1).mean() / (x64 ** 2).sum(1).mean()
              + ((a2 - a64) ** 2).sum(1).mean() / (a64 ** 2).sum(1).mean())
    assert float(anchor(shared)) == pytest.approx(direct, rel=1e-4)
    fresh = SharedCorrectionModel(30, 40, 6, initial_user_embeddings=torch.as_tensor(x),
                                  initial_actions_embeddings=torch.as_tensor(a), temperature=0.7)
    assert float(anchor(fresh)) == 0.0  # the source itself
    # rescaling the source vectors with the correction expressed in the same units leaves R unchanged
    big = SharedCorrectionModel(30, 40, 6, initial_user_embeddings=torch.as_tensor(3 * x),
                                initial_actions_embeddings=torch.as_tensor(3 * a), temperature=0.7)
    with torch.no_grad():
        for t_big, t in ((big.user_transform, shared.user_transform), (big.action_transform, shared.action_transform)):
            t_big.delta.copy_(t.delta)
            t_big.bias.copy_(3 * t.bias)
    assert float(SourceAnchor(3 * x, 3 * a)(big)) == pytest.approx(float(anchor(shared)), rel=1e-4)


# ------------------------------------------------------------------ the objectives against direct calculations
@pytest.mark.parametrize("weighting,clip", [("none", None), ("uniform", None), ("uniform", 10.0)])
def test_the_likelihood_objectives_match_a_direct_calculation(weighting, clip):
    _opc, shared, x, a = _pair(mode="click")
    g = torch.Generator().manual_seed(5)
    users = torch.randint(0, 30, (50,), generator=g)
    actions = torch.randint(0, 40, (50,), generator=g)
    rewards = torch.bernoulli(torch.full((50,), 0.3), generator=g).double()
    pscore = (torch.rand(50, generator=g) ** 3 * 0.5 + 1e-4).double()
    anchor = SourceAnchor(x, a)
    lam = 0.01
    loss = LikelihoodLoss(40, weighting=weighting, clip=clip, lam=lam, penalty=lambda: anchor(shared))
    logits = shared(users).squeeze(-1)
    value = float(loss(pscore, None, logits, rewards, actions))
    z = logits.detach().double().numpy()[np.arange(50), actions.numpy()]
    r = rewards.numpy()
    nll = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) - r * z  # −r log σ(z) − (1 − r) log(1 − σ(z)), stable
    w = np.ones(50) if weighting == "none" else 1.0 / (40 * pscore.numpy())
    if clip is not None:
        w = np.minimum(w, clip)
    expected = float(np.mean(w * nll)) + lam * float(anchor(shared))
    assert value == pytest.approx(expected, rel=1e-5)
    assert weighted_nll(z, r, None if weighting == "none" else w) == pytest.approx(float(np.mean(w * nll)), rel=1e-9)
    assert LikelihoodLoss.needs_qhat is False


def test_the_anchored_policy_loss_adds_lambda_times_r():
    opc, shared, x, a = _pair()
    anchor = SourceAnchor(x, a)
    base = DRPolicyLoss(use_log_trick=False, weights="harmonic:0.1")
    g = torch.Generator().manual_seed(7)
    args = ((torch.rand(20, generator=g) * 0.2 + 0.01).double(), torch.rand(20, 40, generator=g),
            shared(torch.randint(0, 30, (20,), generator=g)).squeeze(-1),
            torch.bernoulli(torch.full((20,), 0.2), generator=g).double(), torch.randint(0, 40, (20,), generator=g))
    total = AnchoredPolicyLoss(base, 0.1, lambda: anchor(shared))(*args)
    assert float(total) == pytest.approx(float(base(*args)) + 0.1 * float(anchor(shared)), rel=1e-6)
    with pytest.raises(ValueError):
        AnchoredPolicyLoss(base, 0.0, lambda: anchor(shared))  # λ = 0 uses the base loss itself


# ------------------------------------------------------------------ weights and the click head
def test_uniform_reference_weights_and_their_diagnostics():
    p = np.array([0.5, 0.25, 0.01, 0.001])
    w = iw_weights(p, 100)
    np.testing.assert_allclose(w, 1 / (100 * p))
    np.testing.assert_allclose(iw_weights(p, 100, 10.0), np.minimum(1 / (100 * p), 10))
    torch.testing.assert_close(iw_weights(torch.as_tensor(p), 100, 10.0), torch.as_tensor(np.minimum(w, 10)))
    s = weight_stats(p, 100)
    assert s["w_mean"] == pytest.approx(w.mean()) and s["w_max"] == pytest.approx(10.0)
    assert s["w_ess"] == pytest.approx(w.sum() ** 2 / (w ** 2).sum()) and s["w_ess_share"] == pytest.approx(s["w_ess"] / 4)
    assert s["w_clip_share"] == 0.0  # 1/(100 * 0.001) = 10 is not above the clip
    s2 = weight_stats(np.array([0.5, 0.0001]), 100)
    assert s2["w_clip_share"] == 0.5 and s2["w_clip_mass_share"] == pytest.approx((100 - 10) / (100 + 0.02))


def test_the_click_head_fit_recovers_a_logistic_head():
    rng = np.random.default_rng(0)
    g0 = rng.normal(0, 3, 200_000)
    clicks = rng.random(len(g0)) < 1 / (1 + np.exp(-(0.4 * g0 - 2.0)))
    theta_s, c = fit_click_head(g0, clicks)
    assert np.exp(LOGIT_SCALE_SPEED * theta_s) == pytest.approx(0.4, rel=0.03) and c == pytest.approx(-2.0, abs=0.05)
    w = rng.exponential(1.0, len(g0))  # weights reweight the same model: the same parameters in expectation
    theta_w, c_w = fit_click_head(g0, clicks, w)
    assert np.exp(LOGIT_SCALE_SPEED * theta_w) == pytest.approx(0.4, rel=0.05)
    theta_neg, _ = fit_click_head(g0, rng.random(len(g0)) < 1 / (1 + np.exp(0.4 * g0)))
    assert np.exp(LOGIT_SCALE_SPEED * theta_neg) == pytest.approx(1e-3)  # a negative slope is floored


# ------------------------------------------------------------------ the arms through the trainer and the study
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from training.run_full_study import _dataset_paths
    from utils.simulation_utils import generate_dataset

    root = tmp_path_factory.mktemp("shared_toy")
    _toy_embeddings(root)
    up, ip, _, _ = _dataset_paths(root, "toy")
    return root, generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))


def _trainer(ds, tmp_path, label, **kw):
    from training.trainer_trials import LazyRegressionSplitCache, regression_trainer_trial

    logs = {"trials": tmp_path / f"{label}_trials_long.csv", "runs": tmp_path / f"{label}_runs_long.csv"}
    cache = LazyRegressionSplitCache(ds, [N], val_size=1000, val_min=1000, condition_seed=0, regression_size=2000)
    args = dict(train_sizes=[N], dataset=ds, batch_size=None, val_size=1000, val_min=1000, n_trials=4, slim=True,
                shared_regression_size=2000, search_use_log_trick=False, use_log_trick_fixed=False, log_paths=logs,
                split_cache=cache, policy_loss_types=("dr",), train_weights="harmonic:0.1", select_weights="clip:10",
                learn_logit_scale=True, sampler="random", seed=0, method_label=label)
    summary, _ = regression_trainer_trial(**{**args, **kw})
    return summary, pd.read_csv(logs["trials"])


def test_shared_opc_at_lambda_zero_is_the_opc_arm(toy, tmp_path):
    """λ = 0 only: the shared OPC arm draws OPC's configurations and seeds and trains exactly OPC's trials."""
    _, ds = toy
    opc_summary, opc_trials = _trainer(ds, tmp_path, "opc", seed_label="opc")
    sh_summary, sh_trials = _trainer(ds, tmp_path, "shared_opc", seed_label="opc", shared_objective="opc",
                                     anchor_lambdas=[0.0])
    cols = ["trial_number", "value", "r_hat", "actual_reward", "actual_reward_greedy", "param_lr", "param_num_epochs",
            "param_batch_size", "param_lr_decay", "logit_scale", "ess"]
    pd.testing.assert_frame_equal(opc_trials[cols], sh_trials[cols], check_exact=True)
    assert (sh_trials["param_anchor_lambda"] == 0.0).all() and (sh_trials["diag_anchor_R"] > 0).all()
    for col in ("policy_rewards", "policy_rewards_greedy", "selection_val_score"):
        assert opc_summary.loc[N, col] == sh_summary.loc[N, col]


def test_the_arms_are_paired_and_select_natively(toy, tmp_path):
    _, ds = toy
    out = {}
    for label, kw in (("shared_likelihood", {"shared_objective": "likelihood"}),
                      ("shared_iw_likelihood", {"shared_objective": "iw_likelihood"}),
                      ("shared_iw_likelihood_clip10", {"shared_objective": "iw_likelihood", "iw_clip": 10.0}),
                      ("shared_opc", {"shared_objective": "opc"})):
        out[label] = _trainer(ds, tmp_path, label, seed_label="shared", **kw)
    params = ["param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay", "param_anchor_lambda"]
    ref = out["shared_opc"][1][params]
    for label, (summary, trials) in out.items():
        pd.testing.assert_frame_equal(trials[params], ref)  # trial k: the same configuration in every arm
        assert set(trials["param_anchor_lambda"]) <= set(LAMBDA_GRID)
        row = summary.loc[N]
        native = {"shared_likelihood": "diag_val_nll", "shared_iw_likelihood": "diag_val_iw_nll",
                  "shared_iw_likelihood_clip10": "diag_val_iw_nll_clip"}.get(label)
        best = trials.loc[trials["is_best_in_run"].astype(bool)].iloc[0]
        if native is not None:
            assert int(best["trial_number"]) == int(trials.loc[trials[native].idxmin(), "trial_number"])
            assert row["selection_val_score"] == pytest.approx(-trials[native].min())
            assert row["native_selection"] == native.replace("diag_", "")
        else:
            assert int(best["trial_number"]) == int(trials.loc[trials["value"].idxmax(), "trial_number"])
        assert int(row["selected_trial"]) == int(best["trial_number"])
        assert row["policy_rewards_greedy"] == pytest.approx(best["actual_reward_greedy"], rel=1e-12)  # via CSV
        assert trials["diag_dr_greedy_low"].notna().all() and trials["diag_changed_share"].between(0, 1).all()
    rows = {label: s.loc[N] for label, (s, _) in out.items()}
    for key in ("train_rows_sha1", "val_rows_sha1", "train_click_sum", "val_click_sum", "train_pscore_sum",
                "train_w_ess"):
        assert len({r[key] for r in rows.values()}) == 1, key  # the same rows and propensities for every arm
    # the likelihood head starts at the arm's own objective's minimizer; OPC at the logger (s = 1)
    assert rows["shared_likelihood"]["head_logit_scale"] != rows["shared_iw_likelihood"]["head_logit_scale"]
    assert np.isnan(rows["shared_opc"].get("head_logit_scale", np.nan))


def test_the_shared_arms_need_their_configuration(toy, tmp_path):
    _, ds = toy
    with pytest.raises(ValueError, match="shared_objective"):
        _trainer(ds, tmp_path, "x", shared_objective="iw")
    with pytest.raises(ValueError, match="iw_clip"):
        _trainer(ds, tmp_path, "x", shared_objective="likelihood", iw_clip=10.0)
    with pytest.raises(ValueError, match="need a shared_objective"):
        _trainer(ds, tmp_path, "x", anchor_lambdas=[0.0])
    with pytest.raises(ValueError, match="learned logit scale"):
        _trainer(ds, tmp_path, "x", shared_objective="opc", learn_logit_scale=False)


def test_the_likelihood_arm_trains_on_its_rows_only(toy, tmp_path):
    """The likelihood arms use no reward model in training (needs_qhat False) and their data identity record is the
    split's training rows."""
    from training.trainer_trials import LazyRegressionSplitCache

    _, ds = toy
    summary, _ = _trainer(ds, tmp_path, "shared_likelihood", seed_label="shared", shared_objective="likelihood",
                          n_trials=2)
    train = LazyRegressionSplitCache(ds, [N], val_size=1000, val_min=1000, condition_seed=0,
                                     regression_size=2000)[(N, 0)]["train_data"]
    row = summary.loc[N]
    assert row["train_rows"] == N and row["train_click_sum"] == float(np.sum(train["r"]))
    assert row["train_pscore_sum"] == pytest.approx(float(np.sum(train["pscore"])))


def test_the_runner_refuses_to_post_temper_the_shared_arms(toy, tmp_path):
    from training.run_full_study import _run_condition

    root, _ = toy
    with pytest.raises(ValueError, match="not post-tempered"):
        _run_condition(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[N], n_trials=2,
                       batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                       policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=tmp_path, slim=True,
                       shared_regression_size=2000, methods=("shared_opc",), post_temper=True)


def test_an_arm_with_its_own_range_keeps_the_pairing_by_quantile(toy, tmp_path):
    """The edge rule may give one arm a wider range (docs/shared_objective_study.md §8): its trial k draws the same
    uniform number as the other arms', mapped onto its own range."""
    from training.run_full_study import parse_shared_arm_options, shared_arm_settings

    opts = parse_shared_arm_options(["shared_opc:lr=1e-4,6.32e-3", "shared_opc:num_epochs=5,60"],
                                    ["shared_likelihood=0,0.01,10"])
    assert opts == {"spaces": {"shared_opc": {"lr": (1e-4, 6.32e-3), "num_epochs": (5, 60)}},
                    "arm_lambdas": {"shared_likelihood": [0.0, 0.01, 10.0]}}
    common = {"lr": (1e-4, 2e-3), "num_epochs": (5, 30)}
    space, lams = shared_arm_settings("shared_opc", {"lambdas": [0.0, 1.0], **opts}, common)
    assert space == {"lr": (1e-4, 6.32e-3), "num_epochs": (5, 60)} and lams == [0.0, 1.0]
    assert shared_arm_settings("shared_likelihood", {"lambdas": [0.0, 1.0], **opts}, common) == (common, [0.0, 0.01, 10.0])
    with pytest.raises(ValueError):
        parse_shared_arm_options(["shared_opc:momentum=0,1"])
    _, ds = toy
    _, base = _trainer(ds, tmp_path, "shared_likelihood", seed_label="shared", shared_objective="likelihood",
                       search_space=common)
    _, wide = _trainer(ds, tmp_path, "shared_opc", seed_label="shared", shared_objective="opc", search_space=space)
    u = lambda lr, lo, hi: (np.log(lr) - np.log(lo)) / (np.log(hi) - np.log(lo))
    np.testing.assert_allclose(u(base["param_lr"], 1e-4, 2e-3), u(wide["param_lr"], 1e-4, 6.32e-3), rtol=1e-9)
    assert (base["param_anchor_lambda"] == wide["param_anchor_lambda"]).all()


def test_a_supplementary_round_draws_new_paired_trials(toy, tmp_path, monkeypatch):
    """--shared-seed-tag (docs/shared_objective_study.md §8): another tag gives new configurations, still the same in
    every shared arm, and its own configuration key; the default tag leaves the options and keys as they were."""
    import argparse

    import training.run_full_study as rfs
    from training.run_state import arm_config, config_key

    parser = argparse.ArgumentParser()
    rfs.add_shared_arguments(parser)
    default = rfs.shared_options_from_args(parser.parse_args([]))
    supp = rfs.shared_options_from_args(parser.parse_args(["--shared-seed-tag", "shared_supplement"]))
    assert default == {"lambdas": list(LAMBDA_GRID)} and supp == {**default, "seed_tag": "shared_supplement"}
    key = lambda opts: config_key(arm_config("shared_opc", {"n_trials": 20, "shared_options": opts}))
    assert key(default) == key({"lambdas": list(LAMBDA_GRID)}) and key(supp) != key(default)

    root, ds = toy
    seen = []

    def capture(**kw):
        seen.append((kw["method_label"], kw["seed_label"]))
        raise KeyboardInterrupt  # stop after the first arm: only the label is under test

    monkeypatch.setattr(rfs, "regression_trainer_trial", capture)
    for opts in (default, supp):
        with pytest.raises(KeyboardInterrupt):
            rfs._run_condition(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[N],
                               n_trials=2, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                               policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=tmp_path, slim=True,
                               shared_regression_size=2000, methods=("shared_opc",), shared_options=opts)
    assert seen == [("shared_opc", "shared"), ("shared_opc", "shared_supplement")]

    params = ["param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay", "param_anchor_lambda"]
    monkeypatch.undo()
    for sub in ("first", "supp"):
        (tmp_path / sub).mkdir()
    _, first = _trainer(ds, tmp_path / "first", "shared_opc", seed_label="shared", shared_objective="opc")
    _, again = _trainer(ds, tmp_path / "supp", "shared_likelihood", seed_label="shared_supplement",
                        shared_objective="likelihood")
    _, other = _trainer(ds, tmp_path / "supp", "shared_opc", seed_label="shared_supplement", shared_objective="opc")
    pd.testing.assert_frame_equal(again[params], other[params])  # paired within the round
    assert not np.allclose(first["param_lr"], other["param_lr"])  # new configurations
