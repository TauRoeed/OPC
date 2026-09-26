"""--crossfit-folds K (with --reward-data train): users are split into K folds, fold k's reward model
is fit on the other folds' training rows, and the training losses take each user's q_hat from the
model that never saw that user's rows (CrossFitScoresLookup); validation and selection keep the
full model."""

import numpy as np
import pandas as pd
import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _dataset_paths
    from utils.simulation_utils import generate_dataset

    root = tmp_path_factory.mktemp("toy_cf")
    _toy_embeddings(root)
    up, ip, _, _ = _dataset_paths(root, "toy")
    return root, generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))


def _constant_bundle(ds, c):
    from training.trainer_trials import fit_shared_regression_bundle

    return fit_shared_regression_bundle(ds, {}, reward_model="oracle", q_error=1.0, q_bad_value=c, materialize_qhat="never")


@pytest.mark.parametrize("device", DEVICES)
def test_lookup_routes_each_user_to_its_fold_model(toy, device):
    from training.trainer_trials import CrossFitScoresLookup, _scores_lookup_from_bundle

    _, ds = toy
    consts = (0.1, 0.2, 0.3)
    folds = [_scores_lookup_from_bundle(_constant_bundle(ds, c), device) for c in consts]
    full = _scores_lookup_from_bundle(_constant_bundle(ds, 0.9), device)
    user_fold = np.random.default_rng(3).integers(0, 3, ds["n_users"])
    cf = CrossFitScoresLookup(folds, user_fold, full)
    users = torch.as_tensor(np.random.default_rng(4).integers(0, ds["n_users"], 500), device=device)
    rows = cf[users]
    assert rows.shape == (500, ds["n_actions"]) and rows.device.type == torch.device(device).type
    want = torch.as_tensor(np.asarray(consts)[user_fold[users.cpu().numpy()]], device=rows.device, dtype=rows.dtype)
    torch.testing.assert_close(rows, want[:, None].expand_as(rows))
    # every other use is the full model's: validation scoring, selection, the model itself
    assert cf.regression_model is full.regression_model and cf.user_context is full.user_context
    assert cf.qhat_block_fn(device) is full.qhat_block_fn(device)


def test_fold_models_never_see_their_users(toy, monkeypatch):
    import training.run_full_study as rfs
    from training.trainer_trials import LOGGED_RUN_IDX, LazyRegressionSplitCache

    _, ds = toy
    cache = LazyRegressionSplitCache(ds, [3000], val_size=1000, val_min=1000, condition_seed=0, regression_size=2000)
    train = cache[(3000, LOGGED_RUN_IDX)]["train_data"]
    seen = []
    real = rfs.fit_shared_regression_bundle
    monkeypatch.setattr(rfs, "fit_shared_regression_bundle",
                        lambda d, data, **kw: seen.append(np.asarray(data["x_idx"])) or real(d, data, **kw))
    user_fold = np.random.default_rng(0).integers(0, 4, ds["n_users"])
    bundles = rfs._crossfit_bundles(ds, train, user_fold, 4, reward_model="regression")
    assert len(bundles) == 4 and len(seen) == 4
    row_fold = user_fold[np.asarray(train["x_idx"])]
    for k, users in enumerate(seen):
        assert not np.isin(users, np.flatnonzero(user_fold == k)).any()  # none of fold k's users
        assert len(users) == int((row_fold != k).sum())  # every other training row
    sub = rfs._subset_rows(train, row_fold == 0)
    assert sub["num_data"] == int((row_fold == 0).sum()) and len(sub["a"]) == sub["num_data"]
    with pytest.raises(ValueError, match="rows left"):
        rfs._crossfit_bundles(ds, train, np.zeros(ds["n_users"], dtype=int), 2, reward_model="regression")


def test_study_trains_on_the_cross_fitted_lookup(toy, tmp_path, monkeypatch):
    import training.trainer_trials as tt
    from training.run_full_study import _condition_run_key, _finalize_summary_df, _run_condition

    root, _ = toy
    seen = []
    real_train = tt.train
    monkeypatch.setattr(tt, "train", lambda model, loader, scores, **kw: seen.append(type(scores).__name__) or
                        real_train(model, loader, scores, **kw))
    kw = dict(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[1000, 3000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000)
    out = {}
    for folds in (0, 3):
        run_dir = tmp_path / f"cf{folds}"
        run_dir.mkdir()
        seen.clear()
        opc, nop, _, _, meta, extra = _run_condition(**kw, run_dir=run_dir, reward_data="train", crossfit_folds=folds,
                                                     methods=("opc", "dm"), return_extra=True)
        assert meta["crossfit_folds"] == folds
        assert set(seen) == {"CrossFitScoresLookup" if folds else "RegressionScoresLookup"}
        summary = _finalize_summary_df(opc, nop, meta, extra=extra)
        assert (summary["crossfit_folds"] == folds).all()
        out[folds] = pd.read_csv(run_dir / "trials_long.csv")
    # same splits and trials searched, different q_hat in training: different policies
    assert out[0]["initial_reward"].iloc[0] == out[3]["initial_reward"].iloc[0]
    assert not np.allclose(out[0]["actual_reward"], out[3]["actual_reward"])
    assert _condition_run_key("ml", "medium", 0.05, 1, {}, reward_data="train", crossfit_folds=5).endswith("__qhat=train__cf=5")
    with pytest.raises(ValueError, match="crossfit_folds"):
        _run_condition(**kw, run_dir=tmp_path, reward_data="external", crossfit_folds=3)
    with pytest.raises(ValueError, match="crossfit_folds"):
        _run_condition(**kw, run_dir=tmp_path, reward_data="train", crossfit_folds=1)


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_has_crossfit_folds(module, monkeypatch, capsys):
    import sys

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    assert "--crossfit-folds" in capsys.readouterr().out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("features", ["interaction", "concat"])
def test_fast_forms_equal_the_routing_reference(toy, device, features):
    """The dense matrix and the single linear form give every user its fold model's q_hat, as the
    per-batch routing does (up to float32 rounding)."""
    import training.run_full_study as rfs
    from training.trainer_trials import (
        LOGGED_RUN_IDX,
        CrossFitScoresLookup,
        LazyRegressionSplitCache,
        _scores_lookup_from_bundle,
        fit_shared_regression_bundle,
    )

    _, ds = toy
    cache = LazyRegressionSplitCache(ds, [3000], val_size=1000, val_min=1000, condition_seed=0, regression_size=2000)
    train = cache[(3000, LOGGED_RUN_IDX)]["train_data"]
    user_fold = np.random.default_rng(1).integers(0, 3, ds["n_users"])
    folds = rfs._crossfit_bundles(ds, train, user_fold, 3, reward_model="regression", reward_features=features)
    fold_lookups = [_scores_lookup_from_bundle(b, device) for b in folds]
    users = torch.as_tensor(np.random.default_rng(2).integers(0, ds["n_users"], 700), device=device)
    lazy_full = _scores_lookup_from_bundle(fit_shared_regression_bundle(ds, train, reward_model="regression",
                                                                        reward_features=features, materialize_qhat="never"), device)
    dense_full = _scores_lookup_from_bundle(fit_shared_regression_bundle(ds, train, reward_model="regression",
                                                                         reward_features=features, materialize_qhat="always"), device)
    ref = CrossFitScoresLookup(fold_lookups, user_fold, lazy_full, mode="route")
    assert ref.mode == "route"
    want = ref[users]
    for full, mode in ((dense_full, "dense"), (lazy_full, "linear")):
        cf = CrossFitScoresLookup(fold_lookups, user_fold, full)
        assert cf.mode == mode
        torch.testing.assert_close(cf[users].to(want.device), want, rtol=2e-5, atol=2e-7)
    # a user's row really is its fold model's, not the full model's
    u0 = int(users[0])
    np.testing.assert_allclose(want[0].cpu().numpy(), folds[user_fold[u0]]["regression_model"].predict_user_action_block(
        np.asarray(lazy_full.user_context[[u0]], dtype=np.float32), 0, ds["n_actions"])[0, :, 0], rtol=2e-5, atol=2e-7)
