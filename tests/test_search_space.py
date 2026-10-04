"""The policy search space (trainer_trials.DEFAULT_SEARCH_SPACE; --lr-range, --epochs-range, --lr-decay-range,
--weight-decay-range). The defaults reproduce the earlier runs; under the random sampler a changed range changes
only that parameter's draws, and the weight decay comes from its own stream, so trials stay paired."""
import numpy as np
import pandas as pd
import pytest

from training.trainer_trials import DEFAULT_SEARCH_SPACE, resolve_search_space

PARAMS = ["param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay"]


def test_defaults_and_validation():
    assert DEFAULT_SEARCH_SPACE == {"lr": (1e-4, 1e-3), "num_epochs": (5, 25), "lr_decay": (0.8, 1.0),
                                    "weight_decay": None}
    assert resolve_search_space(None) == DEFAULT_SEARCH_SPACE
    assert resolve_search_space({"lr": None}) == DEFAULT_SEARCH_SPACE  # None keeps a default range
    space = resolve_search_space({"lr": (1e-4, 3e-2), "num_epochs": (5.0, 60.0), "weight_decay": [1e-3, 1.0]})
    assert space["lr"] == (1e-4, 3e-2) and space["num_epochs"] == (5, 60) and space["weight_decay"] == (1e-3, 1.0)
    for bad in ({"lr": (0.0, 1e-3)}, {"lr": (1e-2, 1e-3)}, {"num_epochs": (0, 5)}, {"lr_decay": (0.5, 1.2)},
                {"weight_decay": (0.0, 1.0)}, {"momentum": (0.1, 0.9)}):
        with pytest.raises(ValueError):
            resolve_search_space(bad)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _run_condition

    root = tmp_path_factory.mktemp("space")
    _toy_embeddings(root)
    kw = dict(dataset_name="toy", emb_dir=root, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=4,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc", "dm"), sampler="random")
    out = {}
    for name, space in (("none", None), ("explicit", dict(DEFAULT_SEARCH_SPACE)), ("lr", {"lr": (1e-3, 1e-2)}),
                        ("wd", {"weight_decay": (1e-3, 1.0)})):
        run_dir = root / name
        run_dir.mkdir()
        *_, meta, _extra = _run_condition(**kw, run_dir=run_dir, search_space=space, return_extra=True)
        out[name] = (pd.read_csv(run_dir / "trials_long.csv"), meta)
    return out


def test_default_ranges_reproduce_the_run_without_a_search_space(runs):
    a, b = runs["none"][0], runs["explicit"][0]
    pd.testing.assert_frame_equal(a[PARAMS + ["actual_reward"]], b[PARAMS + ["actual_reward"]])
    assert runs["none"][1]["search_space"] == {"lr": [1e-4, 1e-3], "num_epochs": [5, 25], "lr_decay": [0.8, 1.0],
                                              "weight_decay": None}


def test_a_new_lr_range_changes_only_the_lr_draws(runs):
    a, b = runs["none"][0], runs["lr"][0]
    pd.testing.assert_frame_equal(a[PARAMS[1:]], b[PARAMS[1:]])
    assert (b["param_lr"] >= 1e-3).all() and (b["param_lr"] <= 1e-2).all() and not np.allclose(a["param_lr"], b["param_lr"])
    # log-uniform draws from the same uniforms: the same positions within the two ranges
    pa = np.log(a["param_lr"] / 1e-4) / np.log(10.0)
    pb = np.log(b["param_lr"] / 1e-3) / np.log(10.0)
    np.testing.assert_allclose(pa, pb, rtol=1e-9)
    assert runs["lr"][1]["search_space"]["lr"] == [1e-3, 1e-2]


def test_weight_decay_is_drawn_per_trial_and_paired_across_arms(runs):
    a, b = runs["none"][0], runs["wd"][0]
    pd.testing.assert_frame_equal(a[PARAMS], b[PARAMS])  # the Optuna draws are untouched
    assert "param_weight_decay" not in a.columns
    wd = b.pivot(index="trial_number", columns="method", values="param_weight_decay")
    assert ((wd >= 1e-3) & (wd <= 1.0)).all().all() and wd["opc"].nunique() == len(wd)
    pd.testing.assert_series_equal(wd["opc"], wd["dm"], check_names=False)  # paired: the same decay per trial
    assert not np.allclose(a["actual_reward"], b["actual_reward"])  # and it changes the trained policies


def test_weight_decay_needs_the_random_sampler(tmp_path):
    from training.trainer_trials import regression_trainer_trial

    with pytest.raises(ValueError, match="random"):
        regression_trainer_trial([1000], {}, None, sampler="tpe", search_space={"weight_decay": (1e-3, 1.0)})


def test_training_uses_adamw_only_with_a_decay(monkeypatch):
    import torch

    from training import training_utils

    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training_utils, "run_train_loop", lambda model, loader, opt, *a, **k: seen.append(type(opt).__name__))
    model = torch.nn.Linear(2, 2)
    for wd in (0.0, 0.1):
        opt = training_utils.train(model, [], None, criterion=None, num_epochs=1, lr=1e-3, device="cpu", weight_decay=wd)
        assert type(opt).__name__ == ("AdamW" if wd else "Adam")
        if wd:
            assert opt.param_groups[0]["weight_decay"] == 0.1
    assert seen == ["Adam", "AdamW"]


def test_a_diverging_trial_keeps_its_last_finite_parameters(tmp_path, monkeypatch):
    """A non-finite gradient (clip_grad_norm_ refuses it before the step) ends that trial's training; the trial is
    evaluated at its last finite parameters and flagged, and the condition completes. Other errors propagate."""
    from test_reproducibility import _toy_embeddings

    import training.trainer_trials as tt
    from training.run_full_study import _run_condition

    _toy_embeddings(tmp_path)
    real, calls = tt.train, []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("The total norm of order 2.0 for gradients from `parameters` is non-finite, so it cannot be clipped.")
        return real(*args, **kwargs)

    monkeypatch.setattr(tt, "train", flaky)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",), sampler="random")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_condition(**kw, run_dir=run_dir)
    t = pd.read_csv(run_dir / "trials_long.csv")
    assert t["diverged"].tolist() == [True, False]
    assert np.isfinite(t["actual_reward"]).all()
    assert t.loc[0, "actual_reward"] == pytest.approx(t.loc[0, "initial_reward"], abs=1e-9)  # never stepped: the logger

    def broken(*args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(tt, "train", broken)
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="illegal memory access"):
        _run_condition(**kw, run_dir=run_dir)
