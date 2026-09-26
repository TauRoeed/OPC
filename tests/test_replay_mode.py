"""Replay mode (--sampler random): paired comparisons of training objectives.

With Optuna's TPE, the first 10 trials of a size are seeded random draws and the rest adapt to the
trial values, and each size after the first starts from the previous size's best: runs that differ in
the objective share only part of their configurations. Seeded random search, with no warm start,
proposes the same configurations in every run with the same seeds and search space, and each trial's
seed depends only on (seed, arm, train size, trial number): trial k is then the same configuration with
the same initialization and batch order under every objective."""

import sys

import numpy as np
import optuna
import pandas as pd
import pytest

from utils.seeding import OPTUNA_SAMPLERS, optuna_sampler

PARAMS = ["param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay"]


def _search(kind, sign, n=15, seed=3):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna_sampler(seed, "opc", 5000, kind=kind))

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        batch = trial.suggest_categorical("batch_size", [512, 1024, 2048])
        epochs = trial.suggest_int("num_epochs", 5, 25)
        return sign * (np.log(lr) + batch / 1000 + epochs / 10)

    study.optimize(objective, n_trials=n)
    return [tuple(t.params[k] for k in ("lr", "batch_size", "num_epochs")) for t in study.trials]


def test_random_search_does_not_depend_on_the_objective():
    assert OPTUNA_SAMPLERS == ("tpe", "random")
    assert isinstance(optuna_sampler(0, "opc", 5000), optuna.samplers.TPESampler)  # the default
    assert isinstance(optuna_sampler(0, "opc", 5000, kind="random"), optuna.samplers.RandomSampler)
    with pytest.raises(ValueError, match="sampler"):
        optuna_sampler(0, "opc", kind="grid")
    assert _search("random", +1) == _search("random", -1)  # opposite objectives, the same configurations
    tpe_up, tpe_down = _search("tpe", +1), _search("tpe", -1)
    assert tpe_up[:10] == tpe_down[:10] and tpe_up[10:] != tpe_down[10:]  # TPE: only its start-up is shared
    assert _search("random", +1) != _search("random", +1, seed=4)  # another seed, other configurations


@pytest.fixture(scope="module")
def toy_runs(tmp_path_factory):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _finalize_summary_df, _run_condition

    tmp = tmp_path_factory.mktemp("replay")
    _toy_embeddings(tmp)
    kw = dict(dataset_name="toy", emb_dir=tmp, bias="medium", ctr=0.05, seed=0, train_sizes=[1000, 1500], n_trials=3,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",))
    runs = {}
    for name, extra in (("dr_random", dict(policy_loss_types=("dr",), sampler="random")),
                        ("legacy_random", dict(policy_loss_types=("sndr",), sampler="random")),
                        ("global_random", dict(policy_loss_types=("sndr",), sn_scope="global", sampler="random")),
                        ("dr_tpe", dict(policy_loss_types=("dr",)))):
        run_dir = tmp / name
        run_dir.mkdir()
        opc, nop, _, _, meta = _run_condition(**kw, run_dir=run_dir, **extra)
        trials = pd.read_csv(run_dir / "trials_long.csv")
        runs[name] = (trials[trials["method"] == "opc"], meta, _finalize_summary_df(opc, nop, meta))
    return runs


def test_random_sampler_replays_the_same_trials_under_every_objective(toy_runs):
    ref = toy_runs["dr_random"][0].set_index(["train_size", "trial_number"]).sort_index()
    assert len(ref) == 6  # 2 sizes x 3 trials
    for name in ("legacy_random", "global_random"):
        other = toy_runs[name][0].set_index(["train_size", "trial_number"]).sort_index()
        pd.testing.assert_frame_equal(other[PARAMS], ref[PARAMS])  # every trial, both sizes
        np.testing.assert_array_equal(other["initial_reward"], ref["initial_reward"])  # the same world and logger
        assert not np.allclose(other["actual_reward"], ref["actual_reward"])  # a different objective trained


def test_random_sampler_has_no_warm_start(toy_runs):
    """TPE starts each later size from the previous size's best; random search does not (that would make
    the second size's first trial depend on the objective)."""
    def first_trial_is_previous_best(trials):
        small, large = trials[trials["train_size"] == 1000], trials[trials["train_size"] == 1500]
        best = small.loc[small["is_best_in_run"].astype(bool), PARAMS].iloc[0].to_numpy(dtype=float)
        first = large.loc[large["trial_number"] == 0, PARAMS].iloc[0].to_numpy(dtype=float)
        return bool(np.array_equal(best, first))

    assert first_trial_is_previous_best(toy_runs["dr_tpe"][0])
    for name in ("dr_random", "legacy_random", "global_random"):
        assert not first_trial_is_previous_best(toy_runs[name][0])


def test_sampler_and_stage_are_recorded(toy_runs, tmp_path):
    from training.run_full_study import RUN_STAGES, _run_condition

    assert RUN_STAGES == ("development", "confirmatory")
    for name, (_, meta, summary) in toy_runs.items():
        want = "tpe" if name.endswith("tpe") else "random"
        assert meta["sampler"] == want and (summary["sampler"] == want).all()
        assert meta["stage"] == "development" and (summary["stage"] == "development").all()
    with pytest.raises(ValueError, match="stage"):
        _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000],
                       n_trials=1, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                       policy_reward_mode="exact", policy_reward_mc_sim=8, run_dir=tmp_path, stage="final")


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_defaults(module, monkeypatch, capsys):
    import argparse

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    text = " ".join(capsys.readouterr().out.split())
    assert "--sampler" in text and "--stage" in text and "paired" in text
    real_parse, seen = argparse.ArgumentParser.parse_args, {}

    class Parsed(Exception):
        pass

    def parse(self, args=None, namespace=None):
        seen["default"] = real_parse(self, [], namespace)
        seen["set"] = real_parse(self, ["--sampler", "random", "--stage", "confirmatory"], namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        main()
    assert (seen["default"].sampler, seen["default"].stage) == ("tpe", "development")
    assert (seen["set"].sampler, seen["set"].stage) == ("random", "confirmatory")


def test_parallel_runner_forwards_sampler_and_stage(monkeypatch, tmp_path):
    """The parallel runner's configs carry both settings to every worker's _run_condition call."""
    import training.run_full_study_parallel as par

    captured = {}

    def fake_dispatch(run_configs, **kwargs):
        captured["configs"] = list(run_configs)
        return []

    monkeypatch.setattr(par, "_run_with_memory_cap", fake_dispatch)
    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", "--datasets", "ml", "--seeds", "7", "--bias-configs",
                                      "medium", "--train-sizes", "5000", "--sampler", "random", "--stage",
                                      "confirmatory", "--out-dir", str(tmp_path), "--run-tag", "t", "--no-skip-completed"])
    par.main()
    (config,) = captured["configs"]
    assert (config["sampler"], config["stage"]) == ("random", "confirmatory")

    class Called(Exception):
        pass

    def fake_condition(**kwargs):
        captured["kwargs"] = kwargs
        raise Called

    monkeypatch.setattr(par, "_run_condition", fake_condition)
    with pytest.raises(Called):
        par._execute_run(config)
    assert (captured["kwargs"]["sampler"], captured["kwargs"]["stage"]) == ("random", "confirmatory")
