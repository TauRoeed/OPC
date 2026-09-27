"""Working development defaults since f5cade9 (2026-09-27) (docs/decision_record_opc_objective_weighting.md): the full-study
runners train OPC as DR, differentiated directly, with harmonic:0.1 weights; selection keeps clip:10. The previous
defaults (legacy SNDR, log trick, shrink:100) stay reproducible with explicit flags, and the H1 runner keeps its own
settings."""

import inspect
import sys

import pytest


def _parse_defaults(module, monkeypatch, argv=()):
    import argparse

    real_parse, seen = argparse.ArgumentParser.parse_args, {}

    class Parsed(Exception):
        pass

    def parse(self, args=None, namespace=None):
        seen["args"] = real_parse(self, list(argv), namespace)
        raise Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(Parsed):
        __import__(f"training.{module}", fromlist=["main"]).main()
    monkeypatch.undo()
    return seen["args"]


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_cli_working_defaults(module, monkeypatch):
    from training.run_full_study import STUDY_OPC_GRADIENT, STUDY_POLICY_LOSSES, STUDY_TRAIN_WEIGHTS

    assert (STUDY_POLICY_LOSSES, STUDY_OPC_GRADIENT, STUDY_TRAIN_WEIGHTS) == (("dr",), "direct", "harmonic:0.1")
    a = _parse_defaults(module, monkeypatch)
    assert list(a.policy_losses) == ["dr"] and a.opc_gradient == "direct" and a.train_weights == "harmonic:0.1"
    assert a.select_weights == "clip:10" and a.sn_scope == "batch" and a.sampler == "tpe" and a.stage == "development"
    old = _parse_defaults(module, monkeypatch, ["--policy-losses", "sndr", "--sn-scope", "batch", "--opc-gradient",
                                                "log-trick", "--train-weights", "shrink:100"])
    assert (list(old.policy_losses), old.opc_gradient, old.train_weights) == (["sndr"], "log-trick", "shrink:100")


def test_run_condition_defaults_and_record(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.run_full_study import _run_condition

    sig = inspect.signature(_run_condition).parameters
    assert sig["policy_loss_types"].default == ("dr",) and sig["opc_gradient"].default == "direct"
    assert sig["train_weights"].default is None  # None -> STUDY_TRAIN_WEIGHTS
    _toy_embeddings(tmp_path)
    kw = dict(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000], n_trials=1,
              batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
              policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, methods=("opc",))
    run_dir = tmp_path / "defaults"
    run_dir.mkdir()
    *_, meta = _run_condition(**kw, run_dir=run_dir)
    assert meta["policy_loss_types"] == ["dr"] and meta["opc_gradient"] == "direct" and not meta["opc_use_log_trick_fixed"]
    assert (meta["train_weights"], meta["train_weight_mode"], meta["train_weight_param"]) == ("harmonic:0.1", "harmonic", 0.1)
    assert (meta["select_weights"], meta["select_weight_mode"], meta["select_weight_param"]) == ("clip:10", "clip", 10.0)
    # the previous defaults, with explicit flags
    run_dir = tmp_path / "previous"
    run_dir.mkdir()
    *_, meta = _run_condition(**kw, run_dir=run_dir, policy_loss_types=("sndr",), sn_scope="batch",
                              opc_gradient="log-trick", train_weights="shrink:100")
    assert meta["policy_loss_types"] == ["sndr"] and meta["opc_gradient"] == "log-trick" and meta["opc_use_log_trick_fixed"]
    assert (meta["train_weights"], meta["sn_scope"]) == ("shrink:100", "batch")


def test_h1_runner_keeps_its_own_settings(monkeypatch, tmp_path):
    """H1 (the bounded-error reward-model grid) is not switched to the working defaults: legacy SNDR, the log
    trick and shrink:100, as before."""
    import training.run_h1_study as h1

    a = _parse_defaults("run_h1_study", monkeypatch)
    assert list(a.policy_losses) == ["sndr"] and a.train_weights == "shrink:100"
    captured = {}

    class Called(Exception):
        pass

    def fake(**kwargs):
        captured.update(kwargs)
        raise Called

    monkeypatch.setattr(h1, "_run_condition", fake)
    config = dict(run_dir=str(tmp_path / "cell"), dataset_name="toy", seed=0, target_rand_ctr=0.05, q_error=0.0,
                  logging_uniform_mix=0.0, emb_dir=str(tmp_path), bias="medium", train_sizes=[1000], n_trials=1,
                  batch_size=512, val_size=1000, policy_loss_types=list(a.policy_losses), train_weights=a.train_weights,
                  select_weights=a.select_weights)
    with pytest.raises(Called):
        h1._execute_h1_cell(config)
    assert captured["opc_gradient"] == "log-trick" and captured["policy_loss_types"] == ("sndr",)
    assert captured["train_weights"] == "shrink:100"
