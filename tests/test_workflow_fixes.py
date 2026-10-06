"""Regression tests for the review of the live workflows (docs/handoff_20261006.md §5): the world characterization,
the pick diagnostics' logger, dense-policy values, the slim run log's training rows, the OOM detector, the runners'
manifests, flags and help, and the CausE trial diagnostics' autograd."""
from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from test_reproducibility import _toy_embeddings


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy_fixes")
    _toy_embeddings(root)
    return root


def _world_options(*argv) -> dict:
    from utils.representation_bias import add_world_arguments, world_options_from_args

    parser = argparse.ArgumentParser()
    add_world_arguments(parser)
    return world_options_from_args(parser.parse_args(list(argv)))


# ------------------------------------------------------------------ the world characterization (finding 1)
def test_characterize_world_runs_and_builds_the_study_world(toy_root, tmp_path, monkeypatch):
    """The command crashed on every run since the logger's greedy share joined the world options (2026-09-26), and
    ignored the share; it now builds each condition's world exactly as the study does."""
    from training import characterize_world as cw
    from training.run_full_study import build_condition_world
    from utils.seeding import seed_everything

    out = tmp_path / "world"
    monkeypatch.setattr(sys, "argv", ["characterize_world", "--datasets", "toy", "--emb-dir", str(toy_root),
                                      "--out-dir", str(out), "--bias-configs", "none", "medium"])
    cw.main()
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["bias"]) == ["none", "medium"]
    assert "toy__seed=0" in json.loads((out / "calibration.json").read_text())
    temperatures = {}
    for share in ("0.8", "0.6"):
        options = _world_options("--logger-greedy-share", share)
        rows, calibration = cw.characterize("toy", toy_root, 0, 0.05, options, ["medium"])
        seed_everything(0)
        world = build_condition_world("toy", toy_root, "medium", 0.05, 0, world_options=options)[0]["world"]
        assert calibration["logging_temperature"] == world["logging_temperature"]
        assert rows[0]["logging_ctr"] == world["logging_ctr"] and rows[0]["signal_kept"] == world["signal_kept"]
        assert world["logger_greedy_share"] == pytest.approx(float(share))
        temperatures[share] = world["logging_temperature"]
    assert temperatures["0.8"] != temperatures["0.6"]  # the share reaches the world


# ------------------------------------------------------------------ the pick diagnostics' logger (finding 5)
def test_pick_diagnostics_read_the_runs_logger_mix(tmp_path):
    from training.policy_diagnostics import _world_settings

    run = tmp_path / "run_mixed"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"world_options": {"pop_strength": 0.0},
                                                       "logging_uniform_mix": 0.2}))
    assert _world_settings(run) == ({"pop_strength": 0.0}, 0.2)
    oracles = tmp_path / "oracles"
    (oracles / "policies").mkdir(parents=True)
    (oracles / "class_oracle_settings.json").write_text(json.dumps({"world_options": {"pop_strength": 0.0}}))
    assert _world_settings(oracles / "policies") == ({"pop_strength": 0.0}, 0.0)  # class oracles: never mixed
    running = tmp_path / "run_running" / "dataset=toy__bias=medium__ctr=0.05__seed=0"
    running.mkdir(parents=True)
    (running / "run_meta.json").write_text(json.dumps({"params": {"bias": "medium", "ctr": 0.05, "pop_strength": 0.0,
                                                                  "logging_uniform_mix": 0.3}}))
    assert _world_settings(running.parent) == ({"pop_strength": 0.0}, 0.3)


def test_pick_diagnostics_rebuild_the_logger_with_its_mix(tmp_path, monkeypatch):
    import training.run_full_study as rfs
    from training import policy_diagnostics as pdiag

    cond = tmp_path / "run_mixed" / "dataset=toy__bias=medium__ctr=0.05__seed=0"
    cond.mkdir(parents=True)
    (cond.parent / "run_manifest.json").write_text(json.dumps({"world_options": {}, "logging_uniform_mix": 0.2}))
    pdiag.save_selected_policy(cond / f"opc_n1000_r0{pdiag.POLICY_SUFFIX}", np.zeros((3, 2)), np.zeros((4, 2)),
                               arm="opc")
    seen = {}

    class Built(Exception):
        pass

    def fake_world(*args, **kwargs):
        seen.update(kwargs)
        raise Built

    monkeypatch.setattr(rfs, "build_condition_world", fake_world)
    with pytest.raises(Built):
        pdiag.main(["--runs", str(cond.parent), "--out", str(tmp_path / "out")])
    assert seen["logging_uniform_mix"] == 0.2


def test_a_mixed_logger_gets_its_own_condition_folders():
    from training.run_full_study import _condition_run_key

    plain = _condition_run_key("ml", "medium", 0.05, 1, {})
    assert _condition_run_key("ml", "medium", 0.05, 1, {}, logging_uniform_mix=0.0) == plain
    assert _condition_run_key("ml", "medium", 0.05, 1, {}, logging_uniform_mix=0.2) == plain + "__mix=0.2"


# ------------------------------------------------------------------ dense-policy values (finding 7)
def test_a_dense_policy_has_the_value_of_the_same_policy_object(toy_root):
    from training.run_full_study import build_condition_world
    from utils.simulation_utils import calc_reward, calc_reward_mc, ensure_exact_env_q_cache

    dataset = build_condition_world("toy", toy_root, "medium", 0.05, 0, world_options={})[0]
    prior = np.asarray(dataset["user_prior"], dtype=float)
    assert prior.std() > 0.1 * prior.mean()  # the users are not equally likely: the weighting matters
    ensure_exact_env_q_cache(dataset)
    x, a = np.asarray(dataset["our_x"], dtype=np.float64), np.asarray(dataset["our_a"], dtype=np.float64)
    logits = x @ a.T
    dense = np.exp(logits - logits.max(axis=1, keepdims=True))
    dense /= dense.sum(axis=1, keepdims=True)
    exact = float(calc_reward(dataset, SimpleNamespace(user_emb=dataset["our_x"], item_emb=dataset["our_a"],
                                                       temperature=1.0, action_chunk=8192)))
    assert float(calc_reward(dataset, dense)[0]) == pytest.approx(exact, rel=1e-5)
    assert float(calc_reward_mc(dataset, dense)[0]) == pytest.approx(exact, rel=1e-5)
    uniform_mean = float(np.sum(dataset["q_x_a"] * dense, axis=1).mean())
    assert abs(uniform_mean - exact) > 1e-4  # the value before the fix: users weighted equally


# ------------------------------------------------------------------ the slim run log (finding 8)
def test_the_slim_run_log_describes_the_rows_the_run_trained_on(toy_root, tmp_path, monkeypatch):
    """Without a split cache the trainer draws its own split; the winning run's weight diagnostics are computed on
    that split's training rows (before 2026-10-06 they re-simulated rows with an older seed)."""
    import training.trainer_trials as tt
    from training.run_full_study import _dataset_paths
    from utils.simulation_utils import generate_dataset

    up, ip, _, _ = _dataset_paths(toy_root, "toy")
    dataset = generate_dataset({"bias": "medium", "ctr": 0.05}, seed=0, emb_x=np.load(up), emb_a=np.load(ip))
    splits, seen = [], {}
    real_split, real_extras = tt._build_regression_logged_split, tt._append_slim_winning_run_extras

    def split(*args, **kwargs):
        splits.append(real_split(*args, **kwargs))
        return splits[-1]

    def extras(row, d, **kwargs):
        seen.update(kwargs)
        return real_extras(row, d, **kwargs)

    monkeypatch.setattr(tt, "_build_regression_logged_split", split)
    monkeypatch.setattr(tt, "_append_slim_winning_run_extras", extras)
    logs = {"trials": tmp_path / "trials_long.csv", "runs": tmp_path / "runs_long.csv"}
    tt.regression_trainer_trial(train_sizes=[1000], dataset=dataset, batch_size=None, val_size=1000, val_min=1000,
                                n_trials=2, slim=True, shared_regression_size=2000, search_use_log_trick=False,
                                use_log_trick_fixed=False, log_paths=logs)
    trained = splits[-1]["train_data"]  # the size's split (the first one fits the shared reward model)
    assert int(trained["num_data"]) == 1000
    np.testing.assert_array_equal(seen["train_users"], trained["x_idx"])
    np.testing.assert_array_equal(seen["train_actions"], trained["a"])
    np.testing.assert_array_equal(seen["pscore_tr"], np.asarray(trained["pscore"], dtype=np.float32))
    assert pd.read_csv(logs["runs"])["weights_gini"].notna().all()


# ------------------------------------------------------------------ the OOM detector (finding 9)
def test_oom_detection_matches_whole_words():
    from training.run_full_study_parallel import _is_oom_like

    for msg in ("no room left in the bracket", "zoom level", "a bloom filter", "skilled", "killedness"):
        assert not _is_oom_like(ValueError(msg)), msg
    for msg in ("host OOM", "oom-killer", "Worker process killed by SIGKILL", "CUDA out of memory. Tried to allocate"):
        assert _is_oom_like(RuntimeError(msg)), msg


# ------------------------------------------------------------------ manifests, flags, help (findings 10, 11, 13)
@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_the_runners_record_their_arms_and_options(module, toy_root, tmp_path, monkeypatch):
    """Both runners' manifests name the arms and record the CausE / BLOB options only when those arms were asked
    for; every invocation is appended to run_invocations.jsonl."""
    mod = __import__(f"training.{module}", fromlist=["main"])
    if module == "run_full_study_parallel":
        monkeypatch.setattr(mod, "_run_with_memory_cap", lambda configs, **kw: [])
    else:
        import training.run_full_study as rfs

        monkeypatch.setattr(rfs, "execute_condition", lambda cfg, **kw: {"run_key": cfg["run_key"], "ran": []})
    out = tmp_path / "out"
    base = [module, "--datasets", "toy", "--bias-configs", "medium", "--seeds", "0", "--train-sizes", "1000",
            "--emb-dir", str(toy_root), "--out-dir", str(out), "--run-tag", "m", "--sampler", "random"]
    cond = out / "run_m" / "dataset=toy__bias=medium__ctr=0.05__seed=0__qhat=train__cf=5__val=20000"
    cond.mkdir(parents=True)
    pd.DataFrame({"method": ["opc"], "train_size": [1000]}).to_csv(cond / "summary_metrics.csv", index=False)
    for methods, cause, blob in ((["cause", "--cause-family", "cap"], "cap", None), (["opc", "blob"], None, ["nq"])):
        monkeypatch.setattr(sys, "argv", base + ["--methods", *methods])
        mod.main()
        manifest = json.loads((out / "run_m" / "run_manifest.json").read_text())
        arms = [m for m in methods if not m.startswith("--") and m != "cap"]
        runner = {"run_full_study": "serial", "run_full_study_parallel": "parallel"}[module]
        assert manifest["study_methods"] == arms and manifest["runner"] == runner
        assert (manifest["cause_options"] or {}).get("family") == cause
        assert (manifest["blob_options"] or {}).get("families") == blob
        assert manifest["n_trials"] == 20 and manifest["sampler"] == "random" and manifest["code_commit"]["commit"]
    history = [json.loads(x) for x in (out / "run_m" / "run_invocations.jsonl").read_text().splitlines()]
    assert [h["manifest"]["study_methods"] for h in history] == [["cause"], ["opc", "blob"]]


def test_the_parallel_runner_rejects_an_unknown_loss_at_once(monkeypatch, capsys):
    import training.run_full_study_parallel as par

    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", "--policy-losses", "drr"])
    with pytest.raises(SystemExit):
        par.main()
    assert "invalid choice: 'drr'" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["run_full_study", "run_full_study_parallel"])
def test_help_states_the_real_chunk_defaults(module, monkeypatch, capsys):
    from training.trainer_trials import DEFAULT_QHAT_ACTION_CHUNK, DEFAULT_QHAT_USER_CHUNK

    main = __import__(f"training.{module}", fromlist=["main"]).main
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit):
        main()
    text = " ".join(capsys.readouterr().out.split())
    assert f"(default {DEFAULT_QHAT_ACTION_CHUNK})" in text and f"(default {DEFAULT_QHAT_USER_CHUNK})" in text
    assert "--skip-completed" in text and "made with the same settings" in text


# ------------------------------------------------------------------ the CausE trial diagnostics (finding 4)
def test_cause_trial_diagnostics_build_no_autograd_graph(monkeypatch):
    from models.cause import CausELinModel
    from training import cause_trials

    model = CausELinModel(np.random.default_rng(0).standard_normal((5, 3)),
                          np.random.default_rng(1).standard_normal((4, 3)))
    grad_on, real_cat = [], torch.cat

    def cat(*args, **kwargs):
        grad_on.append(torch.is_grad_enabled())
        return real_cat(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", cat)
    out = cause_trials._diagnostics(model, layout=None)
    assert grad_on and not any(grad_on)
    assert set(out) >= {"alpha", "map_norm_user", "mean_l1_treatment_minus_control"}
    assert torch.is_grad_enabled()  # restored after the call
