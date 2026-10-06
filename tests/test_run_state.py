"""The condition-folder state behind --skip-completed and idempotent reruns (training/run_state.py): the labels a
request expects, the configuration key, the status of a folder, and the row merges. The workflow end to end is in
tests/test_resume_idempotence.py."""
from __future__ import annotations

import pandas as pd
import pytest

from training.run_state import (
    CONFIG_KEY_COLUMN,
    append_csv,
    arm_config,
    arm_labels,
    arm_status,
    condition_plan,
    config_key,
    dedupe_trials,
    merge_run_meta,
    merge_summary,
    read_trials_long,
    replace_rows,
    reset_arm_logs,
)


def _cfg(tmp_path, **kw):
    """A condition config with the runners' keys (training/run_full_study.py condition_configs)."""
    base = dict(run_dir=str(tmp_path), train_sizes=[1000], n_trials=20, sampler="random", stage="development",
                val_size=20000, val_frac=0.15, val_min=5000, val_max=None, shared_regression_size=50000,
                logging_uniform_mix=0.0, world_options={"pop_strength": 0.0}, reward_data="train", crossfit_folds=5,
                deterministic=True, cpu_threads=4, policy_loss_types=["dr"], opc_gradient="direct",
                train_weights="harmonic:0.1", sn_scope="batch", select_weights="clip:10", policy_transform="linear",
                learn_logit_scale=True, post_temper=False, search_space=None, optuna_selection="ci_low",
                optuna_batch_sizes=None, batch_size=None, reward_model="regression", reward_features="interaction",
                slim=True, policy_reward_mode="exact", policy_reward_mc_sim=8, qhat_user_chunk=5000,
                qhat_action_chunk=8192, require_cuda=False, save_policies=False,
                cause_options={"family": "cap", "rhos": [0.0, 0.25]}, blob_options={"families": ["nq"]})
    return {**base, **kw}


def test_the_labels_of_each_arm():
    assert arm_labels("opc") == ("opc",) and arm_labels("tempered_logger") == ("tempered_logger",)
    assert set(arm_labels("cause", {"rhos": [0.0, 0.25]})) == {
        "cause_prod_c_r000", "cause_prod_t_r000", "cause_avg_r000", "cause_prod_c_r250", "cause_prod_t_r250",
        "cause_avg_r250"}
    assert set(arm_labels("cause", {"rhos": [0.0], "variants": ["avg"]})) == {"cause_avg_r000"}
    assert set(arm_labels("cause", {"family": "cap", "rhos": [0.05]})) == {"causecap_c_r050", "causecap_t_r050"}
    assert set(arm_labels("cause", {"family": "warm", "rhos": [0.0], "variants": ["avg"]})) == {
        "causewarm_c_r000", "causewarm_t_r000"}  # the warm families train the prod layout whatever the variants
    assert set(arm_labels("blob", blob_options={"families": ["nq", "mnq"], "variants": ["released", "L10"]})) == {
        "blob_nq", "blob_mnq", "blob_l10_nq", "blob_l10_mnq"}
    with pytest.raises(ValueError):
        arm_labels("oracle")


def test_the_configuration_key_covers_the_settings_that_change_results(tmp_path):
    cfg = _cfg(tmp_path)
    key = lambda method, **kw: config_key(arm_config(method, _cfg(tmp_path, **kw)))
    for method in ("opc", "dm", "cause", "blob"):
        assert key(method) == config_key(arm_config(method, cfg)) and key(method).startswith("cfg-")
        # performance and bookkeeping settings do not change it
        assert key(method, qhat_action_chunk=4096, require_cuda=True, save_policies=True) == key(method)
        # the data and the search do
        assert key(method, n_trials=10) != key(method) and key(method, val_size=5000) != key(method)
        assert key(method, logging_uniform_mix=0.2) != key(method)
    assert key("opc", train_weights="none") != key("opc") and key("dm", select_weights="clip:1") != key("dm")
    assert key("cause", train_weights="none") == key("cause")  # an OPC setting
    assert key("blob", train_weights="none") == key("blob")
    assert key("cause", cause_options={"family": "warm", "rhos": [0.0, 0.25]}) != key("cause")
    assert key("cause", cause_options={"family": "cap", "rhos": [0.0, 0.25], "l2": [0.0]}) != key("cause")
    # which rhos, variants and families run, and where, do not change a label's results
    assert key("cause", cause_options={"family": "cap", "rhos": [0.1], "device": "cpu"}) == key("cause")
    assert key("blob", blob_options={"families": ["mnq"], "variants": ["L10"], "pick_diagnostics": True}) == key("blob")
    assert key("blob", blob_options={"families": ["nq"], "seed_tag": "x"}) != key("blob")
    assert key("opc", search_space={"lr": (1e-4, 1e-3)}) == key("opc")  # the default range, given explicitly
    assert key("opc", search_space={"lr": (1e-4, 2e-3)}) != key("opc")


def test_the_status_of_a_folder(tmp_path):
    labels, sizes = ["causecap_c_r000", "causecap_t_r000"], [1000]
    k = "cfg-aaaaaaaaaaaa"
    assert arm_status(None, labels, sizes, k) == ("pending", [(x, 1000) for x in labels])
    summary = pd.DataFrame({"method": labels, "train_size": [1000.0, 1000.0], CONFIG_KEY_COLUMN: [k, k]})
    assert arm_status(summary, labels, sizes, k) == ("complete", [])
    assert arm_status(summary, labels, [1000, 5000], k)[0] == "pending"
    status, info = arm_status(summary, labels, sizes, "cfg-bbbbbbbbbbbb")
    assert status == "conflict" and info[0] == ("causecap_c_r000", 1000, [k])
    legacy = summary.drop(columns=[CONFIG_KEY_COLUMN])  # written before the key: its settings are unrecorded
    assert arm_status(legacy, labels, sizes, "cfg-bbbbbbbbbbbb") == ("complete", [])


def test_a_partly_complete_arm_runs_only_what_it_lacks(tmp_path):
    cfg = _cfg(tmp_path)
    key = config_key(arm_config("cause", cfg))
    pd.DataFrame({"method": ["causecap_c_r000", "causecap_t_r000"], "train_size": 1000,
                  CONFIG_KEY_COLUMN: key}).to_csv(tmp_path / "summary_metrics.csv", index=False)
    plan = condition_plan(cfg, ["opc", "cause"], skip_completed=True)
    assert plan["pending"] == ["opc", "cause"] and not plan["conflicts"]
    assert plan["options"]["cause_options"]["rhos"] == [0.25]  # rho 0 is done
    assert condition_plan(cfg, ["cause"], skip_completed=False) == {"pending": ["cause"], "complete": [],
                                                                    "conflicts": [], "detail": [], "options": {}}
    blob = _cfg(tmp_path, blob_options={"families": ["nq"], "variants": ["released", "L10"]})
    pd.DataFrame({"method": ["blob_nq"], "train_size": 1000, CONFIG_KEY_COLUMN: config_key(arm_config("blob", blob))
                  }).to_csv(tmp_path / "summary_metrics.csv", index=False)
    assert condition_plan(blob, ["blob"], skip_completed=True)["options"]["blob_options"]["variants"] == ["L10"]
    other = _cfg(tmp_path, blob_options={"families": ["nq"], "variants": ["released"], "lr_range": [1e-3, 1e-2]})
    plan = condition_plan(other, ["blob"], skip_completed=True)
    assert plan["conflicts"] == ["blob"] and "blob_nq at train size 1000" in plan["detail"][0]


def test_rows_replace_their_own_and_keep_the_rest():
    old = pd.DataFrame({"method": ["opc", "opc", "dm"], "train_size": [0, 1000, 1000], "v": [1.0, 2.0, 3.0]})
    new = pd.DataFrame({"method": ["opc", "opc"], "train_size": [0.0, 1000.0], "v": [1.5, 2.5], "w": [9, 9]})
    out = replace_rows(old, new)
    assert list(zip(out["method"], out["train_size"], out["v"])) == [("dm", 1000, 3.0), ("opc", 0, 1.5),
                                                                       ("opc", 1000, 2.5)]
    assert out["w"].isna().tolist() == [True, False, False]
    assert replace_rows(None, new).equals(new) and replace_rows(old, None) is old


def test_the_merged_summary_recomputes_the_paired_column():
    base = dict(seed=0, dataset="toy", ctr=0.05)
    old = pd.DataFrame([{"method": "opc", "train_size": 1000, "policy_rewards": 0.2, **base},
                        {"method": "no_propensity", "train_size": 1000, "policy_rewards": 0.1, **base}])
    from training.metrics_utils import add_paired_method_pct_columns

    old = add_paired_method_pct_columns(old)
    new = pd.DataFrame([{"method": "no_propensity", "train_size": 1000, "policy_rewards": 0.16, **base}])
    out = merge_summary(old, new)
    assert list(out.columns).count("opc_vs_noprop_pct") == 1
    assert out.loc[out["method"] == "opc", "opc_vs_noprop_pct"].item() == pytest.approx(25.0)


def test_logs_append_safely_and_reset_per_arm(tmp_path):
    path = tmp_path / "opc_trials_long.csv"
    append_csv(path, pd.DataFrame({"method": "opc", "train_size": 1000, "trial_number": [0, 1], "a": [1, 2]}))
    append_csv(path, pd.DataFrame({"method": "opc", "train_size": 5000, "trial_number": [0], "b": [7], "a": [3]}))
    t = pd.read_csv(path)  # another column set: rewritten with the union, every value in its own column
    assert list(t["a"]) == [1, 2, 3] and t["b"].isna().tolist() == [True, True, False]
    reset_arm_logs({"trials": path}, "opc", [1000])
    assert list(pd.read_csv(path)["train_size"]) == [5000]
    reset_arm_logs({"trials": path}, "dm", [5000])  # another arm's rows are not touched
    assert len(pd.read_csv(path)) == 1
    reset_arm_logs({"trials": path, "runs": tmp_path / "missing.csv"}, "opc", [5000])
    assert not path.exists()


def test_trial_logs_load_one_row_per_trial(tmp_path):
    t = pd.DataFrame({"method": "opc", "train_size": 1000, "run": 0, "trial_number": [0, 1, 0, 1],
                      "actual_reward": [0.1, 0.2, 0.1, 0.2], "trial_time_s": [1.0, 2.0, 3.0, 4.0]})
    t.to_csv(tmp_path / "trials_long.csv", index=False)
    assert list(dedupe_trials(t)["trial_time_s"]) == [3.0, 4.0]  # the last copy: a rerun's
    out = read_trials_long(tmp_path / "trials_long.csv", usecols=["method", "train_size", "actual_reward"])
    assert len(out) == 2 and "trial_time_s" not in out.columns and {"trial_number", "run"} <= set(out.columns)


def test_run_meta_records_every_label():
    old = {"study_methods": ["cause"], "cause": {"family": "cap"}, "blob": None,
           "labels": {"causecap_c_r000": {"arm": "cause", "config_key": "cfg-1"}},
           "arm_configs": {"cfg-1": {"arm": "cause"}, "cfg-0": {"arm": "stale"}}}
    new = {"study_methods": ["blob"], "cause": None, "blob": {"families": ["nq"]}, "world": {"x": 1}}
    out = merge_run_meta(old, new, {"blob_nq": {"arm": "blob", "config_key": "cfg-2"}}, {"cfg-2": {"arm": "blob"}})
    assert out["study_methods"] == ["cause", "blob"] and out["cause"] == {"family": "cap"}
    assert set(out["labels"]) == {"causecap_c_r000", "blob_nq"} and set(out["arm_configs"]) == {"cfg-1", "cfg-2"}
    assert out["world"] == {"x": 1}


def test_kept_rows_keep_their_digits(tmp_path):
    """Merging rewrites the summary: the rows it keeps must come back digit for digit (pandas' default float parser
    reads 3.9057021226737194 as 3.90570212267372)."""
    from training.run_state import atomic_write_csv, read_summary

    path = tmp_path / "summary_metrics.csv"
    path.write_text("method,train_size,policy_rewards_pct_vs_initial\nopc,1000,3.9057021226737194\n")
    new = pd.DataFrame({"method": ["dm"], "train_size": [1000], "policy_rewards_pct_vs_initial": [1.5]})
    atomic_write_csv(merge_summary(read_summary(path), new), path)
    assert path.read_text().splitlines()[1] == "opc,1000,3.9057021226737194"
