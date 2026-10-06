"""Rerunning, resuming and extending a study run is idempotent (training/run_state.py), through the runners' command
lines on the toy world. One condition collects, over several invocations, OPC, DM-only, CausE-capacity-matched (two
rhos), native CausE and BLOB-NQ with a second prior variant, with an interrupted invocation in between. After every
step the folder holds each requested label once per train size and each trial once, rows already completed keep their
values, an interrupted arm leaves nothing behind once resumed, and repeating a request runs nothing. At the end the
folder's rows equal those of the same arms run in one invocation."""
from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd
import pytest

from test_reproducibility import _toy_embeddings

N, TRIALS = 1000, 2
OPC_FAMILY = ("opc", "no_propensity", "dm", "tempered_logger")
CAP = ["causecap_c_r000", "causecap_t_r000", "causecap_c_r250", "causecap_t_r250"]
NATIVE = ["cause_prod_c_r000", "cause_prod_t_r000"]


def _argv(root, tag, *extra):
    return ["run_full_study", "--datasets", "toy", "--bias-configs", "medium", "--seeds", "0", "--train-sizes", str(N),
            "--n-trials", str(TRIALS), "--val-size", "1000", "--val-min", "1000", "--shared-regression-size", "2000",
            "--slim", "--sampler", "random", "--learn-logit-scale", "--emb-dir", str(root / "emb"),
            "--out-dir", str(root / "out"), "--run-tag", tag,
            "--cause-dim", "8", "--cause-trials", "3", "--cause-epochs", "2", "5", "--cause-batch-size", "128",
            "--cause-rhos", "0", "0.25", "--cause-temper",
            "--blob-families", "nq", "--blob-trials", "3", "--blob-epochs", "2", "5", "--blob-batch-size", "128",
            *extra]


def _serial(monkeypatch, argv):
    import training.run_full_study as rfs

    monkeypatch.setattr(sys, "argv", argv)
    rfs.main()


def _parallel(monkeypatch, argv, pool=False):
    """The parallel runner: its own planning and worker body; with ``pool`` its real process pool (one worker),
    otherwise the workers' function called in this process."""
    import training.run_full_study_parallel as par

    if not pool:
        def dispatch(configs, **_kw):
            failures = []
            for cfg in configs:
                try:
                    par._execute_run(cfg)
                except Exception as e:
                    failures.append({"run_key": cfg["run_key"], "error": repr(e)})
            return failures

        monkeypatch.setattr(par, "_run_with_memory_cap", dispatch)
        monkeypatch.setenv("OPC_IN_PARALLEL", "1")  # _execute_run sets it; restored after the test
    monkeypatch.setattr(sys, "argv", ["run_full_study_parallel", *argv[1:], "--max-workers", "1", "--no-memory-cap"])
    par.main()


def _cond(root, tag):
    (cond,) = sorted((root / "out" / f"run_{tag}").glob("dataset=*"))
    return cond


def _summary(cond) -> pd.DataFrame:
    """The summary indexed by (label, train size); the arms trained by trainer_trials also hold the logger's row at
    train size 0."""
    return pd.read_csv(cond / "summary_metrics.csv", dtype={"arm_config_key": str}).set_index(["method", "train_size"])


def _rows(df: pd.DataFrame, labels) -> pd.DataFrame:
    return df[df.index.get_level_values("method").isin(list(labels))].sort_index()


def _hashes(cond) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(cond.iterdir())
            if p.is_file() and p.suffix in (".csv", ".json", ".npz")}


def _invocations(root, tag) -> list[dict]:
    path = root / "out" / f"run_{tag}" / "run_invocations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _check_folder(cond, labels):
    """Each label once, each trial once, the summary's values those of its selected trials, provenance complete."""
    s = pd.read_csv(cond / "summary_metrics.csv", dtype={"arm_config_key": str})
    assert not s.duplicated(["method", "train_size"]).any()
    expected = {(x, N) for x in labels} | {(x, 0) for x in labels if x in OPC_FAMILY}  # + the logger's row
    assert set(zip(s["method"], s["train_size"])) == expected
    s = s[s["train_size"] == N]
    assert s["arm_config_key"].str.startswith("cfg-").all()
    meta = json.loads((cond / "run_meta.json").read_text())
    assert set(meta["labels"]) == set(labels) and set(meta["study_methods"]) >= {meta["labels"][x]["arm"] for x in labels}
    assert {v["config_key"] for v in meta["labels"].values()} == set(meta["arm_configs"])
    family = sorted(x for x in labels if x in OPC_FAMILY)
    t = pd.read_csv(cond / "trials_long.csv") if family else pd.DataFrame(columns=["method"])
    assert sorted(t["method"].unique()) == family
    if family:
        assert not t.duplicated(["method", "train_size", "run", "trial_number"]).any()
        assert (t.groupby("method").size() == TRIALS).all()
        runs = pd.read_csv(cond / "runs_long.csv")
        assert sorted(runs["method"]) == family
    for label in family:
        log = pd.read_csv(cond / f"{dict(opc='opc', no_propensity='no_prop').get(label, label)}_trials_long.csv")
        assert len(log) == TRIALS and not log.duplicated(["train_size", "trial_number"]).any()
        best = t[(t["method"] == label) & t["is_best_in_run"].astype(bool)]
        assert len(best) == 1
        assert best["actual_reward"].iloc[0] == pytest.approx(s.loc[s["method"] == label, "policy_rewards"].iloc[0],
                                                              abs=1e-12)
    for label in (x for x in labels if x not in OPC_FAMILY):
        trials = pd.read_csv(cond / f"{label}_trials.csv")
        row = s[s["method"] == label].iloc[0]
        assert not trials.duplicated(["train_size", "trial"]).any() and len(trials) == int(row["n_trials"])
        assert int(row["selected_trial"]) in set(trials["trial"])
    run_root = cond.parent
    every = pd.read_csv(run_root / "all_summary_metrics.csv")
    assert set(zip(every["method"], every["train_size"])) == expected and len(every) == len(expected)
    assert not (run_root / "failures.csv").exists()


def _same_rows(before: pd.DataFrame, after: pd.DataFrame, labels):
    """The rows of ``labels`` are unchanged: every column they had, bit for bit."""
    pd.testing.assert_frame_equal(_rows(before, labels), _rows(after, labels)[before.columns], check_dtype=False,
                                  check_exact=True)


def test_rerun_resume_and_extend_a_condition(tmp_path, monkeypatch):
    import training.run_full_study as rfs

    (tmp_path / "emb").mkdir()
    _toy_embeddings(tmp_path / "emb")
    run = lambda *extra, tag="inc": _serial(monkeypatch, _argv(tmp_path, tag, *extra))

    # 1. one invocation: OPC and CausE-capacity-matched at two rhos
    run("--methods", "opc", "cause", "--cause-family", "cap")
    cond = _cond(tmp_path, "inc")
    labels = ["opc", *CAP]
    _check_folder(cond, labels)
    first = _summary(cond)
    assert len(_invocations(tmp_path, "inc")) == 1

    # 2. the same request again: nothing runs, nothing changes
    hashes = _hashes(cond)
    run("--methods", "opc", "cause", "--cause-family", "cap")
    assert _hashes(cond) == hashes
    assert _invocations(tmp_path, "inc")[-1]["skipped"] == [cond.name]

    # 3. add DM-only; the invocation dies after DM trained (its logs written), before the folder's files are merged
    real = rfs._finalize_summary_df

    def crash(*a, **kw):
        raise RuntimeError("worker died")

    monkeypatch.setattr(rfs, "_finalize_summary_df", crash)
    run("--methods", "opc", "dm", "cause", "--cause-family", "cap")
    assert (cond.parent / "failures.csv").exists() and len(pd.read_csv(cond / "dm_trials_long.csv")) == TRIALS
    assert set(pd.read_csv(cond / "trials_long.csv")["method"]) == {"opc"}  # the interrupted arm is not listed
    assert set(_summary(cond).index.get_level_values("method")) == set(labels)
    # ... and resumes: only DM runs (its earlier attempt's rows replaced), the failure record is cleared
    monkeypatch.setattr(rfs, "_finalize_summary_df", real)
    run("--methods", "opc", "dm", "cause", "--cause-family", "cap")
    labels.append("dm")
    _check_folder(cond, labels)
    _same_rows(first, _summary(cond), ["opc", *CAP])
    last = _invocations(tmp_path, "inc")[-1]
    assert last["ran"] == [cond.name] and last["manifest"]["study_methods"] == ["opc", "dm", "cause"]

    # 4. another CausE family (only it runs), then a second BLOB prior variant (only the new variant runs)
    run("--methods", "cause", "--cause-family", "native", "--cause-variants", "prod", "--cause-rhos", "0")
    labels += NATIVE
    _check_folder(cond, labels)
    run("--methods", "blob", "--blob-variants", "released")
    labels.append("blob_nq")
    _check_folder(cond, labels)
    before = _summary(cond)
    trials_nq = pd.read_csv(cond / "blob_nq_trials.csv")
    run("--methods", "blob", "--blob-variants", "released", "L10")
    labels.append("blob_l10_nq")
    _check_folder(cond, labels)
    _same_rows(before, _summary(cond), [x for x in labels if x != "blob_l10_nq"])
    pd.testing.assert_frame_equal(trials_nq, pd.read_csv(cond / "blob_nq_trials.csv"))  # not retrained

    # 5. repeat each request: nothing runs
    hashes = _hashes(cond)
    run("--methods", "blob", "--blob-variants", "released", "L10")
    run("--methods", "opc", "dm", "cause", "--cause-family", "cap")
    run("--methods", "cause", "--cause-family", "native", "--cause-variants", "prod", "--cause-rhos", "0")
    assert _hashes(cond) == hashes

    # 6. --no-skip-completed reruns the requested arm and replaces its rows (the same values: deterministic); the
    # other arms' rows stay
    before = _summary(cond)
    run("--methods", "opc", "--no-skip-completed")
    _check_folder(cond, labels)
    after = _summary(cond)
    _same_rows(before, after, [x for x in labels if x != "opc"])
    times = [c for c in before.columns if "time" in c.lower() or "seconds" in c.lower()]
    _same_rows(before.drop(columns=times), after, ["opc"])

    # 7. a request whose settings differ from the rows it would complete stops before running anything
    hashes = _hashes(cond)
    with pytest.raises(SystemExit, match="other settings"):
        run("--methods", "opc", "--train-weights", "none")
    assert _hashes(cond) == hashes

    # 8. the folder equals the same arms run in one invocation (CausE's two families need two). A BLOB prior variant
    # added to a folder trains as it does alone (variants trained together agree with it only to float32 batching,
    # tests/test_blob.py, which can flip a near-tie of the selection), so it is compared with a run of it alone.
    run("--methods", "opc", "dm", "cause", "blob", "--cause-family", "cap", "--blob-variants", "released",
        tag="straight")
    run("--methods", "cause", "--cause-family", "native", "--cause-variants", "prod", "--cause-rhos", "0",
        tag="straight")
    run("--methods", "blob", "--blob-variants", "L10", tag="alone")
    straight = pd.concat([_summary(_cond(tmp_path, "straight")), _summary(_cond(tmp_path, "alone"))]).sort_index()
    final = _summary(cond).sort_index()
    assert list(straight.index) == list(final.index)
    assert (straight["arm_config_key"] == final["arm_config_key"]).all()
    # every column but the wall times and the reward model's features, a tag of the invocation (empty when none of
    # its arms fits a reward model)
    cols = [c for c in straight.columns if c in final.columns and c != "reward_features"
            and not ("time" in c.lower() or "seconds" in c.lower())]
    pd.testing.assert_frame_equal(straight[cols], final[cols], check_dtype=False, check_exact=True)


def test_the_parallel_runner_resumes_the_same_way(tmp_path, monkeypatch):
    """The parallel runner plans and merges like the serial one: its real pool runs a condition, a repeat runs
    nothing, an added variant runs alone; and its rows equal the serial runner's."""
    (tmp_path / "emb").mkdir()
    _toy_embeddings(tmp_path / "emb")
    _parallel(monkeypatch, _argv(tmp_path, "par", "--methods", "opc", "blob", "--blob-variants", "released"), pool=True)
    cond = _cond(tmp_path, "par")
    _check_folder(cond, ["opc", "blob_nq"])
    hashes = _hashes(cond)
    _parallel(monkeypatch, _argv(tmp_path, "par", "--methods", "opc", "blob", "--blob-variants", "released"))
    assert _hashes(cond) == hashes
    before = _summary(cond)
    _parallel(monkeypatch, _argv(tmp_path, "par", "--methods", "opc", "blob", "--blob-variants", "released", "L10"))
    _check_folder(cond, ["opc", "blob_nq", "blob_l10_nq"])
    _same_rows(before, _summary(cond), ["opc", "blob_nq"])
    manifest = json.loads((cond.parent / "run_manifest.json").read_text())
    assert manifest["runner"] == "parallel" and manifest["study_methods"] == ["opc", "blob"]
    assert manifest["blob_options"]["variants"] == ["released", "L10"] and manifest["cause_options"] is None
    assert [r["ran"] for r in _invocations(tmp_path, "par")] == [[cond.name], [], [cond.name]]
    _serial(monkeypatch, _argv(tmp_path, "ser", "--methods", "opc", "blob", "--blob-variants", "released"))
    ser = _summary(_cond(tmp_path, "ser"))
    cols = [c for c in ser.columns if c != "reward_features" and not ("time" in c.lower() or "seconds" in c.lower())]
    pd.testing.assert_frame_equal(_rows(ser, ["opc", "blob_nq"])[cols], _rows(before, ["opc", "blob_nq"])[cols],
                                  check_dtype=False, check_exact=True)


def test_an_arm_extended_later_equals_one_run_of_it(tmp_path, monkeypatch):
    """A CausE arm extended by a rho, and a BLOB arm extended by a family, run only the new rho / family; their rows
    equal those of one run of the whole arm (CausE trains each rho, and BLOB each family, on its own streams)."""
    (tmp_path / "emb").mkdir()
    _toy_embeddings(tmp_path / "emb")
    run = lambda tag, *extra: _serial(monkeypatch, _argv(tmp_path, tag, *extra))
    cap = ("--methods", "cause", "--cause-family", "cap")
    run("later", *cap, "--cause-rhos", "0")
    run("later", *cap, "--cause-rhos", "0", "0.25")
    run("later", "--methods", "blob", "--blob-families", "nq")
    run("later", "--methods", "blob", "--blob-families", "nq", "mnq")
    assert [r["ran"] != [] for r in _invocations(tmp_path, "later")] == [True] * 4
    run("once", *cap, "--cause-rhos", "0", "0.25")
    run("once", "--methods", "blob", "--blob-families", "nq", "mnq")
    later, once = _cond(tmp_path, "later"), _cond(tmp_path, "once")
    a, b = _summary(later).sort_index(), _summary(once).sort_index()
    assert list(a.index.get_level_values("method")) == sorted(CAP + ["blob_nq", "blob_mnq"])
    cols = [c for c in b.columns if c != "reward_features" and not ("time" in c.lower() or "seconds" in c.lower())]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False, check_exact=True)
    for label in CAP + ["blob_nq", "blob_mnq"]:
        ta, tb = (pd.read_csv(c / f"{label}_trials.csv") for c in (later, once))
        keep = [c for c in tb.columns if not ("time" in c.lower() or "seconds" in c.lower())]
        pd.testing.assert_frame_equal(ta[keep], tb[keep], check_exact=True)
