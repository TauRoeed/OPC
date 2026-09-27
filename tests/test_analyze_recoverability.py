"""Recoverability tables (training/analyze_recoverability.py) on hand-made oracle rows."""

import numpy as np
import pandas as pd
import pytest

from training.analyze_recoverability import MIN_LOSS, derive, learned_recovery, summary


def _rows():
    base = dict(dataset="ml", seed=0, logger_ceiling=0.30)
    rows = []
    for bias, lv, lg, ov, og in (("none", 0.24, 0.30, 0.29, 0.30), ("high/none/none", 0.18, 0.22, 0.30, 0.29),
                                 ("none/none/high", 0.17, 0.21, 0.20, 0.24)):
        r = dict(base, bias=bias, logger_value=lv, logger_greedy=lg)
        for cls in ("linear", "linear+scale", "scale"):
            r[f"oracle_{cls}_value"], r[f"oracle_{cls}_greedy"] = ov, (lg if cls == "scale" else og)
        rows.append(r)
    return pd.DataFrame(rows)


def test_derived_quantities():
    df = derive(_rows()).set_index("bias")
    warp = df.loc["w-high.g-none.v-none"]
    assert warp["V_clean"] == pytest.approx(0.24) and warp["V_clean_greedy"] == pytest.approx(0.30)
    assert warp["representation_loss"] == pytest.approx(0.06) and warp["representation_loss_greedy"] == pytest.approx(0.08)
    assert warp["gain_linear+scale_greedy"] == pytest.approx(0.07)
    assert warp["recoverability_linear+scale_greedy"] == pytest.approx(0.875)
    assert warp["recoverability_linear+scale"] == pytest.approx(0.12 / 0.06)  # above 1: not clipped
    none = df.loc["none"]
    assert none["representation_loss_greedy"] == 0 and np.isnan(none["recoverability_linear+scale_greedy"])  # no ratio
    assert df.loc["w-none.g-none.v-high", "recoverability_linear+scale_greedy"] == pytest.approx(0.03 / 0.09)
    assert df.loc["w-high.g-none.v-none", "recoverability_scale_greedy"] == pytest.approx(0.0)  # sharpening keeps the ranking
    # the learner's class takes the better of the two fits (scale fixed, scale learned)
    assert warp["oracle_repair_value"] == pytest.approx(0.30) and warp["recoverability_repair_greedy"] == pytest.approx(0.875)
    assert MIN_LOSS > 0


def test_summary_and_learned_join():
    df = derive(_rows())
    s = summary(df)
    assert list(s.index) == ["none", "w-high.g-none.v-none", "w-none.g-none.v-high"]
    assert s.loc["w-high.g-none.v-none", "gain greedy %"] == pytest.approx(7.0)
    assert s.loc["w-high.g-none.v-none", "bias type"] == "warp only (high)"
    learned = pd.DataFrame([dict(dataset="ml", bias="high/none/none", seed=0, method="opc", train_size=5000,
                                 V_method=0.24, V_method_greedy=0.255)])
    m = learned_recovery(learned, _rows()).iloc[0]
    assert m["learned_gain"] == pytest.approx(0.06) and m["fraction_of_oracle_repair"] == pytest.approx(0.06 / 0.12)
    assert m["fraction_of_oracle_repair_greedy"] == pytest.approx(0.035 / 0.07)


def test_load_learned_reads_a_study_condition(tmp_path):
    """A toy study condition with all four arms (paired random sampler) and a toy oracle row join into
    per-arm recovery rows."""
    from test_reproducibility import _toy_embeddings

    from training.analyze_recoverability import load_learned
    from training.run_full_study import _condition_run_key, _finalize_summary_df, _run_condition

    _toy_embeddings(tmp_path)
    run = tmp_path / "run_toy"
    from utils.representation_bias import resolve_bias_configs

    (bias,) = resolve_bias_configs(["high/none/none"])  # canonical, as the runners name folders
    cond = run / _condition_run_key("toy", bias, 0.05, 0, {}, "1000", reward_data="train", crossfit_folds=5)
    cond.mkdir(parents=True)
    opc, nop, _, _, meta, extra = _run_condition(
        dataset_name="toy", emb_dir=tmp_path, bias=bias, ctr=0.05, seed=0, train_sizes=[1000], n_trials=2,
        batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact",
        policy_reward_mc_sim=8, slim=True, shared_regression_size=2000, run_dir=cond, return_extra=True,
        methods=("opc", "no_propensity", "dm", "tempered_logger"), sampler="random", reward_data="train",
        crossfit_folds=5, learn_logit_scale=True)
    _finalize_summary_df(opc, nop, meta, extra=extra).to_csv(cond / "summary_metrics.csv", index=False)
    rows = load_learned(run)
    assert set(rows["method"]) == {"opc", "no_propensity", "dm", "tempered_logger"} and (rows["n_trials"] == 2).all()
    assert (rows["bias"] == "w-high.g-none.v-none").all() and (rows["regret"] >= 0).all()
    assert rows["V_tempered"].nunique() == 1 and rows[["V_method", "V_method_greedy", "ess_raw"]].notna().all().all()
    oracle = pd.DataFrame([dict(dataset="toy", bias="w-high.g-none.v-none", seed=0, logger_value=0.1, logger_greedy=0.12,
                                logger_ceiling=0.3, **{f"oracle_{c}_{k}": v for c in ("linear", "linear+scale", "scale")
                                                       for k, v in (("value", 0.25), ("greedy", 0.27))}),
                           dict(dataset="toy", bias="none", seed=0, logger_value=0.24, logger_greedy=0.3, logger_ceiling=0.3,
                                **{f"oracle_{c}_{k}": 0.3 for c in ("linear", "linear+scale", "scale") for k in ("value", "greedy")})])
    m = learned_recovery(rows, oracle)
    assert m["fraction_of_oracle_repair"].notna().all()
    np.testing.assert_allclose(m["learned_gain"], m["V_method"] - 0.1)


def _stage2_rows():
    """Two datasets × two seeds of one warp cell with the four arms; OPC beats DM by 1, 2, 3 and 4 points."""
    rows = []
    for i, (ds, seed) in enumerate((("ml", 0), ("ml", 1), ("kuairand", 0), ("kuairand", 1))):
        base = dict(dataset=ds, bias="w-high.g-none.v-none", seed=seed, train_size=5000, n_trials=20)
        for method, v in (("tempered_logger", 0.20), ("no_propensity", 0.21), ("dm", 0.22), ("opc", 0.23 + 0.01 * i)):
            rows.append(dict(base, method=method, V_method=v, V_method_greedy=v + 0.01, learned_gain=v - 0.2,
                             learned_gain_greedy=v - 0.19, fraction_of_oracle_repair=(v - 0.2) / 0.1,
                             fraction_of_oracle_repair_greedy=(v - 0.2) / 0.05, ess_raw=1000.0 + i, w_share_gt10=0.02,
                             w_max=50.0, logit_scale=1.5, sel_error_point=0.01, sel_error_lower=-0.005, regret=0.001))
    return pd.DataFrame(rows)


def test_mean_ci():
    from training.analyze_recoverability import mean_ci

    mean, lo, hi, n = mean_ci([1, 2, 3, 4, np.nan])
    assert (mean, n) == (2.5, 4)
    half = 3.182446305284263 * np.std([1, 2, 3, 4], ddof=1) / 2  # t(0.975, 3)
    assert lo == pytest.approx(2.5 - half) and hi == pytest.approx(2.5 + half)
    assert mean_ci([0.5])[0] == 0.5 and np.isnan(mean_ci([0.5])[1]) and mean_ci([])[3] == 0


def test_stage2_tables():
    from training.analyze_recoverability import stage2_tables

    t = stage2_tables(_stage2_rows())
    f = t["fractions"]
    assert list(f["method"]) == ["opc", "dm", "no_propensity", "tempered_logger"]  # the study's arm order
    assert f["bias type"].eq("warp only (high)").all() and (f["n"] == 4).all()
    opc = f[f["method"] == "opc"].iloc[0]
    assert opc["V %"] == pytest.approx(24.5) and opc["fraction"] == pytest.approx(0.45)
    assert opc["fraction greedy ml"] == pytest.approx(0.7) and opc["fraction greedy kuairand"] == pytest.approx(1.1)
    p = t["paired"].set_index(["contrast", "measure"])
    assert p.loc[("opc - dm", "stochastic"), "mean"] == pytest.approx(2.5) and p.loc[("opc - dm", "stochastic"), "n"] == 4
    assert p.loc[("opc - dm", "stochastic"), "ci_low"] == pytest.approx(2.5 - 3.182446305284263 * np.std([1, 2, 3, 4], ddof=1) / 2)
    assert p.loc[("opc - tempered_logger", "greedy"), "mean"] == pytest.approx(4.5)
    assert list(t["paired"]["measure"])[:3] == ["stochastic"] * 3
    d = t["diagnostics"].set_index("method")
    assert d.loc["opc", "ess_raw"] == pytest.approx(1001.5) and d.loc["opc", "w>10 %"] == pytest.approx(2.0)
    assert d.loc["dm", "sel error lower %"] == pytest.approx(-0.5)


def test_paired_runs_matches_conditions():
    from training.analyze_recoverability import paired_runs

    a = _stage2_rows()
    b = a.assign(V_method=a["V_method"] - 0.002, ess_raw=a["ess_raw"] + 10)
    b = pd.concat([b, b.assign(dataset="anime")])  # a condition missing from ``a`` is not paired
    r = paired_runs(a, b).set_index("measure")
    assert r.loc["V %", "diff"] == pytest.approx(0.2) and r.loc["V %", "n"] == 4
    assert r.loc["V %", "ci_low"] == pytest.approx(0.2) and r.loc["V greedy %", "diff"] == pytest.approx(0.0)
    assert r.loc["ess_raw", "diff"] == pytest.approx(-10.0) and r.loc["V %", "a"] == pytest.approx(24.5)


def test_cli_writes_tables(tmp_path):
    from training.analyze_recoverability import main

    root = tmp_path / "oracle" / "ml"
    root.mkdir(parents=True)
    _rows().to_csv(root / "oracle_repair.csv", index=False)
    main(["stage1", str(tmp_path / "oracle"), "--out", str(tmp_path / "out")])
    got = pd.read_csv(tmp_path / "out" / "stage1_recoverability_by_bias.csv", index_col=0)
    assert list(got.index) == ["none", "w-high.g-none.v-none", "w-none.g-none.v-high"]
    assert got.loc["w-high.g-none.v-none", "recoverability greedy"] == pytest.approx(0.875)
    assert len(pd.read_csv(tmp_path / "out" / "stage1_oracle_rows.csv")) == 3


def _candidates():
    """Validation candidates for the toy warp world: a better linear fit (value and greedy), and a linear+scale fit
    with a higher greedy value but a lower stochastic value (the selection rule keeps the Stage 1 one)."""
    base = dict(dataset="ml", bias="high/none/none", seed=0, logit_scale=1.0)
    return pd.DataFrame([dict(base, cls="linear", lr=0.1, steps=3000, value=0.31, greedy=0.295),
                         dict(base, cls="linear", lr=0.1, steps=9000, value=0.305, greedy=0.296),
                         dict(base, cls="linear+scale", lr=3e-3, steps=9000, value=0.29, greedy=0.299)])


def test_validated_oracle_pools_candidates_with_the_stage1_rule():
    from training.analyze_recoverability import oracle_validation_table, validated_oracle

    oracle = _rows().assign(**{f"oracle_{c}_{k}": v for c in ("linear", "linear+scale") for k, v in (("lr", 0.01), ("logit_scale", 1.0))})
    v = validated_oracle(oracle, _candidates()).set_index("bias")
    w = v.loc["w-high.g-none.v-none"]
    # linear: the 3000-step candidate has the best stochastic value; its greedy value comes with it
    assert w["oracle_linear_value"] == pytest.approx(0.31) and w["oracle_linear_greedy"] == pytest.approx(0.295)
    assert (w["oracle_linear_lr"], w["oracle_linear_steps"]) == (0.1, 3000)
    assert w["oracle_linear_greedy_max"] == pytest.approx(0.296)  # the best greedy value over all candidates
    # linear+scale: the Stage 1 fit keeps the best value, so its greedy value stays; the sensitivity sees 0.299
    assert w["oracle_linear+scale_value"] == pytest.approx(0.30) and w["oracle_linear+scale_greedy"] == pytest.approx(0.29)
    assert w["oracle_linear+scale_greedy_max"] == pytest.approx(0.299) and w["stage1_oracle_linear_value"] == pytest.approx(0.30)
    # worlds without candidates keep their Stage 1 bound
    vec = v.loc["w-none.g-none.v-high"]
    assert vec["oracle_linear_value"] == pytest.approx(0.20) and vec["oracle_linear_steps"] == 3000
    t = oracle_validation_table(validated_oracle(oracle, _candidates())).set_index("bias")
    assert t.loc["w-high.g-none.v-none", "old bound greedy"] == pytest.approx(0.29)
    assert t.loc["w-high.g-none.v-none", "new bound greedy"] == pytest.approx(0.295)
    assert t.loc["w-high.g-none.v-none", "change"] == pytest.approx((0.295 - 0.22) / 0.08 - (0.29 - 0.22) / 0.08)
    assert t.loc["w-high.g-none.v-none", "recoverability greedy, best greedy candidate"] == pytest.approx((0.299 - 0.22) / 0.08)
    assert t.loc["w-none.g-none.v-high", "change"] == pytest.approx(0.0) and "none" not in t.index


def test_per_dataset_tables_and_gap_decomposition():
    from training.analyze_recoverability import gap_decomposition, gap_summary, stage1_by_dataset, stage2_by_dataset

    s1 = stage1_by_dataset(derive(_rows())).set_index("bias")
    assert s1.loc["w-high.g-none.v-none", "logger ranking loss %"] == pytest.approx(8.0)
    assert s1.loc["w-high.g-none.v-none", "oracle repair gain %"] == pytest.approx(7.0)
    assert s1.loc["w-high.g-none.v-none", "structural recoverability"] == pytest.approx(0.875)
    rows = _stage2_rows()
    s2 = stage2_by_dataset(rows).set_index("dataset")
    ml = s2.loc["ml"]
    assert ml["OPC-DM"] == pytest.approx(1.5) and ml["OPC-DM min"] == pytest.approx(1.0) and ml["OPC-DM max"] == pytest.approx(2.0)
    assert ml["OPC gain %"] == pytest.approx(3.5) and ml["tempered gain %"] == pytest.approx(0.0)
    assert ml["OPC fraction greedy"] == pytest.approx(0.7) and ml["n"] == 2
    # decomposition: the ceiling is the target, and the four parts add up
    oracle = _rows().assign(dataset="ml", logger_ceiling=0.30)
    learned = pd.DataFrame([dict(dataset="ml", bias="high/none/none", seed=0, method="opc", train_size=5000,
                                 V_method=0.25, V_method_greedy=0.26)])
    g = gap_decomposition(learned, oracle).iloc[0]
    assert g["V_target_best"] == pytest.approx(0.30) and g["V_logger"] == pytest.approx(0.22)
    assert g["V_oracle_repair"] == pytest.approx(0.29) and g["V_OPC"] == pytest.approx(0.26)
    assert g["structural_gap"] == pytest.approx(0.01) and g["learning_gap"] == pytest.approx(0.03)
    assert g["learned_repair_gain"] == pytest.approx(0.04) and g["representation_loss"] == pytest.approx(0.08)
    assert g["structural_gap"] + g["learning_gap"] + g["learned_repair_gain"] == pytest.approx(g["representation_loss"])
    assert g["structural_gap share"] + g["learning_gap share"] + g["learned_repair_gain share"] == pytest.approx(1.0)
    s = gap_summary(gap_decomposition(learned, oracle)).iloc[0]
    assert s["learning_gap %"] == pytest.approx(3.0) and s["bias type"] == "warp only (high)"


def test_followup_cli_writes_the_tables(tmp_path):
    from test_reproducibility import _toy_embeddings

    from training.analyze_recoverability import main
    from training.oracle_repair import main as oracle_main
    from training.run_full_study import _condition_run_key, _finalize_summary_df, _run_condition
    from utils.representation_bias import resolve_bias_configs

    _toy_embeddings(tmp_path)
    common = ["--datasets", "toy", "--seeds", "0", "--steps", "40", "--fit-users", "400", "--batch-users", "128",
              "--emb-dir", str(tmp_path)]
    oracle_main(common + ["--bias-configs", "none", "high/none/none", "--classes", "linear", "linear+scale",
                          "--lrs", "0.01", "--out", str(tmp_path / "oracle")])
    (bias,) = resolve_bias_configs(["high/none/none"])
    run = tmp_path / "run_toy"
    cond = run / _condition_run_key("toy", bias, 0.05, 0, {}, "1000", reward_data="train", crossfit_folds=2)
    cond.mkdir(parents=True)
    opc, nop, _, _, meta, extra = _run_condition(
        dataset_name="toy", emb_dir=tmp_path, bias=bias, ctr=0.05, seed=0, train_sizes=[1000], n_trials=2, batch_size=None,
        val_size=1000, val_frac=0.15, val_min=1000, val_max=None, policy_reward_mode="exact", policy_reward_mc_sim=8,
        slim=True, shared_regression_size=2000, run_dir=cond, return_extra=True,
        methods=("opc", "no_propensity", "dm", "tempered_logger"), sampler="random", reward_data="train", crossfit_folds=2,
        learn_logit_scale=True)
    _finalize_summary_df(opc, nop, meta, extra=extra).to_csv(cond / "summary_metrics.csv", index=False)
    cand = pd.DataFrame([dict(dataset="toy", bias=bias, seed=0, cls="linear", lr=0.1, steps=120, value=0.0, greedy=0.0,
                              logit_scale=1.0)])
    (tmp_path / "val" / "toy").mkdir(parents=True)
    cand.to_csv(tmp_path / "val" / "toy" / "oracle_candidates.csv", index=False)
    main(["followup", str(tmp_path / "oracle"), "--candidates", str(tmp_path / "val"), "--use-validated",
          "--runs", str(run), "--out", str(tmp_path / "out")])
    for name in ("oracle_validation_by_world", "oracle_validation_candidates", "stage1_by_dataset", "stage2_by_dataset",
                 "gap_decomposition_rows", "gap_decomposition", "gap_decomposition_by_dataset"):
        assert (tmp_path / "out" / f"{name}.csv").exists(), name
    assert (tmp_path / "out" / "gap_decomposition.png").stat().st_size > 1000
    v = pd.read_csv(tmp_path / "out" / "oracle_validation_by_world.csv")
    assert (v["change"] == 0).all()  # the weak candidate cannot lower the bound
    g = pd.read_csv(tmp_path / "out" / "gap_decomposition_rows.csv")
    np.testing.assert_allclose(g["structural_gap"] + g["learning_gap"] + g["learned_repair_gain"], g["representation_loss"])
    s = pd.read_csv(tmp_path / "out" / "oracle_validation_summary.csv")
    assert set(s["dataset"]) == {"toy", "all"} and (s["change"] == 0).all()
    # default: the tables stay on the Stage 1 bounds, plus the decomposition on the validated ones as a sensitivity
    main(["followup", str(tmp_path / "oracle"), "--candidates", str(tmp_path / "val"), "--runs", str(run), "--out", str(tmp_path / "out2")])
    assert (tmp_path / "out2" / "gap_decomposition_validated_bound.csv").exists()
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "out2" / "gap_decomposition.csv"),
                                  pd.read_csv(tmp_path / "out" / "gap_decomposition.csv"))  # nothing to lift here
