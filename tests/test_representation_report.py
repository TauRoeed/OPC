"""The experimental report's figures and tables (training/representation_report.py) rebuild from the committed summary
tables alone, and their values are the source tables' values."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARIES = ROOT / "artifacts" / "full_study" / "summaries_20260927"
REPORT = ROOT / "artifacts" / "full_study" / "report_20260928"

pytestmark = pytest.mark.skipif(not (SUMMARIES / "stage2_learned_rows.csv").exists(), reason="committed summaries missing")


def _build(tmp_path, *, with_committed_raw_tables=False):
    from training.representation_report import main

    out = tmp_path / "report"
    out.mkdir()
    if with_committed_raw_tables:
        for name in ("table_propensity_earlier.csv", "table_weighting_study.csv"):
            if (REPORT / name).exists():
                shutil.copy(REPORT / name, out / name)
    main(["--summaries", str(SUMMARIES), "--runs", str(tmp_path / "no_runs"), "--out", str(out)])
    return out


def test_rebuilds_from_committed_tables(tmp_path):
    out = _build(tmp_path)
    for name in ("fig1_structural_recoverability", "fig2_fraction_of_oracle_repair", "fig3_opc_minus_dm", "fig4_gap_decomposition",
                 "fig6_logging_support", "fig7_per_dataset"):
        for ext in ("png", "pdf", "csv"):
            assert (out / f"{name}.{ext}").exists(), (name, ext)
    assert not (out / "fig5_propensity_value.png").exists()  # needs the earlier runs or their committed table
    text = (out / "tables.md").read_text()
    assert "### T1." in text and "### T5." in text and "### T9." in text and "### T6." not in text


def test_values_are_the_source_tables(tmp_path):
    out = _build(tmp_path)
    t1 = pd.read_csv(out / "fig1_structural_recoverability.csv")
    s1 = pd.read_csv(SUMMARIES / "stage1_recoverability_by_bias.csv", index_col=0)
    agg = t1[t1["dataset"] == "all"].set_index("bias")
    for b in agg.index:
        assert agg.loc[b, "structural recoverability"] == pytest.approx(s1.loc[b, "recoverability greedy"], abs=1e-4)
    val = pd.read_csv(SUMMARIES / "followup" / "oracle_validation_summary.csv")
    v = val[val["dataset"] == "all"].set_index("bias")
    for b in agg.index:
        assert agg.loc[b, "validated recoverability"] == pytest.approx(v.loc[b, "new recoverability greedy"], abs=1e-9)
    f3 = pd.read_csv(out / "fig3_opc_minus_dm.csv").set_index(["bias", "train_size"])
    p = pd.read_csv(SUMMARIES / "stage2_paired.csv")
    p = p[(p["contrast"] == "opc - dm") & (p["measure"] == "stochastic")].set_index(["bias", "train_size"])
    assert len(f3) == 18
    for k in f3.index:
        assert f3.loc[k, "mean"] == pytest.approx(p.loc[k, "mean"], abs=1e-9)
    f4 = pd.read_csv(out / "fig4_gap_decomposition.csv")
    np.testing.assert_allclose(f4["structural_gap %"] + f4["learning_gap %"] + f4["learned_repair_gain %"],
                               f4["representation_loss %"], atol=1e-9)
    f2 = pd.read_csv(out / "fig2_fraction_of_oracle_repair.csv")
    fr = pd.read_csv(SUMMARIES / "stage2_fractions.csv").set_index(["bias", "train_size", "method"])
    assert len(f2) == 45
    for _, r in f2.iterrows():
        assert r["fraction greedy"] == pytest.approx(fr.loc[(r["bias"], r["train_size"], r["method"]), "fraction greedy"], abs=1e-9)


def test_uses_committed_earlier_tables_without_run_folders(tmp_path):
    if not (REPORT / "table_propensity_earlier.csv").exists():
        pytest.skip("report tables not committed yet")
    out = _build(tmp_path, with_committed_raw_tables=True)
    assert (out / "fig5_propensity_value.png").exists()
    pd.testing.assert_frame_equal(pd.read_csv(out / "table_propensity_earlier.csv"), pd.read_csv(REPORT / "table_propensity_earlier.csv"))
    assert "### T6." in (out / "tables.md").read_text() and "### T7." in (out / "tables.md").read_text()
