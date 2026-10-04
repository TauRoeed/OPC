"""The revalidation report (training/revalidation_report.py and training/revalidation_tables.py) rebuilds from the
committed summaries alone, and its old-vs-new numbers are paired per-world differences of the summary rows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "artifacts" / "full_study" / "summaries_20260927"
REVAL = ROOT / "artifacts" / "full_study" / "opc_revalidation_20261004"
NEW = REVAL / "summaries"

pytestmark = pytest.mark.skipif(
    not (OLD / "stage2_learned_rows.csv").exists() or not (NEW / "stage2" / "stage2_learned_rows.csv").exists(),
    reason="committed summaries missing")

FIGURES = ("fig1_recovery_old_vs_new", "fig2_corrected_arms", "fig3_opc_minus_dm_old_vs_new", "fig4_weights_ess",
           "fig5_reward_model_tests", "fig6_support_old_vs_new", "fig7_conclusions")


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    from training.revalidation_report import main

    out = tmp_path_factory.mktemp("report")
    main(["--old", str(OLD), "--new", str(NEW), "--tuning", str(REVAL / "tuning"), "--phase0", str(REVAL / "phase0"),
          "--out", str(out)])
    return out


def test_rebuilds_from_committed_summaries(report):
    for name in FIGURES + (("fig8_simulator_vs_retuning",) if (NEW / "decomposition").exists() else ()):
        for ext in ("png", "pdf", "csv"):
            assert (report / f"{name}.{ext}").exists(), (name, ext)
    text = (report / "tables.md").read_text()
    for k in range(1, 14):
        assert f"### R{k}." in text, k


def test_findings_are_paired_world_differences(report):
    f = pd.read_csv(report / "findings_old_vs_new.csv").set_index("finding")
    keys = ["dataset", "bias", "seed", "train_size"]

    def opc_minus_dm(rows):
        r = rows[(rows["train_size"] == 25000) & (rows["bias"] != "none")]
        return 100 * (r[r["method"] == "opc"].set_index(keys)["V_method"] - r[r["method"] == "dm"].set_index(keys)["V_method"])

    old = opc_minus_dm(pd.read_csv(OLD / "stage2_learned_rows.csv")).dropna()
    new = opc_minus_dm(pd.read_csv(NEW / "stage2" / "stage2_learned_rows.csv")).dropna()
    common = old.index.intersection(new.index)
    row = f.loc["OPC − DM-only, biased, 25k"]
    assert row["worlds"] == len(common)
    assert np.isclose(row["old"], old.loc[common].mean()) and np.isclose(row["new"], new.loc[common].mean())
    assert np.isclose(row["change"], (new.loc[common] - old.loc[common]).mean())
