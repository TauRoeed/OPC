"""Arithmetic of the CausE vs OPC analysis (training/analyze_cause_comparison.py) on a toy condition table."""
import numpy as np
import pandas as pd
import pytest

from training.analyze_cause_comparison import condition_table, paired_table, summary_table


def _toy():
    rows = []
    for ds, seed, base in (("ml", 100, 0.20), ("ml", 101, 0.22)):
        common = {"dataset": ds, "bias": "w-high.g-none.v-none", "seed": seed, "train_size": 25000, "initial_reward": base - 0.01}
        rows.append({**common, "method": "opc", "arm": "opc", "rho": np.nan, "policy_rewards": base + 0.02,
                     "policy_rewards_greedy": base + 0.04})
        for rho, g in ((0.0, base - 0.05), (0.1, base - 0.03)):
            rows.append({**common, "method": f"cause_avg_r{int(rho*1000):03d}", "arm": "avg", "rho": rho,
                         "policy_rewards": g - 0.01, "policy_rewards_greedy": g, "exploration_cost_expected": rho * 1000.0})
    bounds = pd.DataFrame({"dataset": ["ml", "ml"], "bias": ["w-high.g-none.v-none"] * 2, "seed": [100, 101],
                           "ceiling": [0.30, 0.32], "V_logger_greedy": [0.20, 0.22], "oracle_linear_greedy": [0.28, 0.30]})
    return pd.DataFrame(rows), bounds


def test_gains_and_fractions():
    df, bounds = _toy()
    t = condition_table(df, bounds)
    opc = t[t.arm == "opc"].sort_values("seed")
    assert opc.gain_greedy.tolist() == pytest.approx([0.04, 0.04])
    assert opc.frac_mismatch.tolist() == pytest.approx([0.04 / 0.10, 0.04 / 0.10])
    assert opc.frac_opc_oracle.tolist() == pytest.approx([0.04 / 0.08, 0.04 / 0.08])
    assert opc.gain.tolist() == pytest.approx([0.03, 0.03])  # stochastic gain over V_logger


def test_paired_differences_and_summary():
    df, bounds = _toy()
    t = condition_table(df, bounds)
    p = paired_table(t)
    row = p[(p.cause == "avg") & np.isclose(p.rho, 0.0)].iloc[0]
    assert row["OPC minus CausE (pts)"] == pytest.approx(9.0) and row["n"] == 2 and row["OPC better in"] == 2
    s = summary_table(t)
    assert s[(s.arm == "avg") & np.isclose(s.rho.astype(float), 0.1)]["gain_greedy"].iloc[0] == pytest.approx(-0.03)
