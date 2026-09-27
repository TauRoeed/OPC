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
    warp = df.loc["high/none/none"]
    assert warp["V_clean"] == pytest.approx(0.24) and warp["V_clean_greedy"] == pytest.approx(0.30)
    assert warp["representation_loss"] == pytest.approx(0.06) and warp["representation_loss_greedy"] == pytest.approx(0.08)
    assert warp["gain_linear+scale_greedy"] == pytest.approx(0.07)
    assert warp["recoverability_linear+scale_greedy"] == pytest.approx(0.875)
    assert warp["recoverability_linear+scale"] == pytest.approx(0.12 / 0.06)  # above 1: not clipped
    none = df.loc["none"]
    assert none["representation_loss_greedy"] == 0 and np.isnan(none["recoverability_linear+scale_greedy"])  # no ratio
    assert df.loc["none/none/high", "recoverability_linear+scale_greedy"] == pytest.approx(0.03 / 0.09)
    assert df.loc["high/none/none", "recoverability_scale_greedy"] == pytest.approx(0.0)  # sharpening keeps the ranking
    assert MIN_LOSS > 0


def test_summary_and_learned_join():
    df = derive(_rows())
    s = summary(df)
    assert list(s.index) == ["none", "high/none/none", "none/none/high"]
    assert s.loc["high/none/none", "gain greedy %"] == pytest.approx(7.0)
    learned = pd.DataFrame([dict(dataset="ml", bias="high/none/none", seed=0, method="opc", train_size=5000,
                                 V_method=0.24, V_method_greedy=0.255)])
    m = learned_recovery(learned, _rows()).iloc[0]
    assert m["learned_gain"] == pytest.approx(0.06) and m["fraction_of_oracle_repair"] == pytest.approx(0.06 / 0.12)
    assert m["fraction_of_oracle_repair_greedy"] == pytest.approx(0.035 / 0.07)
