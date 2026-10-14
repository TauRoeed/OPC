"""§11's per-family decision (training/analyze_adaptive.py family_decision): the highest mean native TrueGain among
stable variants; within one paired SE, the earlier in the pre-registered order (family C: the smallest γ, the ungated
C1 first at equal γ); and the estimator each arm's objective is built on (training_estimator)."""
import pandas as pd
import pytest

from training.analyze_adaptive import family_decision, training_estimator


def _worlds(gains: dict, diverged: dict | None = None) -> pd.DataFrame:
    rows = []
    for arm, g in gains.items():
        for i, x in enumerate(g):
            rows.append({"dataset": "ml", "bias": "s-small.r-strong", "seed": i, "arm": arm, "family": "C",
                         "native_gain": x, "diverged": (diverged or {}).get(arm, 0), "trials": 10})
    return pd.DataFrame(rows)


def _paired(worlds: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"arm": worlds["arm"], "vs": "B3", "rule": "native", "d": 0.0})


def test_a_tie_between_harmonic_crm_variants_takes_the_smallest_gamma():
    w = _worlds({"shared_lr_opc_hcrm2": [1.0, 1.2, 0.9, 1.1], "shared_lr_opc_hcrm1": [1.0, 1.1, 0.95, 1.08],
                 "shared_lr_opc_crm1": [0.5, 0.6, 0.4, 0.5]})
    assert family_decision(w, _paired(w), "C")["decision"] == "shared_lr_opc_hcrm1"


def test_a_clear_winner_is_kept_and_an_unstable_variant_is_excluded():
    w = _worlds({"shared_lr_opc_hcrm2": [2.0, 2.1, 2.2, 2.05], "shared_lr_opc_crm0.5": [0.5, 0.6, 0.4, 0.5],
                 "shared_lr_opc_hcrm0.5": [3.0, 3.1, 3.2, 3.05]}, diverged={"shared_lr_opc_hcrm0.5": 3})
    assert family_decision(w, _paired(w), "C")["decision"] == "shared_lr_opc_hcrm2"


def test_at_equal_gamma_the_ungated_crm_comes_first():
    w = _worlds({"shared_lr_opc_hcrm1": [1.0, 1.1, 0.9, 1.05], "shared_lr_opc_crm1": [1.0, 1.08, 0.92, 1.04]})
    assert family_decision(w, _paired(w), "C")["decision"] == "shared_lr_opc_crm1"


def test_an_arm_outside_the_order_is_an_error():
    w = _worlds({"shared_lr_opc_crm7": [1.0, 1.1, 0.9, 1.05], "shared_lr_opc_crm1": [1.0, 1.08, 0.92, 1.04]})
    with pytest.raises(ValueError):
        family_decision(w, _paired(w), "C")


def test_each_arm_maps_to_the_estimator_its_objective_is_built_on():
    assert training_estimator("opc") == "dr_h0.1"
    assert training_estimator("opc_h0.2") == "dr_h0.2"
    assert training_estimator("opc_raw") == "dr_raw"
    assert training_estimator("opc_s2b0.01") == "dr_raw" and training_estimator("opc_crm1") == "dr_raw"
    assert training_estimator("opc_hcrm1") == "dr_h0.1" and training_estimator("opc_d1") == "dr_h0.1"
    assert training_estimator("opc_oq") == "dr_oracle" and training_estimator("opc_a2") == "dr_hA2"
    assert training_estimator("likelihood") is None and training_estimator("value_star") is None
