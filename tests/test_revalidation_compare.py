"""The revalidation's old-vs-new classification (training/revalidation_compare.py) and the simulator-vs-retuning
decomposition (training/revalidation_report.decomposition_table)."""

import numpy as np
import pandas as pd
import pytest

from training.revalidation_compare import KEYS, classify, old_new_table


@pytest.mark.parametrize("old, new, change, verdict", [
    ((1.0, 0.5, 1.5), (1.1, 0.6, 1.6), (0.1, -0.2, 0.4), "unchanged"),
    ((0.1, -0.5, 0.7), (0.2, -0.4, 0.8), (0.1, -0.2, 0.4), "unchanged"),  # not significant before or after
    ((1.0, 0.5, 1.5), (2.0, 1.5, 2.5), (1.0, 0.6, 1.4), "same direction, different magnitude"),
    ((1.0, 0.5, 1.5), (0.4, -0.1, 0.9), (-0.6, -1.0, -0.2), "weakened"),
    ((1.0, 0.5, 1.5), (1.2, -0.3, 2.7), (0.2, -0.8, 1.2), "unchanged size, less precise"),
    ((1.0, 0.5, 1.5), (-0.2, -0.7, 0.3), (-1.2, -1.6, -0.8), "unsupported"),
    ((1.0, 0.5, 1.5), (-1.0, -1.5, -0.5), (-2.0, -2.5, -1.5), "reversed"),
    ((0.2, -0.3, 0.7), (-1.0, -1.5, -0.5), (-1.2, -1.6, -0.8), "reversed"),  # significant only after, opposite sign
    ((0.2, -0.3, 0.7), (1.0, 0.5, 1.5), (0.8, 0.3, 1.3), "new"),
    ((2.0, 1.5, 2.5), (2.2, 1.7, 2.7), (0.2, 0.1, 0.3), "unchanged"),  # significant but below 20% of the effect
    ((1.0, 0.5, 1.5), (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), "no data"),
])
def test_classify(old, new, change, verdict):
    assert classify(old, new, change) == verdict


def test_an_absolute_materiality_threshold_for_log_scale_findings():
    old, new, change = (3.2, 3.1, 3.3), (2.9, 2.8, 3.0), (-0.3, -0.45, -0.15)  # log10 ESS halved
    assert classify(old, new, change) == "unchanged"  # under 20% of 3.2
    assert classify(old, new, change, material_abs=np.log10(1.2)) == "same direction, different magnitude"


def _rows(values: dict, sizes=(5000, 25000), seeds=(100, 101), datasets=("ml", "kuairand")) -> pd.DataFrame:
    """Learned rows: one row per world × arm with V_method = values[arm](world index)."""
    out, i = [], 0
    for d in datasets:
        for s in seeds:
            for n in sizes:
                for arm, f in values.items():
                    out.append(dict(dataset=d, bias="high", seed=s, train_size=n, method=arm, V_method=f(i)))
                i += 1
    return pd.DataFrame(out)


def test_old_new_table_pairs_by_world():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.001, 64)
    old = _rows({"opc": lambda i: 0.10 + noise[i], "dm": lambda i: 0.09})
    new = _rows({"opc": lambda i: 0.12 + noise[i], "dm": lambda i: 0.09})
    t = old_new_table(old, new, {"OPC - DM": ("opc", "dm", "V_method")})
    assert set(t["train_size"]) == {5000, 25000} and (t["worlds"] == 4).all()
    assert np.allclose(t["change"], 2.0) and np.allclose(t["change_lo"], 2.0) and np.allclose(t["change_hi"], 2.0)
    assert (t["verdict"] == "same direction, different magnitude").all()


def test_decomposition_effects_add_up_and_are_paired():
    from training.revalidation_report import decomposition_table

    rng = np.random.default_rng(1)
    world = rng.normal(0, 0.02, 64)  # large world-to-world spread, cancelled by the pairing
    old = _rows({"opc": lambda i: 0.10 + world[i], "dm": lambda i: 0.09 + world[i]})
    mid = _rows({"opc": lambda i: 0.11 + world[i], "dm": lambda i: 0.09 + world[i]})  # simulator: OPC +1 point
    new = _rows({"opc": lambda i: 0.115 + world[i], "dm": lambda i: 0.095 + world[i]})  # retuning: both +0.5
    mid = mid[~((mid["dataset"] == "ml") & (mid["seed"] == 101))]  # a world missing from one run is dropped everywhere
    t = decomposition_table(old, mid, new).set_index(["quantity", "train_size"])
    assert (t["worlds"] == 3).all()
    for q, sim, conf in (("OPC", 1.0, 0.5), ("DM-only", 0.0, 0.5), ("OPC - DM-only", 1.0, 0.0)):
        for n in (5000, 25000):
            r = t.loc[(q, n)]
            assert r["simulator"] == pytest.approx(sim) and r["configuration"] == pytest.approx(conf)
            assert r["total"] == pytest.approx(sim + conf)
            assert r["simulator_hi"] - r["simulator_lo"] == pytest.approx(0, abs=1e-9)  # paired: no world spread
    assert set(KEYS) <= set(old.columns)
