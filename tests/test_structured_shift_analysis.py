"""The structured-shift study's analysis (training/analyze_structured_shift.py): the decomposition identity, the paired
differences, the expansion gate and the sanity-check filter, on small synthetic frames."""
import numpy as np
import pandas as pd

from training.analyze_structured_shift import (
    expansion_gate,
    harmonic_dominance,
    paired_cells,
    paired_worlds,
    population_cells,
    population_checks,
    world_table,
)


def _trials(world=("ml", "s-moderate.r-moderate", 100), arms=("shared_lr_likelihood", "shared_lr_opc"), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for arm in arms:
        greedy = 0.20 + 0.05 * rng.random(4)
        native_col = rng.random(4)
        best = int(np.argmin(native_col)) if "likelihood" in arm else int(np.argmax(native_col))
        for k in range(4):
            rows.append({"dataset": world[0], "bias": world[1], "seed": world[2], "method": arm, "trial_number": k,
                         "is_best_in_run": k == best, "actual_reward_greedy": greedy[k], "actual_reward": greedy[k] - 0.01,
                         "gain_greedy": 100 * (greedy[k] - 0.18), "V_logger_greedy": 0.18, "V_logger": 0.15,
                         "param_anchor_lambda": 0.01, "diag_dr_greedy_low": rng.random(),
                         "diag_val_nll": native_col[k], "value": native_col[k], "diverged": False,
                         **{c: 1.0 for c in ("param_lr", "param_num_epochs", "param_batch_size", "param_lr_decay")}})
    return pd.DataFrame(rows)


def _pop(world=("ml", "s-moderate.r-moderate", 100)):
    r = {"dataset": world[0], "bias": world[1], "seed": world[2], "shift": "moderate", "response": "moderate",
         "Vg_source": 0.18, "Vg_truth": 0.27, "Vg_value": 0.269, "Vg_likelihood": 0.255, "Vg_calib": 0.269,
         "Vg_harmonic": 0.25}
    r.update({"M_L": 100 * (r["Vg_value"] - r["Vg_likelihood"]), "M_calib": 0.0,
              "M_harm": 100 * (r["Vg_value"] - r["Vg_harmonic"]), "optimizer_gap": 0.1, "available_gain": 9.0,
              "qinf_rmse_logging": 0.05, "qinf_rmse_target": 0.1})
    r["M_L_minus_M_calib"] = r["M_L"] - r["M_calib"]
    r["M_harm_minus_M_L"] = r["M_harm"] - r["M_L"]
    return pd.DataFrame([r])


def test_decomposition_identity_and_population_mapping():
    wt = world_table(_trials(), pd.DataFrame(), _pop())
    np.testing.assert_allclose(wt["gap_native"], wt["M"] + wt["EO"] + wt["S_native"], atol=1e-12)
    np.testing.assert_allclose(wt["gap_common"], wt["M"] + wt["EO"] + wt["S_common"], atol=1e-12)
    m = wt.set_index("short")["M"]
    np.testing.assert_allclose(m["likelihood"], 1.4, atol=1e-9)  # θ_lik*
    np.testing.assert_allclose(m["opc"], 1.9, atol=1e-9)  # harmonic OPC: θ_harm*
    np.testing.assert_allclose(wt["logger_check"], 0.0, atol=1e-12)


def test_paired_differences_by_rule():
    wt = world_table(_trials(), pd.DataFrame(), _pop())
    pw = paired_worlds(wt)
    idx = wt.set_index("short")
    for rule, col in (("native", "native_gain"), ("best", "best_gain"), ("mean", "trial_mean_gain")):
        d = pw[(pw["a"] == "opc") & (pw["b"] == "likelihood") & (pw["rule"] == rule)]["d"].iloc[0]
        np.testing.assert_allclose(d, idx.at["opc", col] - idx.at["likelihood", col])


def _cells(m_l, m_calib, m_harm, opc_lik, calib_lik_ci):
    pop = pd.concat([_pop((ds, f"s-moderate.r-{r}", 100)).assign(response=r, dataset=ds, M_L=m_l[r],
                                                                  M_calib=m_calib, M_harm=m_harm)
                     for r in ("none", "moderate", "strong") for ds in ("ml", "kuairand")], ignore_index=True)
    pop["M_L_minus_M_calib"] = pop["M_L"] - pop["M_calib"]
    pop["M_harm_minus_M_L"] = pop["M_harm"] - pop["M_L"]
    cells = population_cells(pop)
    rows = []
    for r in ("none", "moderate", "strong"):
        rows.append({"scope": "cell", "dataset": "all", "shift": "moderate", "response": r, "a": "opc",
                     "b": "likelihood", "rule": "native", "a_minus_b": opc_lik[r], "ci_lo": opc_lik[r] - 1,
                     "ci_hi": opc_lik[r] + 1, "worlds": 2, "a_higher": 0})
        lo, hi = calib_lik_ci[r]
        rows.append({"scope": "cell", "dataset": "all", "shift": "moderate", "response": r, "a": "likelihood_calib",
                     "b": "likelihood", "rule": "native", "a_minus_b": (lo + hi) / 2, "ci_lo": lo, "ci_hi": hi,
                     "worlds": 2, "a_higher": 0})
    return pop, cells, pd.DataFrame(rows)


def test_expansion_gate_criteria():
    pop, cells, pc = _cells({"none": 0.0, "moderate": 1.0, "strong": 3.0}, 0.0, 2.0,
                            {"none": -0.5, "moderate": -0.2, "strong": 0.4}, {"none": (-0.1, 0.1),
                                                                              "moderate": (-0.1, 0.3),
                                                                              "strong": (0.2, 0.9)})
    g = expansion_gate(pc, cells).set_index("response")
    assert list(g["a"]) == [False, True, True]  # M_L − M_calib ≥ 0.5
    assert g["b"].all()  # OPC − likelihood changes sign across response levels at this shift
    assert list(g["c"]) == [False, False, False]  # M_harm < M_L only at strong, where OPC does not lose
    assert list(g["d"]) == [False, False, True]  # the calibration-aware head's CI excludes 0
    assert g["expand"].all()


def test_harmonic_dominance_and_check_filter():
    pop, _cells_, _pc = _cells({"none": 0.0, "moderate": 1.0, "strong": 3.0}, 0.0, 2.0,
                               {"none": 0, "moderate": 0, "strong": 0}, {r: (0, 0) for r in ("none", "moderate", "strong")})
    ac = pd.DataFrame([{"scope": "all", "dataset": "all", "arm": a, "gap_native": g}
                       for a, g in (("likelihood", 2.0), ("opc", 3.0), ("opc_raw", 2.5), ("opc_oq", 9.0))])
    d = harmonic_dominance(pop, ac)
    assert d["largest_gap_arm"] == "opc"  # the oracle-q arm is excluded
    assert np.isclose(d["share_worlds_M_harm_ge_M_L"], 4 / 6) and d["dominant"]
    checks = pop.assign(check_A=[None, True, True, True, True, True], check_B=[True, True, False, True, True, True])
    assert len(population_checks(checks)) == 1
    assert len(paired_cells(paired_worlds(world_table(_trials(), pd.DataFrame(), _pop())))) > 0
