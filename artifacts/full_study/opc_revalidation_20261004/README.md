# Simulator fix and OPC revalidation: artifacts (2026-10-04)

The results here come from the **fixed** logging simulator (dbc401b on `CRM`). The document they belong to is
[docs/simulator_fix_opc_revalidation_20261004.md](../../../docs/simulator_fix_opc_revalidation_20261004.md).

The run folders and their console logs stay local, as for every earlier run. They are listed with their code
commit, purpose and pairing in `../run_registry.csv`. Every Phase 2–3 run was made from a pinned worktree at
d791849, and each records `code_commit` and `search_space` in its `run_meta.json`. Everything below is regenerated
from the repository root.

## phase0/: what the buggy logs were

```bash
python -m training.logging_coupling_diagnostic --sample --logger-greedy-share 0.8 \
  --bias-configs none medium high high/none/none none/high/none none/none/high \
  --out artifacts/full_study/opc_revalidation_20261004/phase0/coupling_exact_lgs0.8.csv
python -m training.logging_coupling_diagnostic --sample --logger-greedy-share 0.6 --datasets ml kuairand \
  --bias-configs high/none/none none/high/none none/none/high \
  --out artifacts/full_study/opc_revalidation_20261004/phase0/coupling_exact_lgs0.6.csv   # and 0.95
```

`coupling_exact_lgs{0.6,0.8,0.95}.csv` has one row per world (seed 100) and logger share. Each row compares the
distribution the pre-fix sampler actually drew from (P_eff) with the logger π0:
- support, largest probability, collision probabilities (`coll_eff`, `coll_pi0`) and total variation;
- recorded vs generating propensity, and V under P_eff vs V(π0);
- the infinite-data IPS limits for three fixed targets;
- old vs fixed logs drawn from the production path: users always given the same action, the user–action index
  correlation, distinct actions.

The `.log` files hold each run's console output.

## tuning/: Phase 2

`weights_screen_s201.csv` is the training-weight screen of §2.7: 16 weightings, paired with raw DR on tuning seed
201. It holds the selected and per-trial differences, the diagnostics and the selection regret. It is built with
`training/analyze_revalidation.py` (`screen_table`).

## summaries/: Phase 3 (the corrected runs, in the old summaries' format)

```bash
python -m training.revalidation_phase3 --out artifacts/full_study/opc_revalidation_20261004/summaries
```

| folder | content |
|---|---|
| `stage2/` | `analyze_recoverability stage2` of the corrected Stage 2: one learned row per world × size × arm (`stage2_learned_rows.csv`); fractions, paired contrasts and diagnostics; the robustness arm (`opc_shrink100`) as `stage2_robust_shrink100_minus_default.csv` |
| `followup/` | the gap decomposition with the unchanged Stage 1 oracle (and the validated bounds), per dataset |
| `stage3_lgs_0_6/`, `stage3_lgs_0_95/` | the logging-support sweep at 25k; `stage3_lgs_0_8/` holds the corrected Stage 2's 25k single-type rows |
| `reward_model/` | learned rows of the external-q̂ (`learned_rows_external.csv`) and misspecified-q̂ (`learned_rows_concat.csv`) reruns |
| `decomposition/` | learned rows of the old search space on the corrected logs (OPC and DM-only, ml and kuairand) |
| `old/` | the old (buggy-log) counterparts not in `summaries_20260927`: the reward-model tests of the older pipeline and the Su robustness slice, rebuilt from their run folders |
| `m5/` | the OPC side of the CausE M5 comparison, read for the Phase 5 assessment only |

## report/: Phase 4 (old vs corrected)

```bash
python -m training.revalidation_report   # also writes report/tables.md (training/revalidation_tables.py)
```

- **Figures** `fig1`–`fig8` (PNG, PDF), each with its plotted values in a CSV of the same name.
- **Tables.**
  - `old_new_stage2*.csv`: per-world paired old-vs-new tables, each finding classified (`training/revalidation_compare.py`).
  - `findings_old_vs_new.csv`: every major finding of the old report.
  - `decomposition_simulator_vs_retuning.csv`: the simulator fix vs the retuning.
  - `reward_model_tests_{old,new}.csv`, `support_{old,new}.csv`: the reward-model tests and the logging-support
    sweep, old and new.
  - `phase5_m5_opc_side_vs_revalidated.csv`: the Phase 5 comparison.
- **`tables.md`**: every table of the document (R1–R14), rendered from the files above.

The report needs only the committed summaries (`summaries_20260927/`, `summaries/`, `tuning/`, `phase0/`).
`tests/test_revalidation_report.py` rebuilds it into a temporary folder and checks a finding against the per-world
rows.
