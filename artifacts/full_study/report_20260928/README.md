# Experimental report: figures and source tables (2026-09-28)

> **Historical: buggy logging simulator.**
> - **The bug.** The learned results here were produced by the logging simulator of commits 69fffab..c11b2b3
>   (2026-09-24 to 2026-10-04). In it, each logged action reused the random draw that picked its user. A user
>   therefore received a nearly fixed action, while the stored propensity was the logger's softmax probability.
> - **Scope.** Figure 1 (Stage 1) is unaffected; the figures of learned results are affected.
> - **Superseded by** [the revalidation on the fixed simulator](../../../docs/simulator_fix_opc_revalidation_20261004.md).
> - **This document** is kept unchanged as the record of what was run and concluded at the time.

These are the figures and tables of
[docs/representation_repair_experimental_report_20260928.md](../../../docs/representation_repair_experimental_report_20260928.md).
Everything is built by one script from the repository root:

```bash
python -m training.representation_report --out artifacts/full_study/report_20260928
```

- **Inputs:**
  - the committed summaries in `artifacts/full_study/summaries_20260927/`;
  - for the earlier development tests (Tables 6 and 7, Figure 5), the local run folders listed in
    `artifacts/full_study/run_registry.csv`.
- **Without the run folders,** the committed `table_propensity_earlier.csv` and `table_weighting_study.csv` in
  this folder are used instead, so every figure rebuilds from committed tables alone.
- **Tests:** `tests/test_representation_report.py` checks the rebuild and that plotted values equal the source
  tables.

| file | contents |
|---|---|
| `fig1_structural_recoverability.{png,pdf,csv}` | Stage 1 greedy structural recoverability per dataset and bias; Stage 1 vs validated bound |
| `fig2_fraction_of_oracle_repair.*` | Stage 2 fraction of the oracle ranking repair vs training size (OPC, DM-only, no-propensity) |
| `fig3_opc_minus_dm.*` | OPC − DM-only in true CTR by bias and size (paired, 95% CI) |
| `fig4_gap_decomposition.*` | representation loss = structural gap + learning gap + learned repair (greedy) |
| `fig5_propensity_value.*` | OPC − DM-only under the earlier reward-model settings (external 50k, budget-fair, misspecified) |
| `fig6_logging_support.*` | Stage 3 logger-share sweep: oracle bound, OPC fraction, OPC − DM-only, ESS |
| `fig7_per_dataset.*` | per-dataset OPC fraction and OPC − DM-only |
| `table_stage1.csv` … `table_stage3.csv` | source tables (per dataset and overall) |
| `table_propensity_earlier.csv` | Table 6, derived from `run_logger_explore`, `run_logger_explore_budget`, `run_qhat_concat` |
| `table_weighting_study.csv` | Table 7, derived from the `run_gradcmp_*`, `run_replay_*` and `run_final_dr_harmonic_*` runs |
| `tables.md` | all report tables in markdown |

Every figure's CSV holds exactly the plotted values. All runs are development runs (seeds 100/101).
