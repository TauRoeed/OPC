# CausE vs OPC, bounded development comparison (25k, 2026-10-04)

The tables and figures of `docs/cause_dev_report_20261004.md`, built from the local run folders `run_cause_dev_25k_opc_20261004/`
(OPC, DM-only, tempered logger) and `run_cause_dev_25k_cause_20261004/` (native CausE, ρ ∈ {0, .01, .05, .10, .15, .25}):

```bash
python -m training.analyze_cause_comparison --best-item --out artifacts/full_study/cause_dev_25k_20261004 \
    --runs artifacts/full_study/run_cause_dev_25k_opc_20261004 artifacts/full_study/run_cause_dev_25k_cause_20261004
```

| file | contents |
|---|---|
| `table_conditions.csv` | one row per condition (dataset, bias, seed) × arm × ρ: true greedy and stochastic values, gains over the logger, fraction of the mismatch repaired, fraction of OPC's linear-class oracle, budget and collection-reward fields |
| `table_summary.csv` | mean and 95% CI over the 6 dataset × seed conditions per bias, arm and ρ |
| `table_paired.csv` | OPC − CausE (greedy gain, CTR points), paired over dataset × seed, per bias, CausE prediction and ρ |
| `table_best_single_item.csv` | each world's best single-item value (the best policy without personalization) |
| `fig_rho_curve.{png,pdf,csv}` | target performance vs randomized share ρ, per mismatch family |
| `fig_exploration_cost.{png,pdf,csv}` | final target value vs expected clicks given up during collection |

Every run is a development run (seeds 100, 101), on the fixed log simulator (local commit `5c011a9`).
