# OPC adaptive estimators: tables and figures

The pre-registration and report is `docs/opc_adaptive_estimator_study.md`. Every run it uses is in
`artifacts/full_study/run_registry.csv` (`run_ae_*`). Every file here is rebuilt from the run folders by the commands
below, from the repository root with the BPR embeddings in `BPR/embeddings`. World builds pin 4 CPU threads; analyses
run with BLAS threads capped (`OPENBLAS_NUM_THREADS=2`).

## `world_scan/`: the worlds before any method (§5.6)

```bash
python scripts/ss_world_scan.py --shifts small --seeds 100 101 300 301 --out artifacts/full_study/opc_adaptive_estimator/world_scan/world_scan_small.csv
python scripts/ss_world_scan.py --shifts none moderate strong --seeds 300 301 --out artifacts/full_study/opc_adaptive_estimator/world_scan/world_scan_pilot_seeds.csv
```

| file | content |
|---|---|
| `world_scan_small.csv` | the new small shift level (ρ = 0.95) × response none / moderate / strong on the development and pilot seeds: 36 worlds, all calibrated, none pathological |
| `world_scan_pilot_seeds.csv` | the pilot seeds 300 and 301 at the existing levels: 54 worlds, all calibrated, none pathological |

The confirmation seeds (900–903) are never scanned before Stage F.

## `stage_a_R50/` (and `stage_a_R200/`): the fixed-policy oracle evaluator diagnostics (§8, §11 Stage A)

```bash
python -m training.oracle_eval --datasets ml --seeds 100 101 --labels s-moderate.r-strong s-moderate.r-moderate s-none.r-strong s-strong.r-strong --replicates 50 --run artifacts/full_study/run_ss_main_25k --population artifacts/full_study/run_ss_population --out artifacts/full_study/run_ae_stage_a
# (one job per dataset; then the same with --replicates 200)
python -m training.analyze_adaptive stage_a --run artifacts/full_study/run_ae_stage_a --replicates 50 --trials artifacts/full_study/run_ss_main_25k --out artifacts/full_study/opc_adaptive_estimator/stage_a_R50
```

Per world: the selection-time reward model (the full-data model of the world's 25k training rows) and 14 fixed
policies (the five Stage-1 arms' selected policies, the source and θ_value*, each as softmax and greedy). The linear
estimators' moments are exact. R fresh logs of 25,000 rows, shared by every policy and estimator of a world.

| file | content |
|---|---|
| `table_world_policy_estimator.csv` | per world, policy and estimator: the replicates' bias (± SE), variance (χ² interval), MSE, RMSE, absolute-error quantiles, coverage of the plug-in interval, the same for the paired improvement over the source; the exact moments (`exact_*`, linear estimators); the replicates against them (`bias_z`, `var_ratio`) |
| `table_world_policy_support.csv` | per world and policy: the true value and gain over the source, exact S2 and ESS fraction, the target mass on μ < 10⁻⁴ / 10⁻³ and on w > 10 / 100, the reward model's RMSE under the policy and the logger, A2's population λ |
| `table_world_policy_log_stats.csv` | per world and policy, medians over the logs: A2's and A3's chosen λ, the maximum weight and its 99th / 99.9th percentiles, the Kish ESS fraction |
| `regime_summary.csv` | per regime (pooled and per dataset), policy kind, arm and estimator: mean bias, SD, RMSE (points; exact for linear estimators), coverage, the improvement's bias and RMSE |
| `replicate_checks.csv` | the replicates against the exact moments; `ok` for bounded-weight estimators |
| `delta_paired_vs_independent.csv` | the paired improvement's exact variance against two independent logs', and the correlation of the two estimates |
| `adaptivity.csv` | per world and arm: the selected trial's validation estimate minus its true value, the mean over all trials, the evaluator's exact conditional bias at the selected policy, and the difference (selection adaptivity) |
| `tables.md` | the tables as markdown |
| `fig_lambda_curve_softmax.*`, `fig_lambda_curve_greedy.*` | harmonic DR's exact bias, SD and RMSE against λ at fixed policies, per regime |
