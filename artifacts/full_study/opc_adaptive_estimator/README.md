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

## `stage_b_{A,B,C,D}/`, `stage_b_C_C2/`: the Stage B pilots (§11, §16.3–§16.6)

```bash
python -m training.run_full_study_parallel --run-tag ae_pilot_A --world-family structured_shift --ctr-levels 0.05 --n-trials 10 --sampler random --stage development --slim --learn-logit-scale --lr-range 1e-4 6.3246e-3 --epochs-range 5 30 --seeds 300 301 --datasets ml kuairand --bias-configs s-small.r-strong s-moderate.r-strong s-moderate.r-moderate --train-sizes 25000 --methods <the family's arms> --max-workers 3 --save-policies
# (then --datasets anime; the same for ae_pilot_D, ae_pilot_B, ae_pilot_C and the gated ae_pilot_C2)
python -m training.analyze_adaptive stage_b --runs artifacts/full_study/run_ae_pilot_C artifacts/full_study/run_ae_pilot_C2 artifacts/full_study/run_ae_pilot_A --out artifacts/full_study/opc_adaptive_estimator/stage_b_C_C2
```

| file | content |
|---|---|
| `table_worlds.csv` | per world and arm: native / best / trial-mean TrueGain (points), recovery, divergences, the extension's diagnostics (λ, S2, ν) |
| `paired_worlds.csv`, `paired_summary.csv` | arm − B3 / B1 / B2 per world and rule; mean [95% CI] and worlds higher, pooled, per regime, dataset and regime × dataset |
| `decisions.json`, `tables.md` | §11's per-family decision and the tables |

## `stage_b_*_oracle/`, `stage_c_oracle/`: the oracle evaluator at the selected policies (§8)

```bash
python -m training.oracle_eval --datasets ml --seeds 300 301 --labels s-small.r-strong s-moderate.r-strong s-moderate.r-moderate --replicates 50 --stage stage_b --run artifacts/full_study/run_ae_pilot_B artifacts/full_study/run_ae_pilot_A --arms <arms> --population artifacts/full_study/run_ae_population --out artifacts/full_study/run_ae_oracle_ae_pilot_B
# (--reward-model lowrank | hetero for family D's q̂ variants)
python -m training.oracle_eval --datasets ml --seeds 100 101 --labels s-small.r-strong s-moderate.r-strong s-moderate.r-moderate s-moderate.r-none s-none.r-strong s-strong.r-strong --replicates 200 --stage stage_c --run artifacts/full_study/run_ae_main_ABD artifacts/full_study/run_ae_main_C artifacts/full_study/run_ae_main_R1_baselines artifacts/full_study/run_ss_main_25k --arms shared_lr_likelihood shared_lr_likelihood_calib shared_lr_opc shared_lr_opc_raw shared_lr_opc_oq shared_lr_opc_h0.2 shared_lr_opc_s2b0.01 shared_lr_opc_hcrm1 shared_lr_opc_d1 --population artifacts/full_study/run_ae_population --out artifacts/full_study/run_ae_oracle_stage_c
python -m training.analyze_adaptive stage_a --run artifacts/full_study/run_ae_oracle_stage_c --replicates 200 --trials artifacts/full_study/run_ae_main_ABD artifacts/full_study/run_ae_main_C artifacts/full_study/run_ae_main_R1_baselines artifacts/full_study/run_ss_main_25k --out artifacts/full_study/opc_adaptive_estimator/stage_c_oracle
```

The files are those of `stage_a_*`; `tables.md` adds every evaluated arm, each arm's selection and training
estimator at its selected policy, and (Stage C) the adaptivity of selection over the four training runs.

## `gradients/`: the oracle gradient diagnostics (§10, §10a)

`python -m training.ae_gradients` (defaults: ml, kuairand × seed 300 × R1, R2, B3's selected policy) into
`run_ae_gradients`, then `python -m training.analyze_adaptive gradients --run artifacts/full_study/run_ae_gradients --out artifacts/full_study/opc_adaptive_estimator/gradients`.

## `stage_c/`: the controlled main comparison (§11, §16.9)

```bash
python -m training.run_full_study_parallel --run-tag ae_main_ABD ... --n-trials 20 --seeds 100 101 --bias-configs s-small.r-strong s-moderate.r-strong s-moderate.r-moderate s-moderate.r-none s-none.r-strong s-strong.r-strong --methods shared_lr_opc_h0.2 shared_lr_opc_s2b0.01 shared_lr_opc_d1 --save-policies
# (ae_main_C: shared_lr_opc_hcrm1; ae_main_R1_baselines: the five baselines on s-small.r-strong; the launchers are in artifacts/full_study/ae_logs/launchers/)
python -m training.analyze_adaptive stage_c --runs artifacts/full_study/run_ae_main_ABD artifacts/full_study/run_ae_main_C artifacts/full_study/run_ae_main_R1_baselines artifacts/full_study/run_ss_main_25k --out artifacts/full_study/opc_adaptive_estimator/stage_c
```

The 36 development worlds only (`run_ss_main_25k` holds more); the files are those of `stage_b_*`.

## `stage_b_population/`, `stage_c_population/`: the population objectives (§9)

```bash
python -m training.ae_population --datasets ml --seeds 100 101 --labels <labels> --out artifacts/full_study/run_ae_population
python -m training.analyze_adaptive population --population artifacts/full_study/run_ae_population --seeds 100 101 --out artifacts/full_study/opc_adaptive_estimator/stage_c_population
```

`table_worlds.csv`: per world, M = V_g(θ_value*) − V_g(θ_obj*) (points) for the likelihood, the calibration-aware
likelihood, harmonic λ = 0.005–0.2, harmonic 0.1 with the D1 / D2 population fits, the S2 penalty path and caps; the
optima's S2; the available gain. Stage C's R2–S1 base optima are `run_ss_population`'s (copied; bit-identical when
recomputed).
