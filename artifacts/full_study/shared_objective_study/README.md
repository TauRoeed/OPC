# Three training objectives on one global correction model: tables and figures

The report is `docs/shared_objective_study.md`. The run registry has `run_shared_tune_s200`,
`run_shared_tune_s200_supp`, `run_shared_oracles_20261006` and `run_shared_main_25k`. Every file here is rebuilt by
`python -m training.analyze_shared_objectives` from the run folders and the reused tables named below.

## `tuning/`: the first tuning round on the 18 tuning worlds (§11)

```bash
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200 --out artifacts/full_study/shared_objective_study/tuning
```

| file | content |
|---|---|
| `tuning_selected.csv` | per world and arm: the native, common-DR and best trial (configuration, greedy and stochastic value, λ), the mean of the 20 trials, the regrets, the selected trial's diagnostics |
| `tuning_summary.csv` | per arm, all worlds and per bias: mean gains of the three selections and of the 20 trials, the regrets, the selected λ |
| `tuning_marginals.csv` | per arm, dimension and bin: trials, distinct configurations, the mean gap to each world's best trial |
| `tuning_edges.csv` | the pre-registered edge rule per arm and dimension |
| `tuning_weights.csv` | per world and arm: the uniform-reference weights' profile on the training and validation rows |
| `tuning_trials_long.csv.gz` | every trial |

## `tuning_round2/` and `tuning_round2_supplement_only/`: the supplementary round (§11.2)

The same files: both rounds pooled (the decision), and the supplementary round alone (the check).

```bash
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200 artifacts/full_study/run_shared_tune_s200_supp --out artifacts/full_study/shared_objective_study/tuning_round2
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_shared_tune_s200_supp --out artifacts/full_study/shared_objective_study/tuning_round2_supplement_only
```

## `population/`: the four population optima on the 30 development worlds (§13)

```bash
python -m training.analyze_shared_objectives oracles --out artifacts/full_study/shared_objective_study/population
```

It reads the saved oracle policies of `run_class_oracles_20261005` (θ_value*, θ_log*) and `run_shared_oracles_20261006`
(θ_uniform*, θ_clip10*), and checks each optimum's greedy value against its oracle run.

| file | content |
|---|---|
| `oracles.csv` | per world and optimum: greedy value and gain, the value optimum's stochastic value, L_log, L_uniform and L_clip10 (the value optimum after its recalibration under L_log), M's deviation from a multiple of I, the item term's spread |
| `oracle_distances.csv` | per world and pair of optima: the cosine distance between their M, the prior-weighted share of users with the same greedy item |
| `oracle_summary.csv` | per bias and pooled: V(θ_value*) − V(θ_obj*) for each likelihood optimum and the differences between them, mean, 95% CI, worlds where the first is higher |
| `oracle_profile.csv`, `oracle_distance_summary.csv` | per bias and optimum (pair): the means and CIs of the columns above |

## `compare/`: the 25k comparison on the 30 development worlds (§14–§17)

```bash
python -m training.analyze_shared_objectives compare --runs artifacts/full_study/run_shared_main_25k --oracles artifacts/full_study/shared_objective_study/population --out artifacts/full_study/shared_objective_study/compare
```

The reused comparator rows (historical OPC, DM-only, CausE-cap-C at ρ = 0, BLOB-Pnorm-NQ) and the target-best ceilings
come from `../blob_prior_calibration/compare/table_conditions.csv`.

| file | content |
|---|---|
| `tables.md` | every table of §13–§17 |
| `table_conditions.csv` | per world and arm: the three selections, the population optima, the decomposition (objective mismatch, training gap, selection gaps), the capacity gap, the share of the value optimum's gain recovered, the weight profile; the comparator rows after them |
| `table_summary.csv` | per bias and pooled: mean and 95% CI of every column above |
| `table_paired.csv` | paired contrasts between arms by world, per selection rule (native, common, best of 20, mean of 20) |
| `table_per_dataset.csv` | per dataset (its 8 biased worlds): native and best-of-20 gains |
| `table_data_identity.csv` | per world: identical training and validation rows across the four arms, the same click sums as the BLOB run, the same logger as the comparator rows |
| `table_population*.csv` | the population tables, as in `population/` |
| `fig1_native_gain`, `fig1b_common_gain` | greedy gain over the logger per bias, native and common selection; dashed: the value optimum |
| `fig2_decomposition` | objective mismatch, training gap and selection gaps per arm, biased worlds |
| `fig3_population_mismatch` | V(θ_value*) − V(θ_obj*) per bias for the three likelihood optima |
| `trials_long.csv.gz` | every trial of the main grid |

Figures are PNG and PDF, with the plotted values in a CSV of the same name.
