# The OPC gradient and regime study: tables and figures

The report is `docs/opc_gradient_regime_study.md`. Every run it uses is in `artifacts/full_study/run_registry.csv`
(`run_opc_*`). Every file here is rebuilt from the run folders by the commands below. The diagnostic folders
`run_opc_gradients_diag_*` (§12.4) belong to no analysis family. All analyses run with BLAS threads capped
(`--threads`); world rebuilds pin 4 CPU threads, as the runs did.

## `tuning_raw/`: the raw-DR arm on the 18 tuning worlds (§7–§8)

```bash
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_opc_raw_tune_s200 --arms shared_opc_raw --out artifacts/full_study/opc_gradient_regime/tuning_raw
```

The files are as in `../shared_objective_study/tuning/`. The edge rule extended no dimension.

## `support/`: the logging-support levels (§5)

```bash
python -m training.opc_support_levels --states-run artifacts/full_study/run_opc_gradients_8a --emb-dir BPR/embeddings --out artifacts/full_study/opc_gradient_regime/support
```

| file | content |
|---|---|
| `support_diagnostics.csv` | per world and candidate logger greedy share: whether the click model is unchanged (it is, in all 105 rows); effective items per user; propensity quantiles; the population ESS share and low-π0 mass of the fixed 8A targets |
| `support_summary.csv` | per share: the geometric means over worlds |
| `support_levels.json` | the levels by the pre-registered rule: poor 0.9, current 0.8, better 0.6 |

## `gradients_8a/` and `gradients_regime/`: the gradient benchmark (§4, §8, §12.2–§12.4)

```bash
python -m training.opc_visible_gradients --runs artifacts/full_study/run_opc_gradients_8a artifacts/full_study/run_opc_gradients_N*_lgs*
python -m training.analyze_opc_gradients --runs artifacts/full_study/run_opc_gradients_8a --out artifacts/full_study/opc_gradient_regime/gradients_8a --threads 4
python -m training.analyze_opc_gradients --runs artifacts/full_study/run_opc_gradients_8a artifacts/full_study/run_opc_gradients_N*_lgs* --out artifacts/full_study/opc_gradient_regime/gradients_regime --threads 4
```

The first command writes the supported-region gradients (`gstar_visible.*`) into the low-overlap worlds of the run
folders (§12.3).

| file | content |
|---|---|
| `table_gradients_world.csv` | per world, cell (N, support), state and estimator G1–G5: debiased relative bias (to the state's and to the source's ‖g*‖) with its Monte Carlo floor, bias ratio, calibrated p-values (Satterthwaite, split Hotelling, along g*), SNR, cosines, P(⟨ĝ, g*⟩ > 0), norm ratio, variance, MSE, minibatch error, weights, the reward model's error, bootstrap intervals; G5's bias paired against G3 and conditional on q̂ |
| `table_gradients_summary.csv` | per cell, state and estimator: mean and 95% t-interval over worlds, per bias, over the biased worlds, over all, and per dataset |
| `table_unbiasedness.csv` | §11's first stop condition: per cell, state and estimator G1–G4, the calibrated p-values with Holm's adjustment; the supported-region tests and their 0.1 and 10 sensitivity; the heavy-tail diagnostics; the verdict (consistent / practical support failure / low overlap, unresolved / biased) and the stop flag |
| `fig_g1_snr`, `fig_g2_alignment`, `fig_g3_bias_noise` | SNR; P(⟨ĝ, g*⟩ > 0) and the mean cosine; bias against noise, per estimator and state (PNG, PDF and the plotted data as CSV) |

## `regime_map/`: OPC against the likelihood over N × support × corruption (§8–§10)

```bash
python -m training.analyze_opc_regimes map --runs artifacts/full_study/run_opc_regime_lgs* --levels artifacts/full_study/opc_gradient_regime/support/support_levels.json --states artifacts/full_study/run_opc_gradients_8a artifacts/full_study/run_opc_gradients_N25000_lgs* --gradients artifacts/full_study/opc_gradient_regime/gradients_regime/table_gradients_world.csv --out artifacts/full_study/opc_gradient_regime/regime_map
```

| file | content |
|---|---|
| `table_selected.csv` | per world, cell and arm: the native, common-DR and best trial, the mean of the 20 trials |
| `table_population_states.csv` | per world and support level: the population states' greedy values, their population ESS share and low-π0 mass |
| `table_margins.csv` | per world, cell, OPC arm and rule: F = M_L − (T_O − T_L) − (S_O − S_L) and its parts, the observed OPC − likelihood difference (equal to F), θ_value*'s overlap (N × ESS share) |
| `table_regime_summary.csv` | per cell, OPC arm, rule and panel (bias type, pooled, per dataset): means and 95% t-intervals, the worlds where OPC is ahead |
| `table_dataset_corruption.csv` | per dataset × corruption cell: the same quantities |
| `table_gradient_features.csv`, `table_correlations.csv` | G3 / G5 gradient features per world and cell; their Spearman correlations with the observed difference (§10) |
| `fig_r1_<arm>`, `fig_r2_<arm>` | OPC − likelihood against N per bias panel and support level; the margin's parts M_L against dT + dS |

## `decomposition_25k/`: the 25k gap decomposition (§6)

```bash
python -m training.analyze_opc_regimes decompose --runs artifacts/full_study/run_opc_regime_lgs0.8 artifacts/full_study/run_opc_interventions_25k artifacts/full_study/run_opc_interventions_bfull_25k --states artifacts/full_study/run_opc_gradients_8a --empirical artifacts/full_study/run_opc_empirical_25k --out artifacts/full_study/opc_gradient_regime/decomposition_25k
```

| file | content |
|---|---|
| `table_decomposition.csv` | per world, arm and rule: V* − V(selected) = M + E + O + S (objective mismatch, empirical-objective gap, optimization gap, selection gap), with the empirical references' greedy values and the oracle-stopped path's best |
| `table_decomposition_summary.csv` | per arm, rule and panel: means and 95% t-intervals |
| `table_decomposition_dataset_corruption.csv` | per dataset × corruption cell |

## `interventions_25k/`: oracle q, larger and full batches, raw against harmonic (§7)

```bash
python -m training.analyze_opc_regimes interventions --runs artifacts/full_study/run_opc_regime_lgs0.8 artifacts/full_study/run_opc_interventions_25k artifacts/full_study/run_opc_interventions_bfull_25k --out artifacts/full_study/opc_gradient_regime/interventions_25k
```

| file | content |
|---|---|
| `table_interventions.csv` | per contrast (a − b paired by world), selection rule and panel: mean, 95% t-interval, the worlds where a is higher |
| `table_interventions_dataset_corruption.csv` | per contrast, rule and dataset × corruption cell |
| `table_selected.csv` | the selections of every arm |
