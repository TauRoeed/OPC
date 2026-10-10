# The structured scenario-shift study: tables and figures

The pre-registration and report is `docs/structured_scenario_shift_study.md`. Every run it uses is in
`artifacts/full_study/run_registry.csv` (`run_ss_*`). Every file here is rebuilt from the run folders by the commands
below, from the repository root with the BPR embeddings in `BPR/embeddings`. World builds pin 4 CPU threads; analyses
run with BLAS threads capped (`OPENBLAS_NUM_THREADS=2`).

## `world_scan/`: the worlds before any method was trained (§13.1)

```bash
python scripts/ss_world_scan.py --datasets ml kuairand anime --seeds 100 101 200 201 --out artifacts/full_study/structured_scenario_shift/world_scan/world_scan_amended.csv
```

| file | content |
|---|---|
| `world_scan_preregistered.csv` | the first response levels (sd log α 0.25 / 0.5, sd β 0.5 / 1.0) with κ and c solved for both CTR targets: the worlds that fail to calibrate or are pathological (§13.1, "What failed"); written by the scan's earlier version, before the amendment |
| `world_scan_amended.csv` | the amended levels (sd log α 0.125 / 0.25, sd β 0.25 / 0.5), κ of the homogeneous world, c for the 5% reference CTR: 108 worlds, none pathological. Per world: the best-item CTR and its homogeneous value, the reference and logging CTR, the pathology shares, ρ, top-1 agreement, the available gain, the response factors' spread, κ and c |

## `tuning/`: the tuning round (§7, §13.2)

```bash
python -m training.analyze_shared_objectives tune --runs artifacts/full_study/run_ss_tune_s200 --arms shared_lr_likelihood shared_lr_likelihood_calib shared_lr_opc shared_lr_opc_raw --out artifacts/full_study/structured_scenario_shift/tuning
```

| file | content |
|---|---|
| `tuning_trials_long.csv.gz` | the 960 trials, with the world tags and each trial's greedy gain over the logger |
| `tuning_selected.csv` | per world and arm: the native, common and oracle-best trials, the mean of the 20 trials, the regrets |
| `tuning_summary.csv` | per arm, the 12 worlds pooled: the mean gains and regrets with 95% CIs |
| `tuning_marginals.csv` | per arm, searched dimension and bin: the mean gap of a trial below its world's best trial, the trials and the distinct configurations |
| `tuning_edges.csv` | the edge rule per arm and dimension: the best bin, its edge, the margin over its neighbour, `extend` |
| `tuning_weights.csv` | the logged rows' importance-weight statistics per world |

The rule fires on the learning rate at the top for all four arms; the main grid's range is 1e-4–6.3246e-3 (§13.2).

## `population/`: the population optima of Stage 1 (§6)

```bash
python -m training.ss_population --datasets ml --seeds 100 101 --labels s-none.r-none s-none.r-moderate s-none.r-strong s-moderate.r-none s-moderate.r-moderate s-moderate.r-strong s-strong.r-none s-strong.r-moderate s-strong.r-strong --emb-dir BPR/embeddings --out artifacts/full_study/run_ss_population
# (one job per dataset; the run used extra jobs per seed, with the labels reversed so they never overlap)
python -m training.ss_population --finite-qhat --datasets ml kuairand anime --seeds 100 101 --labels <the nine labels> --emb-dir BPR/embeddings --out artifacts/full_study/run_ss_population
python -m training.analyze_structured_shift population --run artifacts/full_study/run_ss_population --out artifacts/full_study/structured_scenario_shift/population
```

| file | content |
|---|---|
| `table_population.csv` | per world: the greedy (`Vg_*`) and stochastic (`V_*`) values of the source, the truth adapter, θ_value*, θ_lik*, θ_calib* and θ_harm*; M_L, M_calib, M_harm, the class optimizer's gap and the available gain (points); q̂_∞'s RMSE under the logger and under θ_value*'s policy; the 25k grid's cross-fitted q̂'s RMSE (`qhat25k_*`); the checks A–C; the world's statistics (`world_*`) |
| `population_cells.csv` | mean and 95% CI over worlds per shift × response, pooled and per dataset, and per shift, response and dataset |
| `population_check_failures.csv` | the worlds where a check fails (none) |
| `population.md` | the grids as markdown |
| `fig_population_mismatch.*` | the three mismatches against the response level, one panel per shift level |

## `compare/`: the 25k grid of Stage 1 (§7, §8, §10)

```bash
python -m training.analyze_structured_shift compare --population artifacts/full_study/run_ss_population --runs artifacts/full_study/run_ss_main_25k --out artifacts/full_study/structured_scenario_shift/compare
```

| file | content |
|---|---|
| `table_worlds.csv` | per world and arm: the gains over the logger under the native, common, oracle-best and mean-of-20 rules; the selected trials' diagnostics (λ, the regularizer, the correction's norms, the logit scale, ESS, the changed decisions); the arm's weight statistics and row hashes; the world's statistics; its population optimum and the decomposition gap = M + E+O + S |
| `paired_worlds.csv` | per world, pair and rule: a − b |
| `paired_cells.csv` | a − b paired by world: mean, 95% CI, worlds and the worlds where a is higher; per shift × response cell, shift, response and dataset, pooled and per dataset |
| `arm_cells.csv` | per panel and arm: the gains, the decomposition and the diagnostics, mean and 95% CI |
| `mechanism.csv` | per rule: OPC − likelihood against M_L, M_harm, M_L − M_harm and the available gain over worlds (Spearman, OLS slope), pooled and per dataset |
| `expansion_gate.csv` | the §7 gate per pooled cell: criteria (a)–(d), `expand` |
| `data_identity.csv`, `divergence.csv` | identical rows and paired configurations per world; the diverged share per cell and arm |
| `compare_info.json` | counts, the harmonic-dominance criterion of the gate, the population-trainer identity check |
| `tables.md` | the tables as markdown |
| `fig_paired_25k.*`, `fig_mechanism.*`, `fig_decomposition_25k.*` | the figures, each with its data as CSV |
