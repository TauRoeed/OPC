# BLOB's catalog-size prior: calibration and the 25k check

The report is `docs/blob_prior_calibration.md`. The run registry has `run_blob_calib_s200` and
`run_blob_pnorm_main_25k`. The controlled study's folder (`../blob_controlled_25k/`) is unchanged.

## `calibration/`: the pre-registered calibration on the 18 tuning worlds (§2–§3)

```bash
python -m training.analyze_blob calib --runs artifacts/full_study/run_blob_calib_s200 --out artifacts/full_study/blob_prior_calibration/calibration
```

| file | content |
|---|---|
| `calibration_decision.csv` | per prior variant: the simulated 20-trial score, paired differences against the released prior and P₀ = 100, the choice |
| `calibration_mechanism.csv` | per variant, selected trials and all trials: the correction's size, the click model's fit, where the greedy policy recommends |
| `calibration_lr_ranges.csv`, `calibration_lr_marginals.csv` | the learning-rate edge: the previous against the extended range; the half-decade marginals |
| `calibration_edges.csv` | the edge rule on the chosen variant's trials |
| `calibration_trials_long.csv.gz` | every trial |

## `diagnostics/` and `compare/`: the calibrated arm on the 30 main worlds (§4–§7)

```bash
python -m training.policy_diagnostics --emb-dir BPR/embeddings --out artifacts/full_study/blob_prior_calibration/diagnostics --expect 12 --runs artifacts/full_study/run_blob_main_25k_nq artifacts/full_study/run_blob_main_25k_mnq artifacts/full_study/run_blob_pnorm_main_25k artifacts/full_study/run_replay_opc_25k_mlkr artifacts/full_study/run_replay_opc_25k_anime artifacts/full_study/run_replay_cap_rho0_25k_mlkr artifacts/full_study/run_replay_cap_rho0_25k_anime artifacts/full_study/run_class_oracles_20261005/ml/policies artifacts/full_study/run_class_oracles_20261005/kuairand/policies artifacts/full_study/run_class_oracles_20261005/anime/policies
```

```bash
python -m training.analyze_blob compare --blob-runs artifacts/full_study/run_blob_main_25k_nq artifacts/full_study/run_blob_main_25k_mnq artifacts/full_study/run_blob_pnorm_main_25k --diagnostics artifacts/full_study/blob_prior_calibration/diagnostics --out artifacts/full_study/blob_prior_calibration/compare
```

`compare/` has the same files as `../blob_controlled_25k/`, with the calibrated arm `blob_l10_nq` (BLOB-Pnorm-NQ,
P₀ = 10) added to every table and figure; `../blob_controlled_25k/README.md` describes each file. The accounting figure shows the calibrated arm
against OPC, CausE-cap and default BLOB-NQ.
