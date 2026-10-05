# BLOB in the controlled environment, 25k: tables, figures and how to rebuild them

The report is `docs/blob_controlled_integration.md`. Its run registry rows are in `artifacts/full_study/run_registry.csv`.
The run folders (`artifacts/full_study/run_*`) stay local and are not in git. Everything here is rebuilt from them by
the commands below.

## Tuning (§3.1–§3.2)

- `tuning/`: the first round on the 18 tuning worlds (seeds 200/201).
- `tuning_round2/`: the edge rule's supplementary round, pooled with the first.
- `tuning_round2_supplement_only/`: the supplementary trials alone, as a check.

```bash
python -m training.analyze_blob tune --runs artifacts/full_study/run_blob_tune_s200 --out artifacts/full_study/blob_controlled_25k/tuning
```

```bash
python -m training.analyze_blob tune --spaces supplement --runs artifacts/full_study/run_blob_tune_s200 artifacts/full_study/run_blob_tune_s200_supp_nq artifacts/full_study/run_blob_tune_s200_supp_mnq --out artifacts/full_study/blob_controlled_25k/tuning_round2
```

```bash
python -m training.analyze_blob tune --spaces supplement --runs artifacts/full_study/run_blob_tune_s200_supp_nq artifacts/full_study/run_blob_tune_s200_supp_mnq --out artifacts/full_study/blob_controlled_25k/tuning_round2_supplement_only
```

## Pick diagnostics (§5)

These use the selected policies saved by the BLOB main grid, the OPC and CausE-cap replays (`--save-policies`) and
the class oracles (`training/class_oracles.py --save-policies`).

```bash
python -m training.policy_diagnostics --emb-dir BPR/embeddings --out artifacts/full_study/blob_controlled_25k/diagnostics --runs artifacts/full_study/run_blob_main_25k_nq artifacts/full_study/run_blob_main_25k_mnq artifacts/full_study/run_replay_opc_25k_mlkr artifacts/full_study/run_replay_opc_25k_anime artifacts/full_study/run_replay_cap_rho0_25k_mlkr artifacts/full_study/run_replay_cap_rho0_25k_anime artifacts/full_study/run_class_oracles_20261005/ml/policies artifacts/full_study/run_class_oracles_20261005/kuairand/policies artifacts/full_study/run_class_oracles_20261005/anime/policies
```

## The comparison (§5–§6)

```bash
python -m training.analyze_blob compare --blob-runs artifacts/full_study/run_blob_main_25k_nq artifacts/full_study/run_blob_main_25k_mnq --diagnostics artifacts/full_study/blob_controlled_25k/diagnostics --out artifacts/full_study/blob_controlled_25k
```

| file | content |
|---|---|
| `tables.md` | the report's tables |
| `table_conditions.csv` | one row per world × arm: values, gains, shares, regret, oracles, diagnostics of the selected trial |
| `table_summary.csv` | means and 95% CIs per bias × arm, and pooled over the 24 biased worlds |
| `table_paired.csv` | paired differences (BLOB − each arm; OPC − CausE-cap, DM) |
| `table_class_oracles.csv` | the class oracles per bias: value and likelihood objectives, class and objective contrasts |
| `table_accounting.csv` | Δgain = Δceiling − Δtraining − Δselection |
| `table_pick_diagnostics.csv`, `table_pick_pairs.csv`, `policy_pairs.csv` | where the selected policies recommend; pairwise agreement and its value |
| `table_data_identity.csv` | BLOB's training and validation rows against CausE-cap's warm rows |
| `diagnostics/policy_diagnostics.csv`, `diagnostics/policy_pairs.csv` | the pick diagnostics of every saved policy, the logger and BLOB's prior mean, per world |
| `class_oracles/class_oracles_<ds>.csv` | every class-oracle fit kept (class × objective × world), copied from `run_class_oracles_20261005`, with its settings |
| `fig1_greedy_gain.*`, `fig1b_tempered_gain.*` | target value per arm and bias |
| `fig2_accounting.*` | the accounting |
| `fig3_picks.*` | the pick diagnostics |
