# CausE vs the revalidated OPC: the fair 25k comparison (artifacts, 2026-10-05)

The tables and figures of [docs/cause_fair_comparison_25k.md](../../../docs/cause_fair_comparison_25k.md). Development
stage, seeds 100/101 (main grid) and 200/201 (tuning). The run folders and their console logs (`logs/`) stay local,
as for every earlier run; they are listed with their code commit and purpose in `../run_registry.csv`.

## tuning/: the CausE-warm and CausE-capacity-matched search spaces (§3)

```bash
python -m training.analyze_cause_fair tune --out artifacts/full_study/cause_fair_25k/tuning \
    --runs artifacts/full_study/run_cause_tune_warm_s200 artifacts/full_study/run_cause_tune_cap_s200
```

| file | contents |
|---|---|
| `tuning_trials_long.csv.gz` | every tuning trial, one row per (trial, prediction): hyperparameters, validation NLL / AUC, the DR estimates, the exact true values and gains over the logger (CTR points) |
| `tuning_marginals.csv` | per family, dimension and value: trials, the share that diverged, the gain below the cell's best trial, and the shares of cells whose NLL-selected and best-true trials have this value |
| `tuning_selected.csv` | the NLL-selected trial of every (world, ρ, prediction) and its regret |
| `tuning_decision.csv` | every candidate sub-space under the resampled 10-trial protocol: mean selected greedy gain, regret, trials available, the paired difference from the wide space (95% CI over worlds), and the chosen candidate |
| `tuning_boundaries.csv` | where the selected and best-true trials sit on each dimension's range |

## The main comparison (§5, §6)

```bash
python -m training.analyze_cause_fair compare --out artifacts/full_study/cause_fair_25k \
    --cause-runs artifacts/full_study/run_cause_fair_warm_25k artifacts/full_study/run_cause_fair_cap_25k \
    --raw-runs artifacts/full_study/run_cause_fair_opc_raw_25k \
    --dm-own-runs artifacts/full_study/run_cause_fair_dm_oldspace_anime
```

It also reads the native CausE rows (`run_cause_dev_25k_cause_20261004`), the corrected Stage 2 runs
(`run_reval_stage2_{opc,base}_{mlkr,anime}`), the old-space DM-only run (`run_reval_stage2_oldspace_dm_mlkr`), the
Stage 1 oracle (`run_oracle_repair_20260927`) and the best single items (`../cause_dev_25k_20261004`).

| file | contents |
|---|---|
| `table_conditions.csv` | one row per world × arm (× ρ for CausE): true greedy, stochastic and tempered values; gains over the logger (CTR points); the fractions of the representation loss and of the arm's structural oracle repaired; selection estimates and regrets; the data composition, collection rewards and exploration cost; the selected hyperparameters |
| `table_summary.csv` | mean and 95% CI over the 6 dataset × seed worlds per bias (and pooled over the 24 biased worlds), arm and ρ |
| `table_paired_opc.csv` | OPC − each CausE arm, paired by world: greedy, stochastic and (fair variants) tempered |
| `table_paired_references.csv` | OPC − raw-DR OPC, DM-only (both ranges), no-propensity and the tempered logger |
| `table_rho_effect.csv` | each CausE arm at ρ minus the same arm at ρ = 0 |
| `table_variant_contrasts.csv` | CausE-warm − native CausE-prod, CausE-cap − CausE-warm, at equal ρ and prediction side |
| `table_selection_rule.csv` | diagnostic: CausE's NLL selection vs the DR lower bound of each trial's greedy policy vs the best trial |
| `table_oracle_check.csv` | the best CausE-cap and OPC trial of every world against the Stage 1 linear-repair oracle |
| `table_data_identity.csv` | every CausE family trained on the same rows at each (world, ρ) |
| `cause_trials_long.csv.gz` | every main-grid CausE-warm / CausE-cap trial |
| `fig1_rho_greedy`, `fig1b_rho_stochastic_tempered`, `fig2_exploration_cost`, `fig3_cause_variants`, `fig4_opc_minus_cause` (`.png`, `.pdf`, `.csv`) | the figures of §6 and their plotted values |
