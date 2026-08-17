# Experimental Regimes

Paper-facing names for settings already used in tags / scripts.

## Embedding noise severity

| Name | Code `noise_level` | Combined (ε1, ε2, ε_meta) |
|------|--------------------|---------------------------|
| MildNoise | `low` | (0.05, 0.05, 0) |
| ModerateNoise | `medium` | (0.10, 0.15, 0.05) |
| StrongNoise | `high` | (0.20, 0.25, 0.10) |
| ExtremeNoise | `extreme` | (0.35, 0.40, 0.20) |
| BrutalNoise | `brutal` | (0.50, 0.50, 0.30) |

Axes: `CombinedAxis`, `ContextOnly`, `ActionOnly`, `MetadataOnly` ↔ `combined` / `context` / `action` / `metadata`.

## Logging damage

| Name | Settings | Meaning |
|------|----------|---------|
| CleanLog | mix=0, temp=1 | Softmax logging from noisy embeddings |
| HurtLog | mix≈0.3, temp≈2 | Uniform mix + flatter softmax (harder propensities) |

Script tags: `hurtlog`, `abl_*_hurtlog`, `abl_*_hurtlog_v2`.

## Reward-model regimes

| Name | `--reward-model` |
|------|------------------|
| RegQ | `regression` |
| LogScoreQ | `logging_score` |
| OracleQ | `oracle` |

## Selection regimes

| Name | `--optuna-selection` |
|------|----------------------|
| SelectCILow | `ci_low` |
| SelectRHat | `r_hat` |
| SelectActual | `actual_reward` |

## Method arms

| Name | Training |
|------|----------|
| OPC | logged propensities; default `kl_crm` |
| NoProp | uniform propensities; pure `naive` |

## Loss ablations (under HurtLog)

| Tag | Loss |
|-----|------|
| AblIPW | `ipw` |
| AblSNDR | `sndr` |
| AblKLCRM | `kl_crm` control |

## Data-size / CTR (descriptive)

- **SparseCTR** / **DenseCTR** — e.g. ctr=0.02 vs 0.1
- **SmallTrain** / **LargeTrain** — e.g. 5k–25k vs ≥100k
- Use measured SNR + n when reporting complexity interactions (see workplan theory track).
