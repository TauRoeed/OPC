# Experimental Regimes

Paper-facing names for settings already used in tags / scripts.

## Representation bias severity

| Name | Code `noise_level` / `--bias-configs` | Signal kept (all three types) |
|------|---------------------------------------|-------------------------------|
| NoBias | `none` | 1.00 |
| MildBias | `low` | 0.90 |
| ModerateBias | `medium` | 0.75 |
| StrongBias | `high` | 0.50 |

Types: `WarpOnly` / `GroupOnly` / `VectorOnly` ↔ `high/none/none` / `none/high/none` /
`none/none/high` (any level). Bias always applies to users and items. See
[representation_bias.md](representation_bias.md).

## Logging damage

| Name | Settings | Meaning |
|------|----------|---------|
| CleanLog | mix=0, spread 0.5 | Softmax logging from biased vectors |
| HurtLog | mix≈0.3, spread≈0.8 | Uniform mix + flatter softmax (harder propensities) |

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
