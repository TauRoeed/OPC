### T1. Structural recoverability (Stage 1, greedy)

Each cell: logger ranking loss (points) / oracle repair gain (points) / structural recoverability (validated bound in brackets). Datasets: mean of 2 seeds; all: mean of 6.

| bias | MovieLens | KuaiRand | Anime | all |
|---|---|---|---|---|
| warp | 8.01 / 7.96 / 0.994 (0.998) | 4.90 / 4.87 / 0.994 (0.999) | 8.90 / 8.83 / 0.993 (1.000) | 7.27 / 7.22 / 0.994 (0.999) |
| group | 6.94 / 5.37 / 0.775 (0.788) | 3.89 / 2.72 / 0.701 (0.710) | 6.84 / 3.87 / 0.564 (0.571) | 5.89 / 3.99 / 0.680 (0.690) |
| vector | 8.29 / 5.08 / 0.612 (0.626) | 5.37 / 3.11 / 0.576 (0.580) | 7.16 / 2.21 / 0.308 (0.309) | 6.94 / 3.46 / 0.499 (0.505) |
| combined medium | 12.36 / 9.11 / 0.737 (0.752) | 7.49 / 5.35 / 0.713 (0.714) | 11.79 / 6.56 / 0.556 (0.562) | 10.55 / 7.00 / 0.669 (0.676) |
| combined high | 19.18 / 13.82 / 0.721 (0.734) | 15.35 / 11.56 / 0.752 (0.756) | 18.28 / 10.89 / 0.595 (0.605) | 17.60 / 12.09 / 0.689 (0.698) |

### T2. Learned repair (Stage 2), mean over 3 datasets × 2 seeds

True gain over the logger in CTR points: greedy OPC / DM-only / no-propensity (the tempered logger's greedy gain is 0), stochastic OPC / DM-only / no-propensity / tempered logger, and the greedy fraction of the oracle repair.

| bias | train | greedy gain: OPC / DM / no-prop | stochastic gain: OPC / DM / no-prop / tempered | fraction of oracle repair: OPC / DM / no-prop |
|---|---|---|---|---|
| no bias | 5k | -0.86 / -1.62 / -0.45 | +4.43 / +4.07 / +5.17 / +5.88 | — |
| no bias | 25k | -0.45 / -0.54 / -0.41 | +5.27 / +5.40 / +5.55 / +5.85 | — |
| no bias | 100k | -0.35 / -0.14 / -0.43 | +5.54 / +5.79 / +5.53 / +5.93 | — |
| warp | 5k | +1.92 / +0.71 / +0.44 | +6.08 / +4.98 / +4.64 / +4.39 | 0.27 / 0.08 / 0.07 |
| warp | 25k | +3.04 / +2.07 / +0.64 | +7.50 / +6.54 / +5.12 / +4.42 | 0.43 / 0.28 / 0.09 |
| warp | 100k | +3.58 / +2.49 / +0.76 | +8.04 / +6.96 / +5.24 / +4.48 | 0.50 / 0.34 / 0.11 |
| group | 5k | +0.54 / -0.30 / +0.18 | +5.06 / +4.30 / +4.69 / +4.73 | 0.14 / -0.08 / 0.06 |
| group | 25k | +1.13 / +0.63 / +0.11 | +5.92 / +5.40 / +4.90 / +4.73 | 0.28 / 0.14 / 0.04 |
| group | 100k | +1.56 / +1.10 / +0.13 | +6.31 / +5.86 / +4.91 / +4.70 | 0.39 / 0.27 / 0.05 |
| vector | 5k | +0.34 / -0.31 / +0.23 | +4.66 / +4.11 / +4.55 / +4.49 | 0.07 / -0.16 / 0.07 |
| vector | 25k | +0.90 / +0.65 / +0.15 | +5.46 / +5.20 / +4.70 / +4.48 | 0.24 / 0.16 / 0.05 |
| vector | 100k | +1.28 / +1.03 / +0.14 | +5.83 / +5.57 / +4.70 / +4.56 | 0.37 / 0.29 / 0.04 |
| combined medium | 5k | +1.70 / +0.92 / +0.83 | +5.40 / +4.72 / +4.53 / +3.72 | 0.25 / 0.13 / 0.13 |
| combined medium | 25k | +3.06 / +2.30 / +0.70 | +6.96 / +6.21 / +4.61 / +3.85 | 0.43 / 0.32 / 0.11 |
| combined medium | 100k | +3.52 / +2.74 / +0.83 | +7.40 / +6.62 / +4.73 / +3.89 | 0.51 / 0.39 / 0.12 |
| combined high | 5k | +3.23 / +3.82 / +1.37 | +5.55 / +6.25 / +3.71 / +2.42 | 0.27 / 0.32 / 0.11 |
| combined high | 25k | +5.01 / +4.63 / +1.17 | +7.51 / +7.12 / +3.64 / +2.42 | 0.42 / 0.39 / 0.10 |
| combined high | 100k | +5.84 / +5.38 / +1.10 | +8.34 / +7.87 / +3.60 / +2.48 | 0.49 / 0.45 / 0.09 |

### T3. OPC minus each baseline (Stage 2), true CTR points, mean and 95% t-interval over the 6 paired conditions

| bias | train | OPC − DM-only | OPC − no-propensity | OPC − tempered logger |
|---|---|---|---|---|
| no bias | 5k | +0.36 [-0.31, +1.03] | -0.74 [-1.15, -0.34] | -1.46 [-1.74, -1.17] |
| no bias | 25k | -0.13 [-0.27, +0.02] | -0.27 [-0.39, -0.15] | -0.57 [-0.83, -0.31] |
| no bias | 100k | -0.25 [-0.33, -0.18] | +0.01 [-0.19, +0.20] | -0.39 [-0.49, -0.29] |
| warp | 5k | +1.09 [+0.06, +2.13] | +1.43 [+0.89, +1.98] | +1.68 [+1.23, +2.13] |
| warp | 25k | +0.95 [+0.52, +1.38] | +2.37 [+1.51, +3.23] | +3.08 [+2.06, +4.10] |
| warp | 100k | +1.08 [+0.75, +1.42] | +2.80 [+1.89, +3.72] | +3.56 [+2.61, +4.52] |
| group | 5k | +0.76 [+0.04, +1.48] | +0.37 [-0.16, +0.89] | +0.33 [-0.05, +0.72] |
| group | 25k | +0.52 [+0.21, +0.83] | +1.02 [+0.40, +1.63] | +1.19 [+0.76, +1.62] |
| group | 100k | +0.45 [+0.32, +0.57] | +1.40 [+0.68, +2.13] | +1.60 [+1.06, +2.15] |
| vector | 5k | +0.56 [-0.08, +1.20] | +0.11 [-0.23, +0.46] | +0.18 [-0.19, +0.55] |
| vector | 25k | +0.26 [+0.08, +0.44] | +0.76 [+0.13, +1.39] | +0.97 [+0.23, +1.71] |
| vector | 100k | +0.25 [+0.11, +0.40] | +1.12 [+0.62, +1.63] | +1.27 [+0.72, +1.81] |
| combined medium | 5k | +0.68 [-0.16, +1.52] | +0.86 [+0.27, +1.46] | +1.68 [+1.04, +2.31] |
| combined medium | 25k | +0.76 [+0.33, +1.19] | +2.36 [+1.39, +3.33] | +3.11 [+2.09, +4.14] |
| combined medium | 100k | +0.78 [+0.55, +1.00] | +2.67 [+1.99, +3.36] | +3.51 [+2.72, +4.29] |
| combined high | 5k | -0.70 [-2.28, +0.87] | +1.84 [+0.95, +2.74] | +3.13 [+1.74, +4.53] |
| combined high | 25k | +0.39 [-0.16, +0.93] | +3.87 [+2.97, +4.77] | +5.09 [+3.99, +6.18] |
| combined high | 100k | +0.46 [+0.19, +0.74] | +4.73 [+3.96, +5.51] | +5.86 [+4.98, +6.74] |

### T4. Per dataset (Stage 2, mean of 2 seeds)

Each cell: fraction of the oracle ranking repair OPC / DM-only; OPC − DM-only / OPC − no-propensity in true CTR points.

| bias | train | MovieLens | KuaiRand | Anime |
|---|---|---|---|---|
| warp | 5k | 0.27 / 0.12; +1.08 / +1.80 | 0.30 / 0.03; +1.08 / +0.96 | 0.24 / 0.10; +1.13 / +1.54 |
| warp | 25k | 0.47 / 0.35; +0.90 / +3.01 | 0.45 / 0.21; +1.18 / +1.63 | 0.37 / 0.27; +0.77 / +2.49 |
| warp | 100k | 0.53 / 0.41; +0.95 / +3.58 | 0.52 / 0.28; +1.16 / +1.79 | 0.45 / 0.32; +1.14 / +3.04 |
| group | 5k | 0.13 / 0.04; +0.39 / +0.77 | 0.14 / -0.00; +0.32 / +0.03 | 0.14 / -0.28; +1.56 / +0.30 |
| group | 25k | 0.28 / 0.21; +0.37 / +1.63 | 0.26 / 0.02; +0.67 / +0.36 | 0.30 / 0.18; +0.51 / +1.07 |
| group | 100k | 0.35 / 0.27; +0.46 / +2.19 | 0.36 / 0.19; +0.47 / +0.68 | 0.47 / 0.34; +0.41 / +1.34 |
| vector | 5k | 0.14 / -0.01; +0.53 / +0.39 | 0.13 / 0.14; -0.05 / +0.02 | -0.05 / -0.60; +1.20 / -0.07 |
| vector | 25k | 0.30 / 0.25; +0.23 / +1.44 | 0.33 / 0.19; +0.41 / +0.69 | 0.08 / 0.02; +0.13 / +0.15 |
| vector | 100k | 0.36 / 0.29; +0.28 / +1.68 | 0.44 / 0.33; +0.37 / +1.00 | 0.31 / 0.27; +0.11 / +0.69 |
| combined medium | 5k | 0.22 / 0.16; +0.34 / +1.05 | 0.34 / 0.27; +0.34 / +1.04 | 0.19 / -0.03; +1.35 / +0.50 |
| combined medium | 25k | 0.46 / 0.38; +0.71 / +3.46 | 0.45 / 0.31; +0.71 / +1.61 | 0.39 / 0.26; +0.86 / +2.00 |
| combined medium | 100k | 0.48 / 0.41; +0.65 / +3.36 | 0.54 / 0.35; +1.04 / +2.07 | 0.50 / 0.39; +0.64 / +2.59 |
| combined high | 5k | 0.24 / 0.19; +0.60 / +1.97 | 0.37 / 0.52; -1.79 / +2.41 | 0.18 / 0.25; -0.92 / +1.15 |
| combined high | 25k | 0.35 / 0.32; +0.45 / +3.94 | 0.53 / 0.47; +0.72 / +4.43 | 0.37 / 0.38; -0.01 / +3.23 |
| combined high | 100k | 0.41 / 0.39; +0.40 / +4.90 | 0.57 / 0.52; +0.64 / +5.03 | 0.48 / 0.45; +0.35 / +4.27 |

### T5. Structural gap vs learning gap (greedy, CTR %, mean over 3 datasets × 2 seeds)

| bias | train | V_target_best | V_logger | V_oracle_repair | V_OPC | representation loss | structural gap | learning gap | learned repair | shares: structural / learning / learned |
|---|---|---|---|---|---|---|---|---|---|---|
| warp | 5k | 29.94 | 22.67 | 29.89 | 24.59 | 7.27 | 0.05 | 5.30 | 1.92 | 0.01 / 0.73 / 0.27 |
| warp | 25k | 29.94 | 22.67 | 29.89 | 25.71 | 7.27 | 0.05 | 4.18 | 3.04 | 0.01 / 0.57 / 0.42 |
| warp | 100k | 29.94 | 22.67 | 29.89 | 26.26 | 7.27 | 0.05 | 3.64 | 3.58 | 0.01 / 0.50 / 0.50 |
| group | 5k | 29.94 | 24.06 | 28.04 | 24.60 | 5.89 | 1.90 | 3.45 | 0.54 | 0.32 / 0.59 / 0.09 |
| group | 25k | 29.94 | 24.06 | 28.04 | 25.19 | 5.89 | 1.90 | 2.85 | 1.13 | 0.32 / 0.49 / 0.19 |
| group | 100k | 29.94 | 24.06 | 28.04 | 25.62 | 5.89 | 1.90 | 2.43 | 1.56 | 0.32 / 0.42 / 0.26 |
| vector | 5k | 29.94 | 23.00 | 26.47 | 23.34 | 6.94 | 3.48 | 3.13 | 0.34 | 0.50 / 0.45 / 0.05 |
| vector | 25k | 29.94 | 23.00 | 26.47 | 23.90 | 6.94 | 3.48 | 2.56 | 0.90 | 0.50 / 0.37 / 0.13 |
| vector | 100k | 29.94 | 23.00 | 26.47 | 24.28 | 6.94 | 3.48 | 2.18 | 1.28 | 0.50 / 0.31 / 0.19 |
| combined medium | 5k | 29.94 | 19.40 | 26.40 | 21.10 | 10.55 | 3.54 | 5.31 | 1.70 | 0.33 / 0.50 / 0.17 |
| combined medium | 25k | 29.94 | 19.40 | 26.40 | 22.46 | 10.55 | 3.54 | 3.94 | 3.06 | 0.33 / 0.38 / 0.29 |
| combined medium | 100k | 29.94 | 19.40 | 26.40 | 22.92 | 10.55 | 3.54 | 3.48 | 3.52 | 0.33 / 0.33 / 0.34 |
| combined high | 5k | 29.94 | 12.34 | 24.43 | 15.57 | 17.60 | 5.52 | 8.86 | 3.23 | 0.31 / 0.50 / 0.19 |
| combined high | 25k | 29.94 | 12.34 | 24.43 | 17.35 | 17.60 | 5.52 | 7.07 | 5.01 | 0.31 / 0.40 / 0.29 |
| combined high | 100k | 29.94 | 12.34 | 24.43 | 18.18 | 17.60 | 5.52 | 6.25 | 5.84 | 0.31 / 0.35 / 0.34 |

Shares at 100k by dataset (structural / learning / learned):

| bias | MovieLens | KuaiRand | Anime |
|---|---|---|---|
| warp | 0.01 / 0.47 / 0.53 | 0.01 / 0.48 / 0.52 | 0.01 / 0.55 / 0.45 |
| group | 0.23 / 0.50 / 0.27 | 0.30 / 0.45 / 0.25 | 0.44 / 0.30 / 0.26 |
| vector | 0.39 / 0.39 / 0.22 | 0.42 / 0.32 / 0.26 | 0.69 / 0.21 / 0.09 |
| combined medium | 0.26 / 0.38 / 0.36 | 0.29 / 0.32 / 0.39 | 0.44 / 0.28 / 0.28 |
| combined high | 0.28 / 0.42 / 0.30 | 0.25 / 0.32 / 0.43 | 0.40 / 0.31 / 0.29 |

### T6. Earlier reward-model tests (previous defaults: legacy SNDR, log trick, shrink:100, TPE; pre-fix code)

OPC − DM-only in true CTR points (mean [95% CI] over ml, kuairand, anime × 2 seeds), and the change in each arm between paired settings.

| setting | bias | 5k | 25k | 100k |
|---|---|---|---|---|
| external 50k q_hat | combined medium | -1.13 [-1.61, -0.65] | +0.21 [-0.22, +0.65] | +0.86 [+0.40, +1.32] |
| external 50k q_hat | combined high | -2.44 [-3.12, -1.75] | -0.25 [-0.85, +0.34] | +0.37 [-0.05, +0.79] |
| budget-fair q_hat (cf5) | combined medium | +0.35 [-0.60, +1.29] | +0.65 [+0.25, +1.04] | +0.75 [+0.30, +1.21] |
| budget-fair q_hat (cf5) | combined high | -0.85 [-2.26, +0.55] | +0.37 [-0.42, +1.17] | -0.06 [-0.92, +0.81] |
| budget-fair q_hat, interaction features (cf5, val 20k) | combined medium | +1.05 [+0.07, +2.03] | +0.50 [-0.05, +1.04] | +0.75 [+0.42, +1.09] |
| budget-fair q_hat, interaction features (cf5, val 20k) | combined high | -0.85 [-2.42, +0.73] | +0.24 [-0.35, +0.83] | +0.05 [-0.77, +0.88] |
| budget-fair q_hat, misspecified concat features (cf5, val 20k) | combined medium | +5.25 [+1.49, +9.01] | +8.82 [+5.43, +12.21] | +7.97 [+5.36, +10.58] |
| budget-fair q_hat, misspecified concat features (cf5, val 20k) | combined high | +1.09 [-1.32, +3.51] | +1.74 [+0.13, +3.35] | +2.57 [+0.93, +4.20] |
| Δ external 50k q_hat -> budget-fair q_hat (cf5) | combined medium | DM -1.50, OPC -0.03 | DM -0.65, OPC -0.22 | DM +0.16, OPC +0.05 |
| Δ external 50k q_hat -> budget-fair q_hat (cf5) | combined high | DM -1.38, OPC +0.20 | DM -0.45, OPC +0.18 | DM +0.40, OPC -0.03 |
| Δ budget-fair q_hat, interaction features (cf5, val 20k) -> budget-fair q_hat, misspecified concat features (cf5, val 20k) | combined medium | DM -4.11, OPC +0.09 | DM -9.05, OPC -0.73 | DM -8.13, OPC -0.91 |
| Δ budget-fair q_hat, interaction features (cf5, val 20k) -> budget-fair q_hat, misspecified concat features (cf5, val 20k) | combined high | DM -1.68, OPC +0.25 | DM -2.06, OPC -0.56 | DM -2.87, OPC -0.36 |

### T7. Objective / gradient / weighting study (paired, random sampler; OPC only; medium, high × 5k, 25k, 100k)

Range over the 6 cells of the mean per-trial difference in true CTR points (identical configurations and seeds), cells whose 95% CI excludes 0, and the range of the selected-policy difference.

| comparison | per trial | CI excludes 0 | selected |
|---|---|---|---|
| legacy SNDR - dr (log trick, shrink:100) | -0.32 to -0.03 | 6 of 6 | -0.41 to -0.07 |
| global SNDR - dr (log trick, shrink:100) | -0.26 to -0.02 | 6 of 6 | -0.41 to -0.07 |
| direct - log trick (dr, shrink:100) | -0.02 to +0.14 | 2 of 6 | -0.04 to +0.27 |
| direct - log trick (dr, none: control) | -0.00 to +0.00 | 0 of 6 | -0.04 to +0.03 |
| shrink:100 - raw (dr, direct) | -0.03 to +0.51 | 4 of 6 | -0.15 to +0.65 |
| harmonic:0.1 - raw (dr, direct) | +0.07 to +0.74 | 6 of 6 | -0.02 to +0.73 |
| harmonic:0.1 - shrink:100 (dr, direct) | +0.09 to +0.23 | 6 of 6 | +0.09 to +0.24 |

### T8. Robustness slice: OPC with harmonic:0.1 minus OPC with shrink:100 (25k; ml, kuairand × 2 seeds)

| bias | measure | harmonic:0.1 | shrink:100 | difference [95% CI] |
|---|---|---|---|---|
| warp | V % | 26.520 | 26.451 | +0.069 [-0.084, +0.222] |
| warp | V greedy % | 26.554 | 26.514 | +0.041 [-0.095, +0.176] |
| warp | fraction greedy | 0.457 | 0.452 | +0.005 [-0.017, +0.026] |
| group | V % | 25.717 | 25.681 | +0.036 [-0.101, +0.173] |
| group | V greedy % | 25.734 | 25.717 | +0.018 [-0.128, +0.163] |
| group | fraction greedy | 0.269 | 0.261 | +0.008 [-0.038, +0.054] |
| vector | V % | 24.460 | 24.425 | +0.035 [-0.186, +0.257] |
| vector | V greedy % | 24.482 | 24.459 | +0.023 [-0.211, +0.257] |
| vector | fraction greedy | 0.315 | 0.303 | +0.012 [-0.067, +0.091] |

### T9. Logging-support sweep (Stage 3; ml, kuairand × 2 seeds; 25k)

| bias | logger share | logger value % | oracle ranking gain | OPC fraction [95% CI] | DM fraction | OPC − DM-only [95% CI] | OPC raw-weight ESS | OPC weights > 10 (%) |
|---|---|---|---|---|---|---|---|---|
| warp | 0.6 | 14.24 | 6.41 | 0.39 [0.24, 0.54] | 0.25 | +0.83 [+0.33, +1.34] | 290 | 2.86 |
| warp | 0.8 | 18.95 | 6.42 | 0.46 [0.27, 0.64] | 0.28 | +1.04 [+0.24, +1.84] | 536 | 2.36 |
| warp | 0.95 | 22.44 | 6.42 | 0.46 [0.31, 0.60] | 0.26 | +1.09 [+0.38, +1.80] | 743 | 1.56 |
| group | 0.6 | 14.84 | 4.05 | 0.17 [-0.07, 0.41] | 0.05 | +0.51 [+0.06, +0.96] | 427 | 3.11 |
| group | 0.8 | 19.72 | 4.05 | 0.27 [0.20, 0.33] | 0.12 | +0.52 [+0.03, +1.01] | 939 | 2.77 |
| group | 0.95 | 23.38 | 4.09 | 0.26 [0.22, 0.31] | 0.15 | +0.45 [+0.32, +0.58] | 738 | 1.52 |
| vector | 0.6 | 13.91 | 4.03 | 0.29 [0.20, 0.39] | 0.18 | +0.40 [+0.02, +0.77] | 534 | 3.93 |
| vector | 0.8 | 18.58 | 4.09 | 0.32 [0.26, 0.37] | 0.22 | +0.32 [+0.04, +0.60] | 983 | 2.74 |
| vector | 0.95 | 22.04 | 4.20 | 0.32 [0.24, 0.41] | 0.23 | +0.34 [-0.08, +0.76] | 1825 | 1.56 |
