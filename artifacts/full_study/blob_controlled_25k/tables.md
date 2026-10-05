**Greedy value − the logger's greedy value (primary)** (CTR points; mean over worlds, 95% CI for the pooled biased worlds)

| arm | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| BLOB-NQ (supplied source) | +1.42 [+0.85, +1.98] | -0.17 | +1.58 | +0.70 | +0.84 | +2.56 |
| BLOB-MNQ (supplied source) | +0.94 [+0.53, +1.35] | -0.11 | +0.71 | +0.54 | +0.75 | +1.75 |
| CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | +3.09 [+2.10, +4.08] | -0.55 | +4.23 | +1.22 | +1.03 | +5.89 |
| CausE-cap-T, ρ = 0 | +3.04 [+2.09, +4.00] | -0.55 | +4.22 | +1.22 | +1.03 | +5.70 |
| CausE-warm-C, ρ = 0 (likelihood, free vectors) | +1.65 [+0.82, +2.49] | -0.15 | +0.46 | +0.72 | +1.15 | +4.30 |
| OPC (harmonic:0.1) | +2.69 [+1.84, +3.53] | -0.61 | +3.03 | +1.32 | +0.96 | +5.42 |
| DM-only (own range) | +2.15 [+1.30, +3.01] | -0.55 | +2.09 | +0.81 | +0.64 | +5.07 |
| DM-only (OPC's range) | +2.08 [+1.23, +2.94] | -0.58 | +2.00 | +0.77 | +0.57 | +4.99 |
| tempered logger | -0.00 [-0.00, +0.00] | -0.00 | -0.00 | +0.00 | -0.00 | -0.00 |

**Stochastic value − the logger's value (BLOB and CausE: raw softmax, τ = 1)** (CTR points; mean over worlds, 95% CI for the pooled biased worlds)

| arm | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| BLOB-NQ (supplied source) | -12.30 [-13.85, -10.75] | -18.35 | -13.90 | -14.76 | -14.04 | -6.49 |
| BLOB-MNQ (supplied source) | -12.26 [-13.82, -10.71] | -18.30 | -13.99 | -14.66 | -13.97 | -6.43 |
| CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -11.80 [-13.42, -10.18] | -18.74 | -12.76 | -14.46 | -14.13 | -5.85 |
| CausE-cap-T, ρ = 0 | -11.81 [-13.43, -10.19] | -18.74 | -12.81 | -14.46 | -14.13 | -5.85 |
| CausE-warm-C, ρ = 0 (likelihood, free vectors) | -12.10 [-13.63, -10.57] | -18.41 | -13.68 | -14.44 | -13.91 | -6.36 |
| OPC (harmonic:0.1) | +6.75 [+6.18, +7.33] | +5.22 | +7.49 | +6.08 | +5.52 | +7.93 |
| DM-only (own range) | +6.23 [+5.70, +6.76] | +5.39 | +6.56 | +5.59 | +5.20 | +7.57 |
| DM-only (OPC's range) | +6.17 [+5.64, +6.71] | +5.39 | +6.49 | +5.56 | +5.14 | +7.50 |
| tempered logger | +3.99 [+3.53, +4.45] | +5.92 | +4.45 | +4.70 | +4.51 | +2.30 |

**Stochastic value − the logger's value, BLOB and CausE tempered by the DR lower bound (OPC: its learned scale)** (CTR points; mean over worlds, 95% CI for the pooled biased worlds)

| arm | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| BLOB-NQ (supplied source) | +5.45 [+4.90, +6.00] | +5.79 | +6.07 | +5.49 | +5.21 | +5.03 |
| BLOB-MNQ (supplied source) | +4.99 [+4.50, +5.47] | +5.81 | +5.18 | +5.34 | +5.20 | +4.23 |
| CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | +7.13 [+6.36, +7.89] | +5.38 | +8.59 | +5.97 | +5.56 | +8.39 |
| CausE-cap-T, ρ = 0 | +7.08 [+6.34, +7.82] | +5.38 | +8.60 | +5.97 | +5.56 | +8.20 |
| CausE-warm-C, ρ = 0 (likelihood, free vectors) | +5.70 [+5.08, +6.32] | +5.80 | +4.94 | +5.52 | +5.58 | +6.77 |
| OPC (harmonic:0.1) | +6.75 [+6.18, +7.33] | +5.22 | +7.49 | +6.08 | +5.52 | +7.93 |
| tempered logger | +3.99 [+3.53, +4.45] | +5.92 | +4.45 | +4.70 | +4.51 | +2.30 |

**Paired differences, greedy value** (CTR points; mean [95% CI] over worlds; in parentheses the worlds where a is higher)

| a − b | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| BLOB-NQ (supplied source) − BLOB-MNQ (supplied source) | +0.48 [+0.21, +0.75] (21/24) | -0.06 [-0.20, +0.08] (3/6) | +0.87 [+0.17, +1.57] (6/6) | +0.16 [-0.07, +0.39] (4/6) | +0.09 [-0.03, +0.21] (5/6) | +0.81 [-0.10, +1.72] (6/6) |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -1.67 [-2.29, -1.06] (2/24) | +0.38 [+0.20, +0.56] (6/6) | -2.65 [-3.44, -1.86] (0/6) | -0.52 [-0.85, -0.19] (0/6) | -0.20 [-0.51, +0.12] (2/6) | -3.33 [-3.85, -2.81] (0/6) |
| BLOB-NQ (supplied source) − CausE-cap-T, ρ = 0 | -1.62 [-2.21, -1.03] (2/24) | +0.38 [+0.20, +0.56] (6/6) | -2.64 [-3.42, -1.86] (0/6) | -0.52 [-0.85, -0.19] (0/6) | -0.20 [-0.51, +0.12] (2/6) | -3.14 [-3.78, -2.50] (0/6) |
| BLOB-NQ (supplied source) − CausE-warm-C, ρ = 0 (likelihood, free vectors) | -0.24 [-0.84, +0.37] (6/24) | -0.03 [-0.17, +0.11] (4/6) | +1.12 [-0.61, +2.85] (5/6) | -0.02 [-0.23, +0.19] (1/6) | -0.31 [-0.61, -0.00] (0/6) | -1.73 [-3.11, -0.36] (0/6) |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | -1.27 [-1.77, -0.76] (2/24) | +0.43 [+0.15, +0.72] (5/6) | -1.45 [-2.12, -0.79] (0/6) | -0.62 [-1.02, -0.23] (0/6) | -0.13 [-0.34, +0.09] (2/6) | -2.86 [-3.80, -1.93] (0/6) |
| BLOB-NQ (supplied source) − DM-only (own range) | -0.74 [-1.31, -0.16] (9/24) | +0.38 [+0.32, +0.44] (6/6) | -0.52 [-1.22, +0.19] (1/6) | -0.11 [-0.52, +0.30] (3/6) | +0.19 [-0.14, +0.53] (5/6) | -2.51 [-4.20, -0.81] (0/6) |
| BLOB-NQ (supplied source) − DM-only (OPC's range) | -0.66 [-1.25, -0.08] (10/24) | +0.40 [+0.34, +0.46] (6/6) | -0.42 [-1.12, +0.27] (2/6) | -0.07 [-0.50, +0.35] (3/6) | +0.27 [-0.06, +0.60] (5/6) | -2.43 [-4.17, -0.70] (0/6) |
| BLOB-NQ (supplied source) − tempered logger | +1.42 [+0.85, +1.98] (24/24) | -0.17 [-0.30, -0.05] (0/6) | +1.58 [+0.28, +2.88] (6/6) | +0.70 [+0.29, +1.10] (6/6) | +0.84 [+0.44, +1.23] (6/6) | +2.56 [+0.52, +4.61] (6/6) |
| BLOB-MNQ (supplied source) − BLOB-NQ (supplied source) | -0.48 [-0.75, -0.21] (3/24) | +0.06 [-0.08, +0.20] (3/6) | -0.87 [-1.57, -0.17] (0/6) | -0.16 [-0.39, +0.07] (2/6) | -0.09 [-0.21, +0.03] (1/6) | -0.81 [-1.72, +0.10] (0/6) |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -2.15 [-2.92, -1.39] (2/24) | +0.44 [+0.16, +0.72] (6/6) | -3.52 [-4.08, -2.96] (0/6) | -0.68 [-1.07, -0.28] (0/6) | -0.28 [-0.65, +0.08] (2/6) | -4.14 [-5.00, -3.28] (0/6) |
| BLOB-MNQ (supplied source) − CausE-cap-T, ρ = 0 | -2.10 [-2.84, -1.37] (2/24) | +0.44 [+0.16, +0.72] (6/6) | -3.51 [-4.06, -2.95] (0/6) | -0.68 [-1.07, -0.28] (0/6) | -0.28 [-0.65, +0.08] (2/6) | -3.95 [-4.73, -3.17] (0/6) |
| BLOB-MNQ (supplied source) − CausE-warm-C, ρ = 0 (likelihood, free vectors) | -0.72 [-1.37, -0.06] (2/24) | +0.03 [-0.08, +0.14] (5/6) | +0.25 [-1.50, +2.00] (2/6) | -0.18 [-0.30, -0.06] (0/6) | -0.40 [-0.80, +0.01] (0/6) | -2.54 [-4.19, -0.90] (0/6) |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | -1.75 [-2.37, -1.13] (2/24) | +0.49 [+0.27, +0.72] (6/6) | -2.32 [-2.75, -1.90] (0/6) | -0.78 [-1.24, -0.32] (0/6) | -0.21 [-0.52, +0.09] (2/6) | -3.67 [-4.60, -2.75] (0/6) |
| BLOB-MNQ (supplied source) − DM-only (own range) | -1.22 [-1.90, -0.53] (5/24) | +0.44 [+0.25, +0.62] (6/6) | -1.39 [-2.09, -0.68] (0/6) | -0.27 [-0.65, +0.11] (2/6) | +0.11 [-0.30, +0.52] (3/6) | -3.32 [-5.12, -1.51] (0/6) |
| BLOB-MNQ (supplied source) − DM-only (OPC's range) | -1.15 [-1.84, -0.46] (6/24) | +0.46 [+0.28, +0.64] (6/6) | -1.29 [-2.00, -0.58] (0/6) | -0.23 [-0.64, +0.18] (2/6) | +0.18 [-0.23, +0.60] (4/6) | -3.24 [-5.08, -1.40] (0/6) |
| BLOB-MNQ (supplied source) − tempered logger | +0.94 [+0.53, +1.35] (24/24) | -0.11 [-0.18, -0.05] (0/6) | +0.71 [+0.03, +1.39] (6/6) | +0.54 [+0.25, +0.84] (6/6) | +0.75 [+0.44, +1.06] (6/6) | +1.75 [+0.04, +3.46] (6/6) |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -0.41 [-0.66, -0.16] (6/24) | -0.05 [-0.39, +0.28] (2/6) | -1.19 [-1.44, -0.95] (0/6) | +0.10 [-0.14, +0.34] (3/6) | -0.07 [-0.22, +0.08] (2/6) | -0.47 [-1.05, +0.12] (1/6) |
| OPC (harmonic:0.1) − DM-only (own range) | +0.53 [+0.26, +0.81] (22/24) | -0.06 [-0.33, +0.22] (1/6) | +0.94 [+0.55, +1.32] (6/6) | +0.51 [+0.24, +0.78] (6/6) | +0.32 [+0.15, +0.49] (6/6) | +0.36 [-0.90, +1.61] (4/6) |

**Paired differences, stochastic value (BLOB tempered)** (CTR points; mean [95% CI] over worlds; in parentheses the worlds where a is higher)

| a − b | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -1.68 [-2.26, -1.09] (1/24) | +0.41 [+0.22, +0.59] (6/6) | -2.52 [-3.18, -1.85] (0/6) | -0.47 [-0.77, -0.18] (0/6) | -0.36 [-0.63, -0.08] (1/6) | -3.36 [-3.84, -2.88] (0/6) |
| BLOB-NQ (supplied source) − CausE-warm-C, ρ = 0 (likelihood, free vectors) | -0.25 [-0.86, +0.36] (6/24) | -0.01 [-0.19, +0.17] (3/6) | +1.13 [-0.58, +2.85] (5/6) | -0.02 [-0.22, +0.18] (1/6) | -0.38 [-0.83, +0.08] (0/6) | -1.74 [-3.10, -0.38] (0/6) |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | -1.30 [-1.80, -0.81] (1/24) | +0.57 [+0.43, +0.70] (6/6) | -1.42 [-2.09, -0.74] (0/6) | -0.59 [-0.97, -0.20] (0/6) | -0.31 [-0.62, -0.01] (1/6) | -2.90 [-3.80, -2.00] (0/6) |
| BLOB-NQ (supplied source) − tempered logger | +1.46 [+0.87, +2.05] (23/24) | -0.13 [-0.30, +0.03] (1/6) | +1.62 [+0.30, +2.94] (6/6) | +0.79 [+0.32, +1.26] (6/6) | +0.70 [+0.19, +1.20] (5/6) | +2.73 [+0.69, +4.78] (6/6) |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | -2.14 [-2.89, -1.39] (1/24) | +0.43 [+0.20, +0.66] (6/6) | -3.41 [-3.94, -2.88] (0/6) | -0.63 [-0.99, -0.26] (0/6) | -0.37 [-0.60, -0.13] (1/6) | -4.16 [-5.04, -3.28] (0/6) |
| BLOB-MNQ (supplied source) − CausE-warm-C, ρ = 0 (likelihood, free vectors) | -0.72 [-1.37, -0.06] (2/24) | +0.01 [-0.10, +0.12] (4/6) | +0.24 [-1.51, +1.99] (2/6) | -0.17 [-0.30, -0.05] (0/6) | -0.39 [-0.75, -0.02] (0/6) | -2.54 [-4.22, -0.86] (0/6) |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | -1.77 [-2.39, -1.15] (1/24) | +0.59 [+0.37, +0.80] (6/6) | -2.31 [-2.77, -1.85] (0/6) | -0.74 [-1.19, -0.29] (0/6) | -0.32 [-0.58, -0.07] (1/6) | -3.70 [-4.63, -2.77] (0/6) |
| BLOB-MNQ (supplied source) − tempered logger | +1.00 [+0.56, +1.43] (23/24) | -0.11 [-0.18, -0.05] (0/6) | +0.73 [+0.03, +1.42] (6/6) | +0.64 [+0.32, +0.96] (6/6) | +0.69 [+0.22, +1.15] (5/6) | +1.93 [+0.22, +3.65] (6/6) |

**Shares repaired, ceilings, best trials and selection regret, biased worlds** (greedy; mean [95% CI] over the 24 worlds)

| arm | share of the representation loss | share of its class's value oracle | class ceiling (pts) | best of its 20 trials (pts) | selection regret (pts) |
|---|---|---|---|---|---|
| BLOB-NQ (supplied source) | 0.15 [0.11, 0.20] | 0.21 [0.16, 0.25] | +6.70 | +1.98 | 0.56 |
| BLOB-MNQ (supplied source) | 0.10 [0.07, 0.13] | 0.15 [0.11, 0.19] | +6.70 | +1.54 | 0.61 |
| CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | 0.33 [0.24, 0.41] | 0.42 [0.35, 0.50] | +6.62 | +3.22 | 0.13 |
| CausE-cap-T, ρ = 0 | 0.32 [0.24, 0.40] | 0.42 [0.35, 0.49] | +6.62 | +3.19 | 0.15 |
| CausE-warm-C, ρ = 0 (likelihood, free vectors) | 0.15 [0.09, 0.21] | — | — | +1.91 | 0.25 |
| OPC (harmonic:0.1) | 0.27 [0.22, 0.33] | 0.37 [0.32, 0.42] | +6.62 | +2.84 | 0.16 |
| DM-only (own range) | 0.20 [0.15, 0.25] | 0.26 [0.20, 0.33] | +6.62 | +2.58 | 0.42 |
| DM-only (OPC's range) | 0.19 [0.14, 0.24] | 0.25 [0.18, 0.31] | +6.62 | +2.57 | 0.49 |
| tempered logger | -0.00 [-0.00, 0.00] | — | — | +0.00 | 0.00 |

**Class oracles** (truth-trained, no logged data; greedy gain over the logger's greedy value, CTR points; value = the value oracle, likelihood = the infinite-data likelihood fit under π0)

| quantity | biased (24) | no bias (6) | warp high (6) | group high (6) | vector high (6) | combined high (6) |
|---|---|---|---|---|---|---|
| value:affine_bilinear | +6.62 [+5.04, +8.21] | -0.01 | +7.15 | +3.95 | +3.43 | +11.97 |
| likelihood:affine_bilinear | +5.21 [+3.92, +6.49] | -0.02 | +7.19 | +2.72 | +2.31 | +8.61 |
| value−likelihood:affine_bilinear | +1.42 [+0.81, +2.02] | +0.01 | -0.04 | +1.23 | +1.12 | +3.36 |
| value:blob | +6.70 [+5.13, +8.27] | -0.02 | +7.15 | +4.07 | +3.55 | +12.02 |
| likelihood:blob | +5.49 [+4.07, +6.91] | -0.01 | +7.21 | +2.47 | +2.56 | +9.73 |
| value−likelihood:blob | +1.21 [+0.72, +1.69] | -0.00 | -0.05 | +1.60 | +0.98 | +2.30 |
| value:bilinear | +6.53 [+4.96, +8.10] | -0.04 | +7.08 | +3.86 | +3.37 | +11.82 |
| likelihood:bilinear | +4.75 [+3.52, +5.98] | -0.16 | +6.91 | +2.35 | +2.00 | +7.72 |
| value−likelihood:bilinear | +1.79 [+1.09, +2.48] | +0.12 | +0.17 | +1.51 | +1.37 | +4.10 |
| value:blob−affine | +0.07 [+0.05, +0.10] | -0.00 | +0.01 | +0.11 | +0.12 | +0.05 |
| value:affine−bilinear | +0.09 [+0.07, +0.11] | +0.03 | +0.07 | +0.09 | +0.06 | +0.15 |
| likelihood:blob−affine | +0.28 [-0.01, +0.57] | +0.01 | +0.01 | -0.25 | +0.26 | +1.11 |
| likelihood:affine−bilinear | +0.46 [+0.34, +0.58] | +0.14 | +0.28 | +0.36 | +0.31 | +0.89 |

**Accounting: Δgain = Δceiling − Δtraining − Δselection** (greedy, CTR points; mean [95% CI] over worlds)

| a − b | worlds | gain | ceiling | training | selection |
|---|---|---|---|---|---|
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | biased (24) | -1.27 [-1.77, -0.76] | +0.07 [+0.05, +0.10] | +0.94 [+0.61, +1.26] | +0.40 [-0.02, +0.83] |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | no bias (6) | +0.43 [+0.15, +0.72] | -0.00 [-0.01, -0.00] | -0.14 [-0.21, -0.08] | -0.30 [-0.57, -0.02] |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | warp high (6) | -1.45 [-2.12, -0.79] | +0.01 [-0.01, +0.02] | +1.62 [+0.87, +2.36] | -0.16 [-0.27, -0.05] |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | group high (6) | -0.62 [-1.02, -0.23] | +0.11 [+0.07, +0.16] | +0.63 [+0.46, +0.80] | +0.11 [-0.21, +0.42] |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | vector high (6) | -0.13 [-0.34, +0.09] | +0.12 [+0.08, +0.16] | +0.16 [-0.03, +0.34] | +0.09 [-0.09, +0.26] |
| BLOB-NQ (supplied source) − OPC (harmonic:0.1) | combined high (6) | -2.86 [-3.80, -1.93] | +0.05 [-0.01, +0.12] | +1.34 [+0.51, +2.17] | +1.58 [+0.00, +3.15] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | biased (24) | -1.67 [-2.29, -1.06] | +0.07 [+0.05, +0.10] | +1.31 [+0.80, +1.82] | +0.44 [+0.02, +0.85] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | no bias (6) | +0.38 [+0.20, +0.56] | -0.00 [-0.01, -0.00] | +0.06 [-0.03, +0.16] | -0.45 [-0.70, -0.19] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | warp high (6) | -2.65 [-3.44, -1.86] | +0.01 [-0.01, +0.02] | +2.65 [+1.87, +3.44] | +0.00 [-0.05, +0.05] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | group high (6) | -0.52 [-0.85, -0.19] | +0.11 [+0.07, +0.16] | +0.64 [+0.43, +0.85] | -0.01 [-0.35, +0.33] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | vector high (6) | -0.20 [-0.51, +0.12] | +0.12 [+0.08, +0.16] | +0.13 [-0.26, +0.52] | +0.18 [-0.06, +0.42] |
| BLOB-NQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | combined high (6) | -3.33 [-3.85, -2.81] | +0.05 [-0.01, +0.12] | +1.82 [+0.60, +3.04] | +1.56 [-0.01, +3.14] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | biased (24) | -1.75 [-2.37, -1.13] | +0.07 [+0.05, +0.10] | +1.37 [+0.94, +1.80] | +0.45 [-0.01, +0.91] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | no bias (6) | +0.49 [+0.27, +0.72] | -0.00 [-0.01, -0.00] | -0.17 [-0.29, -0.06] | -0.32 [-0.52, -0.13] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | warp high (6) | -2.32 [-2.75, -1.90] | +0.01 [-0.01, +0.02] | +2.48 [+2.04, +2.92] | -0.15 [-0.27, -0.03] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | group high (6) | -0.78 [-1.24, -0.32] | +0.11 [+0.07, +0.16] | +0.89 [+0.58, +1.20] | +0.00 [-0.23, +0.24] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | vector high (6) | -0.21 [-0.52, +0.09] | +0.12 [+0.08, +0.16] | +0.29 [+0.09, +0.50] | +0.04 [-0.18, +0.26] |
| BLOB-MNQ (supplied source) − OPC (harmonic:0.1) | combined high (6) | -3.67 [-4.60, -2.75] | +0.05 [-0.01, +0.12] | +1.82 [+0.72, +2.91] | +1.91 [+0.49, +3.33] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | biased (24) | -2.15 [-2.92, -1.39] | +0.07 [+0.05, +0.10] | +1.75 [+1.12, +2.37] | +0.48 [+0.03, +0.93] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | no bias (6) | +0.44 [+0.16, +0.72] | -0.00 [-0.01, -0.00] | +0.03 [-0.05, +0.11] | -0.47 [-0.77, -0.18] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | warp high (6) | -3.52 [-4.08, -2.96] | +0.01 [-0.01, +0.02] | +3.52 [+2.94, +4.09] | +0.00 [-0.01, +0.02] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | group high (6) | -0.68 [-1.07, -0.28] | +0.11 [+0.07, +0.16] | +0.90 [+0.53, +1.27] | -0.11 [-0.45, +0.23] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | vector high (6) | -0.28 [-0.65, +0.08] | +0.12 [+0.08, +0.16] | +0.27 [-0.14, +0.67] | +0.13 [-0.13, +0.40] |
| BLOB-MNQ (supplied source) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | combined high (6) | -4.14 [-5.00, -3.28] | +0.05 [-0.01, +0.12] | +2.30 [+0.90, +3.70] | +1.90 [+0.46, +3.33] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | biased (24) | -0.41 [-0.66, -0.16] | +0.00 [+0.00, +0.00] | +0.38 [+0.16, +0.60] | +0.03 [-0.05, +0.11] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | no bias (6) | -0.05 [-0.39, +0.28] | +0.00 [+0.00, +0.00] | +0.20 [+0.07, +0.34] | -0.15 [-0.57, +0.27] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | warp high (6) | -1.19 [-1.44, -0.95] | +0.00 [+0.00, +0.00] | +1.04 [+0.83, +1.24] | +0.16 [+0.05, +0.27] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | group high (6) | +0.10 [-0.14, +0.34] | +0.00 [+0.00, +0.00] | +0.01 [-0.10, +0.11] | -0.11 [-0.43, +0.20] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | vector high (6) | -0.07 [-0.22, +0.08] | +0.00 [+0.00, +0.00] | -0.02 [-0.24, +0.19] | +0.09 [-0.04, +0.23] |
| OPC (harmonic:0.1) − CausE-cap-C, ρ = 0 (plain likelihood, OPC's class) | combined high (6) | -0.47 [-1.05, +0.12] | +0.00 [+0.00, +0.00] | +0.48 [-0.07, +1.03] | -0.01 [-0.08, +0.05] |
