# Structured scenario shift: the 25k grid

18 worlds, 1800 trials. Greedy CTR points; mean [95% CI] over worlds, paired by world. Gains are over the logger's greedy value.

## 1. Harmonic OPC − likelihood (ordinary head), by rule

Cells: mean [95% CI] (worlds where OPC is higher / worlds).

### native rule

Pooled:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.78 [-3.02, -0.55] (0/6) | -1.11 [-1.92, -0.31] (0/6) | +0.42 [-0.90, +1.75] (3/6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.90 [-11.21, +7.42] (0/2) | -1.54 [-7.96, +4.88] (0/2) | -0.36 [-1.55, +0.84] (0/2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -2.50 [-19.08, +14.09] (0/2) | -1.11 [-12.44, +10.22] (0/2) | +0.46 [-18.96, +19.89] (1/2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.96 [-2.15, +0.23] (0/2) | -0.69 [-3.64, +2.25] (0/2) | +1.17 [-7.84, +10.18] (2/2) |
| strong | – | – | – |

### common rule

Pooled:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.08 [-1.63, -0.54] (0/6) | -0.30 [-0.80, +0.21] (2/6) | +0.99 [+0.24, +1.75] (6/6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.23 [-4.54, +2.08] (0/2) | -0.44 [-4.76, +3.89] (0/2) | +0.72 [-7.61, +9.05] (2/2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.34 [-7.30, +4.63] (0/2) | -0.11 [-3.45, +3.24] (1/2) | +1.24 [-0.46, +2.93] (2/2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.68 [-5.38, +4.02] (0/2) | -0.35 [-7.71, +7.02] (1/2) | +1.03 [-9.75, +11.81] (2/2) |
| strong | – | – | – |

### best rule

Pooled:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.87 [-3.02, -0.72] (0/6) | -1.01 [-1.91, -0.11] (1/6) | +0.34 [-0.74, +1.42] (3/6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.90 [-11.21, +7.42] (0/2) | -1.47 [-7.00, +4.06] (0/2) | -0.18 [-1.46, +1.11] (0/2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -2.40 [-19.40, +14.59] (0/2) | -0.89 [-15.03, +13.26] (1/2) | +0.12 [-13.49, +13.72] (1/2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.30 [-4.36, +1.76] (0/2) | -0.68 [-3.82, +2.46] (0/2) | +1.09 [-8.92, +11.09] (2/2) |
| strong | – | – | – |

### mean rule

Pooled:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.42 [-0.78, -0.06] (1/6) | +0.04 [-0.23, +0.30] (4/6) | +0.78 [+0.32, +1.24] (6/6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.06 [-2.35, +2.23] (1/2) | +0.08 [-0.21, +0.36] (2/2) | +0.78 [-2.55, +4.11] (2/2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.74 [-2.33, +0.86] (0/2) | -0.03 [-2.62, +2.56] (1/2) | +0.70 [-3.02, +4.43] (2/2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.46 [-1.68, +0.76] (0/2) | +0.07 [-4.13, +4.26] (1/2) | +0.86 [-6.32, +8.04] (2/2) |
| strong | – | – | – |

## 2. The other paired differences (native rule, pooled)

### harmonic OPC − likelihood (calibration-aware head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -2.03 [-3.13, -0.93] (0/6) | -2.06 [-2.91, -1.21] (0/6) | -1.56 [-2.71, -0.41] (0/6) |
| strong | – | – | – |

### raw DR − likelihood (ordinary head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -2.06 [-3.14, -0.99] (0/6) | -1.57 [-2.48, -0.65] (0/6) | -0.20 [-1.46, +1.06] (3/6) |
| strong | – | – | – |

### harmonic OPC, oracle q − likelihood (ordinary head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -1.44 [-2.59, -0.28] (0/6) | -0.78 [-1.54, -0.02] (0/6) | +0.74 [-0.38, +1.86] (5/6) |
| strong | – | – | – |

### likelihood (calibration-aware head) − likelihood (ordinary head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.24 [-0.12, +0.61] (5/6) | +0.95 [+0.53, +1.37] (6/6) | +1.99 [+0.99, +2.98] (6/6) |
| strong | – | – | – |

### harmonic OPC, oracle q − harmonic OPC

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.35 [+0.10, +0.59] (6/6) | +0.33 [-0.08, +0.75] (5/6) | +0.32 [-0.05, +0.68] (5/6) |
| strong | – | – | – |

### raw DR − harmonic OPC

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.28 [-0.55, -0.01] (1/6) | -0.45 [-0.70, -0.20] (0/6) | -0.62 [-1.19, -0.05] (0/6) |
| strong | – | – | – |

## 3. Per dataset, pooled over cells (native and best rules)

| pair | rule | ml | kuairand | anime | all |
|---|---|---|---|---|---|
| opc − likelihood | native | -1.26 [-2.22, -0.30] (0/6) | -1.05 [-3.06, +0.97] (1/6) | -0.16 [-1.36, +1.04] (2/6) | -0.82 [-1.52, -0.13] (3/18) |
| opc − likelihood | best | -1.18 [-2.20, -0.16] (0/6) | -1.06 [-2.86, +0.75] (2/6) | -0.30 [-1.59, +1.00] (2/6) | -0.84 [-1.50, -0.19] (4/18) |
| opc − likelihood_calib | native | -2.40 [-3.38, -1.42] (0/6) | -2.14 [-3.17, -1.10] (0/6) | -1.11 [-1.55, -0.67] (0/6) | -1.88 [-2.36, -1.41] (0/18) |
| opc − likelihood_calib | best | -2.32 [-3.29, -1.36] (0/6) | -2.01 [-3.09, -0.94] (0/6) | -1.19 [-1.64, -0.74] (0/6) | -1.84 [-2.30, -1.38] (0/18) |
| opc_raw − likelihood | native | -1.72 [-2.52, -0.92] (0/6) | -1.33 [-3.36, +0.70] (1/6) | -0.78 [-1.62, +0.06] (2/6) | -1.28 [-1.91, -0.65] (3/18) |
| opc_raw − likelihood | best | -1.63 [-2.43, -0.84] (0/6) | -1.36 [-3.15, +0.44] (1/6) | -0.79 [-1.82, +0.23] (2/6) | -1.26 [-1.86, -0.66] (3/18) |
| opc_oq − likelihood | native | -0.81 [-1.92, +0.30] (2/6) | -0.92 [-2.58, +0.73] (1/6) | +0.25 [-0.93, +1.43] (2/6) | -0.49 [-1.15, +0.16] (5/18) |
| opc_oq − likelihood | best | -0.51 [-1.67, +0.65] (2/6) | +0.38 [-1.15, +1.92] (5/6) | +0.15 [-1.08, +1.38] (2/6) | +0.01 [-0.61, +0.63] (9/18) |
| likelihood_calib − likelihood | native | +1.14 [+0.08, +2.19] (6/6) | +1.09 [-0.00, +2.18] (5/6) | +0.95 [-0.05, +1.95] (6/6) | +1.06 [+0.59, +1.53] (17/18) |
| likelihood_calib − likelihood | best | +1.14 [+0.09, +2.20] (6/6) | +0.96 [+0.13, +1.78] (5/6) | +0.89 [-0.14, +1.92] (5/6) | +1.00 [+0.56, +1.43] (16/18) |
| opc_oq − opc | native | +0.46 [+0.22, +0.69] (6/6) | +0.12 [-0.30, +0.55] (4/6) | +0.41 [+0.20, +0.62] (6/6) | +0.33 [+0.18, +0.49] (16/18) |
| opc_oq − opc | best | +0.67 [+0.41, +0.93] (6/6) | +1.44 [+1.05, +1.84] (6/6) | +0.45 [+0.29, +0.61] (6/6) | +0.85 [+0.60, +1.11] (18/18) |
| opc_raw − opc | native | -0.46 [-0.69, -0.22] (0/6) | -0.28 [-0.56, -0.01] (1/6) | -0.62 [-1.19, -0.04] (0/6) | -0.45 [-0.64, -0.26] (1/18) |
| opc_raw − opc | best | -0.45 [-0.78, -0.12] (0/6) | -0.30 [-0.57, -0.03] (0/6) | -0.49 [-0.93, -0.06] (0/6) | -0.42 [-0.58, -0.25] (0/18) |

## 4. Gains over the logger (native rule, pooled)

### likelihood (ordinary head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +3.68 [+1.76, +5.60] | +3.12 [+1.53, +4.71] | +1.77 [+0.18, +3.36] |
| strong | – | – | – |

### likelihood (calibration-aware head)

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +3.92 [+1.99, +5.86] | +4.07 [+2.51, +5.63] | +3.75 [+2.26, +5.24] |
| strong | – | – | – |

### harmonic OPC

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +1.90 [+0.69, +3.10] | +2.00 [+0.78, +3.23] | +2.19 [+1.03, +3.35] |
| strong | – | – | – |

### raw DR

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +1.62 [+0.30, +2.93] | +1.55 [+0.46, +2.64] | +1.57 [+0.53, +2.60] |
| strong | – | – | – |

### harmonic OPC, oracle q

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +2.24 [+0.99, +3.49] | +2.34 [+1.03, +3.64] | +2.51 [+1.47, +3.54] |
| strong | – | – | – |

## 5. Decomposition of the gap to θ_value* (native rule, pooled)

gap = M (objective) + E+O (finite sample + optimization, to the best of 20) + S (selection).

| shift | response | arm | gap | M | E+O | S |
|---|---|---|---|---|---|---|
| moderate | none | likelihood | +2.70 [+1.69, +3.71] | -0.00 [-0.00, -0.00] | +2.58 [+1.48, +3.68] | +0.12 [-0.17, +0.41] |
| moderate | none | likelihood_calib | +2.46 [+1.74, +3.18] | -0.00 [-0.00, -0.00] | +2.36 [+1.61, +3.12] | +0.10 [-0.11, +0.31] |
| moderate | none | opc | +4.49 [+3.73, +5.24] | +2.38 [+1.72, +3.04] | +2.07 [+1.72, +2.42] | +0.04 [-0.02, +0.09] |
| moderate | none | opc_raw | +4.77 [+4.17, +5.37] | +0.00 [+0.00, +0.00] | +4.62 [+4.00, +5.24] | +0.15 [+0.00, +0.30] |
| moderate | none | opc_oq | +4.14 [+3.32, +4.96] | +0.00 [+0.00, +0.00] | +3.65 [+2.83, +4.47] | +0.49 [-0.17, +1.14] |
| moderate | moderate | likelihood | +3.21 [+2.35, +4.07] | +1.05 [+0.49, +1.60] | +2.17 [+1.57, +2.76] | +0.00 [+0.00, +0.00] |
| moderate | moderate | likelihood_calib | +2.26 [+1.61, +2.91] | -0.00 [-0.01, +0.00] | +2.26 [+1.62, +2.90] | +0.00 [-0.00, +0.01] |
| moderate | moderate | opc | +4.32 [+3.58, +5.07] | +2.36 [+1.88, +2.85] | +1.86 [+1.18, +2.53] | +0.10 [-0.08, +0.29] |
| moderate | moderate | opc_raw | +4.78 [+4.14, +5.41] | +0.00 [+0.00, +0.00] | +4.71 [+4.06, +5.37] | +0.07 [-0.01, +0.14] |
| moderate | moderate | opc_oq | +3.99 [+3.08, +4.91] | +0.00 [+0.00, +0.00] | +3.25 [+2.21, +4.30] | +0.74 [-0.14, +1.62] |
| moderate | strong | likelihood | +4.19 [+2.71, +5.67] | +2.54 [+1.08, +4.01] | +1.48 [+0.43, +2.53] | +0.17 [-0.17, +0.51] |
| moderate | strong | likelihood_calib | +2.20 [+1.50, +2.91] | -0.00 [-0.00, +0.00] | +2.20 [+1.50, +2.91] | +0.00 [+0.00, +0.00] |
| moderate | strong | opc | +3.77 [+2.57, +4.96] | +2.22 [+1.77, +2.67] | +1.46 [+0.56, +2.36] | +0.09 [-0.00, +0.17] |
| moderate | strong | opc_raw | +4.39 [+3.19, +5.59] | +0.00 [+0.00, +0.00] | +4.27 [+3.06, +5.47] | +0.12 [-0.04, +0.28] |
| moderate | strong | opc_oq | +3.45 [+2.39, +4.51] | +0.00 [+0.00, +0.00] | +2.89 [+1.70, +4.07] | +0.56 [-0.06, +1.19] |

## 6. Expansion gate (§7)

| shift | response | M_L | M_calib | M_harm | M_L_minus_M_calib | opc_minus_lik | calib_minus_lik | calib_minus_lik_lo | calib_minus_lik_hi | a | b | c | d | expand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moderate | none | -0.0 | -0.0 | 2.38 | -0.0 | -1.78 | 0.24 | -0.12 | 0.61 | False | True | False | False | True |
| moderate | moderate | 1.05 | -0.0 | 2.36 | 1.05 | -1.11 | 0.95 | 0.53 | 1.37 | True | True | False | True | True |
| moderate | strong | 2.54 | -0.0 | 2.22 | 2.54 | 0.42 | 1.99 | 0.99 | 2.98 | True | True | False | True | True |

Harmonic dominance: {"share_worlds_M_harm_ge_M_L": 0.7777777777777778, "worlds": 18, "share_M_harm_gt_M_L_beyond_0.25": 0.8235294117647058, "worlds_beyond_0.25": 17, "gap_native_pooled": {"likelihood": 3.3681372797338995, "likelihood_calib": 2.308342977379855, "opc": 4.191999792783349, "opc_raw": 4.644866541907611}, "largest_gap_arm": "opc_raw", "dominant": false, "dominant_beyond_0.25": false}

## 7. Data identity and divergences

Worlds with identical rows and paired configurations: 18 of 18.
Cells over the 5% divergence limit: 0.
Largest |population source − trainer logger| greedy value: 2.0795316024724286e-07 points.
