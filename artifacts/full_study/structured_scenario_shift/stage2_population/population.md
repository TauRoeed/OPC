# Structured scenario shift: population optima

18 worlds. Greedy CTR points; mean [95% CI] over worlds (paired by world). M = V_g(θ_value*) − V_g(θ_obj*): positive is a loss of the objective's population optimum against the value optimum.

## M_L, ordinary likelihood

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (6) | +1.05 [+0.49, +1.60] (6) | +2.54 [+1.08, +4.01] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.01, +0.00] (2) | +1.08 [-6.77, +8.92] (2) | +2.07 [-11.09, +15.23] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (2) | +0.82 [-5.25, +6.89] (2) | +2.92 [-20.32, +26.15] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (2) | +1.24 [+0.34, +2.15] (2) | +2.65 [-0.71, +6.00] (2) |
| strong | – | – | – |

## M_calib, calibration-aware likelihood

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (6) | -0.00 [-0.01, +0.00] (6) | -0.00 [-0.00, +0.00] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.01, +0.00] (2) | -0.01 [-0.02, +0.00] (2) | -0.00 [-0.02, +0.01] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (2) | -0.00 [-0.03, +0.02] (2) | +0.00 [-0.02, +0.02] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, -0.00] (2) | -0.00 [-0.00, +0.00] (2) | +0.00 [-0.01, +0.02] (2) |
| strong | – | – | – |

## M_harm, harmonic DR with q̂_∞

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +2.38 [+1.72, +3.04] (6) | +2.36 [+1.88, +2.85] (6) | +2.22 [+1.77, +2.67] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +2.58 [-3.05, +8.20] (2) | +2.74 [+2.57, +2.91] (2) | +2.53 [-2.08, +7.15] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +2.82 [-1.30, +6.94] (2) | +2.52 [-1.36, +6.40] (2) | +2.28 [-0.61, +5.16] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +1.74 [-0.92, +4.39] (2) | +1.84 [+1.24, +2.44] (2) | +1.85 [-0.67, +4.38] (2) |
| strong | – | – | – |

## M_L − M_calib

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, +0.00] (6) | +1.05 [+0.49, +1.61] (6) | +2.54 [+1.08, +4.01] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, +0.00] (2) | +1.08 [-6.77, +8.94] (2) | +2.07 [-11.10, +15.25] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, +0.00] (2) | +0.82 [-5.23, +6.87] (2) | +2.92 [-20.30, +26.13] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | -0.00 [-0.00, +0.00] (2) | +1.24 [+0.34, +2.14] (2) | +2.65 [-0.70, +5.99] (2) |
| strong | – | – | – |

## available gain V_g(truth) − V_g(source)

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +6.38 [+4.69, +8.08] (6) | +6.33 [+5.21, +7.46] (6) | +5.96 [+4.78, +7.14] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +6.54 [-8.57, +21.65] (2) | +6.76 [+5.63, +7.89] (2) | +6.52 [-3.40, +16.43] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +7.87 [+3.40, +12.34] (2) | +7.16 [+4.58, +9.74] (2) | +6.23 [+4.22, +8.24] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +4.74 [+1.31, +8.17] (2) | +5.08 [-2.79, +12.94] (2) | +5.13 [-10.08, +20.34] (2) |
| strong | – | – | – |

## the class optimizer's gap V_g(truth) − V_g(θ_value*)

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.00 [+0.00, +0.00] (6) | +0.00 [+0.00, +0.01] (6) | +0.00 [-0.00, +0.00] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.00 [-0.00, +0.01] (2) | +0.01 [-0.00, +0.02] (2) | +0.00 [-0.02, +0.03] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.00 [+0.00, +0.00] (2) | +0.00 [-0.01, +0.02] (2) | +0.00 [-0.01, +0.01] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.00 [+0.00, +0.00] (2) | +0.00 [-0.00, +0.01] (2) | +0.00 [-0.00, +0.00] (2) |
| strong | – | – | – |

## q̂_∞ RMSE under θ_value*'s policy

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.149 [+0.109, +0.190] (6) | +0.151 [+0.124, +0.178] (6) | +0.150 [+0.127, +0.173] (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.146 [-0.126, +0.417] (2) | +0.154 [+0.135, +0.174] (2) | +0.157 [-0.075, +0.389] (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.188 [-0.098, +0.473] (2) | +0.175 [-0.029, +0.380] (2) | +0.162 [+0.038, +0.286] (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | +0.115 [-0.026, +0.255] (2) | +0.123 [+0.044, +0.203] (2) | +0.130 [-0.054, +0.315] (2) |
| strong | – | – | – |

## the 25k cross-fitted q̂'s RMSE under the logger

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (6) | – (6) | – (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

## the 25k cross-fitted q̂'s RMSE under θ_value*'s policy

Pooled over datasets and seeds (worlds):

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (6) | – (6) | – (6) |
| strong | – | – | – |

ml:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

kuairand:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

anime:

| shift \ response | none | moderate | strong |
|---|---|---|---|
| none | – | – | – |
| moderate | – (2) | – (2) | – (2) |
| strong | – | – | – |

## Sanity checks (§6)

Every check passes.

## Per world

| dataset | bias | seed | shift | response | M_L | M_calib | M_harm | optimizer_gap | available_gain | qinf_rmse_logging | qinf_rmse_target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ml | s-moderate.r-none.gated | 100 | moderate | none | -0.003 | -0.003 | 2.132 | 0.003 | 5.352 | 0.056 | 0.124 |
| ml | s-moderate.r-none.gated | 101 | moderate | none | -0.002 | -0.002 | 3.018 | 0.002 | 7.729 | 0.085 | 0.167 |
| ml | s-moderate.r-moderate.gated | 100 | moderate | moderate | 1.694 | -0.008 | 2.725 | 0.009 | 6.671 | 0.065 | 0.156 |
| ml | s-moderate.r-moderate.gated | 101 | moderate | moderate | 0.459 | -0.007 | 2.752 | 0.007 | 6.849 | 0.086 | 0.153 |
| ml | s-moderate.r-strong.gated | 100 | moderate | strong | 3.107 | -0.004 | 2.898 | 0.006 | 7.299 | 0.074 | 0.175 |
| ml | s-moderate.r-strong.gated | 101 | moderate | strong | 1.035 | -0.002 | 2.172 | 0.002 | 5.738 | 0.086 | 0.139 |
| kuairand | s-moderate.r-none.gated | 100 | moderate | none | -0.001 | -0.001 | 3.143 | 0.001 | 8.221 | 0.093 | 0.21 |
| kuairand | s-moderate.r-none.gated | 101 | moderate | none | -0.001 | -0.001 | 2.494 | 0.001 | 7.518 | 0.074 | 0.165 |
| kuairand | s-moderate.r-moderate.gated | 100 | moderate | moderate | 0.341 | -0.004 | 2.822 | 0.004 | 7.364 | 0.09 | 0.191 |
| kuairand | s-moderate.r-moderate.gated | 101 | moderate | moderate | 1.296 | -0.0 | 2.211 | 0.002 | 6.958 | 0.072 | 0.159 |
| kuairand | s-moderate.r-strong.gated | 100 | moderate | strong | 1.087 | -0.001 | 2.504 | 0.002 | 6.389 | 0.085 | 0.171 |
| kuairand | s-moderate.r-strong.gated | 101 | moderate | strong | 4.744 | 0.002 | 2.049 | 0.001 | 6.072 | 0.071 | 0.152 |
| anime | s-moderate.r-none.gated | 100 | moderate | none | -0.0 | -0.0 | 1.526 | 0.0 | 4.47 | 0.05 | 0.104 |
| anime | s-moderate.r-none.gated | 101 | moderate | none | -0.0 | -0.0 | 1.944 | 0.0 | 5.009 | 0.059 | 0.126 |
| anime | s-moderate.r-moderate.gated | 100 | moderate | moderate | 1.172 | -0.001 | 1.887 | 0.001 | 5.697 | 0.06 | 0.129 |
| anime | s-moderate.r-moderate.gated | 101 | moderate | moderate | 1.314 | -0.0 | 1.792 | 0.0 | 4.459 | 0.057 | 0.117 |
| anime | s-moderate.r-strong.gated | 100 | moderate | strong | 2.912 | 0.003 | 2.052 | 0.0 | 6.329 | 0.069 | 0.145 |
| anime | s-moderate.r-strong.gated | 101 | moderate | strong | 2.383 | 0.0 | 1.655 | 0.0 | 3.934 | 0.059 | 0.116 |
