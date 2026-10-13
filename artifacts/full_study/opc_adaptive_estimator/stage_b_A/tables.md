# Stage B pilots

18 worlds; 1980 trials; points; TrueGain = the selected policy's greedy gain over the source's.

## Mean over worlds

| family | arm | native | best | mean | recovery | diverged | sec_per_trial |
|---|---|---|---|---|---|---|---|
| A | shared_lr_opc_a2 | 0.805 | 0.87 | 0.574 | 0.211 | 0 | 14.44 |
| A | shared_lr_opc_a3 | 0.938 | 1.098 | 0.677 | 0.235 | 0 | 18.428 |
| A | shared_lr_opc_h0.005 | 0.794 | 0.881 | 0.551 | 0.202 | 0 | 9.573 |
| A | shared_lr_opc_h0.01 | 0.835 | 0.895 | 0.567 | 0.216 | 0 | 9.696 |
| A | shared_lr_opc_h0.03 | 0.899 | 0.98 | 0.623 | 0.22 | 0 | 9.322 |
| A | shared_lr_opc_h0.2 | 1.128 | 1.234 | 0.771 | 0.289 | 0 | 9.255 |
| B1 | shared_lr_likelihood | 0.765 | 1.227 | 0.202 | 0.13 | 0 | 8.489 |
| B2 | shared_lr_likelihood_calib | 2.38 | 2.539 | 0.959 | 0.601 | 0 | 9.32 |
| B3 | shared_lr_opc | 1.037 | 1.104 | 0.705 | 0.275 | 0 | 8.782 |
| B4 | shared_lr_opc_raw | 0.755 | 0.845 | 0.549 | 0.19 | 0 | 9.097 |
| B5 | shared_lr_opc_oq | 1.376 | 1.539 | 0.887 | 0.357 | 0 | 9.439 |

### R1: native TrueGain per arm

| arm | native_gain |
|---|---|
| shared_lr_likelihood | -0.82 |
| shared_lr_likelihood_calib | 1.21 |
| shared_lr_opc | 0.62 |
| shared_lr_opc_a2 | 0.46 |
| shared_lr_opc_a3 | 0.5 |
| shared_lr_opc_h0.005 | 0.48 |
| shared_lr_opc_h0.01 | 0.47 |
| shared_lr_opc_h0.03 | 0.41 |
| shared_lr_opc_h0.2 | 0.55 |
| shared_lr_opc_oq | 0.78 |
| shared_lr_opc_raw | 0.38 |

### R2: native TrueGain per arm

| arm | native_gain |
|---|---|
| shared_lr_likelihood | 0.73 |
| shared_lr_likelihood_calib | 2.68 |
| shared_lr_opc | 1.22 |
| shared_lr_opc_a2 | 1.0 |
| shared_lr_opc_a3 | 0.92 |
| shared_lr_opc_h0.005 | 0.84 |
| shared_lr_opc_h0.01 | 0.99 |
| shared_lr_opc_h0.03 | 1.12 |
| shared_lr_opc_h0.2 | 1.31 |
| shared_lr_opc_oq | 1.62 |
| shared_lr_opc_raw | 0.87 |

### R3: native TrueGain per arm

| arm | native_gain |
|---|---|
| shared_lr_likelihood | 2.39 |
| shared_lr_likelihood_calib | 3.25 |
| shared_lr_opc | 1.27 |
| shared_lr_opc_a2 | 0.95 |
| shared_lr_opc_a3 | 1.4 |
| shared_lr_opc_h0.005 | 1.06 |
| shared_lr_opc_h0.01 | 1.04 |
| shared_lr_opc_h0.03 | 1.17 |
| shared_lr_opc_h0.2 | 1.52 |
| shared_lr_opc_oq | 1.72 |
| shared_lr_opc_raw | 1.02 |

## arm − B3 (native; mean over worlds, by regime)

| arm | R1 | R2 | R3 | all |
|---|---|---|---|---|
| shared_lr_likelihood | -1.45 | -0.49 | 1.12 | -0.27 |
| shared_lr_likelihood_calib | 0.59 | 1.46 | 1.98 | 1.34 |
| shared_lr_opc_a2 | -0.17 | -0.22 | -0.31 | -0.23 |
| shared_lr_opc_a3 | -0.12 | -0.31 | 0.13 | -0.1 |
| shared_lr_opc_h0.005 | -0.14 | -0.38 | -0.2 | -0.24 |
| shared_lr_opc_h0.01 | -0.15 | -0.23 | -0.23 | -0.2 |
| shared_lr_opc_h0.03 | -0.21 | -0.11 | -0.1 | -0.14 |
| shared_lr_opc_h0.2 | -0.07 | 0.09 | 0.25 | 0.09 |
| shared_lr_opc_oq | 0.15 | 0.4 | 0.46 | 0.34 |
| shared_lr_opc_raw | -0.24 | -0.36 | -0.25 | -0.28 |

## arm − B1 (native; mean over worlds, by regime)

| arm | R1 | R2 | R3 | all |
|---|---|---|---|---|
| shared_lr_likelihood_calib | 2.04 | 1.95 | 0.86 | 1.61 |
| shared_lr_opc | 1.45 | 0.49 | -1.12 | 0.27 |
| shared_lr_opc_a2 | 1.28 | 0.28 | -1.44 | 0.04 |
| shared_lr_opc_a3 | 1.32 | 0.19 | -0.99 | 0.17 |
| shared_lr_opc_h0.005 | 1.3 | 0.12 | -1.33 | 0.03 |
| shared_lr_opc_h0.01 | 1.3 | 0.27 | -1.35 | 0.07 |
| shared_lr_opc_h0.03 | 1.24 | 0.39 | -1.22 | 0.13 |
| shared_lr_opc_h0.2 | 1.37 | 0.58 | -0.87 | 0.36 |
| shared_lr_opc_oq | 1.6 | 0.9 | -0.67 | 0.61 |
| shared_lr_opc_raw | 1.2 | 0.14 | -1.37 | -0.01 |

## arm − B2 (native; mean over worlds, by regime)

| arm | R1 | R2 | R3 | all |
|---|---|---|---|---|
| shared_lr_likelihood | -2.04 | -1.95 | -0.86 | -1.61 |
| shared_lr_opc | -0.59 | -1.46 | -1.98 | -1.34 |
| shared_lr_opc_a2 | -0.76 | -1.67 | -2.29 | -1.57 |
| shared_lr_opc_a3 | -0.71 | -1.76 | -1.85 | -1.44 |
| shared_lr_opc_h0.005 | -0.74 | -1.84 | -2.18 | -1.59 |
| shared_lr_opc_h0.01 | -0.74 | -1.68 | -2.21 | -1.54 |
| shared_lr_opc_h0.03 | -0.8 | -1.56 | -2.08 | -1.48 |
| shared_lr_opc_h0.2 | -0.66 | -1.37 | -1.72 | -1.25 |
| shared_lr_opc_oq | -0.44 | -1.05 | -1.52 | -1.0 |
| shared_lr_opc_raw | -0.83 | -1.81 | -2.23 | -1.62 |

## Decisions (§11)

```
{
 "A": {
  "family": "A",
  "decision": "shared_lr_opc_h0.2",
  "reason": "highest mean native TrueGain",
  "ranking": {
   "shared_lr_opc_h0.2": 1.1279,
   "shared_lr_opc_a3": 0.9382,
   "shared_lr_opc_h0.03": 0.8988,
   "shared_lr_opc_h0.01": 0.8345,
   "shared_lr_opc_a2": 0.8051,
   "shared_lr_opc_h0.005": 0.7945
  },
  "vs_B3_native": {
   "mean": 0.09051711008865439,
   "lo": -0.014469353350138409,
   "hi": 0.19550357352744718,
   "worlds": 18
  }
 }
}
```
