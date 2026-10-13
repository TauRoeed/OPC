# Stage B pilots

18 worlds; 3780 trials; points; TrueGain = the selected policy's greedy gain over the source's.

## Mean over worlds

| family | arm | native | best | mean | recovery | diverged | sec_per_trial |
|---|---|---|---|---|---|---|---|
| A | shared_lr_opc_a2 | 0.805 | 0.87 | 0.574 | 0.211 | 0 | 14.44 |
| A | shared_lr_opc_a3 | 0.938 | 1.098 | 0.677 | 0.235 | 0 | 18.428 |
| A | shared_lr_opc_h0.005 | 0.794 | 0.881 | 0.551 | 0.202 | 0 | 9.573 |
| A | shared_lr_opc_h0.01 | 0.835 | 0.895 | 0.567 | 0.216 | 0 | 9.696 |
| A | shared_lr_opc_h0.03 | 0.899 | 0.98 | 0.623 | 0.22 | 0 | 9.322 |
| A | shared_lr_opc_h0.2 | 1.128 | 1.234 | 0.771 | 0.289 | 0 | 9.255 |
| B | shared_lr_opc_klb0.001 | 0.821 | 0.885 | 0.563 | 0.212 | 0 | 6.757 |
| B | shared_lr_opc_klb0.003 | 0.841 | 0.933 | 0.581 | 0.214 | 0 | 6.586 |
| B | shared_lr_opc_s2b0.0003 | 0.783 | 0.875 | 0.571 | 0.197 | 0 | 6.749 |
| B | shared_lr_opc_s2b0.001 | 0.825 | 0.902 | 0.595 | 0.215 | 0 | 7.407 |
| B | shared_lr_opc_s2b0.003 | 0.96 | 1.014 | 0.655 | 0.255 | 0 | 6.54 |
| B | shared_lr_opc_s2b0.01 | 1.15 | 1.216 | 0.734 | 0.299 | 0 | 6.912 |
| B | shared_lr_opc_s2c10 | 0.739 | 0.842 | 0.558 | 0.191 | 0 | 9.775 |
| B | shared_lr_opc_s2c20 | 0.777 | 0.854 | 0.549 | 0.198 | 0 | 9.987 |
| B | shared_lr_opc_s2c5 | 0.952 | 0.974 | 0.621 | 0.253 | 0 | 10.271 |
| B | shared_lr_opc_s2c50 | 0.754 | 0.847 | 0.557 | 0.19 | 0 | 9.169 |
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
| shared_lr_opc_klb0.001 | 0.47 |
| shared_lr_opc_klb0.003 | 0.43 |
| shared_lr_opc_oq | 0.78 |
| shared_lr_opc_raw | 0.38 |
| shared_lr_opc_s2b0.0003 | 0.37 |
| shared_lr_opc_s2b0.001 | 0.47 |
| shared_lr_opc_s2b0.003 | 0.52 |
| shared_lr_opc_s2b0.01 | 0.59 |
| shared_lr_opc_s2c10 | 0.38 |
| shared_lr_opc_s2c20 | 0.38 |
| shared_lr_opc_s2c5 | 0.51 |
| shared_lr_opc_s2c50 | 0.38 |

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
| shared_lr_opc_klb0.001 | 0.99 |
| shared_lr_opc_klb0.003 | 0.99 |
| shared_lr_opc_oq | 1.62 |
| shared_lr_opc_raw | 0.87 |
| shared_lr_opc_s2b0.0003 | 0.96 |
| shared_lr_opc_s2b0.001 | 1.01 |
| shared_lr_opc_s2b0.003 | 1.17 |
| shared_lr_opc_s2b0.01 | 1.44 |
| shared_lr_opc_s2c10 | 0.93 |
| shared_lr_opc_s2c20 | 0.96 |
| shared_lr_opc_s2c5 | 1.19 |
| shared_lr_opc_s2c50 | 0.86 |

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
| shared_lr_opc_klb0.001 | 1.0 |
| shared_lr_opc_klb0.003 | 1.11 |
| shared_lr_opc_oq | 1.72 |
| shared_lr_opc_raw | 1.02 |
| shared_lr_opc_s2b0.0003 | 1.02 |
| shared_lr_opc_s2b0.001 | 1.0 |
| shared_lr_opc_s2b0.003 | 1.18 |
| shared_lr_opc_s2b0.01 | 1.42 |
| shared_lr_opc_s2c10 | 0.9 |
| shared_lr_opc_s2c20 | 0.99 |
| shared_lr_opc_s2c5 | 1.16 |
| shared_lr_opc_s2c50 | 1.02 |

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
| shared_lr_opc_klb0.001 | -0.15 | -0.23 | -0.27 | -0.22 |
| shared_lr_opc_klb0.003 | -0.19 | -0.24 | -0.16 | -0.2 |
| shared_lr_opc_oq | 0.15 | 0.4 | 0.46 | 0.34 |
| shared_lr_opc_raw | -0.24 | -0.36 | -0.25 | -0.28 |
| shared_lr_opc_s2b0.0003 | -0.25 | -0.26 | -0.25 | -0.25 |
| shared_lr_opc_s2b0.001 | -0.16 | -0.21 | -0.26 | -0.21 |
| shared_lr_opc_s2b0.003 | -0.1 | -0.05 | -0.09 | -0.08 |
| shared_lr_opc_s2b0.01 | -0.03 | 0.22 | 0.15 | 0.11 |
| shared_lr_opc_s2c10 | -0.24 | -0.29 | -0.36 | -0.3 |
| shared_lr_opc_s2c20 | -0.24 | -0.27 | -0.27 | -0.26 |
| shared_lr_opc_s2c5 | -0.11 | -0.03 | -0.11 | -0.09 |
| shared_lr_opc_s2c50 | -0.24 | -0.36 | -0.25 | -0.28 |

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
| shared_lr_opc_klb0.001 | 1.3 | 0.26 | -1.39 | 0.06 |
| shared_lr_opc_klb0.003 | 1.25 | 0.26 | -1.29 | 0.08 |
| shared_lr_opc_oq | 1.6 | 0.9 | -0.67 | 0.61 |
| shared_lr_opc_raw | 1.2 | 0.14 | -1.37 | -0.01 |
| shared_lr_opc_s2b0.0003 | 1.19 | 0.23 | -1.37 | 0.02 |
| shared_lr_opc_s2b0.001 | 1.29 | 0.28 | -1.39 | 0.06 |
| shared_lr_opc_s2b0.003 | 1.35 | 0.45 | -1.21 | 0.2 |
| shared_lr_opc_s2b0.01 | 1.42 | 0.71 | -0.97 | 0.39 |
| shared_lr_opc_s2c10 | 1.2 | 0.21 | -1.49 | -0.03 |
| shared_lr_opc_s2c20 | 1.2 | 0.23 | -1.4 | 0.01 |
| shared_lr_opc_s2c5 | 1.33 | 0.46 | -1.23 | 0.19 |
| shared_lr_opc_s2c50 | 1.2 | 0.13 | -1.37 | -0.01 |

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
| shared_lr_opc_klb0.001 | -0.74 | -1.69 | -2.25 | -1.56 |
| shared_lr_opc_klb0.003 | -0.78 | -1.69 | -2.14 | -1.54 |
| shared_lr_opc_oq | -0.44 | -1.05 | -1.52 | -1.0 |
| shared_lr_opc_raw | -0.83 | -1.81 | -2.23 | -1.62 |
| shared_lr_opc_s2b0.0003 | -0.84 | -1.72 | -2.23 | -1.6 |
| shared_lr_opc_s2b0.001 | -0.75 | -1.67 | -2.24 | -1.55 |
| shared_lr_opc_s2b0.003 | -0.69 | -1.5 | -2.06 | -1.42 |
| shared_lr_opc_s2b0.01 | -0.62 | -1.24 | -1.83 | -1.23 |
| shared_lr_opc_s2c10 | -0.83 | -1.74 | -2.34 | -1.64 |
| shared_lr_opc_s2c20 | -0.83 | -1.72 | -2.25 | -1.6 |
| shared_lr_opc_s2c5 | -0.7 | -1.49 | -2.09 | -1.43 |
| shared_lr_opc_s2c50 | -0.83 | -1.82 | -2.23 | -1.63 |

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
 },
 "B": {
  "family": "B",
  "decision": "shared_lr_opc_s2b0.01",
  "reason": "highest mean native TrueGain",
  "ranking": {
   "shared_lr_opc_s2b0.01": 1.1501,
   "shared_lr_opc_s2b0.003": 0.9602,
   "shared_lr_opc_s2c5": 0.9522,
   "shared_lr_opc_klb0.003": 0.8405,
   "shared_lr_opc_s2b0.001": 0.8255,
   "shared_lr_opc_klb0.001": 0.8208,
   "shared_lr_opc_s2b0.0003": 0.7831,
   "shared_lr_opc_s2c20": 0.777,
   "shared_lr_opc_s2c50": 0.7538,
   "shared_lr_opc_s2c10": 0.739
  },
  "vs_B3_native": {
   "mean": 0.1127224992764456,
   "lo": -0.025456052184771524,
   "hi": 0.2509010507376627,
   "worlds": 18
  }
 }
}
```
