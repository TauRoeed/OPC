# Population objectives (§9)

36 worlds (seeds 100, 101). M = V_g(θ_value*) − V_g(θ_obj*), points: what each objective's population optimum gives up against the value optimum (greedy policies). M_h0.1 is B3's target; M_h0.2 A's; M_h0.1_D1 D's (harmonic 0.1 with the D1 population fit); M_s2pen0.01 B's (raw value − 0.01 S2). Harmonic CRM has no population objective here (its penalty vanishes as N grows).

## M by regime (mean over worlds)

| quantity | R1 | R2 | R3 | C1 | C2 | S1 | all |
|---|---|---|---|---|---|---|---|
| available_gain | 2.65 | 4.99 | 5.59 | 5.99 | 0.0 | 10.61 | 4.97 |
| M_likelihood | 1.84 | 2.01 | 0.93 | -0.0 | 1.47 | 2.37 | 1.44 |
| M_calib | 0.0 | 0.0 | 0.0 | -0.0 | 0.0 | 0.0 | 0.0 |
| M_h0.005 | 0.1 | 0.4 | 0.44 | 0.46 | 0.0 | 1.91 | 0.55 |
| M_h0.01 | 0.16 | 0.52 | 0.59 | 0.61 | 0.0 | 2.4 | 0.71 |
| M_h0.03 | 0.29 | 0.81 | 0.89 | 0.94 | 0.0 | 3.03 | 0.99 |
| M_h0.1 | 0.55 | 1.29 | 1.44 | 1.52 | 0.01 | 4.37 | 1.53 |
| M_h0.2 | 0.75 | 1.65 | 1.86 | 2.01 | 0.02 | 5.23 | 1.92 |
| M_h0.1_D1 | 0.07 | 0.11 | 0.47 | 0.56 | 0.02 | 1.15 | 0.4 |
| M_h0.1_D2 | 0.0 | 0.0 | 0.0 | 0.01 | 0.0 | 1.31 | 0.22 |
| M_s2pen0.001 | 0.29 | 0.91 | 1.04 | 1.11 | 0.01 | 3.81 | 1.2 |
| M_s2pen0.01 | 0.87 | 1.99 | 2.25 | 2.4 | 0.02 | 5.67 | 2.2 |
| M_s2cap_5 | 0.53 | 1.9 | 2.25 | 2.4 | 0.01 | 5.67 | 2.13 |
| M_s2cap_50 | 0.12 | 0.55 | 0.73 | 0.82 | 0.0 | 3.65 | 0.98 |

## M by dataset (mean over worlds)

| quantity | ml | kuairand | anime |
|---|---|---|---|
| available_gain | 4.96 | 5.16 | 4.79 |
| M_likelihood | 1.61 | 1.23 | 1.47 |
| M_calib | 0.0 | 0.0 | 0.0 |
| M_h0.005 | 0.55 | 0.59 | 0.51 |
| M_h0.01 | 0.78 | 0.71 | 0.64 |
| M_h0.03 | 1.05 | 0.96 | 0.97 |
| M_h0.1 | 1.66 | 1.36 | 1.57 |
| M_h0.2 | 2.2 | 1.63 | 1.94 |
| M_h0.1_D1 | 0.55 | 0.57 | 0.07 |
| M_h0.1_D2 | 0.66 | 0.01 | 0.0 |
| M_s2pen0.001 | 1.03 | 1.49 | 1.06 |
| M_s2pen0.01 | 2.09 | 2.51 | 2.0 |
| M_s2cap_5 | 2.04 | 2.39 | 1.95 |
| M_s2cap_50 | 0.78 | 1.28 | 0.87 |

## The optima's S2 (median over worlds)

| regime | S2_value_star | S2_h0.005 | S2_h0.2 | S2_s2pen0.01 |
|---|---|---|---|---|
| R1 | 58.0 | 22.2 | 15.1 | 2.5 |
| R2 | 2340.6 | 51.3 | 32.5 | 2.6 |
| R3 | 2876.9 | 58.9 | 37.2 | 2.8 |
| C1 | 3197.5 | 65.4 | 35.6 | 2.9 |
| C2 | 7.9 | 7.8 | 7.8 | 2.2 |
| S1 | 22226552.5 | 797.3 | 8860.0 | 2.9 |
