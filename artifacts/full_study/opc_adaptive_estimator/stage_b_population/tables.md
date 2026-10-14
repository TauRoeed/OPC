# Population objectives (§9)

18 worlds (seeds 300, 301). M = V_g(θ_value*) − V_g(θ_obj*), points: what each objective's population optimum gives up against the value optimum (greedy policies). M_h0.1 is B3's target; M_h0.2 A's; M_h0.1_D1 D's (harmonic 0.1 with the D1 population fit); M_s2pen0.01 B's (raw value − 0.01 S2). Harmonic CRM has no population objective here (its penalty vanishes as N grows).

## M by regime (mean over worlds)

| quantity | R1 | R2 | R3 | all |
|---|---|---|---|---|
| available_gain | 2.33 | 4.39 | 4.81 | 3.84 |
| M_likelihood | 2.69 | 2.91 | 1.24 | 2.28 |
| M_calib | 0.0 | 0.0 | 0.0 | 0.0 |
| M_h0.005 | 0.08 | 0.32 | 0.34 | 0.25 |
| M_h0.01 | 0.13 | 0.43 | 0.47 | 0.34 |
| M_h0.03 | 0.27 | 0.74 | 0.77 | 0.59 |
| M_h0.1 | 0.55 | 1.28 | 1.37 | 1.07 |
| M_h0.2 | 0.82 | 1.75 | 1.86 | 1.48 |
| M_h0.1_D1 | 0.07 | 0.12 | 0.04 | 0.08 |
| M_h0.1_D2 | 0.43 | 0.0 | 0.0 | 0.14 |
| M_s2pen0.001 | 0.23 | 0.72 | 0.79 | 0.58 |
| M_s2pen0.01 | 0.76 | 1.68 | 1.84 | 1.43 |
| M_s2cap_5 | 0.44 | 1.53 | 1.84 | 1.27 |
| M_s2cap_50 | 0.04 | 0.41 | 0.49 | 0.31 |

## M by dataset (mean over worlds)

| quantity | ml | kuairand | anime |
|---|---|---|---|
| available_gain | 4.24 | 3.76 | 3.52 |
| M_likelihood | 2.08 | 3.3 | 1.45 |
| M_calib | 0.0 | 0.0 | 0.0 |
| M_h0.005 | 0.29 | 0.23 | 0.22 |
| M_h0.01 | 0.39 | 0.31 | 0.32 |
| M_h0.03 | 0.68 | 0.57 | 0.52 |
| M_h0.1 | 1.22 | 1.03 | 0.96 |
| M_h0.2 | 1.69 | 1.47 | 1.26 |
| M_h0.1_D1 | 0.06 | 0.1 | 0.06 |
| M_h0.1_D2 | 0.0 | 0.43 | 0.0 |
| M_s2pen0.001 | 0.62 | 0.56 | 0.56 |
| M_s2pen0.01 | 1.59 | 1.38 | 1.32 |
| M_s2cap_5 | 1.46 | 1.19 | 1.16 |
| M_s2cap_50 | 0.29 | 0.28 | 0.37 |

## The optima's S2 (median over worlds)

| regime | S2_value_star | S2_h0.005 | S2_h0.2 | S2_s2pen0.01 |
|---|---|---|---|---|
| R1 | 46.5 | 20.3 | 14.2 | 2.5 |
| R2 | 982.2 | 44.4 | 28.3 | 2.6 |
| R3 | 1040.4 | 49.1 | 30.1 | 2.7 |
