# Hybrid results by dataset × misspec (95% t CIs)

Estimates are mean greedy CTR (levels) or CTR-point contrasts over seeds. CI = mean ± t_{0.975,n−1}·SE.

## When to use hybrid vs likelihood

Prefer hybrid_seq_f0.5 as the default vs pure likelihood when residual misspecification strength is unknown. Across dataset×misspec cells, summed regret of always-NLL is much larger than always-hybrid: well-spec (γ=0) costs are typically ≤0.5–0.8pt, while mid/extreme φ misspec yields +0.5 to +6pt H−NLL with CI LB>0 on most R* and extreme SoftAND/SoftThresh cells. Use pure NLL only when you have a strong prior of bilinear well-specification and need to squeeze the last γ=0 points (e.g. kuairand full-catalog γ=0). OPC−NLL≥0.5 is a conservative 'misspec is loud' flag but is not a good gate for switching away from hybrid—it misses many mid-γ hybrid wins where OPC still trails NLL.

**Best regret rule:** `hybrid_unless_OPC_NLL_le_-1.0`
- Sign agreement with H−NLL: 41
- Summed regret (pt): always-NLL=74.65, always-hybrid=3.52, best-rule=2.95
- Clear hybrid wins (CI LB>0) caught: 26/28

### Regime `extreme` — mean H−NLL=+4.97pt (5 cells; CI LB>0 in 5)

| dataset | misspec | γ | n | H−NLL | 95% CI | signs |
|---|---|---:|---:|---:|---|---:|
| ml | SoftAND*-adapt | 100 | 20 | +3.59 | [+2.22,+4.97] | 18/20 |
| ml | SoftThresh-adapt | 100 | 20 | +6.51 | [+4.95,+8.07] | 20/20 |
| ml | R* | 100 | 36 | +4.58 | [+3.89,+5.26] | 36/36 |
| ml | SoftThreshτ1.5 | 100 | 40 | +6.90 | [+5.85,+7.95] | 40/40 |
| ml | SoftAND* | 100 | 40 | +3.29 | [+2.41,+4.17] | 36/40 |

### Regime `mid` — mean H−NLL=+1.64pt (30 cells; CI LB>0 in 23)

| dataset | misspec | γ | n | H−NLL | 95% CI | signs |
|---|---|---:|---:|---:|---|---:|
| anime | SoftAND* | 0.75 | 5 | +0.92 | [-0.78,+2.61] | 4/5 |
| anime | SoftAND* | 0.25 | 9 | +0.35 | [-1.04,+1.75] | 4/9 |
| kuairec | R*-mlsize | 0.5 | 20 | +0.58 | [+0.12,+1.04] | 15/20 |
| kuairand | R*-mlsize | 0.5 | 20 | +0.53 | [+0.14,+0.92] | 13/20 |
| myket | R* | 0.5 | 20 | +0.77 | [+0.29,+1.25] | 17/20 |
| anime | R* | 0.6 | 20 | +0.72 | [+0.39,+1.05] | 17/20 |
| kuairand | R*-full | 0.5 | 20 | +0.03 | [-0.41,+0.47] | 11/20 |
| kuairec | R*-full | 0.5 | 20 | +0.61 | [+0.09,+1.15] | 14/20 |
| ml | R*-adapt | 0.5 | 20 | +0.99 | [+0.33,+1.65] | 14/20 |
| ml | R* | 10 | 36 | +4.66 | [+3.86,+5.46] | 34/36 |
| ml | SoftThreshτ1.5 | 4 | 40 | +5.33 | [+4.71,+5.95] | 40/40 |
| ml | SoftThreshτ1.5 | 10 | 40 | +6.70 | [+5.78,+7.62] | 40/40 |
| ml | SoftThreshτ1.5 | 1 | 40 | +1.57 | [+1.10,+2.04] | 34/40 |
| ml | SoftThreshτ1.5 | 0.75 | 40 | +1.11 | [+0.74,+1.49] | 34/40 |
| ml | SoftThreshτ1.5 | 0.5 | 40 | +0.48 | [+0.14,+0.82] | 24/40 |
| ml | R* | 2 | 40 | +3.46 | [+2.87,+4.04] | 39/40 |
| ml | R* | 0.5 | 40 | +0.94 | [+0.53,+1.35] | 31/40 |
| ml | R* | 4 | 40 | +4.05 | [+3.38,+4.71] | 38/40 |
| ml | R* | 0.75 | 40 | +1.56 | [+1.07,+2.06] | 34/40 |
| ml | R* | 1 | 40 | +2.11 | [+1.62,+2.59] | 37/40 |
| ml | SoftAND* | 0.5 | 40 | +0.19 | [-0.02,+0.40] | 22/40 |
| ml | SoftAND* | 0.25 | 40 | -0.10 | [-0.35,+0.16] | 15/40 |
| ml | R* | 0.25 | 40 | +0.19 | [-0.11,+0.50] | 23/40 |
| ml | SoftAND* | 0.75 | 40 | +0.36 | [+0.08,+0.65] | 28/40 |
| ml | SoftAND* | 2 | 40 | +1.53 | [+1.16,+1.90] | 37/40 |
| ml | SoftAND* | 10 | 40 | +3.33 | [+2.57,+4.10] | 37/40 |
| ml | SoftAND* | 1 | 40 | +0.64 | [+0.36,+0.92] | 29/40 |
| ml | SoftThreshτ1.5 | 0.25 | 40 | -0.10 | [-0.41,+0.20] | 16/40 |
| ml | SoftAND* | 4 | 40 | +2.35 | [+1.87,+2.84] | 40/40 |
| ml | SoftThreshτ1.5 | 2 | 40 | +3.32 | [+2.75,+3.90] | 39/40 |

### Regime `well_spec` — mean H−NLL=-0.23pt (13 cells; CI LB>0 in 0)

| dataset | misspec | γ | n | H−NLL | 95% CI | signs |
|---|---|---:|---:|---:|---|---:|
| anime | SoftAND* | 0 | 9 | +0.08 | [-1.12,+1.28] | 2/9 |
| ml | SoftAND*-adapt | 0 | 20 | -0.20 | [-0.51,+0.10] | 7/20 |
| kuairec | R*-mlsize | 0 | 20 | -0.13 | [-0.80,+0.53] | 10/20 |
| kuairand | R*-mlsize | 0 | 20 | -0.78 | [-1.04,-0.52] | 2/20 |
| myket | R* | 0 | 20 | +0.08 | [-0.28,+0.45] | 12/20 |
| anime | R* | 0 | 20 | +0.22 | [-0.10,+0.55] | 13/20 |
| kuairand | R*-full | 0 | 20 | -0.58 | [-0.86,-0.30] | 3/20 |
| kuairec | R*-full | 0 | 20 | -0.68 | [-1.24,-0.12] | 5/20 |
| ml | SoftThresh-adapt | 0 | 20 | -0.20 | [-0.51,+0.10] | 7/20 |
| ml | R*-adapt | 0 | 20 | -0.20 | [-0.51,+0.10] | 7/20 |
| ml | R* | 0 | 40 | -0.17 | [-0.40,+0.06] | 15/40 |
| ml | SoftAND* | 0 | 40 | -0.18 | [-0.40,+0.04] | 15/40 |
| ml | SoftThreshτ1.5 | 0 | 40 | -0.18 | [-0.40,+0.04] | 15/40 |

## DiD (H−NLL and OPC−NLL)

| dataset | misspec | γ0→γ1 | contrast | n | DiD | 95% CI | signs |
|---|---|---|---|---:|---:|---|---:|
| anime | R* | 0→0.6 | H-NLL | 20 | +0.50 | [+0.08,+0.92] | 16/20 |
| anime | R* | 0→0.6 | OPC-NLL | 20 | +0.10 | [-0.26,+0.46] | 8/20 |
| anime | SoftAND* | 0→0.25 | A-NLL | 9 | +0.17 | [-0.43,+0.78] | 5/9 |
| anime | SoftAND* | 0→0.25 | H-NLL | 9 | +0.27 | [-0.13,+0.68] | 7/9 |
| anime | SoftAND* | 0→0.25 | OPC-NLL | 9 | -0.36 | [-0.71,-0.02] | 3/9 |
| anime | SoftAND* | 0→0.75 | A-NLL | 5 | +0.60 | [-1.62,+2.82] | 4/5 |
| anime | SoftAND* | 0→0.75 | H-NLL | 5 | +1.47 | [-0.33,+3.27] | 4/5 |
| anime | SoftAND* | 0→0.75 | OPC-NLL | 5 | -0.50 | [-1.53,+0.53] | 2/5 |
| kuairand | R*-full | 0→0.5 | H-NLL | 20 | +0.61 | [+0.12,+1.11] | 16/20 |
| kuairand | R*-full | 0→0.5 | OPC-NLL | 20 | +0.17 | [-0.28,+0.62] | 11/20 |
| kuairand | R*-mlsize | 0→0.5 | H-NLL | 20 | +1.31 | [+0.92,+1.69] | 20/20 |
| kuairand | R*-mlsize | 0→0.5 | OPC-NLL | 20 | +1.18 | [+0.69,+1.67] | 17/20 |
| kuairec | R*-full | 0→0.5 | H-NLL | 20 | +1.30 | [+0.72,+1.88] | 18/20 |
| kuairec | R*-full | 0→0.5 | OPC-NLL | 20 | +1.56 | [+0.92,+2.20] | 19/20 |
| kuairec | R*-mlsize | 0→0.5 | H-NLL | 20 | +0.72 | [-0.06,+1.50] | 14/20 |
| kuairec | R*-mlsize | 0→0.5 | OPC-NLL | 20 | +0.94 | [+0.46,+1.42] | 17/20 |
| ml | R* | 0→0.25 | H-NLL | 40 | +0.36 | [+0.12,+0.61] | 24/40 |
| ml | R* | 0→0.25 | OPC-NLL | 40 | +0.26 | [+0.03,+0.49] | 25/40 |
| ml | R* | 0→0.5 | H-NLL | 40 | +1.11 | [+0.72,+1.50] | 32/40 |
| ml | R* | 0→0.5 | OPC-NLL | 40 | +1.12 | [+0.78,+1.47] | 33/40 |
| ml | R* | 0→0.75 | A-NLL | 36 | +1.76 | [+1.37,+2.15] | 34/36 |
| ml | R* | 0→0.75 | H-NLL | 40 | +1.73 | [+1.23,+2.24] | 32/40 |
| ml | R* | 0→0.75 | OPC-NLL | 40 | +2.01 | [+1.59,+2.42] | 38/40 |
| ml | R* | 0→1 | A-NLL | 36 | +2.25 | [+1.74,+2.76] | 33/36 |
| ml | R* | 0→1 | H-NLL | 40 | +2.28 | [+1.78,+2.77] | 36/40 |
| ml | R* | 0→1 | OPC-NLL | 40 | +2.66 | [+2.22,+3.09] | 40/40 |
| ml | R* | 0→2 | A-NLL | 36 | +3.28 | [+2.69,+3.88] | 35/36 |
| ml | R* | 0→2 | H-NLL | 40 | +3.63 | [+2.95,+4.31] | 38/40 |
| ml | R* | 0→2 | OPC-NLL | 40 | +4.46 | [+3.96,+4.96] | 40/40 |
| ml | R* | 0→4 | A-NLL | 36 | +4.15 | [+3.39,+4.91] | 36/36 |
| ml | R* | 0→4 | H-NLL | 40 | +4.22 | [+3.46,+4.98] | 37/40 |
| ml | R* | 0→4 | OPC-NLL | 40 | +5.56 | [+5.02,+6.09] | 40/40 |
| ml | R* | 0→10 | A-NLL | 36 | +4.39 | [+3.62,+5.15] | 36/36 |
| ml | R* | 0→10 | H-NLL | 36 | +4.84 | [+3.93,+5.74] | 34/36 |
| ml | R* | 0→10 | OPC-NLL | 36 | +6.24 | [+5.67,+6.82] | 36/36 |
| ml | R* | 0→100 | A-NLL | 36 | +4.36 | [+3.51,+5.21] | 36/36 |
| ml | R* | 0→100 | H-NLL | 36 | +4.76 | [+3.95,+5.56] | 34/36 |
| ml | R* | 0→100 | OPC-NLL | 36 | +6.55 | [+5.89,+7.21] | 36/36 |
| ml | R*-adapt | 0→0.5 | A-NLL | 20 | +1.44 | [+1.03,+1.85] | 19/20 |
| ml | R*-adapt | 0→0.5 | H-NLL | 20 | +1.19 | [+0.59,+1.79] | 17/20 |
| ml | R*-adapt | 0→0.5 | OPC-NLL | 20 | +1.35 | [+0.88,+1.83] | 19/20 |
| ml | SoftAND* | 0→0.25 | A-NLL | 40 | +0.22 | [-0.00,+0.45] | 23/40 |
| ml | SoftAND* | 0→0.25 | H-NLL | 40 | +0.09 | [-0.11,+0.28] | 19/40 |
| ml | SoftAND* | 0→0.25 | OPC-NLL | 40 | +0.00 | [-0.19,+0.19] | 19/40 |
| ml | SoftAND* | 0→0.5 | A-NLL | 40 | +0.38 | [+0.11,+0.65] | 28/40 |
| ml | SoftAND* | 0→0.5 | H-NLL | 40 | +0.35 | [+0.13,+0.58] | 29/40 |
| ml | SoftAND* | 0→0.5 | OPC-NLL | 40 | +0.13 | [-0.08,+0.34] | 20/40 |
| ml | SoftAND* | 0→0.75 | A-NLL | 40 | +0.76 | [+0.43,+1.09] | 28/40 |
| ml | SoftAND* | 0→0.75 | H-NLL | 40 | +0.55 | [+0.23,+0.86] | 28/40 |
| ml | SoftAND* | 0→0.75 | OPC-NLL | 40 | +0.35 | [+0.10,+0.60] | 24/40 |
| ml | SoftAND* | 0→1 | A-NLL | 40 | +0.81 | [+0.47,+1.15] | 33/40 |
| ml | SoftAND* | 0→1 | H-NLL | 40 | +0.82 | [+0.52,+1.13] | 30/40 |
| ml | SoftAND* | 0→1 | OPC-NLL | 40 | +0.47 | [+0.21,+0.73] | 29/40 |
| ml | SoftAND* | 0→2 | A-NLL | 40 | +1.53 | [+1.10,+1.96] | 36/40 |
| ml | SoftAND* | 0→2 | H-NLL | 40 | +1.71 | [+1.24,+2.18] | 34/40 |
| ml | SoftAND* | 0→2 | OPC-NLL | 40 | +1.05 | [+0.72,+1.38] | 32/40 |
| ml | SoftAND* | 0→4 | A-NLL | 40 | +2.42 | [+1.86,+2.98] | 38/40 |
| ml | SoftAND* | 0→4 | H-NLL | 40 | +2.54 | [+1.94,+3.13] | 36/40 |
| ml | SoftAND* | 0→4 | OPC-NLL | 40 | +1.88 | [+1.43,+2.32] | 36/40 |
| ml | SoftAND* | 0→10 | A-NLL | 40 | +3.38 | [+2.57,+4.20] | 36/40 |
| ml | SoftAND* | 0→10 | H-NLL | 40 | +3.51 | [+2.71,+4.32] | 35/40 |
| ml | SoftAND* | 0→10 | OPC-NLL | 40 | +3.16 | [+2.56,+3.76] | 39/40 |
| ml | SoftAND* | 0→100 | A-NLL | 40 | +3.40 | [+2.48,+4.32] | 35/40 |
| ml | SoftAND* | 0→100 | H-NLL | 40 | +3.38 | [+2.50,+4.25] | 36/40 |
| ml | SoftAND* | 0→100 | OPC-NLL | 40 | +3.92 | [+3.22,+4.62] | 39/40 |
| ml | SoftAND*-adapt | 0→100 | A-NLL | 20 | +3.64 | [+2.22,+5.05] | 19/20 |
| ml | SoftAND*-adapt | 0→100 | H-NLL | 20 | +3.80 | [+2.33,+5.26] | 18/20 |
| ml | SoftAND*-adapt | 0→100 | OPC-NLL | 20 | +3.71 | [+2.59,+4.83] | 19/20 |
| ml | SoftThresh-adapt | 0→100 | A-NLL | 20 | +6.39 | [+5.22,+7.55] | 20/20 |
| ml | SoftThresh-adapt | 0→100 | H-NLL | 20 | +6.71 | [+5.22,+8.21] | 20/20 |
| ml | SoftThresh-adapt | 0→100 | OPC-NLL | 20 | +6.20 | [+5.11,+7.29] | 20/20 |
| ml | SoftThreshτ1.5 | 0→0.25 | A-NLL | 40 | +0.33 | [+0.08,+0.58] | 27/40 |
| ml | SoftThreshτ1.5 | 0→0.25 | H-NLL | 40 | +0.08 | [-0.20,+0.36] | 24/40 |
| ml | SoftThreshτ1.5 | 0→0.25 | OPC-NLL | 40 | -0.04 | [-0.27,+0.19] | 19/40 |
| ml | SoftThreshτ1.5 | 0→0.5 | A-NLL | 36 | +0.87 | [+0.49,+1.25] | 28/36 |
| ml | SoftThreshτ1.5 | 0→0.5 | H-NLL | 40 | +0.66 | [+0.25,+1.07] | 29/40 |
| ml | SoftThreshτ1.5 | 0→0.5 | OPC-NLL | 40 | -0.02 | [-0.32,+0.28] | 18/40 |
| ml | SoftThreshτ1.5 | 0→0.75 | H-NLL | 40 | +1.30 | [+0.87,+1.73] | 34/40 |
| ml | SoftThreshτ1.5 | 0→0.75 | OPC-NLL | 40 | +0.13 | [-0.29,+0.54] | 21/40 |
| ml | SoftThreshτ1.5 | 0→1 | A-NLL | 34 | +1.68 | [+1.12,+2.24] | 30/34 |
| ml | SoftThreshτ1.5 | 0→1 | H-NLL | 40 | +1.75 | [+1.21,+2.30] | 36/40 |
| ml | SoftThreshτ1.5 | 0→1 | OPC-NLL | 40 | +0.33 | [-0.16,+0.81] | 24/40 |
| ml | SoftThreshτ1.5 | 0→2 | A-NLL | 40 | +3.11 | [+2.50,+3.71] | 38/40 |
| ml | SoftThreshτ1.5 | 0→2 | H-NLL | 40 | +3.50 | [+2.86,+4.15] | 38/40 |
| ml | SoftThreshτ1.5 | 0→2 | OPC-NLL | 40 | +1.99 | [+1.32,+2.65] | 33/40 |
| ml | SoftThreshτ1.5 | 0→4 | H-NLL | 40 | +5.51 | [+4.82,+6.20] | 40/40 |
| ml | SoftThreshτ1.5 | 0→4 | OPC-NLL | 40 | +4.31 | [+3.60,+5.02] | 39/40 |
| ml | SoftThreshτ1.5 | 0→10 | H-NLL | 40 | +6.88 | [+5.97,+7.79] | 40/40 |
| ml | SoftThreshτ1.5 | 0→10 | OPC-NLL | 40 | +5.75 | [+4.91,+6.60] | 40/40 |
| ml | SoftThreshτ1.5 | 0→100 | A-NLL | 34 | +6.13 | [+5.29,+6.97] | 34/34 |
| ml | SoftThreshτ1.5 | 0→100 | H-NLL | 40 | +7.09 | [+6.06,+8.12] | 40/40 |
| ml | SoftThreshτ1.5 | 0→100 | OPC-NLL | 40 | +5.95 | [+4.98,+6.91] | 39/40 |
| myket | R* | 0→0.5 | H-NLL | 20 | +0.69 | [+0.16,+1.22] | 14/20 |
| myket | R* | 0→0.5 | OPC-NLL | 20 | +0.55 | [-0.20,+1.31] | 11/20 |

## Absolute levels (hybrid & baselines)

| dataset | misspec | γ | method | n | mean CTR% | 95% CI |
|---|---|---:|---|---:|---:|---|
| anime | R* | 0 | hybrid_seq_f0.5 | 20 | 16.71 | [15.93,17.48] |
| anime | R* | 0 | nll | 20 | 16.49 | [15.83,17.14] |
| anime | R* | 0 | nll_dr | 20 | 12.13 | [10.54,13.73] |
| anime | R* | 0 | opc | 20 | 15.93 | [15.51,16.34] |
| anime | R* | 0.6 | hybrid_seq_f0.5 | 20 | 17.37 | [16.60,18.13] |
| anime | R* | 0.6 | nll | 20 | 16.65 | [16.02,17.27] |
| anime | R* | 0.6 | nll_dr | 20 | 12.39 | [10.64,14.14] |
| anime | R* | 0.6 | opc | 20 | 16.19 | [15.61,16.76] |
| anime | SoftAND* | 0 | hybrid_seq_adapt | 33 | 16.33 | [15.44,17.22] |
| anime | SoftAND* | 0 | hybrid_seq_f0.25 | 33 | 16.18 | [15.55,16.82] |
| anime | SoftAND* | 0 | hybrid_seq_f0.5 | 33 | 16.44 | [15.77,17.11] |
| anime | SoftAND* | 0 | hybrid_seq_f0.75 | 33 | 16.26 | [15.38,17.14] |
| anime | SoftAND* | 0 | nll | 33 | 16.31 | [15.70,16.91] |
| anime | SoftAND* | 0 | opc | 33 | 15.92 | [15.56,16.28] |
| anime | SoftAND* | 0.25 | hybrid_seq_adapt | 9 | 17.06 | [14.82,19.29] |
| anime | SoftAND* | 0.25 | hybrid_seq_f0.25 | 9 | 16.63 | [15.05,18.22] |
| anime | SoftAND* | 0.25 | hybrid_seq_f0.5 | 9 | 17.19 | [15.51,18.87] |
| anime | SoftAND* | 0.25 | hybrid_seq_f0.75 | 9 | 17.09 | [14.84,19.35] |
| anime | SoftAND* | 0.25 | nll | 9 | 16.84 | [15.91,17.76] |
| anime | SoftAND* | 0.25 | opc | 9 | 16.23 | [15.05,17.41] |
| anime | SoftAND* | 0.75 | hybrid_seq_adapt | 5 | 17.59 | [12.25,22.92] |
| anime | SoftAND* | 0.75 | hybrid_seq_f0.25 | 5 | 17.41 | [14.54,20.28] |
| anime | SoftAND* | 0.75 | hybrid_seq_f0.5 | 5 | 18.75 | [16.51,20.99] |
| anime | SoftAND* | 0.75 | hybrid_seq_f0.75 | 5 | 18.77 | [16.09,21.44] |
| anime | SoftAND* | 0.75 | nll | 5 | 17.84 | [14.94,20.73] |
| anime | SoftAND* | 0.75 | opc | 5 | 16.73 | [14.39,19.06] |
| kuairand | R*-full | 0 | hybrid_seq_f0.5 | 20 | 21.57 | [21.29,21.85] |
| kuairand | R*-full | 0 | nll | 20 | 22.15 | [21.78,22.52] |
| kuairand | R*-full | 0 | nll_dr | 20 | 22.15 | [21.78,22.53] |
| kuairand | R*-full | 0 | opc | 20 | 20.78 | [20.36,21.19] |
| kuairand | R*-full | 0.5 | hybrid_seq_f0.5 | 20 | 21.86 | [21.06,22.66] |
| kuairand | R*-full | 0.5 | nll | 20 | 21.83 | [21.04,22.62] |
| kuairand | R*-full | 0.5 | nll_dr | 20 | 21.82 | [21.03,22.61] |
| kuairand | R*-full | 0.5 | opc | 20 | 20.62 | [19.90,21.35] |
| kuairand | R*-mlsize | 0 | hybrid_seq_f0.5 | 20 | 20.79 | [20.34,21.23] |
| kuairand | R*-mlsize | 0 | nll | 20 | 21.57 | [21.15,21.98] |
| kuairand | R*-mlsize | 0 | nll_dr | 20 | 21.58 | [21.17,21.98] |
| kuairand | R*-mlsize | 0 | opc | 20 | 20.00 | [19.52,20.49] |
| kuairand | R*-mlsize | 0.5 | hybrid_seq_f0.5 | 20 | 21.86 | [21.21,22.51] |
| kuairand | R*-mlsize | 0.5 | nll | 20 | 21.34 | [20.69,21.98] |
| kuairand | R*-mlsize | 0.5 | nll_dr | 20 | 21.41 | [20.78,22.05] |
| kuairand | R*-mlsize | 0.5 | opc | 20 | 20.95 | [20.33,21.57] |
| kuairec | R*-full | 0 | hybrid_seq_f0.5 | 20 | 21.59 | [20.67,22.50] |
| kuairec | R*-full | 0 | nll | 20 | 22.27 | [21.55,23.00] |
| kuairec | R*-full | 0 | nll_dr | 20 | 22.24 | [21.48,22.99] |
| kuairec | R*-full | 0 | opc | 20 | 20.54 | [19.74,21.34] |
| kuairec | R*-full | 0.5 | hybrid_seq_f0.5 | 20 | 22.60 | [21.53,23.67] |
| kuairec | R*-full | 0.5 | nll | 20 | 21.98 | [20.90,23.07] |
| kuairec | R*-full | 0.5 | nll_dr | 20 | 21.92 | [20.83,23.02] |
| kuairec | R*-full | 0.5 | opc | 20 | 21.81 | [20.96,22.65] |
| kuairec | R*-mlsize | 0 | hybrid_seq_f0.5 | 20 | 22.87 | [22.22,23.53] |
| kuairec | R*-mlsize | 0 | nll | 20 | 23.01 | [22.25,23.76] |
| kuairec | R*-mlsize | 0 | nll_dr | 20 | 23.00 | [22.25,23.75] |
| kuairec | R*-mlsize | 0 | opc | 20 | 22.02 | [21.40,22.63] |
| kuairec | R*-mlsize | 0.5 | hybrid_seq_f0.5 | 20 | 22.87 | [21.86,23.88] |
| kuairec | R*-mlsize | 0.5 | nll | 20 | 22.29 | [21.34,23.24] |
| kuairec | R*-mlsize | 0.5 | nll_dr | 20 | 22.27 | [21.31,23.23] |
| kuairec | R*-mlsize | 0.5 | opc | 20 | 22.24 | [21.41,23.08] |
| ml | R* | 0 | hybrid_seq_adapt | 36 | 18.41 | [18.05,18.78] |
| ml | R* | 0 | hybrid_seq_f0.25 | 36 | 18.03 | [17.63,18.43] |
| ml | R* | 0 | hybrid_seq_f0.5 | 40 | 18.37 | [18.08,18.65] |
| ml | R* | 0 | hybrid_seq_f0.75 | 36 | 18.42 | [18.08,18.76] |
| ml | R* | 0 | nll | 40 | 18.54 | [18.16,18.92] |
| ml | R* | 0 | nll_dr | 20 | 15.54 | [13.38,17.70] |
| ml | R* | 0 | opc | 40 | 17.35 | [17.06,17.63] |
| ml | R* | 0.25 | hybrid_seq_f0.5 | 40 | 19.13 | [18.64,19.61] |
| ml | R* | 0.25 | nll | 40 | 18.94 | [18.41,19.46] |
| ml | R* | 0.25 | opc | 40 | 18.01 | [17.59,18.42] |
| ml | R* | 0.5 | hybrid_seq_f0.25 | 10 | 19.10 | [17.81,20.40] |
| ml | R* | 0.5 | hybrid_seq_f0.5 | 40 | 20.06 | [19.43,20.68] |
| ml | R* | 0.5 | hybrid_seq_f0.75 | 10 | 19.56 | [18.76,20.36] |
| ml | R* | 0.5 | nll | 40 | 19.12 | [18.49,19.74] |
| ml | R* | 0.5 | nll_dr | 20 | 16.20 | [13.92,18.48] |
| ml | R* | 0.5 | opc | 40 | 19.05 | [18.58,19.52] |
| ml | R* | 0.75 | hybrid_seq_adapt | 40 | 20.53 | [19.82,21.23] |
| ml | R* | 0.75 | hybrid_seq_f0.25 | 40 | 20.15 | [19.46,20.84] |
| ml | R* | 0.75 | hybrid_seq_f0.5 | 40 | 20.47 | [19.78,21.16] |
| ml | R* | 0.75 | hybrid_seq_f0.75 | 40 | 20.45 | [19.76,21.14] |
| ml | R* | 0.75 | nll | 40 | 18.91 | [18.21,19.61] |
| ml | R* | 0.75 | opc | 40 | 19.65 | [19.14,20.16] |
| ml | R* | 1 | hybrid_seq_adapt | 40 | 20.80 | [20.02,21.57] |
| ml | R* | 1 | hybrid_seq_f0.25 | 40 | 20.54 | [19.78,21.30] |
| ml | R* | 1 | hybrid_seq_f0.5 | 40 | 20.71 | [20.00,21.41] |
| ml | R* | 1 | hybrid_seq_f0.75 | 40 | 20.71 | [19.94,21.48] |
| ml | R* | 1 | nll | 40 | 18.64 | [17.89,19.38] |
| ml | R* | 1 | opc | 40 | 20.04 | [19.46,20.62] |
| ml | R* | 2 | hybrid_seq_adapt | 40 | 20.20 | [19.35,21.04] |
| ml | R* | 2 | hybrid_seq_f0.25 | 40 | 19.97 | [19.10,20.84] |
| ml | R* | 2 | hybrid_seq_f0.5 | 40 | 20.48 | [19.61,21.36] |
| ml | R* | 2 | hybrid_seq_f0.75 | 40 | 20.25 | [19.36,21.13] |
| ml | R* | 2 | nll | 40 | 17.08 | [16.37,17.78] |
| ml | R* | 2 | opc | 40 | 20.35 | [19.74,20.96] |
| ml | R* | 4 | hybrid_seq_adapt | 40 | 19.58 | [18.65,20.50] |
| ml | R* | 4 | hybrid_seq_f0.25 | 40 | 19.44 | [18.53,20.35] |
| ml | R* | 4 | hybrid_seq_f0.5 | 40 | 19.64 | [18.78,20.50] |
| ml | R* | 4 | hybrid_seq_f0.75 | 40 | 19.65 | [18.78,20.51] |
| ml | R* | 4 | nll | 40 | 15.65 | [15.01,16.29] |
| ml | R* | 4 | opc | 40 | 20.02 | [19.44,20.60] |
| ml | R* | 10 | hybrid_seq_adapt | 40 | 18.89 | [17.98,19.79] |
| ml | R* | 10 | hybrid_seq_f0.25 | 40 | 18.86 | [17.95,19.77] |
| ml | R* | 10 | hybrid_seq_f0.5 | 40 | 19.27 | [18.30,20.24] |
| ml | R* | 10 | hybrid_seq_f0.75 | 40 | 18.97 | [18.04,19.90] |
| ml | R* | 10 | nll | 40 | 14.70 | [14.06,15.33] |
| ml | R* | 10 | opc | 40 | 19.59 | [18.96,20.22] |
| ml | R* | 100 | hybrid_seq_adapt | 40 | 18.33 | [17.38,19.27] |
| ml | R* | 100 | hybrid_seq_f0.25 | 40 | 18.40 | [17.51,19.30] |
| ml | R* | 100 | hybrid_seq_f0.5 | 40 | 18.68 | [17.86,19.51] |
| ml | R* | 100 | hybrid_seq_f0.75 | 40 | 18.42 | [17.53,19.31] |
| ml | R* | 100 | nll | 40 | 14.13 | [13.48,14.78] |
| ml | R* | 100 | opc | 40 | 19.39 | [18.75,20.03] |
| ml | R*-adapt | 0 | hybrid_seq_adapt | 20 | 18.05 | [17.57,18.54] |
| ml | R*-adapt | 0 | hybrid_seq_f0.5 | 20 | 18.07 | [17.71,18.42] |
| ml | R*-adapt | 0 | nll | 20 | 18.27 | [17.74,18.81] |
| ml | R*-adapt | 0 | opc | 20 | 17.17 | [16.79,17.54] |
| ml | R*-adapt | 0.5 | hybrid_seq_adapt | 20 | 19.86 | [19.18,20.53] |
| ml | R*-adapt | 0.5 | hybrid_seq_f0.5 | 20 | 19.62 | [18.98,20.27] |
| ml | R*-adapt | 0.5 | nll | 20 | 18.63 | [17.86,19.41] |
| ml | R*-adapt | 0.5 | opc | 20 | 18.88 | [18.32,19.44] |
| ml | SoftAND* | 0 | hybrid_seq_adapt | 40 | 18.40 | [18.04,18.76] |
| ml | SoftAND* | 0 | hybrid_seq_f0.25 | 40 | 18.00 | [17.63,18.37] |
| ml | SoftAND* | 0 | hybrid_seq_f0.5 | 40 | 18.36 | [18.07,18.64] |
| ml | SoftAND* | 0 | hybrid_seq_f0.75 | 40 | 18.42 | [18.08,18.76] |
| ml | SoftAND* | 0 | nll | 40 | 18.54 | [18.16,18.92] |
| ml | SoftAND* | 0 | nll_dr | 10 | 16.99 | [14.26,19.71] |
| ml | SoftAND* | 0 | opc | 40 | 17.32 | [17.03,17.60] |
| ml | SoftAND* | 0.25 | hybrid_seq_adapt | 40 | 18.79 | [18.44,19.15] |
| ml | SoftAND* | 0.25 | hybrid_seq_f0.25 | 40 | 18.43 | [18.04,18.82] |
| ml | SoftAND* | 0.25 | hybrid_seq_f0.5 | 40 | 18.61 | [18.25,18.98] |
| ml | SoftAND* | 0.25 | hybrid_seq_f0.75 | 40 | 18.78 | [18.42,19.14] |
| ml | SoftAND* | 0.25 | nll | 40 | 18.71 | [18.33,19.09] |
| ml | SoftAND* | 0.25 | opc | 40 | 17.49 | [17.17,17.81] |
| ml | SoftAND* | 0.5 | hybrid_seq_adapt | 40 | 18.87 | [18.46,19.29] |
| ml | SoftAND* | 0.5 | hybrid_seq_f0.25 | 40 | 18.49 | [18.04,18.94] |
| ml | SoftAND* | 0.5 | hybrid_seq_f0.5 | 40 | 18.83 | [18.41,19.24] |
| ml | SoftAND* | 0.5 | hybrid_seq_f0.75 | 40 | 18.93 | [18.52,19.35] |
| ml | SoftAND* | 0.5 | nll | 40 | 18.64 | [18.23,19.04] |
| ml | SoftAND* | 0.5 | nll_dr | 10 | 17.13 | [14.28,19.98] |
| ml | SoftAND* | 0.5 | opc | 40 | 17.54 | [17.24,17.84] |
| ml | SoftAND* | 0.75 | hybrid_seq_adapt | 40 | 19.12 | [18.62,19.62] |
| ml | SoftAND* | 0.75 | hybrid_seq_f0.25 | 40 | 18.68 | [18.23,19.13] |
| ml | SoftAND* | 0.75 | hybrid_seq_f0.5 | 40 | 18.87 | [18.46,19.28] |
| ml | SoftAND* | 0.75 | hybrid_seq_f0.75 | 40 | 19.09 | [18.60,19.58] |
| ml | SoftAND* | 0.75 | nll | 40 | 18.51 | [18.08,18.93] |
| ml | SoftAND* | 0.75 | opc | 40 | 17.64 | [17.30,17.97] |
| ml | SoftAND* | 1 | hybrid_seq_adapt | 40 | 19.10 | [18.57,19.63] |
| ml | SoftAND* | 1 | hybrid_seq_f0.25 | 40 | 18.63 | [18.16,19.09] |
| ml | SoftAND* | 1 | hybrid_seq_f0.5 | 40 | 19.07 | [18.62,19.53] |
| ml | SoftAND* | 1 | hybrid_seq_f0.75 | 40 | 19.13 | [18.62,19.64] |
| ml | SoftAND* | 1 | nll | 40 | 18.43 | [18.00,18.87] |
| ml | SoftAND* | 1 | opc | 40 | 17.68 | [17.32,18.04] |
| ml | SoftAND* | 2 | hybrid_seq_adapt | 40 | 19.81 | [19.22,20.41] |
| ml | SoftAND* | 2 | hybrid_seq_f0.25 | 40 | 19.36 | [18.70,20.03] |
| ml | SoftAND* | 2 | hybrid_seq_f0.5 | 40 | 19.95 | [19.34,20.57] |
| ml | SoftAND* | 2 | hybrid_seq_f0.75 | 40 | 19.83 | [19.26,20.39] |
| ml | SoftAND* | 2 | nll | 40 | 18.42 | [17.92,18.93] |
| ml | SoftAND* | 2 | opc | 40 | 18.25 | [17.79,18.72] |
| ml | SoftAND* | 4 | hybrid_seq_adapt | 40 | 20.94 | [20.24,21.64] |
| ml | SoftAND* | 4 | hybrid_seq_f0.25 | 40 | 20.55 | [19.76,21.35] |
| ml | SoftAND* | 4 | hybrid_seq_f0.5 | 40 | 21.02 | [20.26,21.77] |
| ml | SoftAND* | 4 | hybrid_seq_f0.75 | 40 | 20.97 | [20.26,21.67] |
| ml | SoftAND* | 4 | nll | 40 | 18.66 | [18.03,19.29] |
| ml | SoftAND* | 4 | opc | 40 | 19.32 | [18.81,19.82] |
| ml | SoftAND* | 10 | hybrid_seq_adapt | 40 | 21.93 | [21.02,22.83] |
| ml | SoftAND* | 10 | hybrid_seq_f0.25 | 40 | 21.55 | [20.67,22.44] |
| ml | SoftAND* | 10 | hybrid_seq_f0.5 | 40 | 22.02 | [21.15,22.88] |
| ml | SoftAND* | 10 | hybrid_seq_f0.75 | 40 | 21.99 | [21.09,22.89] |
| ml | SoftAND* | 10 | nll | 40 | 18.68 | [17.96,19.40] |
| ml | SoftAND* | 10 | opc | 40 | 20.62 | [20.03,21.21] |
| ml | SoftAND* | 100 | hybrid_seq_adapt | 40 | 21.37 | [20.48,22.27] |
| ml | SoftAND* | 100 | hybrid_seq_f0.25 | 40 | 21.07 | [20.15,21.99] |
| ml | SoftAND* | 100 | hybrid_seq_f0.5 | 40 | 21.40 | [20.54,22.27] |
| ml | SoftAND* | 100 | hybrid_seq_f0.75 | 40 | 21.45 | [20.55,22.34] |
| ml | SoftAND* | 100 | nll | 40 | 18.11 | [17.43,18.80] |
| ml | SoftAND* | 100 | nll_dr | 10 | 18.01 | [16.60,19.42] |
| ml | SoftAND* | 100 | opc | 40 | 20.85 | [20.27,21.43] |
| ml | SoftAND*-adapt | 0 | hybrid_seq_adapt | 20 | 18.05 | [17.57,18.54] |
| ml | SoftAND*-adapt | 0 | hybrid_seq_f0.5 | 20 | 18.07 | [17.71,18.42] |
| ml | SoftAND*-adapt | 0 | nll | 20 | 18.27 | [17.74,18.81] |
| ml | SoftAND*-adapt | 0 | opc | 20 | 17.17 | [16.79,17.54] |
| ml | SoftAND*-adapt | 100 | hybrid_seq_adapt | 20 | 21.91 | [20.58,23.24] |
| ml | SoftAND*-adapt | 100 | hybrid_seq_f0.5 | 20 | 22.08 | [20.85,23.31] |
| ml | SoftAND*-adapt | 100 | nll | 20 | 18.49 | [17.64,19.34] |
| ml | SoftAND*-adapt | 100 | opc | 20 | 21.09 | [20.21,21.98] |
| ml | SoftThresh-adapt | 0 | hybrid_seq_adapt | 20 | 18.05 | [17.57,18.54] |
| ml | SoftThresh-adapt | 0 | hybrid_seq_f0.5 | 20 | 18.07 | [17.71,18.42] |
| ml | SoftThresh-adapt | 0 | nll | 20 | 18.27 | [17.74,18.81] |
| ml | SoftThresh-adapt | 0 | opc | 20 | 17.17 | [16.79,17.54] |
| ml | SoftThresh-adapt | 100 | hybrid_seq_adapt | 20 | 26.53 | [24.51,28.55] |
| ml | SoftThresh-adapt | 100 | hybrid_seq_f0.5 | 20 | 26.87 | [24.49,29.26] |
| ml | SoftThresh-adapt | 100 | nll | 20 | 20.36 | [18.72,22.01] |
| ml | SoftThresh-adapt | 100 | opc | 20 | 25.46 | [23.68,27.23] |
| ml | SoftThreshτ1.5 | 0 | hybrid_seq_adapt | 40 | 18.40 | [18.04,18.76] |
| ml | SoftThreshτ1.5 | 0 | hybrid_seq_f0.25 | 40 | 18.00 | [17.63,18.37] |
| ml | SoftThreshτ1.5 | 0 | hybrid_seq_f0.5 | 40 | 18.36 | [18.07,18.64] |
| ml | SoftThreshτ1.5 | 0 | hybrid_seq_f0.75 | 40 | 18.42 | [18.08,18.76] |
| ml | SoftThreshτ1.5 | 0 | nll | 40 | 18.54 | [18.16,18.92] |
| ml | SoftThreshτ1.5 | 0 | nll_dr | 10 | 16.99 | [14.26,19.71] |
| ml | SoftThreshτ1.5 | 0 | opc | 40 | 17.32 | [17.03,17.60] |
| ml | SoftThreshτ1.5 | 0.25 | hybrid_seq_adapt | 40 | 20.14 | [19.48,20.80] |
| ml | SoftThreshτ1.5 | 0.25 | hybrid_seq_f0.25 | 40 | 19.69 | [18.95,20.43] |
| ml | SoftThreshτ1.5 | 0.25 | hybrid_seq_f0.5 | 40 | 19.85 | [19.20,20.50] |
| ml | SoftThreshτ1.5 | 0.25 | hybrid_seq_f0.75 | 40 | 20.07 | [19.44,20.71] |
| ml | SoftThreshτ1.5 | 0.25 | nll | 40 | 19.95 | [19.29,20.61] |
| ml | SoftThreshτ1.5 | 0.25 | opc | 40 | 18.69 | [18.12,19.26] |
| ml | SoftThreshτ1.5 | 0.5 | hybrid_seq_adapt | 36 | 21.93 | [21.00,22.85] |
| ml | SoftThreshτ1.5 | 0.5 | hybrid_seq_f0.25 | 36 | 21.50 | [20.50,22.49] |
| ml | SoftThreshτ1.5 | 0.5 | hybrid_seq_f0.5 | 40 | 21.65 | [20.66,22.64] |
| ml | SoftThreshτ1.5 | 0.5 | hybrid_seq_f0.75 | 36 | 21.93 | [20.97,22.88] |
| ml | SoftThreshτ1.5 | 0.5 | nll | 40 | 21.17 | [20.31,22.03] |
| ml | SoftThreshτ1.5 | 0.5 | opc | 40 | 19.93 | [19.07,20.80] |
| ml | SoftThreshτ1.5 | 0.75 | hybrid_seq_f0.5 | 40 | 23.02 | [21.85,24.18] |
| ml | SoftThreshτ1.5 | 0.75 | nll | 40 | 21.90 | [20.90,22.90] |
| ml | SoftThreshτ1.5 | 0.75 | opc | 40 | 20.81 | [19.74,21.87] |
| ml | SoftThreshτ1.5 | 1 | hybrid_seq_adapt | 34 | 23.72 | [22.57,24.87] |
| ml | SoftThreshτ1.5 | 1 | hybrid_seq_f0.25 | 34 | 22.95 | [21.76,24.14] |
| ml | SoftThreshτ1.5 | 1 | hybrid_seq_f0.5 | 40 | 23.87 | [22.58,25.17] |
| ml | SoftThreshτ1.5 | 1 | hybrid_seq_f0.75 | 34 | 23.76 | [22.62,24.90] |
| ml | SoftThreshτ1.5 | 1 | nll | 40 | 22.30 | [21.26,23.34] |
| ml | SoftThreshτ1.5 | 1 | opc | 40 | 21.41 | [20.29,22.53] |
| ml | SoftThreshτ1.5 | 2 | hybrid_seq_adapt | 40 | 25.68 | [24.25,27.12] |
| ml | SoftThreshτ1.5 | 2 | hybrid_seq_f0.25 | 40 | 25.42 | [23.89,26.95] |
| ml | SoftThreshτ1.5 | 2 | hybrid_seq_f0.5 | 40 | 26.04 | [24.56,27.51] |
| ml | SoftThreshτ1.5 | 2 | hybrid_seq_f0.75 | 40 | 25.64 | [24.23,27.05] |
| ml | SoftThreshτ1.5 | 2 | nll | 40 | 22.72 | [21.56,23.88] |
| ml | SoftThreshτ1.5 | 2 | opc | 40 | 23.48 | [22.16,24.81] |
| ml | SoftThreshτ1.5 | 4 | hybrid_seq_f0.5 | 40 | 27.34 | [25.91,28.77] |
| ml | SoftThreshτ1.5 | 4 | nll | 40 | 22.01 | [20.77,23.26] |
| ml | SoftThreshτ1.5 | 4 | opc | 40 | 25.10 | [23.77,26.43] |
| ml | SoftThreshτ1.5 | 10 | hybrid_seq_f0.5 | 40 | 27.91 | [26.44,29.38] |
| ml | SoftThreshτ1.5 | 10 | nll | 40 | 21.21 | [20.03,22.40] |
| ml | SoftThreshτ1.5 | 10 | opc | 40 | 25.74 | [24.46,27.03] |
| ml | SoftThreshτ1.5 | 100 | hybrid_seq_adapt | 34 | 26.73 | [25.40,28.05] |
| ml | SoftThreshτ1.5 | 100 | hybrid_seq_f0.25 | 34 | 26.77 | [25.09,28.46] |
| ml | SoftThreshτ1.5 | 100 | hybrid_seq_f0.5 | 40 | 27.53 | [26.02,29.05] |
| ml | SoftThreshτ1.5 | 100 | hybrid_seq_f0.75 | 34 | 26.82 | [25.52,28.12] |
| ml | SoftThreshτ1.5 | 100 | nll | 40 | 20.63 | [19.51,21.75] |
| ml | SoftThreshτ1.5 | 100 | nll_dr | 10 | 19.49 | [16.60,22.38] |
| ml | SoftThreshτ1.5 | 100 | opc | 40 | 25.36 | [24.12,26.60] |
| myket | R* | 0 | hybrid_seq_f0.5 | 20 | 21.39 | [20.81,21.98] |
| myket | R* | 0 | nll | 20 | 21.31 | [20.71,21.91] |
| myket | R* | 0 | nll_dr | 20 | 18.63 | [15.92,21.35] |
| myket | R* | 0 | opc | 20 | 19.90 | [19.23,20.58] |
| myket | R* | 0.5 | hybrid_seq_f0.5 | 20 | 21.37 | [20.33,22.41] |
| myket | R* | 0.5 | nll | 20 | 20.60 | [19.64,21.55] |
| myket | R* | 0.5 | nll_dr | 20 | 19.48 | [17.41,21.54] |
| myket | R* | 0.5 | opc | 20 | 19.74 | [18.72,20.77] |
