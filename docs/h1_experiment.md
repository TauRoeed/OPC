# H1 Experiment

**Hypothesis.** If logging is strongly biased, train size `n` is large, and the reward model `q_hat` is bad, **OPC can lose to naive** on true policy value. Naive is still biased. Correction variance / a bad DM term can dominate.

**One-liner (full ablation).** Sweeps datasets, noise, train size, rand_CTR, q-error, logging mix, val size, and 15 seeds. Then analyzes.

```bash
python -m training.run_h1_study --datasets ml anime myket kuairec --run-tag h1_full --train-sizes 5000 25000 100000 250000 400000 --target-rand-ctrs 0.02 0.08 0.18 --q-errors 0.0 0.25 0.5 0.75 1.0 --logging-mixes 0.0 0.3 --noise-levels low medium high extreme brutal --val-sizes 50000 100000 200000 --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --n-trials 15 --policy-losses sndr --policy-temperature 2.0 --qhat-user-chunk 10000 --qhat-action-chunk 10000 --num-gpus 2 --workers-per-gpu 16 --require-cuda --slim && python -m training.analyze_h1_study --root artifacts/h1_study/run_h1_full
```

Uses **2 GPUs** and **16 workers per GPU** (32 processes). User–item scoring chunks are **10k**. OOM backoff drops workers if VRAM dies.

Smoke:

```bash
SMOKE=1 ./scripts/run_h1_study.sh
```

---

## Grid

| Axis | Values | Meaning |
|------|--------|---------|
| dataset | `ml`, `anime`, `myket`, `kuairec` | BPR embedding catalogs |
| `noise_level` | low, medium, high, extreme, brutal | embedding mix vs ground truth |
| `train_size` `n` | 5k, 25k, 100k, 250k, 400k | logged trajectories used to learn `pi` |
| `val_size` | 50k, 100k, 200k | validation logged trajectories |
| `target_rand_ctr` `rho` | 0.02, 0.08, 0.18 | Sparse / Moderate / Dense. Sim `ctr` is **calibrated** so random-item mean reward ≈ `rho` |
| `q_error` `eps` | 0, 0.25, 0.5, 0.75, 1 | how wrong `q_hat` is vs true `q*` |
| `logging_mix` `alpha` | 0, 0.3 | CleanLog vs HurtLog (`alpha=0.3` and temperature `T=2`) |
| seeds | 0–14 | repeats |

Fixed: noise mode `kmeans_templates` / axis `combined`, OPC loss `sndr` (no KL/CRM), naive loss `naive`, qhat chunks 10k, 16 workers/GPU.

Outcome:

```text
delta  =  V(OPC)  -  V(naive)
```

`V` is **true** expected reward on the clean environment (not the OPE estimate). H1: `delta < 0` more often when `eps` is large, `alpha` is large, and `n` is large.

---

## One simulation, start to finish

BPR embeddings already exist: clean user factors `e_x` and item factors `e_a`.

### 1. Calibrate density

Pick target `rho` (rand_CTR). Choose sim link parameter `ctr` so that, under **uniform** user and item,

```text
E[ q_link(e_x[u], e_a[a]; ctr) ]  ≈  rho
```

`rho` is **mean reward under random items**. It is **not** “optimal CTR”. `ctr` is only a link ceiling given scores.

### 2. True world

User traffic:

```text
P(u)  proportional to  Exp(1)    then normalize
```

True mean reward uses **clean** embeddings (`SyntheticBanditEnv`):

```text
s*(u,a)  =  e_x[u] · e_a[a] / T_env     (T_env = 1)
q*(u,a)  =  1 / ( 1/ctr + exp(-s*) )
r        ~  Bernoulli(q*)
```

### 3. What the logger and learner see

Noisy copies of the embeddings:

```text
our  =  (1 - eps1 - eps2 - eps_meta) * gt
     +  eps1 * N_linear
     +  eps2 * N_cluster
     +  eps_meta * N_meta
```

Rewards stay `q*` on clean `e_x`, `e_a`. Logging and the CF model start from `our_x`, `our_a`.

### 4. Logging policy

Softmax on **noisy** dots, temperature `T` (`T=2` on HurtLog):

```text
pi_soft(a|u)  =  softmax_a( our_x[u] · our_a[a] / T )
pi_b(a|u)     =  (1 - alpha) * pi_soft(a|u)  +  alpha / |A|
```

`alpha = 0` → CleanLog. `alpha = 0.3` → HurtLog (uniform mix).  
Sample from the mix; **logged propensity `p` is always `pi_b(a|u)`**.

### 5. Draw the log

For each of `n` train rows (plus val / extra slices):

```text
u  ~  P(u)
a  ~  pi_b(· | u)
p  =  pi_b(a | u)
r  ~  Bernoulli(q*(u,a))
```

Logged tuple: `(u, a, r, p)`.

### 6. Frozen reward model (H1 dial)

Oracle `q*` mixed with a constant `b = rho`:

```text
q_hat(u,a)  =  (1 - eps) * q*(u,a)  +  eps * b
```

Because rewards are in `[0, 1]`:

```text
max |q_hat - q*|  <=  eps
```

`eps = 0` → perfect `q*`. `eps = 1` → constant. **Not trained** during policy learning. Used only in OPC’s SNDR/DM terms.

### 7. What we learn

Collaborative-filtering policy. Init from noisy `our`. Freeze those tables. Train residual MLPs:

```text
u' = our_x[u] + MLP_u( LN(our_x[u]) )
a' = our_a[a] + MLP_a( LN(our_a[a]) )

pi_theta(a|u)  =  softmax_a( u' · a' / T )
```

Same `T` as logging.

### 8. Learning (one batch)

Batch: users, logged actions `a`, rewards `r`, propensities `p`.  
Model outputs `pi_theta(·|u)` over all items.

**OPC** (`sndr`, logged `p`, log-trick on, **no** KL, **no** CRM):

```text
w_i     =  pi_theta(a_i | u_i) / p_i
DM_i    =  sum_a  q_hat(u_i, a) * pi_theta(a | u_i)
SNDR_i  =  w_i * (r_i - q_hat(u_i, a_i)) / mean(w)  +  DM_i

L_OPC   =  - mean(SNDR surrogate)
```

Adam on MLP weights. Optuna does **not** search `gamma`, `M`, `lambda`.

**Naive** (same data, same model family, **no** propensities, no `q_hat`, no KL/CRM):

```text
L_naive  =  - mean( r_i * pi_theta(a_i | u_i) )
```

Optuna searches lr, epochs, batch, decay. Selection uses val OPE (`r_hat` / `ci_low`), **not** true `V`, unless you set oracle selection.

### 9. Score on the true world

After training, true value uses **clean** `q*` and the learned softmax (no uniform mix):

```text
V(pi)  =  sum_u P(u) * sum_a pi_theta(a|u) * q*(u,a)
```

Compare OPC vs naive on the **same** splits and search budget.

---

## Why H1 can happen (sketch)

Naive bias from logging shift **does not vanish** as `n` grows:

```text
B_naive  ~  mismatch between pi and pi_b
```

OPC error has two extra pieces:

```text
DM harm        ~  (q_hat - q*) weighted by (pi / pi_b - 1)     grows with eps
IPS variance   ~  E[w^2] / n                                    grows when p is small (HurtLog)
```

Naive wins when:

```text
B_naive  <  c1 * eps  +  c2 * E[w^2] / n
```

Large `n` kills the variance term. Then you need **large `eps`** and/or **HurtLog** (heavy `w`) for naive to win.

`rho` enters as: reward sparsity, the calibrated `ctr`, and the constant bad predictor `b = rho`.

---

## Thresholds (after the run)

`python -m training.analyze_h1_study` writes `h1_threshold_grid.csv`. Look for cells with `frac_naive_wins >= 0.5`:

```text
eps         >=  eps_star
n           >=  n_star
alpha       >=  0.3
rho         in  some band from data
```

Those cutoffs come from the grid, not from a theorem.

---

## Code map

| File | Role |
|------|------|
| `training/run_h1_study.py` | grid |
| `training/analyze_h1_study.py` | delta + thresholds |
| `utils/rand_ctr.py` | estimate `rho`, calibrate `ctr` |
| `utils/bounded_q_model.py` | `eps`-bounded `q_hat` |
| `scripts/run_h1_study.sh` | smoke / full wrapper |
| `utils/simulation_utils.py` | `q*`, logs, true `V` |
| `models/custom_losses.py` | `sndr` vs `naive` |
| `models/models.py` | `CFModel` residual MLPs |
