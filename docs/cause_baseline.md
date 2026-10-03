# CausE as a prior-work baseline in OPC: specification and design

*Status: design note (2026-10-03), before implementation. Development stage only. Nothing here changes the
OPC method; CausE is added as a separate arm.*

## 1. Sources

| source | version used | where |
|---|---|---|
| Paper | Bonner & Vasile, "Causal Embeddings for Recommendation", RecSys '18 = arXiv:1706.07639 **v6** (3 Aug 2018). v4 (Sep 2017) differs (§2.8) | `~/code/CausE/repro/.local/paper/cause_v{4,5,6}.pdf` |
| Official code | `criteo-research/CausE`, upstream `master` = **`957e556`** (2018-10-07), TensorFlow 1.x | `~/code/CausE/src/` (outside OPC) |
| Standalone audit | branch `repro/local-audit-2026-09-27`, commit **`f536ea6`** (2026-09-28): the original code runs unmodified on Python 3.6 / TF 1.9 and is bit-for-bit deterministic | `~/code/CausE/repro/AUDIT.md` |

Equation numbers below are v6's.

## 2. Mathematical specification

### 2.1 Prediction model (both CausE variants and the SP2V baselines; eq. 21)

```text
ŷ(i, j) = σ( α·⟨u_i, p_j⟩ + b_i + b_j + b )
```

* σ is the logistic function.
* u_i is the user vector and b_i the user bias.
* p_j is the item ("product") vector and b_j the item bias.
* b is a global bias and α a learned scalar scale.

### 2.2 Representations
* **User:** one matrix U, shared by both tasks.
  * Eq. 12 and eq. 18 fix the users: "the exposure change … is explained by the difference in product representations".
  * Eq. 19 also splits and ties the users (Γ_t, Γ_c). It is defined, but "for our experiments we optimize L^prod_CausE".
* **Control items:** Θ_c, with one row θ^c_j per item, fit on S_c (the large sample from the logging/control policy π_c).
* **Treatment items:** Θ_t, with one row θ^t_j per item, fit on S_t (the small sample from the fully randomized policy π_t).

### 2.3 Paper objective (eq. 14–18)

```text
L^prod_CausE = L(UΘ_t, Y_t) + Ω(Θ_t)        treatment task on S_t                 (eq. 15)
             + L(UΘ_c, Y_c) + Ω(Θ_c)        control task on S_c                   (eq. 17)
             + Ω_dist(Θ_t − Θ_c)            discrepancy between the two representations
```

* L is the cross-entropy.
* Best reported configuration: Ω_{t,c} = L2 and Ω_dist = L1. No λ values are reported.
* Residual reading (eq. 16): θ^c_j = θ^t_j − θ^Δ_j, with a penalty on θ^Δ_j. The quantity ⟨θ^Δ_j, u_i⟩ is called the ITE (eq. 13)
  but is never estimated or evaluated.

### 2.4 Released objective (`src/models.py`, verified by the audit)

**CausE-prod (`CausalProd2Vec2i`).**
* The item table has 2N rows: row j is θ^c_j and row j+N is θ^t_j. **Each row has its own bias**, so the biases are not tied.
* S_c rows enter as (i, j, y) and S_t rows as (i, j+N, y), mixed in one shuffled stream.
* Per minibatch B, the loss is:

```text
L_B = mean_{(i,k,y)∈B} CE(y, ŷ(i,k))
    + l2_pen·(½‖U‖² + ½‖P‖² + ½‖b_users‖² + ½‖b_items‖²)                 (full matrices; b and α unpenalized)
    + cf_pen · mean_{(i,k,y)∈B} ‖ P_k − sg(P_{r(k)}) ‖₁,   r(k) = k+N if k < N else k
```

* **One-way tie.** sg is stop-gradient, so only θ^c receives tie gradient and θ^t receives none.
* **Frequency-weighted and lazy.** Only items in the batch are tied, each in proportion to its count in the batch. S_t rows contribute 0
  to the tie (r(k) = k).
* **Defaults:**
  * initialization: α = 1e-8 (trainable); U and P Xavier-uniform on their full shapes (the doubled table gives a √2-smaller scale);
    biases 0;
  * optimizer: plain SGD at lr = 1.0, with no momentum or decay;
  * batch 512, `num_epochs` = 1 (the README uses 10), `l2_pen` = 0, `cf_pen` = 1, `embedding_size` = 50, `cf_distance` = `l1`.
* **Batch order.** `tf.data` caches the first shuffled order and replays it every epoch.
* **Bugs (not used).** The `l2` and `cos` distance options normalize across the batch axis and leak gradient into θ^t.

**CausE-avg (`CausalProd2Vec`).**
* Every S_t row is mapped to **one pooled item row** (id 0), so S_t item identities are discarded.
* The tie applies to every batch row:

  ```text
  cf_pen · mean_B ‖ P_k/‖P_k‖ − sg(P_0)/‖P_0‖ ‖₁
  ```

* The ℓ2 normalization is not in the paper. TF's `l2_normalize` uses `x / sqrt(max(Σx², 1e-12))`.

### 2.5 Variants and prediction
| variant | trained model | prediction row for item j |
|---|---|---|
| CausE-prod-C | prod | control row j |
| CausE-prod-T | prod (the same trained model as prod-C) | treatment row j+N |
| CausE-avg | avg | control row j (the pooled row is not an action) |

The paper also lists SP2V-no, blend and test, WSP2V (IPS-weighted, weights capped at 10), BPR and BanditNet as baselines.

### 2.6 Strongest variant
* **v6 Table 2:** CausE-prod-C is best on both datasets:
  * ML-10M: MSE lift +15.48%, NLL lift +19.12%, AUC 0.814;
  * Netflix: +17.82%, +17.19%, 0.821.
* The paper's explanation is that fitting the large S_c models user responses, while the tie controls the deviation from the target task.
* **v4** called CausE-prod-T "clearly" best (§2.8).
* **The audit:** at the README configuration on ML-10M, CausE-avg is best and prod-C is the weakest non-T method. The tie is inert
  in the released recipe.

### 2.7 Optimization, selection, evaluation
* **Optimizer.**
  * v6 says SGD with momentum and a linearly decaying learning rate.
  * The code uses constant-lr plain SGD.
  * The audit implemented the v6 description and found no change in results (≤ 0.04 pt).
* **Selection.**
  * The paper does not describe it. Its split holds out 10% validation, "all from π_c".
  * The code has optional early stopping on an S_t-validation loss, off by default.
  * The audit selected by π_c-validation NLL; the S_t rule picks the same configurations.
* **Evaluation.**
  * Metrics: MSE and NLL "lift" over the average predictor (eq. 20), plus AUC, on the uniform-exposure test set.
  * The code's average predictor feeds the test **odds** p/(1−p) instead of the rate, which inflates every lift slightly and equally.
  * The paper's ± is the std over 30 bootstrap resamples (80%, with replacement) of one model's test set.
  * The policy the paper's argument implies (eq. 11) is **argmax_j ŷ^rand(i, j)**, i.e. greedy ranking by the predicted
    outcome under random exposure.

### 2.8 Version differences that matter
* v4 → v6: the best variant flipped (prod-T → prod-C), the optimizer description changed, and the main-table datasets changed:
  * v4 used ML-100K and ML-10M;
  * v6 uses ML-10M and Netflix, with ML-100K kept for the Fig. 1 dose-response.
* The code matches neither optimizer description.

## 3. Ambiguities: proposed handling (for your decision)

| # | question | proposed default | alternative |
|---|---|---|---|
| A1 | Which objective is "native CausE": eq. 18 (symmetric tie, Ω on Θ) or the released code (one-way stop-gradient tie, untied biases, α scale)? | **The released objective (§2.4)**, numerically verified against the TF code. The symmetric tie is available as a flag (`tie=symmetric`) | eq. 18 as the default |
| A2 | Optimizer in the OPC comparison | **v6's description:** SGD, momentum 0.9, linear decay to 0, with a tuned initial lr. The released plain SGD (lr 1.0) is used for the reproduction (§6) | the released recipe everywhere |
| A3 | Search space and budget | **20 trials per variant**, the same count as each OPC arm, with the same Optuna sampler. Space: lr log[1e-3, 1]; epochs {1, 3, 10, 30, 100, 300}; `l2_pen` {0, 1e-6, 1e-5, 1e-4, 1e-3}; `cf_pen` {0, 0.1, 1, 10, 100}; batch 512 (the code's) | also search the batch size |
| A4 | Embedding dimension | **d = 32**: OPC's representation dimension and the rank of the true click model | d = 50 (the code's default) |
| A5 | Source representation | **None (from scratch)**, as in the paper. CausE never sees the logger's pre-trained vectors | initialize from the biased vectors: a modified CausE that needs your approval |
| A6 | Selection | **Validation NLL** of each variant's own predictions on the **same 20k warm-logger validation rows** OPC uses ("validation from π_c") | the code's S_t-validation: would need uniform validation rows, charged to the budget |
| A7 | Stochastic policy of an outcome predictor | Greedy (eq. 11) is **primary**. The "stochastic value" is reported for softmax over CausE's own logits (τ = 1); this is a convention, not in the paper | report CausE greedy only |
| A8 | Validation rows and the budget N | **Not part of N** (OPC's existing convention): 20k warm rows per condition, identical for every arm, used only for selection | count them in N |
| A9 | Netflix | Requires Kaggle credentials and is not on this machine. Reproduce on **ML-100K** (Fig. 1 protocol and the code's default dataset) and **ML-10M** (Table 2) | you provide the Netflix data |

## 4. Mapping into OPC (the controlled-mismatch simulator)

### 4.1 Data roles
| CausE role | OPC source | notes |
|---|---|---|
| S_c | the first (1−ρ)N rows of the **warm logger's** training split for size N | a uniformly random subset of OPC's N rows, nested across ρ. Logger: the sharpened softmax over the biased vectors (`--logger-greedy-share 0.8`). Its exact propensities are recorded but CausE does not use them |
| S_t | the first ρN rows of a **separate uniform-random simulation** | the same world, user prior and click model, an independent seed stream, pscore = 1/\|A\| exactly; nested across ρ |
| validation | the 20,000 warm-logger validation rows of the same split | shared with OPC, DM-only and the tempered logger |
| test | exact evaluation on the simulator's true q(u, a) | all users, prior-weighted, as for every OPC arm |

Budget per condition and size N: OPC, DM-only and the tempered logger see N warm rows. CausE at ρ sees N_c = N − N_t warm rows plus
N_t = round(ρN) uniform rows, so N_c + N_t = N exactly. At N = 25k, N_t ∈ {0, 250, 1250, 2500, 3750, 6250}.

### 4.2 From predictions to policies
* The logit of item j for user u is `z_j(u) = α⟨U_u, P_row(j)⟩ + b_row(j)` (+ user and global terms, which do not change the ranking).
* **Greedy value:** the exact value of argmax_j z_j(u), using `calc_greedy_reward` on augmented vectors [αU_u, 1] · [P_j, b_j].
* **Stochastic value:** the exact value of softmax_j z_j(u) (A7).
* For prod, both C and T are evaluated on every trial. Users with no training rows keep their random initial vectors and zero bias, which
  is native CausE behavior.

### 4.3 Outcomes, per arm × ρ × condition
* Greedy and stochastic true values.
* Gain over the original logger's greedy and stochastic values.
* **Fraction of target mismatch repaired:** (V_greedy − V_logger_greedy) / (V_target_best − V_logger_greedy).
* **Fraction of the structural oracle repair**, using each method's own class:
  * OPC: the Stage-1 linear-repair oracle (`stage1_oracle_rows.csv`), which exists for all planned cells.
  * Native CausE: its class contains the truth exactly. q = σ(s·x_u·a_a + c) is matched by U = √s·x and P = √s·a, with biases at 0 and
    global bias c, as long as d ≥ 32. Its structural oracle is therefore the ceiling, and its fraction of oracle repair equals its fraction of
    mismatch repaired.
* **Collection reward:** realized Σr and expected Σq for each subset (warm and uniform). The exploration cost is
  `ρN·(V(π_b) − V(uniform))` in expectation, plus the realized difference.
* Also reported: per-item S_t coverage, and the oracle-selected (best-true-value) trial as an upper-bound diagnostic for every arm.

### 4.4 Fairness rules (recorded in `run_meta.json` and every summary row)
| item | OPC (working default) | DM-only / tempered logger | native CausE at ρ |
|---|---|---|---|
| target interactions N | N warm | N warm | (1−ρ)N warm + ρN uniform |
| propensities | exact, logged | DM: none (only through q̂); tempered: DR score | not used (paper) |
| source / pre-training | frozen biased vectors (the logger's) | the same | none (from scratch) |
| representation capacity | (I+D)x+b per side on d = 32 | DM: the same class; tempered: logit scale only | free U, Θ_c, Θ_t, d = 32, + biases and α |
| search budget | 20 Optuna trials | 20 | 20 per variant (prod-C and prod-T share their 20 trainings) |
| selection | DR lower bound on the 20k warm validation rows | DM score / DR score | validation NLL on the same 20k rows |
| reward model | q̂ cross-fitted on its N rows | the same | none |
| evaluation | exact, all users | the same | the same |

Every run is `--stage development` until the comparison is frozen.

## 5. Implementation plan
* `utils/budget_split.py`: the uniform logger (exact 1/|A|), the nested warm/uniform partition, collection-policy labels, and
  collection-reward records. OPC's existing splits and seeds are unchanged; the uniform stream uses its own derived seed.
* `models/cause.py`: a PyTorch CausE with variants `prod` and `avg`. It implements the released objective exactly (§2.4) and has a
  `symmetric` tie option. The optimizers are `sgd` (released) and `momentum_decay` (v6).
* `training/cause_trials.py`: the Optuna loop (20 trials per variant and ρ, seeded with `derive_seed`), with selection by validation NLL.
  Each trial logs true greedy and stochastic values, α, tie diagnostics and the data counts. It writes the same files as the other arms
  (`*_trials_long.csv`, summary rows, `run_meta.json`).
* `training/run_full_study.py` and `_parallel.py`: an opt-in `--methods cause` with `--cause-rhos`, `--cause-variants`, `--cause-dim`,
  `--cause-optimizer`, `--cause-tie`. The other arms are untouched, and a test checks that their outputs are bit-identical with and
  without `cause`.
* **Validation tests:**
  * the loss against hand calculations on tiny inputs;
  * gradient properties: one-way tie, frequency weighting, no tie gradient for absent items, avg normalization;
  * a **numerical match against the authors' TF graph**. A harness, run in the audit's py3.6/TF 1.9 environment, imports the unmodified
    `models.py`, feeds fixed batches, and dumps the initial and final parameters into a small `.npz` fixture. The OPC test loads the same
    initial values and must match after k SGD steps;
  * the budget tests requested: N_c + N_t = N, uniformity (χ²), no extra rows, same world and seeds.
* `training/cause_protocol.py`: a port of the audit's reconstructed SKEW split (ML-100K, ML-10M), checked row-for-row against the audit's
  files. It reruns the released recipe in PyTorch and compares with the audit's TF numbers (target: within ≈ 0.5 lift pt and 0.005 AUC).

## 6. Not in this stage
* **The capacity-matched CausE variant (Step 4):** proposed in the report, run only after approval.
* **OPC on the CausE protocol:** designed after the reproduction.
* **Not started at all here:** BLOB, the large grid, an OPC-on-mixture arm, a 100k size before the 25k analysis passes, and a
  continuous ρ search.

## 7. Milestones and ETAs (wall clock)
| milestone | content | ETA |
|---|---|---|
| M1 | this note | done at commit |
| M2 | budget split, CausE core, unit tests and TF numerical match; commit and push | +4 h |
| M3 | study-runner arm, determinism and bit-identity tests, smoke run; commit and push | +3 h |
| M4 | ML-100K reproduction (then ML-10M in the background); commit and push | +2 h |
| M5 | bounded comparison at 25k: 3 datasets × 5 biases × seeds 100/101, OPC / DM / tempered / CausE × 6 ρ | +6–8 h of compute |
| M6 | analysis, figures and report; stop | +2 h |
