# CausE as a prior-work baseline in OPC: specification and design

*Status (2026-10-03):*
* *The design was approved with the defaults of §3.*
* *The implementation is on the local branch `cause-baseline`; its code is not pushed (§8).*
* *Development stage only. CausE is a separate arm and does not change the OPC method.*
* *Separately, a seeding bug in OPC's log simulator was found and fixed (§9).*

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

**Decision (2026-10-03):** use the proposed defaults A1–A9.

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

Code milestones are committed on the local branch only. Documentation and results may be pushed to `CRM`.
| milestone | content | ETA |
|---|---|---|
| M1 | this note | done at commit |
| M2 | budget split, CausE core, unit tests and TF numerical match; local commit | done (`5c011a9`, `b2f05d8`, `e497180`) |
| M3 | study-runner arm, determinism and bit-identity tests, smoke run; local commit | done (`5a01501` ... `a0ee395`, local) |
| M4 | ML-100K reproduction (then ML-10M in the background); local commit | done (§10) |
| M5 | bounded comparison at 25k: 3 datasets × 5 biases × seeds 100/101, OPC / DM / tempered / CausE × 6 ρ | done (2026-10-04) |
| M6 | analysis, figures and report; stop | done: `docs/cause_dev_report_20261004.md` |

## 8. Implementation and validation status (local branch `cause-baseline`)

| piece | where | validation |
|---|---|---|
| CausE objective, variants, optimizers | `models/cause.py` | `tests/test_cause_tf_reference.py`: **11 cases from the authors' unmodified TF 1.9 graph match to < 5e-6** in every parameter and per-step loss. They cover prod and avg, plain SGD and momentum with decay, the one-way and symmetric tie, and L2. `tests/test_cause_objective.py`: hand-calculated losses and gradient properties |
| fixed-budget data | `utils/budget_split.py` | `tests/test_budget_split.py`: N_c + N_t = N; χ² uniformity; independence of the user draw; the world's user prior; true q; exact nested prefixes; collection rewards |
| study arm | `training/cause_trials.py` with `--methods ... cause` | `tests/test_cause_trials.py`: the labels and budget metadata; CausE's warm rows are a prefix of OPC's training split; **OPC's outputs are bit-identical with and without `cause`**; determinism |
| CausE's own protocol | `training/cause_protocol.py` | `tests/test_cause_protocol.py`: the reconstructed ML-100K split (all files, including the Fig. 1 levels) is **identical row for row** to the audit's; the bootstrap metrics follow `src/utils.py` |
| TF fixture generator | `scripts/cause_reference/make_tf_fixture.py` | runs only in the audit's Python 3.6 / TF 1.9 environment and refuses to run if `src/` differs from `957e556` |

### 8.1 Where the port differs from the released code
1. **The CausE-avg pooled row's self-tie.**
   * Randomized rows are mapped to the pooled row, so their tie compares the pooled vector with itself.
   * TF normalizes the pooled vector twice, once as batch rows (axis 1) and once alone (axis 0). The two kernels round differently
     (|difference| ≈ 1e-8, nonzero in every coordinate), so `abs()` passes a full ±1 subgradient. In the released code the pooled
     treatment vector is therefore pushed every step in a direction set by rounding.
   * The port uses the exact 0, which is mathematically the same objective.
   * Injecting TF's signs reproduces TF to < 5e-6, so this is the only difference.
   * Expected impact is small: the audit's CausE-avg performs the same with 0% randomized data, where this term never fires.
2. **Batch order.** The port uses a seeded full permutation, replayed every epoch, where TF uses a 10k-row shuffle buffer followed by
   `cache()`. Both replay one order per run; they differ in distribution only.
3. **Random streams.** The Xavier initialization has the same distribution as TF's but a different random stream, so single runs
   differ at the seed level.
4. **Dense updates.** Every update is dense, as in TF. The released graph's L2 term exists even at `l2_pen` = 0, so TF densifies
   every embedding gradient; momentum is therefore applied to every row, verified against TF.

## 9. Separate finding: the OPC log simulator's RNG streams were coupled (fixed locally)
* **The bug.** `_simulate_from_embedding_policy` seeded the logging policy's generator with the same integer as the simulation's.
  Since `69fffab` (2026-09-24), `Policy.sample_actions` draws one uniform per row. Each logged action therefore reused the uniform
  that drew its user.
* **Evidence** (ml, no bias, 25k rows):
  * 95% of users with ≥ 2 rows always got the same action, against 3.8% with independent streams;
  * corr(user index, action index) = +0.72;
  * the logged rows' mean q still matched V(π0) within noise.
* **Consequence.** The recorded pscore π0(a|u) was not the per-user sampling probability.
* **Fix** (local commit `5c011a9`):
  * the policy seed is `derive_seed(random_state, "logging_policy_actions")`;
  * `create_simulation_data_from_policy` rejects a policy whose generator is in the simulation's state;
  * `tests/test_logging_rng_independence.py` fails on the old code.
* **Scope.**
  * Affected: every learned result logged from `69fffab` until the fix (the development Stages 2–3, the weighting study, the paired
    runs behind the objective decision).
  * Not affected: exact values and the Stage-1 oracle bounds.
  * The CausE comparison runs on the fixed simulator, and re-runs OPC, DM-only and the tempered logger in the same runs.

## 10. Reproduction of CausE's own protocol (MovieLens, released recipe)

**Setup.**
* Data: the audit's reconstructed SKEW split (deviations D1–D6), ported to `training/cause_protocol.py` and identical to the audit's
  files row for row. ML-100K, split seed 0.
* Recipe: the released one, i.e. plain SGD at lr 1.0, batch 512, d = 50, tie 1 unless stated.
* Seeds: ours 0, 1, 2; TF is the audit's single run of the unmodified code.
* Metrics: computed as `src/utils.py` does (30 bootstraps; lift over the released "average predictor").
* Agreement: within 0.5 lift point and 0.005 AUC.
* Files: `artifacts/cause_repro/`.

### 10.1 ML-100K, every configuration the audit ran in TF

**Stable configurations (ours, mean ± sd over 3 seeds, vs TF):**
| method | configuration | ours MSE lift | TF MSE lift | ours AUC | TF AUC | agrees |
|---|---|---|---|---|---|---|
| CausE-prod-C | 1 ep (released default) | 0.06 ± 0.28 | 0.22 | 0.7464 | 0.7466 | yes |
| CausE-prod-C | 10 ep (README) | 8.20 ± 0.16 | 8.20 | 0.7540 | 0.7541 | yes |
| CausE-prod-C | 100 ep (validation-selected), tie 0.1 / 1 / 10 | 15.44 ± 0.09 | 15.37 | 0.7764 | 0.7764 | yes |
| CausE-prod-T | 1 ep | −0.36 ± 0.28 | −0.19 | 0.7212 | 0.7213 | yes |
| CausE-prod-T | 10 ep | 5.75 ± 0.15 | 5.75 | 0.7244 | 0.7244 | yes |
| CausE-prod-T | 100 ep, tie 0.1 / 1 / 10 / 100 | 12.40–12.50 | 12.44–12.52 | 0.7481–0.7489 | 0.7480–0.7489 | yes |
| SP2V-no / blend / test | 10 ep | 6.87 / 8.59 / 2.56 | 6.52 / 8.76 / 2.64 | 0.7460 / 0.7589 / 0.7289 | 0.7461 / 0.7590 / 0.7284 | yes |
| SP2V-test | 200 ep, L2 1e-5 | 12.05 ± 0.08 | 12.16 | 0.7429 | 0.7429 | yes |

**Unstable configurations.** These are configurations where the released optimizer diverges in some runs:
* prod with tie 0 at 100 ep;
* prod at 200–300 ep;
* SP2V-no at 100 ep;
* SP2V-blend at 100 ep with L2 1e-5;
* (prod with tie 100 at 100 ep is chaotic: L1 chattering).

Per seed, the converged runs land on TF's numbers, and the others diverge with |α| ≈ 22:
* SP2V-no at 100 ep: 14.31 against TF 14.00 (the other two seeds −7.2 and −12.0);
* prod-C with tie 0 at 100 ep: 15.58 against TF 15.35.

TF diverges as well at 200–300 epochs. The audit reports the same instability (SP2V diverged in 2 of 5 replicates).

### 10.2 CausE-avg: the released code's rounding artifact (§8.1)
Our exact-tie CausE-avg differs from TF beyond seed noise:

| epochs | ours, exact tie | ours, TF rounding emulated | TF |
|---|---|---|---|
| 10 | 8.67 ± 0.65 | 9.50 ± 0.27 | 9.82 |
| 100 | 17.07 ± 0.02 | 15.01 ± 0.07 | 15.92 |
| 300 | 18.01 ± 0.04 | 15.08 ± 0.11 | 16.75 |

* Emulating the artifact (`emulate_tf_pooled_rounding`) moves our result past TF. TF lies between the exact and the emulated versions,
  as expected for an artifact whose strength depends on kernel rounding.
* The Fig. 1 protocol (§10.3) shows it directly. At 0% randomized rows, where the pooled self-tie never fires, our CausE-avg equals TF
  (15.85 vs 15.72). The gap then grows with the randomized share.
* **The decline of CausE-avg with more randomized data that the audit reports (15.7 → 13.6) is produced by the released code's
  rounding artifact, not by the CausE objective.** The exact objective is flat to rising (15.9 → 16.7).
* ML-10M confirms the attribution. With 0% randomized rows (no pooled rows, so no artifact), our CausE-avg gets 15.33 / 15.12 MSE lift
  (seeds 0, 1), 18.03 / 17.92 NLL lift, AUC 0.813 and α ≈ −3. The audit's TF run gets 15.24 / 18.05 / 0.813 with α = −3. With the 10%
  randomized rows, the two differ (§10.5).

### 10.3 Fig. 1 protocol (ML-100K; the randomized share injected into training; 2 seeds)
| share of all events | 0% | 1% | 2.5% | 5% | 7.5% | 10% | 15% |
|---|---|---|---|---|---|---|---|
| CausE-prod-C, ours | 13.03 | 12.80 | 13.29 | 13.10 | 13.96 | 13.97 | 14.40 |
| CausE-prod-C, TF | 12.87 | 12.99 | 13.83 | 13.80 | 13.80 | 13.96 | 14.79 |
| CausE-prod-T, ours | 6.62 | 6.86 | 8.12 | 9.29 | 10.99 | 11.86 | 13.39 |
| CausE-prod-T, TF | 6.44 | 6.99 | 8.34 | 9.81 | 10.92 | 11.86 | 13.46 |
| CausE-avg, ours (exact) | 15.85 | 15.91 | 16.16 | 16.35 | 16.76 | 16.65 | 16.74 |
| CausE-avg, TF | 15.72 | 16.11 | 16.21 | 16.11 | 15.82 | 15.30 | 13.57 |

All values are MSE lift %; prod uses 100 ep, avg 300 ep. SP2V-blend at the fixed configuration (L2 1e-5, 100 ep) diverges in both
implementations at most levels, so it is not compared here.

### 10.4 Reading
* **CausE-prod-C and prod-T** reproduce the original implementation quantitatively, at the selected configurations and along the
  whole Fig. 1 dose-response.
* **CausE-avg** differs only through the released code's floating-point artifact on the pooled row. Our port implements the exact
  objective.
* **The released optimizer's instability** reproduces too: the same configurations diverge.
* **ML-10M** (Table 2 setting) reproduces in the same way (§10.5).
* **Netflix** is not available (A9).

### 10.5 ML-10M (split seed 0; ours: seeds 0 and 1; TF: the audit's single run)
| method | configuration | ours MSE / NLL lift, AUC | TF MSE / NLL lift, AUC | agrees |
|---|---|---|---|---|
| CausE-prod-C | 1 ep (released default) | 4.80 / 5.29, 0.750 | 4.86 / 5.46, 0.750 | yes |
| CausE-prod-C | 10 ep (README) | 13.05 / 15.24, 0.796 | 12.96 / 15.31, 0.796 | yes |
| CausE-prod-T | 1 ep | 2.34 / 2.90, 0.799 | 2.39 / 3.03, 0.799 | yes |
| CausE-prod-T | 10 ep | 12.36 / 14.90, 0.813 | 12.15 / 14.87, 0.813 | yes |
| SP2V-blend | 10 ep | 14.14 / 16.69, 0.805 | 13.93 / 16.71, 0.806 | yes |
| CausE-avg | 1 ep | 8.80 / 9.47, 0.774 | 10.52 / 11.95, 0.782 | no (§10.2) |
| CausE-avg | 10 ep | 15.86 / 18.49, 0.818 | 15.19 / 17.85, 0.817 | no (§10.2) |

The paper reports, for comparison: CausE-prod-C +15.48, CausE-prod-T +7.46, CausE-avg +12.67 and SP2V-blend +4.37 (v6 Table 2). These
are the numbers that neither the audit's TF runs nor the port reproduce.
