# CausE vs OPC: bounded development comparison (report before scaling)

> **Update, 2026-10-05.** The fair comparison is in
> [`docs/cause_fair_comparison_25k.md`](cause_fair_comparison_25k.md). It compares against the revalidated OPC (the
> corrected Stage 2 25k rows) and adds CausE-warm and CausE-capacity-matched; the latter is §11 here, now run.
> - This report's OPC side used the pre-revalidation configuration (old search range, fixed logit scale) and is
>   superseded.
> - Its native CausE rows are reused there unchanged: the data are identical and the code is 77440c5.
> - The CausE code is now on the remote: the archive `origin/cause-baseline` (d31a5dd) and the working branch
>   `cause-fair`.

*Development stage, 2026-10-04. The code is on the local branch `cause-baseline`, not pushed. Specification and mapping:
`docs/cause_baseline.md`. Everything here is development evidence: 3 datasets × 5 bias settings × 2 development seeds, at 25k
interactions.*

## 1. Implementation validation
| check | result |
|---|---|
| Objective vs the authors' unmodified TF 1.9 graph | 11 cases: prod and avg; plain SGD and momentum with linear decay; one-way and symmetric tie; with and without L2. Every parameter and per-step loss matches to < 5e-6 (`tests/test_cause_tf_reference.py`) |
| Hand calculations | loss values, tie gradients (one-way, frequency-weighted, zero for absent items), the direction-only avg tie, layouts, Xavier limits, lr schedule (`tests/test_cause_objective.py`) |
| Speed paths | the CUDA-graph step is bit-identical to the eager step. A batch of trials equals separate runs: bit for bit on CPU, ≤ 2e-6 relative on GPU. A batch of one matches TF (`tests/test_cause_batch.py`) |
| Budget split | N_c + N_t = N; uniform actions pass χ²; uniform actions are independent of the user draw; users follow the world's prior; the q of each row is the true click probability; exact nested prefixes; collection rewards (`tests/test_budget_split.py`) |
| Study arm | CausE's warm rows are a prefix of OPC's training split. **OPC's outputs are bit-identical with and without `cause`.** Deterministic; a CausE-only run reproduces the combined run (`tests/test_cause_trials.py`) |
| Full suite (final local code, `a0ee395`) | 493 passed with the GPU; 464 passed and 11 skipped with it hidden (the skips are GPU-only tests) |

## 2. Does the port behave like the original CausE?
**Yes for CausE-prod (C and T) and SP2V, quantitatively. For CausE-avg, yes once a numerical artifact of the released code is taken
into account.** Details: `docs/cause_baseline.md` §10. Data: the audit's reconstructed split (deviations D1–D6), ported row for row.

| dataset, setting | CausE-prod-C | CausE-prod-T | CausE-avg |
|---|---|---|---|
| ML-100K, validation-selected (100 ep) | 15.44 vs TF 15.37 | 12.50 vs 12.51 | exact tie 17.07 vs TF 15.92 (artifact) |
| ML-100K, Fig. 1 randomized share 0 → 15% | 13.0 → 14.4 vs TF 12.9 → 14.8 | 6.6 → 13.4 vs 6.4 → 13.5 | 15.9 → 16.7 vs 15.7 → 13.6 (matches at 0%) |
| ML-10M, README (10 ep) | 13.05 vs 12.96 | 12.36 vs 12.15 | 15.86 vs 15.19; at 0% randomized rows **15.23 vs 15.24** |

Values are MSE lift % on the uniform-exposure test set, ours (2–3 seeds) vs the audit's single TF run.

* **The artifact.** In the released CausE-avg, the pooled treatment row's tie with itself should be exactly 0. TF normalizes it with
  two kernels that round differently, so it gets a ±1 L1 subgradient at every step with randomized rows.
* **Evidence that it explains the gap:**
  * with 0% randomized rows, where the self-tie never fires, the port equals TF on both datasets;
  * emulating the rounding moves our avg past TF from the other side.
* **A consequence for the audit.** The decline of CausE-avg with more randomized data comes from this artifact, not from the method.
* **Instability.** The released optimizer's instability (plain SGD, lr 1; divergence at long training) reproduces in the same
  configurations.
* **What does not reproduce, by anyone.** The paper's own Table 2 numbers (e.g. CausE-prod-C +15.48 on ML-10M at an unreported
  configuration) are not reproduced by the original code either (audit). Netflix was not available.

## 3. CausE performance vs ρ
**Setup.**
* 3 datasets (ml, kuairand, anime) × 5 bias settings × seeds 100/101; N = 25,000.
* Native CausE (d = 32, from scratch, 20 random-search trials per variant and ρ, selected by validation NLL).
* OPC working default (DR, direct gradient, harmonic:0.1 weights, selection at clip:10), DM-only and the tempered logger, run on the
  same worlds and splits.
* Figure: `artifacts/full_study/cause_dev_25k_20261004/fig_rho_curve.png`.

**Greedy value minus the logger's greedy value, in CTR points (mean over 6 conditions):**
| arm | ρ | no bias | warp | group | vector | combined |
|---|---|---|---|---|---|---|
| CausE-prod-C | 0 / 0.10 / 0.25 | −14.2 / −15.6 / −18.4 | −10.8 / −9.5 / −7.6 | −10.9 / −9.1 / −10.5 | −7.7 / −7.7 / −6.4 | +2.8 / +2.8 / +2.8 |
| CausE-avg | 0 / 0.10 / 0.25 | −13.4 / −13.4 / −14.2 | −6.3 / −7.4 / −5.9 | −8.1 / −7.5 / −7.5 | −6.4 / −6.4 / −8.2 | +1.2 / +1.1 / +1.2 |
| CausE-prod-T | 0 / 0.10 / 0.25 | −27.2 / −22.9 / −23.8 | −19.9 / −15.6 / −16.5 | −21.4 / −17.0 / −17.9 | −20.3 / −15.9 / −16.8 | −9.6 / −6.1 / −8.1 |
| best single item (no personalization) | — | −12.5 | −5.2 | −6.6 | −5.5 | +5.2 |

* **CausE-prod-C and CausE-avg are flat in ρ.** Their validation-selected models are bias-dominated (|α| = 0.004–0.25). Their greedy
  policy recommends one item to everyone, at or near the world's best single item (17–18% CTR, against personalized ceilings of 30%).
  At about 4 interactions per user and 0.02–1.8 uniform rows per item, the from-scratch embeddings never learn personalization. Even
  the best of the 20 trials by true value does no better.
* **CausE-prod-T is the only variant that improves with randomized data** (+3.5 to +4.4 points from ρ = 0 to 0.10, flat after). It
  stays the worst, because its treatment rows see only ρN uniform rows.
* **The stochastic value** (softmax over CausE's logits, decision A7) is −7 to −21 points: the logits of bias-only models are nearly
  flat.

## 4. OPC vs CausE under equal total traffic
**Base arms (greedy gain over the logger, mean [95% CI] over 6 conditions, with the fraction of the mismatch repaired):**
| | no bias | warp | group | vector | combined |
|---|---|---|---|---|---|
| OPC | −1.41 [−1.84, −0.99] | +3.11 [+2.24, +3.98], 0.44 | +1.37 [+0.81, +1.93], 0.23 | +1.14 [+0.18, +2.10], 0.17 | +6.15 [+4.71, +7.59], 0.36 |
| DM-only | −1.11 [−1.50, −0.72] | +1.79 [+0.89, +2.69], 0.25 | +0.62 [+0.39, +0.85], 0.11 | +0.83 [−0.06, +1.72], 0.13 | +5.11 [+3.58, +6.64], 0.30 |
| tempered logger | 0 (a scale does not change the ranking) | 0 | 0 | 0 | 0 |

**OPC minus the best CausE variant at each ρ (paired over dataset × seed):**
| bias | range over ρ ∈ {0, …, 0.25} (CTR points) | OPC better in |
|---|---|---|
| no bias | +11.5 to +12.8 | 6/6 at every ρ |
| warp | +8.8 to +10.5 | 6/6 |
| group | +8.8 to +10.0 | 6/6 |
| vector | +7.6 | 6/6 |
| combined | +3.3 to +4.3 | 6/6 |

* **OPC is ahead in every condition at every ρ.** No randomized share closes the gap, because CausE's limitation here is not its data
  mix (§6).
* **Without bias**, every learned arm loses a little greedy value against the logger, whose ranking is then optimal (OPC −1.4, as in
  earlier stages). CausE loses 12–14 points.
* **Stochastic values.** OPC's softmax policy gains +2.9 to +7.8 points over the logger's stochastic value. The tempered logger
  (sharpening only) gains +2.3 to +5.9.

## 5. Exploration-cost curve
*Figure: `fig_exploration_cost.png`; costs in `table_conditions.csv`.*

**Expected clicks given up during collection, ρN·(V(π0) − V(uniform)):**
| bias | ρ = 0.01 | 0.05 | 0.10 | 0.25 |
|---|---|---|---|---|
| no bias | 53 | 266 | 531 | 1,328 |
| warp / group / vector | 39–41 | 193–207 | 386–413 | 966–1,033 |
| combined | 18 | 89 | 177 | 444 |

* The realized costs agree within 5% at ρ ≥ 0.05 (sampling noise is larger at ρ = 0.01).
* For CausE, every click given up buys at most prod-T's +3.5 to +4.4-point improvement, while the selected variants stay flat. OPC sits
  at zero cost and the highest value in every panel.
* A uniform row costs V(π0) − V(uniform) in expectation:
  * about 21 clicks per 100 uniform rows with no bias;
  * 15–17 under the single biases;
  * about 7 under combined bias, where the logger is poor.

## 6. Data collection or model capacity?
* **Not capacity.** Native CausE's class contains the true click model (its structural oracle is the ceiling). OPC's linear class does
  not: its validated structural bound recovers 100% of warp, 69% of group, 51% of vector and 70% of combined (the Stage-1 oracle). Yet
  OPC reaches 29–51% of its own bound, and CausE reaches nothing.
* **Not the randomized share.** CausE-prod-C and avg are flat in ρ, and prod-T gains only 3.5–4.4 points.
* **What it is: missing prior information.**
  * CausE starts from scratch with about 4 interactions per user. OPC starts from the logger's pre-trained (biased) vectors.
  * Native CausE's best policy at this N is a best-single-item ranker. Without bias it lands on the best single item exactly on ml, and
    within 0.2–2.7 points on kuairand and anime. At combined high bias, where the logger itself is poor, it is 2.4 points short of it on
    average: its item biases come from the biased logger's exposure.
* **What the comparison cannot yet say.** It cannot test the hypothesis about focused warm traffic vs broad random coverage at equal
  information and capacity. That needs the capacity-matched variant (§11), which shares OPC's pre-trained starting point.

## 7. Dataset- and bias-specific observations
* **kuairand, vector:** DM-only edges OPC (+1.48 vs +1.34). It is the only cell where DM ≥ OPC.
* **anime, vector:** OPC +0.10, DM −0.19. OPC repairs 2% of the mismatch, so vector bias is barely repairable on anime at 25k (its
  Stage-1 bound is also the lowest).
* **combined high:** the heavily biased logger (greedy 11–16%) is below the best single item, so even a popularity ranker beats it.
  CausE-prod-C is +2.5 to +3.2 here, still 3–4 points behind OPC.
* **ml, warp:** CausE-prod-C's selected model (−11.2) is worse than the best-item ranker that CausE-avg finds (−5.2). Selection by
  validation NLL does not always pick the better ranker among near-identical bias-only models.
* **Divergence:** about 1 in 20 CausE-avg trials diverged (lr near 1 with momentum 0.9). Selection excluded them.
* **Exploration cost:** largest without bias (the best logger) and smallest at combined high bias.

## 8. Recommended next comparison grid (for approval)
1. **Capacity-matched CausE-lin (§11)** on the same 30 cells × the same 6 ρ values, development seeds, 25k, with OPC's runs reused. This
   is the comparison that tests the data-collection hypothesis at equal capacity and equal prior information. About 2 h of GPU.
2. **Then 100k** for OPC, DM, the tempered logger and the better CausE family, if (1) shows a ρ-dependence worth resolving. 100k gives
   0.9–7 uniform rows per item at ρ = 0.1–0.25.
3. **Native CausE:** keep it as reported. Its curve is flat, so no larger grid is needed. One dense-regime check (e.g. ml at 1M rows,
   about 28 uniform rows per item at ρ = 0.10) would show whether native CausE ever personalizes here, but it is optional and expensive.
4. **The secondary comparison** (CausE given extra randomized rows on top of the warm budget) after (1).
5. **Separately:** decide whether to re-run the development stages affected by the simulator RNG fix (§10).

Provenance:
* runs: `artifacts/full_study/run_cause_dev_25k_opc_20261004/` (OPC, DM, tempered) and `run_cause_dev_25k_cause_20261004/` (CausE),
  local;
* tables and figures: `artifacts/full_study/cause_dev_25k_20261004/`;
* code: local branch `cause-baseline`;
* check: a rebuilt split has exactly the reward sums CausE recorded, and CausE's warm rows are its prefix, so both runs used the same data.

## 9. Fairness of the comparison (Step 7)
| item | OPC (working default) | DM-only | tempered logger | native CausE at ρ |
|---|---|---|---|---|
| total target interactions N | 25,000 warm-logger rows | the same | the same | 25,000 = (1−ρ)·25,000 warm + ρ·25,000 uniform |
| from the warm logger | 25,000 | 25,000 | 25,000 | (1−ρ)·25,000: the first rows of OPC's training split |
| from uniform-random exposure | 0 | 0 | 0 | ρ·25,000, a separate draw from the same world, pscore 1/\|A\| |
| propensities used | exact logged π0 (DR training, DR selection) | none in training (q̂ only); DM selection | DR selection score | none |
| source / pre-training | the logger's frozen biased vectors (BPR v2, d = 32) | the same | the same | none (from scratch) |
| representation capacity | (I+D)x+b per side | the same | logit scale only | free U, Θ_c, Θ_t (d = 32) + biases and α: contains the true click model |
| reward model | q̂ cross-fitted on its 25k rows | the same | the same | none |
| search budget | 20 trials (random search) | 20 | 20 | 20 per variant and ρ (prod-C and prod-T share their 20 trainings) |
| validation / selection | DR lower bound on 20k warm validation rows | DM score on the same rows | DR on the same rows | validation NLL on the same 20k rows |
| evaluation population | every user, prior-weighted, exact | the same | the same | the same |
| world, seeds | one world per (dataset, bias, seed); fixed simulator (§10) | the same | the same | the same world and splits |

Every run is marked `development`. The 20k validation rows are outside N for every arm (decision A8).

## 10. Separate finding: the log simulator's coupled RNG streams (fixed locally)
* **The bug.** Since `69fffab` (24 Sep), every logged action reused the uniform that drew its user. The logging policy's generator
  was seeded like the simulation's.
* **Evidence** (ml, no bias, 25k rows): 95% of users always got the same action, and corr(user index, action index) = +0.72.
  Aggregate values still matched V(π0).
* **Status.**
  * Fixed in local commit `5c011a9`; `tests/test_logging_rng_independence.py` fails on the old code.
  * Affected: every learned result logged between `69fffab` and the fix (Stages 2–3, the weighting study, the paired development
    runs behind the objective decision).
  * Not affected: the Stage-1 oracle bounds and exact values.
  * This comparison re-ran OPC, DM-only and the tempered logger on the fixed simulator.
* **Open question for you:** whether to re-run the affected development stages.

## 11. Capacity-matched CausE: proposal for approval (Step 4; not run)
Native CausE's class is unrestricted: free per-item vectors, which contain the true click model. OPC's correction is one global
linear map per side on the frozen biased vectors.

**The proposed form ("CausE-lin").** It keeps CausE's objective and replaces its free per-item residual (eq. 16:
θ^t_j = θ^c_j + θ^Δ_j, penalized) with OPC's global residual:

```text
users:            u_i   = (I + D_u) x_i + b_u           x_i = the logger's frozen (biased) user vector
control items:    θ^c_j = a_j                           a_j = the logger's frozen item vector (the control representation)
treatment items:  θ^t_j = (I + D_t) a_j + b_t           the same class as OPC's item-side repair
prediction:       ŷ = σ(α⟨u_i, θ_j⟩ + b)                no per-user or per-item biases (OPC has none)
loss:             CE on S_c with θ^c + CE on S_t with θ^t + λ · mean_B ‖θ^t_k − θ^c_k‖₁  (= λ‖D_t a_k + b_t‖₁, eq. 16 residual)
init:             D = 0, b = 0, α = 1 / T_logger        (starts exactly at the logger's scores)
evaluation:       the greedy and softmax policies of θ^t (CausE-lin-T) and θ^c (CausE-lin-C, which uses only D_u)
```

* **Why this is natural and not crippled.** CausE's own derivation writes the target representation as the source representation
  plus a penalized residual. This form keeps that structure and changes only the residual's family, from free per-item vectors to the
  global linear map OPC uses. The source-to-target correction capacity is then exactly OPC's.
* **What it isolates.** The comparison with OPC becomes objective and data only: likelihood on warm plus uniform rows with a
  discrepancy tie, against DR policy value with exact propensities. It also gives CausE the pre-trained vectors OPC has, which removes
  native CausE's from-scratch disadvantage on users with few rows.
* **Alternatives, if you prefer:**
  * (a) also learn the control map, θ^c_j = (I + D_c) a_j + b_c, which doubles the item-side capacity;
  * (b) keep CausE's per-item biases, which adds per-item capacity OPC does not have.

## 12. OPC on CausE's protocol (Step 6, after the reproduction): design, not run
CausE's protocol is a prediction task on real ratings. OPC needs a logging policy with propensities and a frozen biased
representation. The natural mapping:
* **The logger's representation:** BPR (OPC's own `BPR/` pipeline) fit on the S_c training rows.
* **Propensities:** the protocol's own item-level density ratio p_t(j) / p_c(j) (known by construction: `item_propensity.csv`,
  which WSP2V uses), capped at 10 as in the paper.
* **The policy objective:** DR with that ratio.
* **Evaluation:** CausE's metrics on the uniform test set, from OPC's repaired scores, plus a policy-level IPS value on that set.

This needs decisions (which data OPC sees, and how its policy maps to click probabilities), so it is proposed and not run.
