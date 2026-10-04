# CausE vs revalidated OPC: the fair 25k comparison (design, then results)

*Development stage. Branch `cause-fair`: the revalidated `CRM` (d5b4478) with the archived CausE implementation merged
(`origin/cause-baseline` = d31a5dd). This note fixes the design before any new run. Results follow in later
sections. Do not read anything here as confirmatory.*

## 0. The question and the protocol

> Given the same useful source representation and the same total target-interaction budget, is it better to adapt
> using partly randomized target traffic, as in CausE, or ordinary warm-logger traffic with propensities, as in OPC?

**Budget.**
- Per world (dataset × bias × seed) and budget N = 25,000:
  - **OPC, DM-only and the tempered logger** see N warm-logger rows. OPC and the tempered logger also use their
    exact propensities.
  - **Every CausE variant at ρ** sees N_c = N − round(ρN) warm rows, the first rows of OPC's training split, plus
    N_t = round(ρN) uniform-random rows (pscore 1/|A|), a separate draw from the same world. So N_c + N_t = N
    exactly (`utils/budget_split.py`, unchanged from M5).
- ρ ∈ {0, 0.01, 0.05, 0.10, 0.15, 0.25}, prespecified and not tuned. N_t ∈ {0, 250, 1,250, 2,500, 3,750, 6,250}.
- The 20,000 warm validation rows are outside N and shared by every arm (decision A8 of `docs/cause_baseline.md`).
  They are the only rows any method selects on.

**Worlds.** Exactly the M5 grid, so valid rows are reused:
- ml, kuairand, anime;
- no bias, warp high, group high, vector high, combined high;
- seeds 100/101.

That is 30 worlds. Their logs are the corrected logs: on all 30, the tempered logger has exactly the same value in
M5 and in the corrected Stage 2 (revalidation Table R14).

**Notation.**
- x_i, a_j ∈ ℝ³² are the **source vectors**: the logger's biased user and item vectors (`our_x`, `our_a`; BPR v2
  after the representation bias).
- The logger is π0(j | i) = softmax_j(⟨x_i, a_j⟩ / T).
- The truth is q(i, j) = σ(s·⟨x*_i, a*_j⟩ + c) on the clean vectors. No method sees it.
- S_c and S_t are the warm and uniform training rows; V is the validation rows.

## 1. The three CausE variants

All three use the SP2V prediction model of CausE (eq. 21), z(i, k) = α⟨u_i, p_k⟩ + b_i + b_k + b with ŷ = σ(z). They
are trained by the released CausE objective (`docs/cause_baseline.md` §2.4), with momentum and linear decay (decision
A2). Each is selected by validation NLL on V, and evaluated exactly.

### 1.1 Native CausE (M5, reused unchanged)

- **Parameters:** U ∈ ℝ^{n_u×32}; an item table P with 2·n_a rows (row j = control θ^c_j, row j + n_a = treatment
  θ^t_j), each row with its own bias; user biases; b; α.
- **Initialization:** Xavier-uniform U and P, biases 0, α = 1e-8. CausE never sees the source vectors.
- **Loss** (one minibatch B of S_c rows mapped to control rows and S_t rows mapped to treatment rows):

```text
L_B = mean_B CE(y, σ(z(i, k)))
    + l2 · ½(‖U‖² + ‖P‖² + ‖b_users‖² + ‖b_items‖²)
    + cf · mean_B ‖P_k − sg(P_r(k))‖₁,     r(k) = k + n_a for control rows; S_t rows contribute 0
```

- **Rows reused:** `run_cause_dev_25k_cause_20261004`, reported as **Native CausE** (prod-C, prod-T, avg). It answers
  how the published method does natively here. It is not the random-vs-propensity test, because it lacks OPC's source
  information.

### 1.2 CausE-warm: native CausE capacity, OPC's source representation

The same parameters, objective, optimizer and search as native CausE-prod. Only the starting point changes: CausE
starts from the source representation OPC starts from.

| question | CausE-warm |
|---|---|
| fixed or trainable | Everything trainable, as in native CausE: U, both item tables, the per-row item biases, the user biases, b, α. Nothing is frozen |
| user representation | U_i ← x_i (the source user vector), shared by the control and treatment tasks (eq. 12/18), updated by both. Users without training rows keep x_i |
| control items | θ^c_j ← a_j (the source item vector), updated on S_c and by the tie |
| treatment (target) items | θ^t_j ← a_j: the treatment representation starts at the source / control representation. This is CausE's residual reading θ^t = θ^c + θ^Δ (eq. 16) with θ^Δ = 0 at the start. It is updated on S_t (and by the tie under the symmetric form) |
| biases, scale | biases 0 and α = 1e-8, exactly the native initialization: at the start the logit is the bias part, and the embeddings enter as α grows. Starting α at a fitted or logger scale was rejected as an extra, non-CausE step |
| discrepancy regularizer | The released L1 tie cf·mean_B ‖θ^c_k − sg(θ^t_k)‖₁ (one-way: it pulls the control rows toward the treatment rows), or the paper's eq. 18 reading without sg (symmetric). The direction is chosen in the tuning stage (§3) as a structural hyperparameter, then fixed |
| L2 | The native l2·½(‖U‖² + ‖P‖² + biases), which shrinks toward 0, not toward the source. Tuning chooses its strength, 0 included. Shrinking toward the source instead would be an invented advantage |
| ρ = 0 | S_t is empty. Under the one-way tie with l2 = 0 the treatment rows stay at a_j. The T prediction is then ⟨U_trained, a_j⟩: the user vectors are adapted on S_c, the items are the source. Under the symmetric tie the treatment rows follow the control rows. The C prediction is a click model fit on all N warm rows, pulled toward the treatment rows |
| evaluation | Two predictions from one trained model, as native prod. **CausE-warm-C** has logits α⟨U_i, θ^c_j⟩ + b^c_j and **CausE-warm-T** has α⟨U_i, θ^t_j⟩ + b^t_j. Greedy policy = argmax_j; stochastic = softmax_j (§2) |
| capacity | Free per-user and per-item vectors plus per-item biases. The class contains the true click model exactly (U = √s·x*, P = √s·a*, b = c), so its structural ceiling is the target-best value and its fraction of the oracle repair equals its fraction of the mismatch repaired (§4) |

CausE-avg is not warm-started. Its treatment task maps every uniform row to one pooled row, so it has no per-item
target representation to start from the source. Native CausE-avg stays among the native rows.

### 1.3 CausE-capacity-matched: OPC's source representation and OPC's correction family

CausE's objective (control task on S_c, treatment task on S_t, L1 discrepancy) is kept. Each free table is replaced by
OPC's correction family on the frozen source vectors: one global linear map per side, y = (I + D)x + b
(`models/models.py · GlobalLinearCorrection`, which OPC's policy uses).

```text
users:            u_i   = (I + D_u) x_i + b_u
control items:    θ^c_j = (I + D_c) a_j + b_c
treatment items:  θ^t_j = (I + D_t) a_j + b_t
logits:           z_c(i, j) = α⟨u_i, θ^c_j⟩ + b          z_t(i, j) = α⟨u_i, θ^t_j⟩ + b
loss on a batch B of S_c (control) and S_t (treatment) rows:
  mean_B CE(y, σ(z))
  + l2 · ½(‖D_u‖²_F + ‖b_u‖² + ‖D_c‖²_F + ‖b_c‖² + ‖D_t‖²_F + ‖b_t‖²)
  + cf · mean_B 1[k control] ‖θ^c_k − sg(θ^t_k)‖₁        (one-way; symmetric without sg)
init: D = 0, b_u = b_c = b_t = 0, so every vector starts at its source vector; global bias 0, α = 1e-8 (native)
```

| question | CausE-capacity-matched |
|---|---|
| fixed or trainable | The source vectors x, a are frozen. D_u, b_u, D_c, b_c, D_t, b_t, b and α are trained (3·(32² + 32) + 2 = 3,170 parameters) |
| user representation | u_i = (I + D_u)x_i + b_u, shared by both tasks: OPC's user-side map |
| target item update | Through the treatment map only: (D_t, b_t) moves every item's target vector at once |
| discrepancy | The L1 tie of CausE in the treatment–control difference, ‖(D_t − D_c)a_k + (b_t − b_c)‖₁: CausE's eq. 16 residual restricted to the global linear family. Direction as in 1.2 |
| per-row biases | None. The per-user bias would not change any ranking. Per-item biases would add item offsets, which OPC's class does not have |
| ρ = 0 | No S_t rows. Under the one-way tie (D_t, b_t) gets no data gradient and stays at 0, so the T ranking uses the source items with the adapted user map. Under the symmetric tie it follows the control map. C is a likelihood fit of the user and control-item maps on the N warm rows: OPC's class, fit to the clicks, without propensities |
| evaluation | **CausE-cap-C** (θ^c) and **CausE-cap-T** (θ^t); greedy argmax_j z; stochastic softmax_j z (§2) |

**Relationship to OPC.**
- OPC's policy is softmax_j(s·⟨(I + D_u)x_i + b_u, (I + D_a)a_j + b_a⟩ / T) with learned s, D and b.
- CausE-cap's T policy is softmax_j(α⟨(I + D_u)x_i + b_u, (I + D_t)a_j + b_t⟩ + b); the global b cancels in the
  softmax.
- The two policy families are therefore identical: the same greedy rankings, and the same softmax policies up to the
  scale α ↔ s/T. A unit test checks that the two models give equal logits for the same maps.
- What differs is only the learning principle:
  - OPC maximizes the DR estimate of the policy value on N warm rows with exact propensities. Its selection is the DR
    lower bound with clip:10.
  - CausE-cap maximizes the likelihood of the observed clicks on (1 − ρ)N warm and ρN uniform rows, with the
    discrepancy tie. Its selection is validation NLL.

**Relationship to CausE.** The control and treatment tasks, the tie and the selection are CausE's. Only the residual
family changes:
- native free per-item residuals θ^Δ_j become one global residual map;
- free user vectors become a map of the source user vectors.

**Its structural oracle is OPC's.** Its greedy family equals OPC's, so the Stage-1 linear-repair oracle
(`run_oracle_repair_20260927`, the learner's class trained on the truth) is its oracle too. Its fraction of the
oracle repair is computed against the same bound.

## 2. Sharpness and policy evaluation

- **Primary: greedy (ranking) value.** argmax_j of each method's scores, valued exactly on the truth over every user.
  Sharpness cannot affect it.
- **Secondary: stochastic value.**
  - OPC learns a logit scale. The tempered logger is the logger's logits × s with s chosen by the DR lower bound on V.
  - A click predictor's softmax has no defined temperature (CausE's paper defines none; M5 used τ = 1).
  - CausE therefore gets the same mechanism the tempered logger uses. Its selected model's logits are multiplied by s
    from OPC's post-tempering grid (0.25–16), chosen by the DR lower bound with clip:10 on V.
  - The q̂ inside that score is fit on CausE's own N rows at that ρ, so the budget stays fair.
  - Both are reported: CausE's raw softmax (τ = 1) and the tempered one.
- **The tempered logger stays as the sharpening-only reference.** No gain that sharpening alone reaches is attributed
  to representation correction.

## 3. Hyperparameter fairness and the tuning protocol

**OPC.** The revalidated configuration (revalidation §2.9): DR, the direct gradient, harmonic:0.1, clip:10 with the
95% lower bound, lr 1e-4–2e-3, 5–30 epochs, the learned logit scale and the paired random sampler. Its rows are the
corrected Stage 2's 25k rows on the same 30 worlds; they are not rerun.

**Raw DR.** A secondary robustness reference: one OPC run with `--train-weights none` at 25k on the 30 worlds. It
does not double the experiment.

**DM-only.** Reported in its own best range, the old space (lr 1e-4–1e-3, 5–25 epochs), which was its best on the
tuning seeds (revalidation 2.3 and R12b). The ml / kuairand rows come from the decomposition runs; anime gets one
small 25k run. The shared-range DM rows are kept beside them.

**CausE.** Every CausE study draws 20 random-search trials per (variant, ρ, world), the same count as OPC's 20 trials
per world. Each is selected by validation NLL. The search space of each new variant is set on **separate tuning
worlds** before the main grid:
- **tuning worlds:** seeds 200/201 (never in the main grid) × ml, kuairand, anime × warp high, vector high, combined
  high; 25k; ρ ∈ {0.05, 0.25} (ρ is not tuned; two values check that the space works across ρ);
- **a wide search:** 40 trials per (variant, ρ, world) over
  - lr log-uniform on [1e-4, 3];
  - epochs {1, 3, 10, 30, 100, 300};
  - l2 {0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
  - cf {0, 0.01, 0.1, 1, 10, 100, 1000};
  - tie {one-way, symmetric};
- **analysis:** where the validation-selected trials and the best-true-value trials fall in each dimension;
  whether any optimum lies on a boundary; and the 20-trial protocol simulated on candidate sub-spaces by resampling
  the logged trials, as for OPC's range (revalidation 2.3);
- **decision rule, fixed now:** the main space for each variant is the candidate with the best mean selected greedy
  value over the tuning worlds. The tie direction is chosen the same way. Each dimension's range is widened if its
  optimum sits on an edge.

ρ is never tuned. Nothing is tuned on the 30 main worlds.

## 4. Capacity diagnostics

| variant | representation family | structural ceiling used |
|---|---|---|
| OPC, DM-only | (I + D)x + b per side on the source | Stage-1 linear-repair oracle (truth-trained) |
| CausE-capacity-matched | the same family (§1.3) | the same Stage-1 oracle, verified by the logit-equivalence test |
| CausE-warm, native CausE | free U, P and per-item biases | the class contains the true click model: ceiling = target-best value (exact, no training needed) |

Every arm reports both the fraction of the representation loss repaired, (V − V_logger) / (V_target_best −
V_logger) greedy, and the fraction of its own family's structural repair.
- **Equal capacity.** A gap between OPC and CausE-cap is about the learning principle and the data, not capacity.
- **Unequal capacity.** A gap between CausE-warm and CausE-cap is about capacity, given CausE's learning principle.

## 5. Measurements

For every arm × world, and for CausE × ρ:
- true greedy and stochastic CTR (CausE's stochastic both raw and tempered);
- gain over the logger;
- the fraction of the representation loss repaired, and the fraction of the arm's structural oracle repair;
- the selected model's validation estimate (OPC: DR lower bound; CausE: validation NLL, plus the DR estimate of its
  greedy and tempered policies);
- selection regret: the best-true-value trial minus the selected one;
- the data composition N_c / N_t and the trial budget.

For CausE also:
- ρ and the number of uniform rows;
- clicks collected (warm and uniform);
- the exploration cost, expected ρN·(V(π0) − V(uniform)) and realized.

## 6. Figures

1. Greedy value against ρ, one panel per bias, with uncertainty. Lines for OPC, DM-only, the logger and the best
   single item.
2. Final value against exploration cost. OPC sits at zero deliberate exploration cost.
3. Native vs warm vs capacity-matched CausE: the contributions of the source representation, the capacity and the
   randomized rows.
4. OPC − CausE against ρ for the two fair variants (paired).

## 7. Not in this stage

- CausE at 100k;
- new ρ values;
- BLOB;
- OPC redesign;
- richer correction architectures.
