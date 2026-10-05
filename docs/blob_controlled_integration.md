# BLOB in the controlled representation-mismatch environment (design, reproduction, results)

*Development stage. Branch `blob-controlled-integration`, from `representation-mismatch-next` (364c07b). §1–§3 were
written before any comparison ran. Results follow in later sections. Nothing here is confirmatory.*

**The question.** Given the same useful source representation and the same N logged target interactions, how does
BLOB (Bayesian Latent Organic Bandit; Sakhi, Bonner, Rohde & Vasile, KDD 2020) compare with:
- the plain likelihood learner (CausE-capacity-matched at ρ = 0);
- the Direct Method (DM-only);
- the revalidated OPC?

This stage is diagnostic: BLOB is a model-based source-to-target adaptation baseline in the environment we already
understand. No new representation corruption is introduced.

**Sources.**
- Paper: arXiv 2008.12504, KDD 2020, DOI 10.1145/3394486.3403121.
- Code: `criteo-research/blob` at e15cb38, cloned to `~/code/BLOB` and left unmodified. The bandit layer is
  `models/models_organic_bandit.py` (sha256 d2110c7f…).
- The audit environment is outside OPC: `~/code/BLOB_audit/envs/blob-py36`, with Python 3.6, TensorFlow 1.15.2,
  tensorflow-probability 0.7.0, torch 1.10.2 (CPU), pandas 0.25.3, recogym 0.1.3.0 and gym 0.17.3.

## 1. The published model

### 1.1 Data and notation

- **Organic data.** For each user u, an organic session of item views v_{u,1}, …, v_{u,T_u} made without the
  recommender's intervention.
- **Bandit data.** N records (history, action a_n, click c_n) from the recommender's log. In the released dataset
  builder the history of a bandit record is the user's organic views before it.
- **Dimensions.** P items, a K-dimensional latent user state ω_u.

### 1.2 Organic model (BLO)

```text
ω_u ~ N(0_K, I_K)
v_{u,t} | ω_u ~ Categorical(softmax(Ψ ω_u + ρ))            Ψ: P × K organic item embeddings, ρ: P popularity
```

- **Estimation.** Ψ and ρ by maximum likelihood. The posterior over ω_u is approximated by q(ω_u) = N(μ_q, diag
  Σ_q), amortized by a linear encoder: μ_q = Σ_{v ∈ history} W_v + b. Training uses the reparameterization trick.
  The Table 3 and 4 runs use RMSprop, lr 1e-4, 1,000 epochs, K = 20.
- **Recommendation input.** The user embedding is the posterior mean ω̂_u = μ_q (point estimate). The paper's
  "pragmatic compromise" then treats Ψ and ω̂ as observed in the bandit model.

### 1.3 Bandit model (the "B")

```text
c_n | a_n, ω̂_n ~ Bernoulli(σ(β_{a_n} ω̂_n + κ_{a_n}))
β | Ψ, w_a, w_b ~ MN(s+(w_a) Ψ, s+(w_b) Ψ Ψᵀ, s+(w_b) Ψᵀ Ψ / P)        s+(w) = log(1 + e^w)
            implemented as  β = s+(w_a) Ψ + s+(w_b) Ψ ζ Lᵀ,   ζ ~ MN(0_{K×K}, I_K, I_K),   L Lᵀ = Ψᵀ Ψ / P
κ = κ′ + w_c,  κ′ ~ N(0_P, σ_κ² I_P)
w_a ~ N(μ_wa, σ_wa²),  w_b ~ N(μ_wb, σ_wb²),  w_c ~ N(μ_wc, σ_wc²)
```

- **The three "distances".**
  - The prior mean s+(w_a)Ψ: action–history similarity, the organic auto-completion prior.
  - The row covariance ΨΨᵀ: action–action similarity.
  - The column covariance ΨᵀΨ: history–history similarity.
- **Inference: mean-field variational Bayes** over ζ, w_a, w_b, w_c and κ′ (Ψ and ω̂ fixed).
  - **NQ:** one Gaussian per element of ζ (K² variances).
  - **MNQ:** a matrix-normal posterior with diagonal factors (K + K variances).
  - Both use the local reparameterization trick. The loss is the negative ELBO: the mean minibatch Bernoulli NLL
    plus KL(Q | P) / N.
- **Published configuration** (Tables 3 and 4):
  - priors w_a ~ N(−1, 1), w_b ~ N(−6, 1), w_c ~ N(−4.5, 10²), σ_κ = 0.01;
  - lr 1e-3, batch 1,024, 800 epochs (P = 100) or 1,200 (P = 1,000).
- **Point estimate for recommendation:** β̂ = s+(μ_wa)Ψ + s+(μ_wb)Ψ μ_ζ Lᵀ and κ̂ = μ_κ′. The agent recommends
  argmax_a ω̂ β̂_a + κ̂_a, deterministic, with no exploration and no posterior sampling.

### 1.4 The checklist

| question | answer |
|---|---|
| organic / source information | Ψ (how items co-occur in organic sessions) and ω̂_u (the user's organic posterior mean) |
| bandit / target information | the (history → ω̂, action, click) records |
| learned from each | organic: Ψ, ρ and the encoder, by maximum likelihood with a VAE. Bandit: the posterior over ζ, w_a, w_b, w_c and κ′, with Ψ and ω̂ frozen |
| how the source enters the target click model | three ways: ω̂ is the user feature; the prior mean of β is a scaled Ψ; and β is confined to Ψ's column space (β = ΨM with one K × K matrix M = s+(w_a) I + s+(w_b) ζ Lᵀ) |
| Bayesian or point estimate | variational Bayes on the bandit parameters; point estimates for Ψ and ω̂ and for the final recommendation (posterior means) |
| prior tying target to source | the matrix-normal prior centered at s+(w_a)Ψ, its ΨΨᵀ / ΨᵀΨ covariance, and the hyperpriors. The per-item intercepts' tight prior (σ_κ = 0.01) holds them at their common value w_c |
| propensities | **never used.** The released bandit dataset keeps only the history, action and click; the paper argues from the likelihood principle against IPS |
| recommendation at test time | argmax_a of the point-estimate logit ω̂ β̂_a + κ̂_a |

### 1.5 Where the released code differs from the paper's text

All of these are reproduced in the port (`models/blob.py`). The ones that touch the bandit model are checked against
the unmodified graph:

1. **Ψ normalization aliases the prior mean.** With `norm=True` (the released setting) the graph runs
   `Psi_cov = Psi_loc` and then `Psi_cov /= norm(columns)` in place. Both the prior mean and the covariance factor
   therefore use the column-normalized Ψ. The paper's prior mean uses Ψ itself. `alias_loc=False` gives the paper's
   version.
2. **The MNQ noise term uses the prior stds.** It uses `tf.exp(2*zeta_0_std1)`, not the posterior's. ζ's posterior
   stds then enter only the KL. NQ uses its posterior stds correctly.
3. w_b's prior std is `wa_s` in the graph (`wb_0_std = [wa_s]`). The two are equal in the released configuration.
4. The bandit layer uses TensorFlow 1's Adam (lr 1e-3), not RMSprop as the paper states.
5. The point prediction omits w_c, a constant, so rankings are unchanged.
6. The organic encoder's variance uses the mean weights (`CW_inv_Sigmaq_diag` sums `WW_muq`).
7. `results/` in the repository holds a small P = 20 run, not the paper's tables.

### 1.6 The bandit layer's capacity

With Ψ̃ the (normalized) Ψ actually used, the click logit is

```text
logit(u, a) = ω̂_uᵀ W Ψ̃_a + κ_a + w_c,     W = s+(w_a) I_K + s+(w_b) L ζᵀ
```

- L is invertible and ζ free, so **W can be any K × K matrix**: the ranking family is every bilinear form in (ω̂, Ψ),
  plus free per-item intercepts κ_a.
- Ψ̃ is Ψ with rescaled columns, so this is the family x_uᵀ M a_a + κ_a over the raw vectors.
- **Per-user vectors and per-item vectors are not free parameters.** BLOB adapts users and items only through the
  one global K × K map, plus the intercepts. §3 compares this with OPC.

## 2. Reproduction of the published experiment

**Setup.** `~/code/BLOB_audit/run_repro/repro_table3.py` is a copy of the authors' `simulate_abtest_with_bandit.py`.
- It is restricted to one repetition and to five agents: BLO, BLOB-MNQ, BLOB-NQ, logistic regression and random.
- The agents' classes, arguments and evaluation calls are copied unchanged, with the README's Table 3 arguments
  (P = 100, K = 20, 1,000 bandit and 20,000 organic sessions, 1,000 organic and 800 bandit epochs, 4,000 scored
  users, flips 0 and 50).
- **Environment substitution.** recogym 0.1.3.0 is the release whose harness signature (`with_cache`, `reverse_pop`)
  the code calls. Its older 0.1.2.3 pins unavailable `intel-*` packages. Standard numpy, scipy and scikit-learn are
  used instead.
- RecoGym serves here only as BLOB's own reproduction harness. It is not used as a research environment in this
  stage.

**PyTorch port.** `tests/test_blob_tf_reference.py` replays the authors' graph step by step in five cases:
- MNQ and NQ as released;
- MNQ without normalization;
- NQ and MNQ with wider priors.

The fixture comes from `scripts/blob_reference/make_tf_fixture.py`, which executes the released graph-building
block verbatim and fetches each step's noise. It matches the per-step losses to 2e-5 (relative), the final variables
and the point estimate β̂, κ̂ to 1e-4, and the released initialization exactly.

**Result** (2026-10-05 12:19–14:23, one repetition, CPU TensorFlow 1.15; `artifacts/blob_reference/` holds the CSV and
the log). CTR in %; in brackets the evaluation's 2.5–97.5% quantiles, which cover the A/B test's evaluation noise
only, not the variation between training repetitions.

| agent | flips 0: reproduced | paper | flips 50: reproduced | paper |
|---|---|---|---|---|
| BLO (organic) | 2.42 [2.37, 2.48] | 2.42 | 0.76 [0.73, 0.79] | 0.76 |
| BLOB-NQ | 2.42 [2.37, 2.48] | 2.42 | 1.57 [1.53, 1.62] | 1.57 |
| BLOB-MNQ | 2.38 [2.33, 2.44] | 2.40 | 1.57 [1.53, 1.61] | 1.56 |
| logistic regression (bandit) | 1.38 [1.34, 1.42] | 1.37 | 1.38 [1.34, 1.42] | 1.21 |
| random | 1.09 [1.05, 1.13] | 1.09 | 1.11 [1.07, 1.15] | 1.11 |

- **BLOB and BLO reproduce Table 3.** Every BLO and BLOB entry is within 0.02 points of the paper, in both scenarios.
  The paper's central result reproduces as well: with the organic signal intact (flips 0), BLOB matches BLO; when 50
  of the 100 products' organic behaviour is permuted (flips 50), BLO falls below random and BLOB keeps about 2× the
  bandit-only baseline.
- **One discrepancy, in a baseline.** Logistic regression at flips 50 gives 1.38, against the paper's 1.21. It does
  not depend on the organic data, so the same value at flips 0 and 50 is what its definition implies. The paper's
  drop to 1.21 may come from repetition-to-repetition variation or from a configuration the release does not record.
  It does not involve BLOB.
- The faithful implementation used for our comparison is the PyTorch port, not this TensorFlow run. The port is
  verified against the released graph step by step (above); this run verifies that the released configuration
  produces the published numbers.

## 3. BLOB-supplied-source: the controlled variant

**What OPC and CausE-cap receive.** The logger's biased vectors: x_u (users), a_a (items) and the logger π0(a|u) =
softmax(⟨x_u, a_a⟩ / T). Nothing else.

**BLOB-supplied-source.**
- **Source.** BLOB's organic model is replaced by the logger itself. The organic softmax(Ψω + ρ) becomes π0, with
  Ψ = a (the item vectors), ρ = 0 and ω̂_u = x_u / RMS(x). The division puts ω̂ on the scale of BLOB's N(0, I)
  organic prior. BLOB's normalization of Ψ makes the scale of a irrelevant. No organic VAE is trained, so BLOB gets
  no source information beyond what OPC gets.
- **Bandit layer.** Unchanged: the released graph as ported, both families, the released prior structure,
  normalization and optimizer.
- **Target data.** The same N = 25,000 warm logger rows OPC trains on, without their propensities. Selection uses
  the same 20,000 warm validation rows. No randomized rows and no extra interactions.
- **How it differs from native BLOB.**
  1. Ψ and ω̂ come from the supplied source rather than from organic sessions.
  2. ω̂ is rescaled to unit RMS.
  3. The search below tunes the optimizer and three prior hyperparameters that the paper fixed.

**Selection and evaluation.**
- **Selection: validation NLL of the posterior-mean click model**, σ(ω̂ β̂_a + κ̂_a + μ_wc), as for CausE. The
  paper fixed its hyperparameters, so selection is our protocol.
- **Primary metric:** the true greedy value of argmax_a ω̂ β̂_a + κ̂_a, the released recommendation.
- **Secondary metric:** its softmax, raw (τ = 1) and with the fair tempering of the CausE comparison (logits × s,
  with s chosen by the DR lower bound on validation, using q̂ fit on the same N rows).

**Search.** The main grid draws 20 random trials per world and family, OPC's count. They come from a space chosen
on separate tuning worlds by the rule below. The wide tuning space:
- lr log-uniform in [1e-4, 3e-2] (released 1e-3);
- epochs {10, 30, 100, 300} (released 800–1,200 at a different N);
- the prior means μ_wa {−1 (released), 1, 3} and μ_wb {−6 (released), −3, 0}: how strongly the source-aligned term and
  the K × K deviation start;
- σ_κ {0.01 (released), 0.1, 1}: how free the per-item intercepts are.

The rest of the priors stay as released.

**Tuning protocol.** This is the CausE comparison's protocol (§3 of `docs/cause_fair_comparison_25k.md`), adapted to
BLOB's dimensions. It was written at 12:39 on 2026-10-05 (commit 1d3bdbc), while the tuning ran. At that point only the log lines of
one finished cell (its timing and best validation NLL) had been seen, and no value.
- **Tuning worlds:**
  - seeds 200/201 (never in the main grid) × ml, kuairand, anime × warp high, vector high, combined high; N = 25,000;
  - 40 trials per world and family over the wide space (`run_blob_tune_s200`, code e21e9cf);
  - the trial configurations depend on the seed and the family, not on the world.
- **A candidate sub-space's score:**
  - a 10-trial study is simulated inside the sub-space, by 300 resamples of the logged trials;
  - it selects the finite trial with the lowest validation NLL;
  - the score is that trial's true greedy gain over the logger's greedy value, averaged over the 18 worlds.
  - k is 10 rather than 20 because each candidate keeps only part of the 40 trials.
- **Candidates,** per family:
  1. The prior structure, one dimension at a time with the others searched: σ_κ fixed at 0.01, 0.1 or 1; μ_wa fixed
     at −1, 1 or 3; μ_wb fixed at −6, −3 or 0. The wide space is also a candidate.
  2. On the best of these:
     - the 1.5-decade lr windows [1e-4, 3e-3], [3e-4, 1e-2] and [1e-3, 3e-2];
     - the epoch windows {10, 30, 100}, {30, 100, 300} and {100, 300};
     - the best lr window combined with the best epoch window.
  - A candidate needs at least 5 trials per cell on average.
- **Decision rule:** the eligible candidate with the best score is the family's main space.
- **Edges:**
  - In the chosen space, for each searched dimension and value, compute the mean gap between a usable trial's true
    greedy gain and its cell's best trial (lr per half-decade).
  - Suppose the best value lies at an edge of the wide range and beats the adjacent value by more than 0.25 points.
    Then the range is extended one step past that edge in a supplementary 40-trial tuning run on the same worlds,
    and the rule is applied again.
  - One step is a half-decade for lr, 300 → 1,000 for epochs, a decade for σ_κ, and +2 / +3 for μ_wa / μ_wb.
- **Families:** both released families run in the main grid, each with its own chosen space and 20 trials. The
  primary BLOB row is the family with the higher score in its chosen space. The other is reported beside it.
- Nothing is tuned on the 30 main worlds.

### 3.1 Tuning, first round (`run_blob_tune_s200`)

**Run.** Code e21e9cf on a pinned worktree, 2026-10-05 12:31–14:06, 2 workers.
- 18 tuning worlds × 2 families × 40 trials (1,440 trials); no trial diverged.
- Tables: `artifacts/full_study/blob_controlled_25k/tuning/` (`tuning_decision.csv`, `tuning_edges.csv`,
  `tuning_marginals_*.csv`, `tuning_selected.csv`).
- From 13:35 the class-oracle jobs for kuairand and anime were paused (SIGSTOP), because they starved the BLOB
  steps of GPU time slices. A step took 14–20 ms with them paused and 70–85 ms with them running. The computation
  is deterministic, so the pause changes timing only.

**The rule's result** (selected greedy gain over the logger's greedy value, CTR points; 10-trial studies):

| family | wide space | chosen candidate | its gain | minus wide [95% CI] | trials per cell |
|---|---|---|---|---|---|
| MNQ | 0.41 | σ_κ = 0.1; lr 3e-4–1e-2; epochs {10, 30, 100} | 1.73 | +1.32 [+0.81, +1.84] | 7.0 |
| NQ | 1.03 | σ_κ = 0.1; lr 1e-3–3e-2 | 1.39 | +0.36 [−0.46, +1.17] | 5.0 |

- **σ_κ = 0.1 is the best structure for both families.**
  - Free intercepts (σ_κ = 1) are the worst by about 2 points: the selection regret grows to about 3 points.
  - The released σ_κ = 0.01 sits between them.
- **NLL selection is costly for BLOB.** Over all 40 trials, the selected trial is 1.05 (MNQ) and 0.86 (NQ) points
  below the best trial of its world. NLL selects 300-epoch trials in about 80% of the worlds, while the best trials
  are spread over the epoch values.

**The edge rule fires**, so a supplementary round is required before the main grid. In each family's chosen space,
these values sit at an edge of the wide range and beat their neighbour by more than 0.25 points:

| family | dimension | best value (edge) | margin over the neighbour (pts) | extension (one step) |
|---|---|---|---|---|
| NQ | lr | 1e-2–3e-2 (top) | 2.25 | lr up to 1e-1 |
| NQ | epochs | 300 (top) | 2.09 | + 1,000 |
| MNQ | epochs | 10 (bottom) | 0.49 | + 3 |
| MNQ | μ_wa | 3 (top) | 0.50 | + 5 |
| MNQ | μ_wb | −6 (bottom) | 0.65 | + −9 |

- NQ's two flags mean more training: a larger total step lr × steps. MNQ's mean less training and a start closer to
  the source, with a larger s+(w_a) and a smaller K × K deviation.
- The MNQ margins rest on about 2–3 trials per value per cell; the rule applies regardless.
- One step below 10 epochs is 3, on the grid's factor of about 3. "+2 / +3" for the μ's applies at either edge, so
  μ_wb goes to −9.

**Supplementary round (fixed before it runs).**
- **Runs.** `run_blob_tune_s200_supp_nq` and `run_blob_tune_s200_supp_mnq`: the same 18 worlds, 40 new trials per
  world and family.
  - The trials are drawn from the family's extended space, with the other dimensions as in the wide space.
  - They are independent of the first round: `--blob-seed-tag supplement` gives new configurations, batch orders
    and noise.
- **Re-applying the rule.** The rule runs on the first and the supplementary trials pooled, 80 per cell.
  - The candidates are rebuilt over the extended space: each prior value fixed; the 1.5-decade lr windows on the
    half-decade grid, which adds 3e-3–1e-1 for NQ; three consecutive epoch values, and the top two.
  - The 10-trial score, eligibility (at least 5 trials per cell) and the choice are unchanged
    (`training/analyze_blob.py tune --spaces supplement`).
- **No second extension.** If the edge rule fires again, the range is not extended further, and the boundary is
  reported as a limitation.
- **Check.** The supplementary trials alone, analyzed the same way.

### 3.2 Tuning, supplementary round, and the main-grid spaces

**Runs.** `run_blob_tune_s200_supp_nq` (3 workers) and `run_blob_tune_s200_supp_mnq` (1 worker): code 152502b, pinned
worktree, 14:11–15:04. 18 worlds × 40 trials each; no trial diverged. Tables: `tuning_round2/` (pooled, the decision)
and `tuning_round2_supplement_only/` (the check).

**The rule on the pooled trials** (80 per cell; selected greedy gain, CTR points; 10-trial studies):

| family | wide | best structure | chosen candidate | its gain | minus wide [95% CI] | trials per cell |
|---|---|---|---|---|---|---|
| NQ | 1.41 | σ_κ = 0.1 (1.78) | σ_κ = 0.1; lr 3e-3–1e-1 | 1.90 | +0.49 [+0.07, +0.90] | 6.5 |
| MNQ | 0.57 | μ_wa = 5 (1.24) | μ_wa = 5; lr 1e-3–3e-2 | 1.29 | +0.71 [+0.37, +1.06] | 8.0 |

- **NQ** keeps σ_κ = 0.1, as in the first round. Its best lr window moves up, with the extended range. More total
  training is better: the windows with 300–1,000 epochs or lr above 3e-3 lead.
- **MNQ's structure is a near-tie.** μ_wa = 5 (only supplementary trials, 11 per cell) leads σ_κ = 0.1 (1.17) by
  0.07. The supplement-only check picks μ_wa = 1 (1.27), with μ_wa = 5 at 1.24 and σ_κ = 0.1 at 1.13.
  - The rule fixes one structural dimension, so σ_κ stays searched over {0.01, 0.1, 1} in MNQ's space. One third of
    its trials therefore have free intercepts (σ_κ = 1), the worst value in every analysis.
  - MNQ's main-grid result is therefore a lower bound on what a two-dimension structure choice would give. NQ's
    space fixes σ_κ = 0.1.
- **The edge rule fires again, on lr in both families, at the top.** In the chosen spaces the top half-decade leads
  its neighbour by 0.63 (NQ, 3e-2–1e-1) and 0.37 (MNQ, 1e-2–3e-2). As fixed before the round, there is no second
  extension.
  - Limitation: both families might gain from a still larger total step. NQ already searches up to lr 0.1 with
    1,000 epochs.
- **The primary BLOB row is BLOB-NQ**, the family with the higher score in its chosen space (1.90 against 1.29).

**Main-grid spaces** (20 trials per world and family; the 30 main worlds; `--blob-pick-diagnostics --save-policies`):

| family | lr (log-uniform) | epochs | μ_wa | μ_wb | σ_κ |
|---|---|---|---|---|---|
| NQ (primary) | 3e-3–1e-1 | 10, 30, 100, 300, 1,000 | −1, 1, 3 | −6, −3, 0 | 0.1 |
| MNQ | 1e-3–3e-2 | 3, 10, 30, 100, 300 | 5 | −9, −6, −3, 0 | 0.01, 0.1, 1 |

**Tuning budget.** 80 trials per tuning world and family, 40 in each round (2,880 trials). OPC's range came from
the revalidation's range study, and CausE-cap's from 80 trials per tuning world (40 at each of two ρ).

**Capacity, against the other arms** (§1.6; as score functions for the greedy ranking):

| arm | ranking family over the logger's vectors | free parameters (K = 32) |
|---|---|---|
| OPC (linear repair) | ⟨(I + D_u)x + b_u, (I + D_a)a + b_a⟩ = xᵀMa + wᵀa + (user terms) | 2(K² + K) + scale = 2,113 |
| CausE-capacity-matched | the same family (the item side tied) | 3(K² + K) + 2 = 3,170 |
| BLOB-supplied-source | xᵀMa + κ_a (+ w_c) | K² + P + 3 (plus as many variances) |

- **Per-item intercepts make BLOB's class a superset of OPC's.** A linear item term wᵀa is one particular κ.
  Without them, BLOB's class is a subset of OPC's: it has no item-linear term.
- **The released prior σ_κ = 0.01 effectively removes them.** An intercept moves only when an item has tens of
  thousands of rows. So the effective capacity is set by the prior, and the search's σ_κ dimension is the capacity
  dial. Every trial reports how far κ moved.

**Truth-trained oracles** (`training/class_oracles.py`; exact, no logged data; they separate structural capacity
from statistical learnability). For each class:
- **Value oracle:** the class's policy trained on the exact true value (the Stage 1 recipe), for xᵀMa + κ_a.
  OPC's class has its Stage 1 oracle.
- **Likelihood oracle:** the class fit by the infinite-data likelihood under the logging distribution, minimizing
  E_{u ~ prior, a ~ π0(·|u)} CE(q(u, a), σ(f(u, a))) exactly, then graded by its greedy value. This is where the
  plain likelihood learner and BLOB converge with unlimited logs from π0.
  - Fit for: xᵀMa + wᵀa + vᵀx + c (CausE-cap's family), xᵀMa + κ_a + c (BLOB), and xᵀMa + c (BLOB with pinned κ).
- **How to read them.**
  - Value oracle minus likelihood oracle: what the likelihood objective itself gives up at infinite data
    (misspecification under π0's sampling).
  - Likelihood oracle minus the 25k learner: finite-sample learnability.

## 4. The bounded experiment

- **Worlds.** The 30 development worlds of the fair CausE study: ml, kuairand and anime × no bias, warp, group, vector
  and combined high × seeds 100/101. N = 25,000.
- **New runs.** Only BLOB-supplied-source (both families).
- **Reused rows** (identical worlds, logs and splits):
  - tempered logger, OPC (harmonic:0.1) and DM-only in OPC's range: the corrected Stage 2;
  - DM-only in its own validated range: the revalidation's old-space runs plus `run_cause_fair_dm_oldspace_anime`;
  - CausE-capacity-matched at ρ = 0: `run_cause_fair_cap_25k`. At ρ = 0 it has no randomized rows and no
    propensities: it is the plain click-likelihood fit in OPC's class. CausE-cap-C is the primary likelihood
    reference; CausE-cap-T is reported beside it.
  - CausE-warm-C at ρ = 0 (`run_cause_fair_warm_25k`): a likelihood fit with free per-user and per-item vectors, as
    context for the capacity question.
- **Policy replays (for the pick diagnostics only).**
  - The reused runs did not keep their selected policies, and question 4 below needs them. OPC and CausE-cap at
    ρ = 0 are therefore replayed with `--save-policies`. The replays are 25k only, use the same code path (unchanged
    since those runs) and the same configurations, and run from a pinned worktree.
  - Each replayed condition is checked trial by trial against the original run.
    - OPC: the true value, the true greedy value, the selection estimate and the DR estimate of all 20 trials.
    - CausE-cap: the validation NLL and the C and T values of all 20 trials.
  - The reported numbers stay the original rows. The replays only provide the saved vectors.
- **Metrics.**
  - true greedy CTR (primary) and the gain over the logger;
  - stochastic value (raw and tempered);
  - the fraction of the representation loss and of each class's structural (value-oracle) repair recovered;
  - selection regret;
  - target-model prediction quality: validation NLL and AUC, and the likelihood oracle's NLL as the floor;
  - parameter and correction norms;
  - the search budget and the effective capacity (κ's movement).

**Mechanism questions** (Phase 5):
1. Does BLOB behave like the plain likelihood learner under warp?
2. Does its source prior help under group or vector mismatch?
3. Does its richer target capacity (κ) explain any advantage?
4. Does it extrapolate into poorly logged regions better or worse?
5. Where does OPC differ from it given the same source?
6. Is a difference training, selection or capacity?

**Diagnostics for the mechanism questions** (fixed before the comparison's results).
- **Accounting** (questions 3 and 6). Per world and arm, the greedy gain over the logger is split exactly:
  - gain = ceiling − training gap − selection regret;
  - ceiling: the value oracle of the arm's class (§3);
  - training gap: the ceiling minus the best of the arm's own 20 trials;
  - selection regret: the best trial minus the selected one.
  - A paired difference between two arms splits the same way, into Δceiling, Δtraining and Δselection.
  - The likelihood oracle of the arm's class separates the training gap further. Its distance below the value oracle
    is what the likelihood objective gives up at infinite data. The rest is finite-sample learning.
- **Pick diagnostics** (questions 4 and 5; `training/policy_diagnostics.py`; exact). They are computed for the
  selected policy of each arm, for the class oracles and for the logger.
  - Where the greedy policy recommends: the share of users whose pick is the logger's top item or in its top 10;
    the share whose pick the logger shows less often than uniformly (P π0(a*|u) < 1); and the mean log10 P π0(a*|u).
  - What the picks are worth there: the true click probability at the picks above and below uniform propensity.
  - For the click models (BLOB, CausE-cap, the likelihood oracles):
    - the optimism at the picks, Σ prior (σ(f(u, a*)) − q(u, a*));
    - the prediction error by logging-propensity bin (P π0 below 0.1, 0.1–1, 1–10, above 10);
    - the exact infinite-data NLL under π0, against its class's likelihood oracle.
  - Between two arms (question 5): the users where they pick the same item. On the rest, V_A − V_B is split by which
    of the two picks the logger shows less often.
- "Extrapolates better" (question 4) means the true click probability of its picks below uniform propensity is
  higher, with a smaller optimism there.

**Phase 6 trigger** (fixed before the comparison's results). A targeted 5k / 100k extension runs only if one of the
following holds.
- **(i) Close.** The pooled paired 95% CI of BLOB − OPC or BLOB − CausE-cap-C includes 0, with |mean| < 0.25 points.
- **(ii) Size-dependent.** The 25k ordering of BLOB and OPC is the opposite of the ordering of their infinite-data
  references: BLOB's likelihood oracle against OPC's value oracle, pooled over the biased worlds.
- **Scope if triggered.**
  - Only the bias types that meet the condition, both seeds, the three datasets, at 5k and 100k.
  - BLOB only; OPC already has 5k and 100k rows in the corrected Stage 2.
  - The reason is recorded here before launch.
- Otherwise the class oracles stand in for the large-N limit, and nothing more is run.
