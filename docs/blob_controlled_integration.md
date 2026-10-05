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

**Status:** *running (results below when complete).*

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

**Search** (20 random trials per world and family, OPC's count; the space is fixed on tuning seeds 200/201):
- lr log-uniform in [1e-4, 3e-2] (released 1e-3);
- epochs {10, 30, 100, 300} (released 800–1,200 at a different N);
- the prior means μ_wa {−1 (released), 1, 3} and μ_wb {−6 (released), −3, 0}: how strongly the source-aligned term and
  the K × K deviation start;
- σ_κ {0.01 (released), 0.1, 1}: how free the per-item intercepts are.

The rest of the priors stay as released.

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
  - CausE-capacity-matched at ρ = 0: `run_cause_fair_cap_25k`.
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
