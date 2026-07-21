# Introduction and Methods Draft

## Introduction

Modern recommender systems are usually improved using logs of past user interactions. These logs contain rich information about user preferences, but they are also shaped by the recommender system that collected them. A user can only click, rate, install, or consume an item that was shown to them. As a result, the observed data is not a random sample of user-item outcomes. It is the result of a behavior policy that selected which actions were exposed in each context. This makes policy improvement from historical recommendation logs an off-policy learning problem.

The off-policy nature of recommender fine-tuning creates a basic tension. If a new policy is trained directly on observed rewards, it may learn to imitate or amplify the exposure pattern of the logging policy rather than discover actions with higher true reward. Propensity-based correction addresses this by reweighting logged observations according to how likely the learned target policy would have been to choose the observed action relative to the behavior policy. In principle, this correction can reduce bias. In practice, it can introduce high variance when the learned policy assigns large probability to actions that were unlikely under the behavior policy.

This thesis studies that tradeoff in the setting of recommender-policy fine-tuning. The central question is whether using logged behavior propensities during policy optimization improves learned policy value compared with an otherwise identical training procedure that ignores those propensities. I refer to the propensity-aware method as off-policy correction (OPC), and compare it against a no-propensity baseline that optimizes a pure naive reward objective without propensities, importance weights, or doubly robust terms.

The study is built around a controlled semi-synthetic experimental framework. Real recommendation datasets are first transformed into implicit-feedback matrices and used to train Bayesian Personalized Ranking (BPR) matrix-factorization embeddings. These embeddings provide a realistic user-item latent space. The clean embeddings define the ground-truth reward environment, while noisy versions of the embeddings define the information available to the logging and learning procedures. This design preserves recommender-system structure while allowing exact evaluation of learned policies against a known reward model.

Within this environment, a softmax collaborative-filtering policy is initialized from the noisy embeddings and fine-tuned using logged bandit feedback. The code implements several policy-gradient losses (`kl_crm`, `kl`, `ipw`, `sndr`, `crm`, `naive`). The full-study default for OPC is the unified `kl_crm` objective (SNDR surrogate plus KL and CRM variance penalties) with logged propensities. Hyperparameters are selected by Optuna using `--optuna-selection`: `ci_low` (default lower confidence bound), `r_hat`, or `actual_reward` (oracle).

The main contribution of the experimental design is a matched ablation between OPC and no-propensity training. Both methods use the same logged train and validation splits, the same policy model, the same shared reward model, and the same Optuna budget. OPC uses logged propensities in IW/DR-style losses; the no-propensity arm uses pure naive `mean(r * pi)` with `propensity_mode=uniform`. This isolates the empirical effect of propensity-aware off-policy correction.

The rest of the paper is organized as follows. Section 2 reviews the contextual bandit formulation of recommendation and the off-policy estimators used in this work. Section 3 describes the BPR-based semi-synthetic data generation process. Section 4 presents the policy model, reward model, and off-policy fine-tuning objectives. Section 5 describes the experimental protocol and evaluation metrics. Section 6 reports the results of the OPC versus no-propensity comparison, and Section 7 discusses the conditions under which propensity correction helps or hurts.

## Methods

### Problem Formulation

I formulate recommendation as a contextual bandit problem. Each interaction consists of a context `x`, representing a user, an action `a`, representing an item, and a binary reward `r`, representing whether the user interacted positively with the item. Historical data is collected by a behavior policy `pi_b(a | x)`, which chooses the logged action. For each logged interaction, the dataset contains the propensity score:

```text
p_i = pi_b(a_i | x_i)
```

The goal is to learn a target policy `pi_theta(a | x)` that maximizes expected reward:

```text
V(pi_theta) = E_x E_{a ~ pi_theta(. | x)} [r(x, a)].
```

Because rewards are observed only for actions sampled by `pi_b`, directly optimizing the observed rewards can produce a biased estimate of the target policy value. Off-policy correction uses the importance weight:

```text
w_i = pi_theta(a_i | x_i) / pi_b(a_i | x_i)
```

to adjust for the mismatch between the target policy and the behavior policy.

### BPR Embedding Construction

The experimental pipeline begins by training recommender embeddings from real implicit-feedback data. The code supports MovieLens 1M, Myket, Anime, LastFM, and Million Song Dataset style interaction data. Each dataset is loaded into a user-item interaction table and converted into a sparse user-item matrix. A Bayesian Personalized Ranking matrix-factorization model is then trained on this matrix.

The BPR model learns a user factor matrix and an item factor matrix. Training samples triples `(u, i, j)`, where user `u` interacted with positive item `i` and did not interact with negative item `j`. The model updates the factors so that the score of the positive item is larger than the score of the negative item for the same user. The learned user and item factors are saved as NumPy arrays and reused by the simulation pipeline.

In the thesis code, this stage is implemented mainly by `BPR/bpr.py`, `BPR/dataload.py`, and `BPR/generate_artifacts.py`.

### Semi-Synthetic Bandit Environment

The BPR embeddings are used to construct a semi-synthetic contextual bandit environment. Let `e_x` and `e_a` denote the clean BPR user and item embeddings. These clean embeddings define the latent reward model. For a user-action pair, the environment computes a dot-product score and maps it to a reward probability controlled by a click-through-rate parameter:

```text
P(r = 1 | x, a) = 1 / (1 / ctr + exp(-e_x^T e_a)).
```

This construction gives the experiment an exact reward oracle: the true value of any learned policy can be computed by integrating over users and actions under the known reward probabilities.

The learner, however, does not operate directly on the clean embeddings. Instead, noisy embeddings are generated by mixing the clean embeddings with structured noise. The code includes several noise generators:

- linear transform noise,
- random cluster-template noise,
- k-means cluster-template noise,
- optional metadata-projection noise.

Noise can be applied to user embeddings, item embeddings, metadata-derived features, or a combined representation. The resulting noisy embeddings represent imperfect recommender features available to the logging policy and the learned policy. This allows the experiments to vary representation quality while keeping the ground-truth reward model fixed.

This environment is implemented in `utils/simulation_utils.py`.

### Logging Policy and Logged Bandit Data

The logging policy is a softmax dot-product recommender over the noisy user and item embeddings:

```text
pi_b(a | x) = softmax(u_noisy(x)^T v_noisy(a) / tau).
```

Logged bandit data is generated by sampling users from a user-prior distribution and sampling actions from this policy. For each sampled user-action pair, the environment samples a binary reward from the ground-truth reward probability. Each logged tuple therefore contains:

```text
(x_i, a_i, r_i, p_i)
```

where `p_i` is the exact logging propensity assigned to the sampled action. These propensities are used by OPC IW/DR-style losses and ignored by the naive no-propensity baseline.

The scalable policy implementation is in `utils/policies.py`, and logged simulation is handled by `create_simulation_data_from_policy` in `utils/simulation_utils.py`.

### Target Policy Model

The learned target policy is also a softmax collaborative-filtering model. Given trainable or transformed user and item embeddings, the policy assigns probability:

```text
pi_theta(a | x) = softmax(u_theta(x)^T v_theta(a) / tau).
```

The implementation uses a collaborative-filtering model initialized from the noisy BPR embeddings. In the main regression-based trainer, the model applies residual MLP transforms to the initial user and item embeddings. This allows the policy to fine-tune the representation while staying anchored to the recommender structure learned from BPR.

The core policy classes are implemented in `models/models.py`. The main training path uses `CFModel` together with `SingleMLPTransform`.

### Reward Model

Several objectives and validation metrics require an estimate of the conditional mean reward:

```text
q_hat(x, a) ≈ E[r | x, a].
```

The main implementation uses a regression model over context-action features. User context vectors are combined with item/action embeddings, and a classifier predicts the probability of positive reward. The code also includes an MLP reward model and a neighborhood-based reward model, but the main full-study trainer uses a shared regression bundle so that OPC and no-propensity methods are evaluated with the same reward model.

Full-study flag `--reward-model`:

- `regression` (default): fit logistic regression on noisy `our_x` / `our_a`.
- `logging_score`: CTR link `1 / (1/ctr + exp(-(our_x·our_a)/T))` (no fit).
- `oracle`: same CTR link on clean `env.emb_x` / `env.emb_a` (sim diagnosis only).

Details: `docs/training_losses.md` §1.1. Code: `AnalyticRewardModel` /
`fit_shared_regression_bundle` in `training/trainer_trials.py`, plus
`RegressionModel` / `MLPRewardModel` / `NeighborhoodModel` in `models/models.py`.

### Off-Policy Training Objectives

The policy losses are implemented in `models/custom_losses.py`. Full formulas are in
`docs/training_losses.md`. Supported names:

```text
kl_crm (default), kl, ipw, sndr, crm, naive
```

Propensity modes:

- `logged`: use true logged propensities and importance weights (OPC).
- `uniform`: used by the no-propensity arm with the naive loss (no IW/DM/SNDR).

#### Default OPC loss (`kl_crm`)

Unified objective in `KLCRMPolicyLoss`:

```text
Loss = -SNDR surrogate + kl_gamma * KL_MC + crm_lambda * sqrt(Var(u) / n)
```

where the SNDR surrogate is the self-normalized doubly robust policy-gradient
term, `KL_MC` is the batch Monte Carlo KL at logged actions, and
`u_i = -r_i * min(w_i, crm_M)`.

#### Inverse Propensity Weighting (`ipw`)

```text
w_i = pi_theta(a_i | x_i) / p_i
```

Log-trick: minimize `-mean(stopgrad(w) * r * log pi)`. Direct: minimize `-mean(w * r)`.

#### Self-Normalized Doubly Robust (`sndr`)

```text
DM_i = sum_a q_hat(x_i, a) pi_theta(a | x_i)
Correction_i = w_i (r_i - q_hat(x_i, a_i)) / mean(w)
r_hat_i = DM_i + Correction_i
```

Training minimizes the negative surrogate (log-trick or direct). Used in recent
hurt-logging ablations with `--optuna-selection r_hat`.

#### KL-Regularized Objective (`kl`)

```text
Loss = -SNDR surrogate + gamma * mean(log pi_b - log pi_theta)
```

#### Naive Objective (`naive`; no-propensity default)

```text
Loss = -mean(r_i * pi_theta(a_i | x_i))
```

No IW, DM, SNDR, KL, or CRM. Full study fixes `use_log_trick=False` for this arm.

### Log-Trick and Direct-Probability Surrogates

The log-trick version detaches policy-probability coefficients and multiplies by
`log pi_theta`. The direct version keeps probabilities attached. In
`run_full_study.py`, OPC fixes log trick True and no-propensity fixes it False.
`--no-log-trick` disables the searchable log-trick path.

### Hyperparameter Optimization and Policy Selection

Optuna searches over learning rate, epochs, batch size, LR decay, and (when
needed) `kl_gamma`, `crm_M`, `crm_lambda`, and optionally `policy_loss` if
multiple `--policy-losses` are passed. Each trial evaluates a validation vector
and maximizes `--optuna-selection`:

```text
ci_low (default) = R_hat - t_crit * SE
r_hat            = mean(row_value)
actual_reward    = true simulator reward (oracle)
```

For OPC, `row_value` is the DR estimate `DM + w (r - q)`. For no-propensity,
`row_value = r * pi`. After selection, the policy is retrained and scored against
the simulator.

### Main Experimental Comparison

```text
OPC:           propensity_mode=logged,  default loss=kl_crm, log trick fixed True
No propensity: propensity_mode=uniform, loss=naive,         log trick fixed False
```

Both share data splits, policy architecture, reward model, validation criterion,
and Optuna budget. Implemented by `regression_trainer_trial`,
`no_propensity_trainer_trial`, and `_run_condition`.

### Evaluation

The primary evaluation metric is the true policy value under the semi-synthetic environment. Because the simulator has access to the ground-truth reward probabilities, the code can compute the exact expected reward of a learned policy by summing over the action catalog in chunks. This value is logged as `policy_rewards` or `actual_reward`, depending on the stage of the pipeline.

Additional metrics include:

- improvement relative to the initial noisy policy,
- validation-selected policy reward,
- OPC minus no-propensity paired deltas,
- effective sample size of importance weights,
- maximum and minimum importance weights,
- Gini coefficient of importance weights,
- direct-method, IPW, DR, and SNDR validation estimates.

The analysis scripts aggregate these metrics across datasets, noise levels, CTR levels, train sizes, validation sizes, and seeds.

### Experimental Variables

The full-study scripts sweep several dimensions:

- dataset,
- noise mode,
- noise axis,
- noise level (`low` / `medium` / `high` / `extreme` / `brutal`),
- CTR level,
- training set size,
- validation set size,
- random seed,
- policy loss (`--policy-losses`),
- Optuna selection metric (`--optuna-selection`),
- logging–uniform mix (`--logging-uniform-mix α`),
- policy temperature (`--policy-temperature`).

Logging damage: higher noise levels wipe more ground-truth signal from the
embeddings used by the logging policy; `--logging-uniform-mix α` sets
`pi_b = (1-α)·π_softmax + α/|A|`. Recent ablations combine `sndr`/`ipw` with
hurt logging and `r_hat` selection.

The default study uses BPR embeddings, k-means template noise, multiple noise
levels, several training sizes, and repeated seeds. The parallel runner spreads
conditions across workers/GPUs.

## Result Placeholders

The final paper should fill in the following empirical claims after choosing the final artifact run:

- OPC improves true policy reward over no-propensity by `[insert result]`.
- The improvement is strongest under `[insert noise/CTR/train-size regime]`.
- The no-propensity baseline outperforms OPC under `[insert regime, if any]`.
- Effective sample size explains failures or instability when `[insert observation]`.
- KL regularization improves stability by `[insert evidence]`.
- Validation size affects selected policy quality by `[insert result]`.

