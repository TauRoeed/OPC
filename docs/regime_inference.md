# Regime Inference from Random Interactions

## Hypothesis (to test)

**H1 — Bad \(\hat q\), propensity correction hurts:**  
Under strong logging bias and large \(n\), if the reward model \(\hat q\) is poorly specified, propensity-aware OPC (esp. DR/SNDR that lean on \(\hat q\)) can lose to a matched **naive / no-propensity** baseline on *true* policy value — not because naive is unbiased, but because correction variance / bad DM term dominates.

Status: **hypothesis**, not a literature theorem. Closest support is the classic DM bias vs IPS variance tradeoff (Dudík et al. 2011; Su et al. CAB 2019), which we map onto OPC vs naive experimentally.

Test sketch: sweep `reward_model ∈ {oracle, regression, logging_score}` × logging mix × train size; compare OPC − naive true reward. H1 predicts: oracle \(\hat q\) → OPC wins more often; bad \(\hat q\) + HurtLog → naive competitive or better.

---

## What random samples identify

Let \(\rho = \mathbb{E}_{x\sim P_x,\, a\sim\mathrm{Unif}(\mathcal{A})}[r(x,a)]\)  
= **rand_CTR** (mean reward under uniform exposure).

From i.i.d. random exposures \((x_i,a_i,r_i)\):

\[
\hat\rho = \frac1n\sum_i r_i
\]

This estimates **reward density**, not the simulator link parameter `ctr`.

---

## How many random samples?

For a Bernoulli mean, Wald / normal approximation at confidence \(1-\alpha\), absolute error \(\varepsilon\):

\[
n \ge \frac{z_{1-\alpha/2}^2 \,\hat\rho(1-\hat\rho)}{\varepsilon^2}
\]

Conservative bound (\(\hat\rho=0.5\), worst-case variance):

\[
n \ge \frac{z_{1-\alpha/2}^2}{4\varepsilon^2}
\]

**Defaults we use:** \(\alpha=0.05\) (\(z\approx1.96\)), \(\varepsilon=0.01\) → **\(n \ge 9604\)** (≈10k).  
Tighter \(\varepsilon=0.005\) → **\(n \ge 38416\)** (≈40k).

### Relation to \(|U|\times|I|\)

Estimating the **global** mean \(\rho\) does **not** require \(\Omega(|U||I|)\) samples.  
\(n\) depends on \(\rho\) and \(\varepsilon\), not matrix size.

Use \(|U||I|\) only for **coverage** goals, e.g.:

| Goal | Rule of thumb |
|------|----------------|
| Estimate \(\rho\) to \(\pm\varepsilon\) | \(n\) from formula above (often \(10^4\)–\(4\cdot10^4\)) |
| Light user coverage | \(n \ge c\cdot|U|\) with \(c\approx 5\)–\(20\) random items/user |
| Dense matrix probe | \(n \ge f\cdot|U||I|\) with small \(f\) (e.g. \(10^{-3}\)); usually unnecessary for \(\rho\) |

**Practical pick for regime tagging:**

\[
n_{\mathrm{rand}} = \max\Bigl(9604,\; 10\cdot|U|\Bigr)
\]

so we hit \(\varepsilon\approx0.01\) *and* ~10 random items per user on average.

Code: `python -m utils.rand_ctr_sample_size --n-users U --n-items I`

---

## From \(\hat\rho\) → assumed regime

| \(\hat\rho\) (rand_CTR) | Assumed density regime | Suggested sim `ctr` *seed* |
|-------------------------|------------------------|----------------------------|
| \(\hat\rho < 0.03\) | SparseReward | start grid near 0.02–0.05 |
| \(0.03 \le \hat\rho < 0.10\) | ModerateReward | ~0.05–0.15 |
| \(\hat\rho \ge 0.10\) | DenseReward | ~0.15–0.30 |

Logging-bias regime (needs both random + production logs):

\[
\mathrm{Lift} = \frac{\mathbb{E}[r\mid \mathrm{logged}]}{\hat\rho}
\]

| Lift | Assumed bias regime |
|------|---------------------|
| ≈1 | WeakSelection |
| 1.5–3 | StrongSelection |
| ≫3 | ExtremeSelection |

---

## Can we estimate “optimal CTR”?

**No — not as a free observable.** Clarify names:

| Quantity | Identifiable from random data alone? |
|----------|--------------------------------------|
| \(\rho =\) rand_CTR = \(\mathbb{E}[r]\) under random | **Yes** |
| Link ceiling `ctr` in \(P=1/(1/\mathrm{ctr}+e^{-s})\) | **Only with a score model** \(s(x,a)\) (MLE / moment match) |
| “Optimal CTR” as best true click rate of some policy | Needs OPE / online eval of that policy — not from random mean alone |

So:

- Random samples → **\(\hat\rho\)** and a **density regime**.
- Given embeddings/scores → you can **calibrate** sim `ctr` so \(\mathbb{E}[q_{\mathrm{ctr}}]\approx\hat\rho\). That is a **fit**, not a discovered optimum.
- Without scores, the ceiling parameter is **not separately identified** (many `(ctr, score-law)` pairs give the same \(\rho\)).

**Bottom line:** we can estimate rand_CTR well with ~10k–40k random draws (or \(\max(9604, 10|U|)\)). We cannot know an “optimal CTR” from random data alone; we can only calibrate a parametric link if we assume scores.
