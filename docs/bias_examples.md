# Representation Bias — Real-World Explanations

Each bias type maps to a concrete recommender failure mode. Mechanics and calibration:
[representation_bias.md](representation_bias.md).

The simulator starts from clean (centered) BPR vectors, which define the true clicks. The
logger, the reward model and the learned policy see biased copies:

```text
x ← (1 − ε_warp)   · x + ε_warp   · (x_clean @ W)          one map for the whole side
x ← (1 − ε_group)  · x + ε_group  · template[group(x)]     one offset per group
x ← (1 − ε_vector) · x + ε_vector · noise(x)               one offset per user / item
(each step rescaled to the clean RMS)
```

All three types act on **both** users and items. Level `L` of all three together keeps
90 / 75 / 50 % of the signal (low / medium / high).

---

## Global warp (`warp`)

The whole representation space is systematically distorted. The distortion is the same for
everyone, so relative positions partly survive while the directions that matter get mixed.

**Users.** A user model trained for another objective or era is reused. For example,
engagement-trained embeddings are used for purchases, so every user's profile over-weights
binge-watching and under-weights intent. A feature-pipeline change (new normalization,
whitening or PCA merge) hits every user alike.

**Items.** A content or item tower is refreshed, and the new encoder emphasizes different
attributes for the whole catalog: audio embeddings that encode tempo rather than mood, or
thumbnails rather than topics. Cross-domain towers are reused without recalibration (music
embeddings applied to podcasts).

**Why it matters for OPC.** One shared correction applied to every vector (such as a
global linear transform) can in principle undo it. It is the easiest bias to learn away,
given enough logged data.

## Group bias (`group`)

Everyone in a group is shifted the same way. Differences between groups are distorted, and
nuance within a group gets less weight.

**Users.** Stereotyping by segment: users of an age group, country, device type or
subscription tier are pushed toward what "people like them" are assumed to like. Marketing
personas replace individual taste.

**Items.** Category-level priors or mislabels: all titles of a genre, label or seller move
together. New items inherit their category's vector. An editorial rule pushes a whole
genre toward "family friendly".

**Groups.** Default `--bias-groups cluster` uses k-means clusters of the clean vectors (taste
segments). `--bias-groups metadata` clusters the metadata arrays instead (for example ml: user
demographics and item genres). The bias then follows observable attributes, which is
closest to demographic or category stereotyping.

**Why it matters.** Correction needs per-group parameters: a future per-cluster transform,
or a global map that happens to fit.

## Per-vector bias (`vector`)

Each user and item carries its own independent error.

**Users.** Estimation error from sparse or stale histories: new users, users whose taste
changed, shared accounts.

**Items.** Long-tail items with few interactions, noisy content features, wrong metadata on
individual items.

**Why it matters.** Only per-entity parameters can fix it, and they need data about that
entity. A shared transform cannot remove it; the reward model and the propensities absorb
the damage.

---

## Configurations (useful captions)

| `--bias-configs` | Short story |
|------------------|-------------|
| `high/none/none` | A model migration warped every user and item the same way. |
| `none/high/none` | Personalization collapsed users into personas and items into genre prototypes. |
| `none/none/high` | Mostly cold-start: individual vectors are noisy, but no systematic error. |
| `medium` | Messy production: all three at once (the reference world for CTR calibration). |
| `low/high/low` | A segment-level bias dominates; mild global and individual errors. |

---

## Logging / exposure bias (separate knobs)

**Code:** `--logging-spread` (how peaked the logger is; the clean logger's effective items as
a share of the catalog) and `--logging-uniform-mix` (HurtLog).

**Real world.** A production ranker over-exposes what it already believes in. Logged
clicks over-represent its favorites. A sharper logger gives less exploration and larger
importance weights; a uniform mix mimics exploration traffic or noisy propensities. Neither
changes the representation bias. See [regimes.md](regimes.md).

---

## Temporal distribution shift

**Not a dedicated generator yet.** It could be approximated with time-split logs (future)
or with different bias levels for training and evaluation.

**Real world.** Festival weeks, back-to-school, sudden viral genres: both the logging
policy and item popularity drift.

---

## Propensity correction vs naive

**OPC** uses the logged `p_i`; **NoProp** ignores them. The bias type decides what a
correction can learn: a global transform can undo warp, but not per-vector error. It also
decides whether IPW/CRM helps or mostly adds variance. Protocol: [bias_axis_trial.md](bias_axis_trial.md).
