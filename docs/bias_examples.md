# Bias Modeling — Real-World Explanations

Map each code knob to a concrete recommender failure mode. The study uses **two independent knobs**: **axis** (where the catalog is wrong) and **component** (what kind of structured corruption).

Starting point is always clean BPR embeddings; noise is mixed in as:

```text
X̃ = (1 − ε1 − ε2 − ε_meta) · X
   + ε1 · N_linear
   + ε2 · N_cluster
   + ε_meta · N_meta
```

---

## Axis = where (which side of the model is wrong)

### Context / user axis (`noise_axis=context`)

Only **user** factors (`our_x`) are corrupted; item catalog stays GT-clean.

**Real world.** The product knows items well (catalog embeddings, content towers) but **users are misspecified**:

- Incomplete profiles (missing age, locale, device, subscription tier).
- Stale user state (moved cities, new taste after a life event; vectors not refreshed).
- Cold-start / sparse history — BPR user vectors pulled toward the mean.
- Privacy aggregation or ID-mapping bugs that blend several people into one “user”.
- Personalization features that systematically mis-read intent (e.g. treating gift-buying as own taste).

**Why it matters for OPC.** Propensity is a function of logged `(user, item)`. If the **learner’s user view** differs from the world that generated logs, importance weights and Q estimates both see the wrong context — correction can help or amplify error depending on how far user SNR falls.

### Action / item axis (`noise_axis=action`)

Only **item** factors (`our_a`) are corrupted; users stay clean.

**Real world.** Users are well tracked, but the **catalog representation is wrong**:

- Popularity / editorial bias baked into item factors (head titles dominate training).
- Catalog churn: new SKUs, remastered IDs, wrong merges of duplicate titles.
- Content embedding drift (thumbnail/model refresh) while interaction history still points at old IDs.
- Supply-side incentives that push certain genres into denser regions of embedding space.
- Cross-domain item towers reused without re-calibration (music embeddings on podcasts).

**Why it matters.** Logging policy often over-exposes popular items; if item factors are also warped, **exposure bias and representation bias stack**. OPC sees bad `p_i` *and* a distorted action space.

### Combined axis (`noise_axis=combined`)

Both catalogs corrupted with the same component mix.

**Real world.** Typical industrial fine-tune: cold-start users **and** messy catalog at once (new market launch, major app redesign, merged product lines).

---

## Component = what (structure of the corruption)

Independent of axis — same generators, different ε column.

### General / linear (`--noise-components linear`, alias `general`)

**Code:** `generate_linear_transform_noise` — shared linear warp `X @ W` plus isotropic Gaussian (`eps1`).

**Real world — global misspecification / feature-space drift:**

- One bad affine transform on the whole embedding space (wrong whitening, bad PCA merge, accidental scaling).
- Sensor / logging pipeline change that applies the **same** distortion to every vector.
- Domain shift that is mostly “everything moved a bit in the same way” (new UI changes click semantics uniformly).
- Not stereotype-local: **everyone / every item** is nudged by a shared matrix, plus unstructured jitter.

**Intuition.** Like calibrating a thermometer with the wrong slope and offset — relative neighborhoods partly survive, absolute coordinates do not.

### Cluster / stereotype (`--noise-components cluster`)

**Code:** `generate_kmeans_cluster_template_noise` — replace (mix) entity vectors toward **cluster templates** (`eps2`).

**Real world — group-level stereotype bias:**

- Collapsing “all teens”, “all romance titles”, or “all short-form creators” toward a prototype.
- Fairness failure: demographic or genre buckets share nearly the same embedding.
- Editorial shelving: storefront taxonomies force similar items into identical slots.
- Filter bubbles encoded as hard clusters — within-group nuance wiped, between-group gaps exaggerated.

**Intuition.** The model stops seeing individuals and sees **types**. Ranking within a cluster becomes nearly random relative to GT taste; between-cluster rankings may still look “structured”.

### Metadata (`--noise-components metadata`)

**Code:** `generate_metadata_projection_noise` — project toward side-info directions (`eps_meta`).

**Real world — side-info dominance / content-tower takeover:**

- Over-trusting genre, price, duration, language tags vs behavioral signal.
- Content embedding that ignores engagement (pretty thumbnails, wrong ASR topics).
- Business rules injected as features (must-promote SKUs pulled toward a promo subspace).
- Sparse CF signal overwritten by rich but **misaligned** metadata.

**Intuition.** The catalog starts to look like a **feature dump**, not a preference geometry. Useful when metadata is truthy; harmful when tags are marketing fiction.

### Combined component (`--noise-components combined`)

All three ε > 0 at that severity level.

**Real world.** Messy production: global drift + stereotype collapse + noisy tags at once.

---

## Axis × component stories (useful captions)

| Axis × component | Short story |
|------------------|-------------|
| context × linear | Every user’s vector got the same bad affine recalibration after a model migration. |
| context × cluster | Personalization collapsed users into coarse personas (age×geo buckets). |
| context × metadata | Profile fields (income proxy, declared interests) dominate over click history. |
| action × linear | Whole item space rotated/scaled after a content-tower refresh. |
| action × cluster | Titles snapped to genre prototypes; long-tail uniqueness gone. |
| action × metadata | Genre/price tags overwrite collaborative item factors. |
| combined × * | Launch / redesign: users **and** catalog wrong in the same way. |

There is **no** separate “personal” noise generator. **Personal / user-specific bias** = put any component on the **context** axis.

---

## Logging / exposure bias (separate knob)

**Code:** `--logging-uniform-mix`, `--policy-temperature` (HurtLog).

**Real world.** Production ranker over-exposes popular short videos; logged clicks over-represent the head. Flatter softmax / uniform mix mimics under-exploration and noisy propensities **without** changing embedding SNR.

Treat as orthogonal to axis×component embedding noise. See [regimes.md](regimes.md).

---

## Temporal distribution shift

**Not a dedicated generator yet.** Approximate with time-split logs (future) or different noise levels train vs eval.

**Real world.** Festival weeks, back-to-school, sudden viral genres — both logging policy and item popularity drift.

---

## Propensity correction vs naive

**OPC** uses logged `p_i`; **NoProp** ignores it. Embedding axis/component interact with whether IPW/CRM **helps** or mostly **adds variance** (see SNR × HurtLog). Protocol: [bias_axis_trial.md](bias_axis_trial.md).
