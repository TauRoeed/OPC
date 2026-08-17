# Bias Modeling — Real-World Examples

Map framework axes to concrete recommender biases.

## Exposure / logging-policy bias

**Code:** `--logging-uniform-mix`, `--policy-temperature` (HurtLog).

**Example:** Production ranker over-exposes popular short videos. Logged clicks over-represent head content. Uniform mix + flatter softmax mimics under-exploration / noisy propensities.

## Context (user) embedding noise

**Code:** `noise_axis=context`, linear + cluster templates on user factors.

**Example — price sensitivity:** Two users with similar taste embeddings; one is highly price-sensitive. Incomplete user features warp the learner’s user vectors relative to GT preference (metadata / context noise).

## Action (item) embedding noise

**Code:** `noise_axis=action`.

**Example — global preference bias / popularity:** Item factors of viral videos are contaminated by engagement templates shared across clusters → learner sees distorted item geometry while GT reward still reflects true quality.

## Metadata projection noise

**Code:** `eps_meta`, `noise_axis=metadata`.

**Example:** Category / demographic side info leaks into embeddings incorrectly (spurious correlation between genre tags and watch time).

## Temporal distribution shift

**Not a dedicated generator yet.** Approximate with:

- train/val split by time on KuaiRand logs (future work), or
- different noise levels train vs eval (proxy).

**Example:** Spring festival content mix differs from ordinary weeks; logging policy and item popularity both drift.

## Combined structured noise

**Code:** `noise_axis=combined` with Mild→BrutalNoise levels.

**Example:** Simultaneous user cold-start misspecification + item catalog churn + weak side info — typical industrial fine-tuning setting.

## Propensity correction vs naive

**OPC** uses logged `p_i`; **NoProp** ignores it. Bias types above interact with whether importance weighting helps or mostly adds variance (see SNR × HurtLog analysis).
