# Bias Modeling — Real-World Examples

Map framework axes to concrete recommender biases.

## Exposure / logging-policy bias

**Code:** `--logging-uniform-mix`, `--policy-temperature` (HurtLog).

**Example:** Production ranker over-exposes popular short videos. Logged clicks over-represent head content. Uniform mix + flatter softmax mimics under-exploration / noisy propensities.

## Context (user) embedding noise

**Code:** `noise_axis=context` (users only). Pair with `--noise-components linear|cluster|metadata`.

**Example — price sensitivity / personal misspecification:** incomplete user features warp the learner’s user vectors relative to GT preference while item catalog stays clean.

## Action (item) embedding noise

**Code:** `noise_axis=action` (items only).

**Example — global preference bias / popularity:** item factors contaminated while user factors stay clean.

## Mixture components (what gets mixed in)

Independent of axis:

| Component | Flag | Code |
|-----------|------|------|
| general / linear | `--noise-components linear` (alias `general`) | shared warp `X@W` + Gaussian (`eps1`) |
| cluster | `--noise-components cluster` | k-means templates (`eps2`) |
| metadata | `--noise-components metadata` | side-info projection (`eps_meta`) |

See [bias_axis_trial.md](bias_axis_trial.md) for the axis × component study protocol.

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
