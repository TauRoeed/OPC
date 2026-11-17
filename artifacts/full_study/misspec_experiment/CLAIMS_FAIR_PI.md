# Claims lock — fair reasonable π

**Status:** locked for wording / next runs (2026-11-17).  
**Primary report:** [`capacity_fair_pi/REPORT.md`](capacity_fair_pi/REPORT.md).

---

## Locked classes

| Role | `--policy-transform` | Notes |
|------|----------------------|-------|
| **Fair reasonable π (primary)** | `linear+mlp`, `--mlp-hidden 64` | \((I+D)x+b+\mathrm{MLP}(x)\); zero-init last layer; no LN/dropout |
| Identifying small \(\mathcal{F}\) | `linear` | Main hybrid / confirmatory grid; class-misspec vs affine-bilinear |
| Capacity sensitivity | `linear+mlp`, hidden **256** | Not the default; γ=0 hybrid cost worsens with width |

Same transform + width for **NLL, OPC, and `hybrid_seq_f0.5`** inside a cell (matched capacity).  
q̂ stays interaction regression in this wave (isolates **policy-class** capacity).

---

## Banned (unfair or confounding)

1. **Mismatched capacity** — e.g. OPC-mlp vs NLL-linear / CausE-linear.
2. **Old `mlp` transform** — `x + MLP(LN(x))` with dropout and non-identity start.
3. **`blob` as the “MLP residual” answer** — still bilinear \(x^\top Ma+\kappa\); different estimand; do not equate with published BLOB ELBO.
4. **Changing q̂ architecture in the same wave** as promoting π (optional later sensitivity only).
5. **Headline “production π is linear-only”** — linear is identifying, not the fair reasonable head.

---

## How to read results

| Pattern | Interpretation |
|---------|----------------|
| R* DiD stays + under mlp64/256 | Effect persists after residual MLP capacity — not linear-only handicapping |
| SoftThresh mid-γ DiD → 0 under MLP | **Class-relative** absorption by richer NLL; expected |
| SoftThresh loud γ shrinks partly | Partial absorption; not a clean vanish |
| γ=0 H−NLL more negative at width 256 | Capacity/HPO cost — prefer width 64 |
| SoftAND under mlp64 (gap run) | Mid γ = absorption probe; γ=100 = φ-dominated probe |

---

## Wording

**Say:**

> Under matched two-tower residual-MLP capacity (`linear+mlp`, hidden 64), …

> Linear `GlobalLinearCorrection` is the locked identifying score class for bilinear misspec; `linear+mlp@64` is the fair reasonable policy head.

**Do not say:**

> DR/hybrid wins under misspecification (unqualified) — without stating the locked \(\mathcal{F}\).

> SoftAND*/SoftThresh escape all bilinear capacity — they can still be absorbable by MLP/Blob classes.
