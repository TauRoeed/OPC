# What are we actually studying? (framing memo)

This memo answers: **is the residual-dial experiment “model misspecification”?**  
It walks through several standard framings so we do not rubber-stamp one label.

---

## 1. Data-generating process (fixed)

For each seed, embeddings \((u_x,v_a)\) and a residual field \(\phi\) (unit SD) define

\[
\ell_\gamma(x,a)=b'(\gamma)+u_x^\top v_a+\gamma\alpha\phi(x,a),\qquad
q_\gamma=\sigma(\ell_\gamma).
\]

A logging policy \(\pi_0\) (temperature frozen at \(\gamma=0\)) draws actions; we observe
\((x,a,r,p)\) with \(r\sim\mathrm{Bern}(q_\gamma)\).  
**Evaluation** uses exact-greedy CTR under \(q_\gamma\) (simulator oracle for \(V(\pi)\)).

Families (\(R^\star\), SoftAND\(^\star\), SoftThresh) only change the **geometry of \(\phi\)**, not the learner code path.

---

## 2. Learner side (what is “the model”)

Phase-1 / hybrid-grid learners optimize policies in a **score class** \(\mathcal{F}\) built from the same embeddings (linear correction, or later `linear+mlp` / blob).  
Separately, DR uses a **reward model** \(\hat q\) (regression on interaction features) and IPS terms with known \(\pi_0\).

So there are at least three “models”:

| Object | Role |
|--------|------|
| True click logit \(\ell_\gamma\) | DGP |
| Policy score class \(\mathcal{F}\) | What \(\pi_\theta\) can represent |
| Reward model class for \(\hat q\) | Nuisance in DR |

Calling the whole study “model misspecification” without saying *which* model is ambiguous.

---

## 3. Candidate framings

### A. Class misspecification of the policy / click score (most accurate for Phase-1)

**Claim:** the true ranking / logit is not in the learner’s bilinear (or affine-bilinear) class for \(\gamma>0\).

\[
\exists\,\gamma>0:\quad
\ell_\gamma \notin \mathcal{F}_{\mathrm{linear}}.
\]

NLL maximizes a likelihood that is well-specified only if \(\ell_\gamma\in\mathcal{F}\) (up to link).  
DR targets \(V(\pi)\) more directly and can win when likelihood projection ≠ value projection.

**This is real misspecification** in the classical M-estimation sense *relative to \(\mathcal{F}_{\mathrm{linear}}\)*.  
It is **not** misspecification of a generic “neural net”; it is misspecification **of a locked small class**.

### B. Omitted structure / incomplete covariates

\(\phi\) is a function of embeddings (projections of \(u,v\)), not an external confounder.  
π₀ is simulator-true → this is **not** classical unobserved confounding.  
Closer to: **omitted nonlinear structure in the score**, with features still “observed” but not used with the right functional form.

### C. Environment perturbation / sensitivity analysis

\(\gamma\) is a continuous **stress dial** on the environment. DiD asks: *relative to γ=0, how does the method gap move?*  
That is clean **sensitivity / stress-test** language even if one dislikes “misspecification.”

### D. Distribution shift

Logging distribution \(P_{\pi_0,q_\gamma}\) changes with \(\gamma\), so train and eval environments co-move.  
There is shift, but the **designed** object is not covariate shift with a fixed \(q\); both reward and optimal policy change.  
“Shift” alone under-describes the construction.

### E. OPE identification misspec (reward model / propensity)

Propensity is known (simulator). Reward-model misspec can still bias the DM part of DR; IPS remains unbiased for \(V\) under standard conditions.  
Our primary story is **not** “DR fixes confounding”; it is **DR vs NLL under class stress** with known π₀.

---

## 4. When the word “misspecification” is loose

| Loose usage | Why it’s loose |
|-------------|----------------|
| “The world is misspecified” | Worlds aren’t misspecified; **models** are |
| SoftAND*/SoftThresh “escape all bilinear capacity” | After ⊥ they can still be low-rank / absorbable by richer classes (Blob, MLP) |
| Equating oracle Δ_proj gaps with learner DiD | Projection-gap oracles ≠ transferable value gaps (SoftAND mid-γ) |
| Calling γ-dial “confounding” | π₀ known; no hidden confounder |

**Tight usage:** *controlled class misspecification of the click/policy score relative to a locked learner class \(\mathcal{F}\), studied via a residual dial and DiD.*

---

## 5. Classical MLE misspec vs this simulator

Classical: observations from \(P_{\theta^\star}\), estimate in wrong parametric family \(\{P_\theta:\theta\in\Theta\}\); MLE → pseudo-true parameter; risk is KL projection.

Here: we **construct** \(P_\gamma\) so that the pseudo-true NLL policy in \(\mathcal{F}\) is deliberately poor for value when \(\gamma\) grows, then compare to a value-oriented objective (DR) and a hybrid schedule.  
Same spirit (pseudo-true vs target risk), different engineering: structured residual instead of an abstract wrong exponential family.

---

## 6. What DiD identifies (and does not)

\[
\mathrm{DiD}
=
\big(V_{\mathrm{H}}(\gamma)-V_{\mathrm{NLL}}(\gamma)\big)
-
\big(V_{\mathrm{H}}(0)-V_{\mathrm{NLL}}(0)\big).
\]

**Identifies (under CRN / paired seeds):** how the hybrid−NLL gap changes when the residual is turned on, holding HPO/logger protocol fixed.

**Does not identify:** a unique structural “misspecification parameter” in field data; external validity beyond this DGP; that DR is preferable whenever any misspecification exists.

---

## 7. Capacity trial link

If we enlarge \(\mathcal{F}_c\) (`linear+mlp` widths), a γ>0 gap that **shrinks** means the residual was misspecification *relative to a small class*.  
A gap that **persists** means either φ stays outside \(\mathcal{F}_c\), or DR/hybrid wins for reasons other than class fit (optimization geometry, weighting, etc.).  
That is exactly why a capacity sweep is the right next probe: it tests whether our “misspecification” claim is class-relative or more intrinsic.

---

## 8. Recommended wording for the paper

Prefer:

> “We induce **controlled class misspecification** of the true click logit relative to a locked policy score class \(\mathcal{F}\), via a unit-SD residual dial \(\gamma\phi\).”

Avoid as primary:

> “We study model misspecification / confounding in offline RL.”

Secondary allowed:

> “Equivalently, a sensitivity analysis of OPL objectives under structured departures from bilinear truth.”
