# Capacity × misspec dial — lean trial

**Status.** R*/SoftThresh mlp64+mlp256 lean wave **done** (see `capacity_fair_pi/REPORT.md`). SoftAND* mlp64 gap **queued** (`run_capacity_softand_mlp64.sh`). Claims: `CLAIMS_FAIR_PI.md`.

**Question.** Holding the residual dial fixed, does **increasing matched learner capacity** shrink the hybrid/DR advantage over NLL (because a richer score class can absorb φ), or does the DiD pattern persist?

This is complementary to Phase-1 (linear identifying contrast) and Phase-2 (OPC-mlp64 vs CausE). Here all methods share the same capacity within a cell; we vary capacity across cells.

**Fair reasonable \(\pi\):** within a cell, use matched `linear+mlp` @ **hidden 64** as the primary production-like head. Linear is the reference identifying arm (do not re-run). Width 256 is sensitivity only.

---

## Estimands

Same as the hybrid grid. Exact-greedy CTR %; paired over seeds:

\[
\Delta_{m,m'}(\gamma;c)=V_m(\gamma;c)-V_{m'}(\gamma;c),\qquad
\mathrm{DiD}_c=\Delta_{\mathrm{H},\mathrm{NLL}}(\gamma;c)-\Delta_{\mathrm{H},\mathrm{NLL}}(0;c)
\]

where \(c\) indexes capacity. Primary: \(\mathrm{H}=\texttt{hybrid\_seq\_f0.5}\) vs NLL; also OPC−NLL.

**Hypothesis (directional, not a go/no-go gate).**
- At γ=0, richer \(c\) should not hurt much (bilinear truth ∈ all classes).
- At mid/loud γ, if φ is partially representable by `linear+mlp`, then \(|\Delta_{\mathrm{H},\mathrm{NLL}}|\) and DiD should **shrink** as hidden width grows.
- If φ remains outside the expanded class (esp. SoftThresh after ⊥), DiD should **persist**.

---

## Design (lean wave)

| Axis | Levels |
|------|--------|
| dataset | ml (`BPR/embeddings`) |
| misspec | R* (`s=1,m=4,psi=tanh`); SoftThresh τ=1.5 (`m=4,tau=1.5,s_th=0.25,proj_tag=default,norm=v2`); **SoftAND*** gap (`m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2`) |
| γ | R*/SoftThresh: 0, 0.5, 2 (done). SoftAND*: **0, 0.5, 100** (gap) |
| capacity \(c\) | `linear` (reference = existing `nlres_grid_ml_*` / `nlres_h1_*`); `linear+mlp` hidden **64** (primary); `linear+mlp` hidden **256** (sensitivity; R*/ST only) |
| methods | `opc`, `nll`, `hybrid_seq_f0.5` (matched within cell) |
| seeds | 300–309 (n=10 lean); extend 310–339 if pattern clear |
| HPO | same Optuna ranges as hybrid grid; CRN / `--skip-completed` |

**Tags.**
- `nlres_cap_{rstar|st15}_g{γ}_mlp64` / `_mlp256` — **done**
- `nlres_cap_softand_g{0,0.5,100}_mlp64` — **gap to run**
- Linear: do **not** re-run; analyze against `nlres_grid_ml_{rstar|softthresh|softand}_g{γ}` (or h1 locks).

**Cells (R*/ST):** 2 families × 3 γ × 2 widths = **12 tags** (complete).  
**SoftAND gap:** 3 tags × n=10.

---

## What “capacity” means here

`--policy-transform linear+mlp --mlp-hidden H` uses
\(x \mapsto (I+D)x + b + \mathrm{MLP}_H(x)\) on each side (starts at identity + zero MLP).
This expands the **policy score class**, not the reward-model architecture for q̂ (still regression interaction features unless changed).

So this trial asks about **policy-class capacity vs residual geometry**, not arbitrary deep Q-models.

---

## Analysis

For each (family, γ, c): levels + H−NLL / OPC−NLL with 95% t CIs.  
DiD γ:0→0.5 and 0→2 within each capacity.  
Plot / table: DiD vs hidden width (linear as width 0).

**Go for n=40 extend:** mid-γ SoftThresh or R* shows monotone shrink or clear non-shrink of H−NLL DiD across {lin,64,256} with non-overlapping CIs or consistent seed signs.

---

## Relation to misspecification language

Increasing capacity changes the **hypothesis class** \(\mathcal{F}_c\). The DGP residual φ is fixed. Wins that vanish as \(c\) grows are “misspecification relative to a small class”; wins that remain are closer to **intrinsic** non-representability of φ in the expanded class (or optimization/DR benefits unrelated to class fit). See `MISSPEC_FRAMING.md`.
