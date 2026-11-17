# Fair reasonable π — capacity elevation report

**Claim language:** see [`CLAIMS_FAIR_PI.md`](../CLAIMS_FAIR_PI.md).  
**Frozen numbers:** [`did_summary.json`](did_summary.json).  
**Regenerate from run folders:** `python -m artifacts...` → [`analyze_capacity_fair_pi.py`](../analyze_capacity_fair_pi.py).

---

## Verdict

The fair, reasonable policy head for new \(\pi\) is **matched `linear+mlp` with hidden width 64**:

\[
x \mapsto (I+D)x + b + \mathrm{MLP}_{64}(x)
\]

on each tower (last layer zero-init → starts at the logger). Same class for NLL, OPC, and `hybrid_seq_f0.5`.

- **Linear** (`GlobalLinearCorrection`) stays the locked **identifying** arm for misspec relative to affine-bilinear \(\mathcal{F}\).
- **Width 256** is sensitivity only (γ=0 hybrid cost worsens with width).
- Do **not** use old `--policy-transform mlp` (LN + dropout + non-identity start) or mismatched capacity across methods.

---

## What already ran (do not ignore)

| Tag family | Transform | γ | Seeds |
|---|---|---|---|
| `nlres_cap_{rstar\|st15}_g{0,0.5,2}_mlp64` | `linear+mlp` / 64 | 0, 0.5, 2 | 300–309 |
| `nlres_cap_{rstar\|st15}_g{0,0.5,2}_mlp256` | `linear+mlp` / 256 | 0, 0.5, 2 | 300–309 |
| `nlres_grid_ml_{rstar\|softthresh}_g*` | `linear` (reference) | … | grid / h1 |

Methods matched within cell: `opc`, `nll`, `hybrid_seq_f0.5`. q̂ held fixed (interaction regression).

---

## H−NLL DiD (CTR points, paired seeds)

\(\mathrm{DiD}=(\mathrm{H}-\mathrm{NLL})(\gamma)-(\mathrm{H}-\mathrm{NLL})(0)\), H = `hybrid_seq_f0.5`.

### R*

| Capacity | 0→0.5 | signs | 0→2 | signs |
|---|---|---|---|---|
| linear (grid ref) | +1.37 [+0.79, +1.95] (n=18) | 17/18 | +3.59 [+2.92, +4.26] (n=40) | 37/40 |
| **mlp64 (primary)** | **+0.99 [+0.02, +1.96]** (n=10) | **8/10** | **+3.22 [+1.99, +4.45]** | **10/10** |
| mlp256 (sensitivity) | +1.62 [+0.80, +2.45] (n=10) | 9/10 | +4.22 [+2.18, +6.25] | 9/10 |

**Reading:** R* DiD does **not** vanish under the residual MLP head. The confirmatory R* story is not an artifact of refusing the MLP.

### SoftThresh τ=1.5

| Capacity | 0→0.5 | signs | 0→2 | signs |
|---|---|---|---|---|
| linear (grid ref) | +0.66 [+0.27, +1.06] (n=40) | 29/40 | +3.50 [+2.88, +4.13] | 38/40 |
| mlp64 | +0.21 [−1.61, +2.03] (n=10) | 5/10 | +1.40 [−0.95, +3.75] | 8/10 |
| mlp256 | +0.80 [−0.36, +1.97] (n=10) | 6/10 | +1.57 [−0.67, +3.81] | 7/10 |

**Reading:** SoftThresh mid-γ under MLP is **class-relative** — richer NLL absorbs much of φ (CI crosses 0). Loud SoftThresh only partially shrinks. This is expected absorption, not a dial failure.

### γ=0 cost

Mean H−NLL at γ=0 ≈ **−0.63 pt** (mlp64) vs **−1.19 pt** (mlp256): hybrid’s well-spec cost **worsens** with width. Prefer width 64 as the default fair head.

### OPC−NLL (secondary)

Under SoftThresh + mlp64, OPC−NLL DiD at 0→0.5 is **negative** (−1.55 pt): pure DR can trail NLL when the MLP class absorbs φ while q̂ stays linear-interaction. Hybrid remains the primary contrast; do not promote OPC−NLL alone under expanded \(\mathcal{F}\).

---

## SoftAND* gap (next run)

Capacity never included SoftAND*. Driver: [`run_capacity_softand_mlp64.sh`](../run_capacity_softand_mlp64.sh).

| Tag | γ | Kind |
|---|---|---|
| `nlres_cap_softand_g0_mlp64` | 0 | `none` |
| `nlres_cap_softand_g0.5_mlp64` | 0.5 | SoftAND* |
| `nlres_cap_softand_g100_mlp64` | 100 | SoftAND* |

Locked spec: `m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2`. Same matched methods / HPO / CRN as the R*/SoftThresh capacity wave. Mid γ probes absorption; γ=100 probes the φ-dominated regime where linear SoftAND DiD was large.

---

## Claims hygiene (short)

| Say | Do not say |
|---|---|
| Reasonable matched two-tower residual MLP (width 64) | “We only studied linear policies” as the production claim |
| Linear = identifying small \(\mathcal{F}\) | SoftThresh mid-γ MLP null = dial broken |
| SoftThresh mid-γ shrink = class-relative absorption | OPC-mlp vs NLL-linear is fair |
| Keep q̂ fixed in this wave | Old `mlp` (LN+dropout) is the capacity protocol |
