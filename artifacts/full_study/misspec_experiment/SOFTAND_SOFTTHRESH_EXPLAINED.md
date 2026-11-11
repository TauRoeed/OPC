# SoftAND* and SoftThresh — explained

Self-contained note for readers not in the implementation loop.  
PDF twin: `softand_softthresh_explained.pdf` (on the `pdf` branch).

## Where they sit

Truth logit with residual dial γ:

\[
\ell_\gamma(x,a)= b + u_x^\top v_a + \gamma\cdot\alpha\cdot\phi(x,a),\qquad \mathrm{sd}(\phi)=1.
\]

- **γ = 0**: well-specified bilinear world.  
- **Mid γ**: residual competes with the GT channel.  
- **Extreme γ**: residual dominates ranking.

SoftAND* and SoftThresh are two *different residual geometries* used alongside **R*** to check that hybrid NLL→DR is not an R*-only artifact. Learners never see φ as a feature—only logged clicks from \(q=\sigma(\ell_\gamma)\).

## Shared scaffolding

1. **Facet projections** \(t_{u,k}, t_{a,k}\) (\(m=4\), `proj_tag=default`), shared across families on a tag.  
2. **⊥ (bilinear perpendicular):** remove the best affine-bilinear fit in the embeddings, so φ is “what bilinear cannot already soak up.”  
3. **Unit-SD (`norm=v2`):** rescale so catalog pair-SD of φ is 1 → SoftAND*/SoftThresh track **sd_ratio ≈ γ** (R* overshoots ≈2.4×).

---

## SoftAND* — thresholded per-facet concordance

**Intuition:** both user and item must “light up” the *same* facet above a threshold (AND-like), then average over facets.

With softplus \(\mathrm{sp}(z)=\log(1+e^z)\) and locked \(c=1\), \(s=\texttt{gate\_s}=0.25\):

\[
g_{u,k}=\mathrm{sp}\big((t_{u,k}-c)/s\big),\quad
h_{a,k}=\mathrm{sp}\big((t_{a,k}-c)/s\big),\quad
\phi_{\mathrm{raw}}=m^{-1/2}\sum_k g_{u,k}h_{a,k}.
\]

Then double-center → ⊥ → unit-SD.

**Locked spec:** `m=4,c=1,gate_s=0.25,proj_tag=default,norm=v2`  
(KuaiRec mlsize may add `catalog_amp=0.75`.)

**Vs R*:** SoftAND* uses one-sided softplus gates, not an even \(\tanh(s(t^2-1))\) warp. Still separable (low-rank product) *before* ⊥.

---

## SoftThresh (τ = 1.5) — soft threshold on \(z_2\)

**Intuition:** form a second bilinear index, then give a sparse bonus only when it is large.

\[
z_2=m^{-1/2}\sum_k t_{u,k}t_{a,k},\quad
\tilde z_2=(z_2-\mu)/\sigma,\quad
\phi_{\mathrm{raw}}=\mathrm{sp}\big((\tilde z_2-\tau)/s_{\mathrm{th}}\big).
\]

Locked \(\tau=1.5\), \(s_{\mathrm{th}}=0.25\). Then ⊥ → unit-SD.

**Locked spec:** `m=4,tau=1.5,s_th=0.25,proj_tag=default,norm=v2`

**Vs SoftAND*:** SoftThresh only cares that the *sum* of facet products is large, not per-facet co-activation. softplus-of-a-sum is **not** finite-rank separable.

---

## Side-by-side

| | SoftAND* | SoftThresh τ=1.5 |
|---|---|---|
| Nonlinearity | Per facet, then product | On scalar \(\tilde z_2\) |
| AND semantics | Same-facet co-activation | “Second index large” |
| Separable after ψ? | Yes before ⊥ | No |
| Tracks sd_ratio≈γ? | Yes | Yes |
| var_share_φ at γ=0.5 | ≈20% | ≈20% |
| R* at γ=0.5 (ref) | — | sd_ratio≈1.19, var_share≈59% (louder) |

## Dial calibration (ML, selected)

| family | γ | sd_ratio | var_share_φ | CTR lift_φ |
|---|---:|---:|---:|---:|
| SoftAND* | 0.5 | 0.50 | 19.9% | +0.33pt |
| SoftAND* | 2 | 1.99 | 79.9% | +1.86pt |
| SoftThresh | 0.5 | 0.50 | 20.0% | +0.39pt |
| SoftThresh | 2 | 2.00 | 80.0% | +2.17pt |
| R* (ref) | 0.5 | 1.19 | 58.7% | +1.17pt |

Full table: `hybrid_by_cell/REWARD_RATIOS.md`.

## How it shows up in results

On ML (\(n=40\)), SoftThresh typically shows clear hybrid wins from about **γ≥0.5**; SoftAND* from about **γ≥0.75**; both large at γ∈{2,4,10,100}. At γ=0, hybrid vs NLL is near zero / slightly negative (small cost when the world is bilinear).

Per-setup numbers: `hybrid_results_by_setup.pdf`.

## Code & tags

- `utils/residual_env.py` (`kind=softand` / `softthresh`)
- Broader family map: `FAMILIES_EXPLAINED.md`
- Grid tags: `nlres_grid_{ds}_softand_g{γ}`, `nlres_grid_{ds}_softthresh_g{γ}`

## One-liners

1. SoftAND* = soft same-facet AND, then strip bilinear, unit-SD.  
2. SoftThresh = soft threshold on second index \(z_2\), then strip bilinear, unit-SD.  
3. Both track sd_ratio≈γ; R* is louder at the same nominal γ.  
4. They stress-test that hybrid NLL→DR is not an R*-only story.
