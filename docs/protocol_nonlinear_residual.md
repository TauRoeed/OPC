# Protocol NLRES-v1: a nonlinear truth residual, OPC vs CausE-cap vs BLOB

Status: DRAFT until the lock addendum (§9) is committed. After the addendum, only a new version number
(NLRES-v2) may change anything below; every number produced under v1 stays labelled v1.
"Phase-1" always means the finished v2-embedding study (`artifacts/full_study/phase1_protocol_v2emb.txt`).
This protocol's stages are Phase-0 (build, gate, calibrate) and Phase-2 (the full run).

## 0. Question and claim ladder

Phase-1 left one pattern: at combined-high, CausE-cap beats OPC by 0.3-1.0 pt (S3, mlp64), and the class
oracle used there was linear (`affine_bilinear`), so it says nothing about what a nonlinear class could earn.
The question: when the true click logit contains a part that no bilinear map of the logger's vectors can
express, does a nonlinear OPC (DR policy objective, DR-LB selection) gain more from its capacity than a
nonlinear CausE-cap (click likelihood, val-NLL selection)?

Claims allowed, in order of strength (a claim needs every claim above it to pass):
1. C0 The simulator is correct (§2 gates pass).
2. C1 The residual is real headroom for a nonlinear class and not for a linear one (§3 gate).
3. C2 Nonlinear gain G(method, g) = Gain(mlp tier) - Gain(linear tier) grows with g, and not under the placebo.
4. C3 The OPC-minus-CausE difference moves with g (difference-in-differences, §6). Only C3 answers the question.

## 1. Truth model and where it is injected

Current pipeline (`utils/representation_bias.py::build_world`, `utils/simulation_utils.py`):

```
clean vectors X, A --calibrate_world--> alpha, b (reference CTR 5%, best item 30%)
q*(u,a)  = sigmoid(scale * <x_u,a_a> + offset)         SyntheticBanditEnv, on CLEAN vectors
logger   = softmax(<our_x,our_a> / T_log), T_log from sharpen_logger(env)   BIASED vectors
log row  = (u ~ prior, a ~ pi_b(.|u), p = pi_b(a|u), r ~ Bernoulli(q*(u,a)))
```

New truth, added on the logit, on the clean vectors, after calibration:

```
logit*(u,a) = scale * <x_u,a_a> + offset' + g * alpha * phi_std(u,a)
q*(u,a)     = sigmoid(logit*(u,a))
phi(u,a)    = m^-1/2 * sum_{k<m} psi(<p_k, x_u>/sd_p_k) * psi(<r_k, a_a>/sd_r_k)
psi_s(t)    = tanh(s * (t^2 - 1))                      even, bounded, no linear component under symmetric inputs
```

- `x_u`, `a_a` are `env.emb_x`, `env.emb_a` (clean). Never `our_x`, `our_a`: the learner must see the residual only
  through biased vectors, which is the representation-mismatch setting of the repo.
- `p_k, r_k ~ N(0, I_d)`, seeded `derive_seed(world_seed, "residual", spec)`. `sd_p_k`, `sd_r_k` are the std of the
  projections over all users / all items, so `t` is unit scale.
- `phi_std` = (phi - mu_phi)/sd_phi, with mu, sd over 2e6 uniform (user, item) pairs (seed `derive_seed(..., "residual","std")`).
  alpha = `cal["alpha"]` is the sd of the linear logit, so g is "residual sd as a fraction of linear-logit sd".
- `offset'` re-solves the offset so the reference CTR is exactly the Phase-1 target. Use the same pairs `calibrate_world` used
  (`cal["_ctr_users"]`, the reference logger's items, same RNG label) and `_solve_b`: mean sigma(scale*dot + g*alpha*phi_std + offset') = 0.05 to 1e-9.
  Best-item CTR is NOT re-targeted (it drifts; record it).
- Placebo (PL): same code with `phi_bil = m^-1/2 * sum_k <p_k,x>/sd * <r_k,a>/sd` (bilinear, inside the linear class),
  standardized the same way, same g. It isolates "nonlinearity" from "more or different signal".
- g = 0 returns the Phase-1 `reward_prob` expression unchanged (early return, no extra float ops): bitwise identity.

### 1.1 Why the logit, and what must stay untouched

- On the logit: q stays in (0,1), the Bernoulli draw uses q, and q*(u,a) remains a function of (u,a) only, so every exact oracle remains
  `sum_u prior_u * (policy-weighted) q`. No clipping is needed.
- The logger is not touched: `our_x`, `our_a`, `policy_temperature` (`T_log`), `logging_uniform_mix`, the user prior. The propensity
  p = pi_b(a|u) therefore depends only on logger quantities and stays exact by construction (`create_simulation_data_from_policy`
  writes `pscore` from `policy.sample_actions`).
- `T_log` is FROZEN at its g=0 value. `build_world` calls `sharpen_logger(..., env)`, which sets T_log so the logger earns 80% of its greedy CTR
  under the env it is given. With a residual env that would change T_log with g, i.e. change propensities, ESS and the logging
  distribution between arms of the g factor. Required build order in `build_world`:
  1. build env0 (g = 0) and run `sharpen_logger` as today -> `T_log`, `our_x`, `our_a`;
  2. build `ResidualBanditEnv` (g) with re-solved offset';
  3. recompute `logger_greedy_ctr`, `logging_ctr`, `uniform_ctr`, `reference_ctr`, `best_item_ctr` under env_g at the frozen T_log
     (call `sharpen_logger(our_x, our_a, cu, T_log, env_g, share=0.0)`: it keeps f = 1 and reports);
  4. record all of them, plus the residual spec, in `world`.
  Consequence to report, not to fix: the logger's true value changes with g.

### 1.2 Every place that computes q (all must go through one function)

Today q is written inline in several places. A residual env that only overrides `reward_prob` / `reward_prob_block` is wrong in the
inline ones. Add to the env: `logit_block(users, a0, a1)` (float64) and `q_block_torch(user_idx, device)`; route:

| Site | Bypass today | Fix |
|---|---|---|
| `utils/simulation_utils.py` `_exact_value_torch` ~L370-390 | `env_x @ env_a_t * scale + offset` then sigmoid | call `env.q_block_torch` |
| `training/policy_diagnostics.py` ~L64-76 | same inline | same |
| `training/oracle_repair.py::true_q_rows` L55-60 (also used by `training/class_oracles.py`) | same inline | same |
| `training/oracle_repair.py::logger_values` ceiling | `env.reward_prob_block` (fine once overridden) | no change, test it |
| `training/trainer_trials.py::AnalyticRewardModel.from_env` L796, L893 (reward models `oracle`, `logging_score`) | scale/offset only | raise `NotImplementedError` when g != 0 (Phase-1 uses `regression`; do not enable these) |
| `utils/representation_bias.py` `calibrate_world` zs / zmax | linear-only calibration | leave; offset' fixes the reference CTR afterwards |
| `ensure_exact_env_q_cache` (`q_x_a`) | goes through `env_reward_block` | fine; must be built after the env is final; assert the cache is None before the swap |
| `utils/budget_split.py`, `utils/rand_ctr.py`, `training/logging_coupling_diagnostic.py` | `env.reward_prob*` | fine once overridden |

Run identity: add `residual_gamma`, `residual_spec`, `residual_kind` (`nl` | `bil`) to `WORLD_RUN_KEY_TAGS` (`world_run_key_suffix`) so folders
carry `__rg=0.5__rk=nl`. Without it `--skip-completed` silently reuses the g = 0 folder for every g. Also add them to `arm_config_key`.

## 2. Phase-0 correctness gates (CPU unit tests plus one GPU smoke; all must pass, nothing here reads a method result)

New file `tests/test_residual_world.py`, modeled on `tests/test_world.py`, `tests/test_policy_sampling.py`,
`tests/test_logging_propensity_calibration.py`.

| Gate | Check | Pass |
|---|---|---|
| G1 identity | g = 0: `reward_prob`, `reward_prob_block`, `q_x_a`, `_exact_value_torch` equal the pre-change env on 1e6 random pairs; `build_world` for ml and anime seed 100 `combined-high` reproduces Phase-1 `run_meta.json` world fields (`scale`, `offset`, `logging_temperature`, `logger_greedy_ctr`, `logging_ctr`) | bitwise (float64 `==`) |
| G2 logger invariance | for g in {0, 0.5, 2} and kind in {nl, bil}: sha256 of `our_x`, `our_a`, `T_log`, `user_prior` identical | equal |
| G3 propensity exactness | log 1e5 rows; recompute `pi_b(a|u)` independently (`softmax(our_x[u] @ our_a.T / T_log)`, mix 0); compare to logged `pscore`; `sum_a pi_b = 1` on 500 users | max abs diff <= 1e-6 (fp32), sums 1 +- 1e-6 |
| G4 sampler | 300 users x 2e5 draws vs pi_b, chi-square on the top-50 items + remainder bin | no user below p = 1e-4 after Bonferroni |
| G5 single q | the same 2000 x 2000 block of q from: `reward_prob`, `reward_prob_block`, `q_x_a`, `_exact_value_torch` (no cache), `policy_diagnostics.q`, `true_q_rows`, brute-force numpy | pairwise max abs diff <= 2e-6 |
| G6 click calibration | 5e6 logged rows: bucket by q* (20 equal-count buckets); `|mean(r) - mean(q*)|` per bucket | <= 4 binomial SE |
| G7 reference CTR | mean q* over the calibration pairs | 0.05 +- 1e-8 |
| G8 oracle consistency | `true_best = sum_u prior_u max_a q*(u,a)` by brute force on 200 users equals `logger_values()["ceiling"]` restricted to them; for every kept oracle `V <= true_best`; logger exact greedy <= `true_best` | exact / inequality |
| G9 residual bookkeeping | mean(phi_std) = 0 +- 1e-3, sd = 1 +- 1e-3 on fresh uniform pairs; PL has the same sd | tolerance shown |
| G10 provenance | run writes `LOCK.json`: git sha, `git status --porcelain` for tracked files empty, protocol file sha256, spec, g. The Phase-1 manifests show `code_commit: null` and the working tree was dirty (`models/cause.py`, `models/models.py`, `tests/test_cause_fair.py` modified at the time of writing); do not repeat that. Mount `.git` read-only into the container or pass `--code-commit $(git rev-parse HEAD)` | file exists, tree clean |

## 3. Calibrating g and the residual family without method results

Allowed inputs: the exact class oracles, `true_best`, the exact logger value, closed-form diagnostics of phi. Forbidden: any OPC /
CausE / BLOB trial, summary or log. Calibration worlds: ml and anime, seeds 900 and 901 (disjoint from Phase-2 seeds 100-104).

Quantities per world (all exact, from existing code):
- `V_L` = exact greedy value of the logger: `_policy_greedy_reward_from_embeddings(dataset, our_x, our_a)` (the `logger_greedy` column of
  `summary_metrics.csv`; NOT `world["logger_greedy_ctr"]`, which is a 1000-user estimate, see §7).
- `V_lin` = value-objective `affine_bilinear` oracle (`python -m training.class_oracles`; OPC's and CausE-cap-linear's class).
- `V_blob` = `blob` oracle (same script; BLOB's class).
- `V_mlp` = `python -m training.oracle_repair --policy-transform linear+mlp --mlp-hidden 64 --classes linear+scale`, then `max(V_mlp, V_lin)`
  because the class is nested but the fit is warm-started only in `class_oracles.py`; also report both.
- `V*` = `true_best`.

Derived: linear headroom LH = V_lin - V_L; nonlinear headroom NH = V_mlp - V_lin; total headroom TH = V* - V_L.

Step 1, pick the residual family (closed-form, no oracles needed). Grid, in this order: (s, m) in {(0.5,4), (0.5,8), (1,4), (1,8)}.
For each, on 2e6 logger-sampled pairs (users ~ prior, items ~ pi_b) at g = 1, using the BIASED vectors the learner sees:
- R2_lin: fraction of Var(phi_std) explained by least squares on [vec(our_x outer our_a), our_x, our_a, 1] (train 1.5e6 / test 0.5e6);
- R2_mlp: fraction explained by a `LinearPlusMLPCorrection(hidden=64)` pair trained by MSE (3000 Adam steps, lr 3e-3, test split).
Take the first (s, m) with R2_lin <= 0.15 and R2_mlp >= 0.50 on both datasets (mean over the 2 seeds, each seed within 0.1 of the mean).
None qualifies: abort A2.

Step 2, pick g_hi from the grid g in {0.25, 0.5, 1, 2} (nl kind, chosen (s, m)). g_hi = the smallest g such that on both datasets,
mean over seeds 900/901:
- NH >= 1.0 CTR pt, and each seed >= 0.7 pt (1 pt is roughly 2-3 SE of a paired difference, Phase-1 per-cell sd 0.3-0.5 pt);
- LH(g) >= 0.5 * LH(0) (linear learning keeps a job; otherwise the experiment is only about the residual);
- TH(g) <= 25 pt and mean over users of max_a q* <= 0.6 (no saturated, trivial world);
- |V_L(g) - V_L(0)| <= 3 pt (the logger is not wrecked by a frozen T_log).
g_mid = the next lower grid value (0.25 -> use 0.125; list it as a grid point only if g_hi = 0.25).
None qualifies: abort A2. Do not extend the grid after seeing the table.

Step 3: the placebo uses g_hi and kind = bil. Check R2_lin(bil) >= 0.85 on the same pairs; else abort A2.

## 4. Factors and exact arms

Factors (Phase-2): dataset in {ml, anime}; seed in {100, 101, 102, 103, 104} (100-102 are the Phase-1 worlds);
bias = `high` (combined-high, `--bias-configs high`); n = 100000; val = 20000.
Residual level R in {g0, g_mid, g_hi, PL(g_hi)}. Lean variant if budget is cut: drop g_mid. Never drop g0 or PL.
World count = 2 x 5 x 4 = 40 (30 lean).

Arms (per world; a tier is one `--policy-transform`):

| Tier | Arm id (summary `method` / trials file) | Runner flags that differ |
|---|---|---|
| L (linear) | OPC-lin = `opc` | `--policy-transform linear` |
| L | CausE-cap-lin, prediction C = `causecap_c_r000` (primary), T = `causecap_t_r000` (secondary) | `--cause-family cap --cause-transform linear` |
| L | BLOB = `blob_l10_nq` | `--blob-families nq --blob-variants L10` |
| M (mlp) | OPC-mlp64 = `opc` | `--policy-transform linear+mlp --mlp-hidden 64` |
| M | CausE-cap-mlp64, C primary | `--cause-family cap --policy-transform linear+mlp --mlp-hidden 64` (as Phase-1 S3) |

Not in this protocol: OPC-free / CausE-warm (S2), BLOB in tier M (BLOB's class is `x^T M a + kappa_a`; there is no mlp BLOB),
other estimators, other n.

Two separate invocations per R level, as Phase-1 did (one per tier), run tags:
`nlres_<R>_s1_linear` and `nlres_<R>_s3_mlp64`, with R in {g0, gmid, ghi, plhi}. Calibration: `nlres_p0_cal`; smoke: `nlres_p0_smoke`.

### 4.1 Knobs that must equal Phase-1 (copy `ablation_logs/phase1_v2emb_rerun.sh` `COMMON` verbatim)

```
--datasets ml anime  --seeds 100 101 102 103 104  --train-sizes 100000  --val-size 20000  --n-trials 20
--policy-losses dr  --opc-gradient direct  --train-weights harmonic:0.1  --select-weights clip:10
--learn-logit-scale  --sampler random  --lr-range 1e-4 2e-3  --epochs-range 5 30
--cause-rhos 0  --cause-lr-range 3e-4 3e-2  --cause-epochs 30 100 300  --cause-l2 0 1e-5 1e-3  --cause-cf 0 1 10
--cause-tie symmetric  --cause-bias-inits base_rate
--bias-configs high  --stage development  --slim  --require-cuda  --skip-completed
--emb-dir BPR/embeddings  --out-dir artifacts/full_study  --num-gpus 4  --max-workers 2  --min-workers 1
BLOB only: --blob-families nq --blob-variants L10 --blob-lr-range 3e-3 3e-1 --blob-epochs 10 30 100 300 1000
           --blob-wa-m -1 1 3 --blob-wb-m -6 -3 0 --blob-kappa-s 0.1
```

Defaults that the manifest must still show after the run (assert in the report script, fail loudly otherwise):
`optuna_selection = ci_low` (OPC picks the DR lower bound), `reward_model = regression`, `reward_features = interaction`,
`crossfit_folds = 5`, `reward_data = train`, `logging_uniform_mix = 0`, `logger_greedy_share = 0.8`, `pop_strength = 0`,
`ctr_reference = logger`, `cause_options.hidden = 64`, `cause_options.optimizer = momentum_decay`, CausE selection by val NLL.
Add `residual_gamma`, `residual_kind`, `residual_spec`, `code_commit` to the same assertion.
New runner flags (to implement): `--residual-gamma G --residual-kind {nl,bil} --residual-spec s=..,m=..`.
Same flags go to `training.class_oracles` and `training.oracle_repair` (§5).

Selection pairing is part of each arm and is NOT harmonized: OPC = DR lower bound at `clip:10`, CausE-cap and BLOB = val NLL
(Phase-1's mix). That is what is being compared; do not "fix" it in this experiment. §7 lists it as a confound.

## 5. Oracles and ceilings (always the same world as the arms)

| Reference | Used for | Command |
|---|---|---|
| `V_L` exact logger greedy | gain baseline for every arm | the `logger_greedy` column; recomputed and asserted equal to `logger_values()["greedy"]` |
| `V*` true best | headroom-closed % (primary secondary metric, <= 100% by construction) | `logger_values()["ceiling"]` |
| `V_lin` affine_bilinear value oracle | class-restored % for OPC-lin, CausE-lin | `training.class_oracles --classes affine_bilinear blob bilinear --objectives value likelihood` |
| `V_blob` blob value oracle | class-restored % for BLOB | same |
| `V_mlp` = max(linear+mlp64 oracle, `V_lin`) | class-restored % for OPC-mlp, CausE-mlp | `training.oracle_repair --policy-transform linear+mlp --mlp-hidden 64 --classes linear+scale` |

Arm-to-oracle map (hard-coded in the report script; Phase-1's `ARM_ORACLE` mapped the mlp arms to `affine_bilinear`, which is wrong here):
OPC-lin, CausE-lin -> `V_lin`; BLOB -> `V_blob`; OPC-mlp, CausE-mlp -> `V_mlp`.
Oracle rows and arm rows are joined on a world fingerprint (sha256 of `env.scale`, `env.offset`, `p_k`, `r_k`, g, kind, `T_log`, `our_x`)
written into both `run_meta.json` and the oracle CSV. A mismatch aborts the report.

## 6. Metrics, estimands, success criteria

Unit of analysis = world (dataset x seed). All gains in CTR points, 100 x (exact greedy value - `V_L`). The estimate of an arm in a world
is the arm's selected model (the selection rule of that arm), exactly as in Phase-1's `gain_pts`.

Primary metrics:
- Gain(arm, R, world).
- G(method, R) = Gain(M-tier arm) - Gain(L-tier arm), paired within world; methods = OPC, CausE-C.
- D(R) = Gain(OPC-mlp) - Gain(CausE-C-mlp), paired within world, same R.
- DiD = D(g_hi) - D(g0), and DiD_PL = D(PL) - D(g0).
Secondary: headroom closed % = 100 x gain/(V* - V_L); class-restored % (reported only where the class gap > 0.5 pt, and flagged if > 100);
best-of-20 trial ceiling; T prediction of CausE; BLOB - OPC-lin in tier L.

Inference: 10 worlds (pooled over ml/anime) at 5 seeds; two-sided paired t interval with 9 df, plus the per-world list and sign count (the Phase-1
report style); dataset-stratified numbers (5 worlds each) are descriptive. Two primary hypotheses, no multiplicity correction: S1 (DiD) and S2
(G(OPC, g_hi) - G(OPC, PL) is nonzero is NOT tested; see below). Everything else is Holm-corrected within its table. Equivalence margin e = 0.5 pt.

| ID | Statement | Pass |
|---|---|---|
| S0 | control reproduces Phase-1 | at g0, tier M, worlds seeds 100-102: |Gain - Phase-1 S3 gain| <= 0.10 pt per arm (bitwise expected, `deterministic: true`); failing means the code or environment changed: stop, bisect |
| S1 | nonlinearity is exploitable | G(OPC, g_hi) >= 0.5 pt and G(CausE-C, g_hi) >= 0.5 pt, each CI > 0, and each exceeds the same quantity at PL by >= 0.5 pt (CI of the difference > 0) |
| S2 | the exploit grows with g | G(m, g_mid) between G(m, g0) and G(m, g_hi) for both methods (monotone, no CI requirement) |
| S3 (primary) | OPC advantage moves | DiD CI excludes 0 and |DiD| >= 0.5 pt; sign decides the verdict: DiD > 0 "OPC gains more from nonlinearity", < 0 "CausE-cap gains more"; and DiD_PL must be inside [-e, e] (the shift is due to nonlinearity, not extra signal) |
| S4 | null | CI of DiD inside [-e, e] and CI of DiD_PL inside [-e, e] -> "no evidence the methods differ in using nonlinearity" |
| S5 | BLOB reference | BLOB gain does not rise with g (G-like slope vs g0 <= 0.5 pt) as it has no interaction capacity; if it does rise, it is reporting the item-intercept part and the interpretation of S1 for BLOB changes |

Anything else (a CI that straddles e, a PL contrast that fails) is "inconclusive", not a weak version of S3.

## 7. Pitfalls and their pre-declared handling

1. Anime CausE-cap collapse (observed in Phase-1 S1, `run_phase1_v2emb_s1_linear`): on anime seed 102, CausE-cap-lin on every one of the three axis worlds has
   all 20 trials at greedy 0.6-0.9% CTR (true CTR of the top items) against ~21-23% logger, while val NLL is within 0.01-0.02 of the other seeds
   (0.449-0.486). It is not a hyperparameter instability: no trial escapes, and NLL does not see it. The argmax over the whole catalog lands on items with
   no logged support (a linear item map extrapolating). It does not occur in S2/S3 combined-high on anime. Handling:
   - Phase-0 canary: run anime seed 102, g0, CausE-cap-lin, 20 trials, on a calibration-style tag before locking. Record whether it collapses.
   - Define `collapsed` = selected greedy < 0.5 x `V_L`; `all_collapsed` = every one of the 20 trials is. Both go in every table (count per arm x R).
   - Collapsed cells stay in the primary analysis (no exclusion). Report the primary tables twice: all worlds, and the worlds where no arm collapsed;
     the verdict must not depend on the choice, otherwise "inconclusive".
   - No re-seeding, no restarts, no hyperparameter edits to rescue a collapse.
2. Class-restored % above 100: caused by (a) comparing nonlinear arms against a linear oracle (Phase-1 did this for S3), (b) oracles that are lower bounds
   (3000 steps, 20k fit users, 3 lrs), (c) the oracle fit on a user sample while arms are graded on all users, (d) BLOB's class differing from OPC's.
   Handling: headroom-closed % against `V*` is the headline (cannot exceed 100); class-restored % uses the §5 map; if any exceeds 105%,
   rerun that oracle once with 2x steps and 2x fit users and take the max; keep the flag in the table. A negative class gap (V_class < V_L) is printed as `n/a`.
3. Mismatched oracles and baselines: arms and oracles must share the fingerprint (§5). The Phase-1 report's baseline was `world["logger_greedy_ctr"]`, estimated on
   1000 calibration users, while the arm values are exact over all users; the sampling error cancels in paired contrasts but not in absolute gains or in headroom
   fractions. Use the exact `logger_greedy` column and assert it equals `logger_values()["greedy"]`.
4. Capacity confound OPC-mlp vs CausE-cap-mlp: the C prediction uses user map + control item map = `LinearPlusMLPCorrection` x2 with hidden 64, identical init (last MLP layer at zero),
   i.e. the same function class as OPC. Differences that remain and must be named in the report: CausE-cap trains three maps (+50% parameters; the treatment map is the ride-along),
   a tie penalty, an L2 on every map weight including the MLP (0/1e-5/1e-3), global click bias and alpha instead of a learned logit scale, different optimizer, epochs 30-300 versus 5-30,
   and OPC's separate reward model q_hat (regression on interaction features, 5-fold crossfit). q_hat is a third capacity that CausE lacks and that is misspecified under the residual;
   log its validation AUC and NLL per R from `summary_metrics.csv` as a diagnostic, do not tune it. One optional Phase-3 ablation (not in the lock): hidden in {16, 256} at g_hi on the same worlds.
   Do not match capacity by tuning per method; the width is fixed at 64.
5. Selection mismatch: OPC selects by DR-LB at clip:10, CausE-cap and BLOB by val NLL. Report "selected" and "best-of-20" gains side by side so the selection rule's share is visible;
   restored-vs-trial-ceiling percentages from Phase-1 already show it is ~97-100% for every arm, so a large gap there means a selection problem, not a class problem.
6. Mean +- std over 3 seeds hid the anime collapse (std 13.6). Use 5 seeds, print per-world lists, use medians next to means.
7. Skip-completed collisions: see §1.2 run identity. After each invocation assert `run_meta.json` residual fields equal the requested ones.
8. Residual that is not learnable at n = 100k is not evidence about capacity. The Phase-0 gate (NH from oracles) shows representability, not sample efficiency. The
   effective information is ~9k-11k clicks at CTR 8-11%; if the g_hi run shows no tier-M gain for either method (S1 fails for both), the report says "residual not learnable at 100k with this
   class", not "mlp does not help". One pre-declared follow-up: the same g_hi at n = 1M for OPC-mlp and CausE-mlp only (+6 GPU-h, separate version tag).
9. Memory: Phase-1 S1 died with higher worker counts (restart at `max-workers 2`, log line 7941). Keep 2 workers; BLOB epochs 1000 is the memory/time tail.
10. Dirty tree and null commit in Phase-1 manifests: gate G10.

## 8. Run order, commands, abort criteria

Branch: `nlres` from the commit that passes G1-G10. Everything below uses `docker run --rm --gpus all --shm-size=128g -v $PWD:/app -w /app opc:gpu` as `ablation_logs/phase1_v2emb_rerun.sh`.

Phase-0 (about 2 GPU-h):
- P0.1 implement §1, tests G1-G9 pass on CPU.
- P0.2 Phase-1 reproduction at g0 on one world (ml seed 100, tier M, both arms, 20 trials): S0.
- P0.3 family selection (§3 step 1): `python -m training.residual_family --datasets ml anime --seeds 900 901`.
- P0.4 oracle table: `class_oracles` + `oracle_repair` over {ml, anime} x {900, 901} x g in {0, 0.25, 0.5, 1, 2} (nl) -> `nlres_p0_cal`.
- P0.5 pick g_hi, g_mid, placebo check (§3 step 2-3).
- P0.6 smoke (only crashes, finiteness, wall time may be read): 1 world per dataset at seed 900, R in {g0, ghi}, 3 trials per arm, tags `nlres_p0_smoke`. The analysis scripts must not be run on this tag.
- P0.7 canary: anime seed 102, g0, CausE-cap-lin, 20 trials.
- P0.8 lock addendum commit (§9), tag `nlres-lock-v1`.

Phase-2: 4 R levels x 2 tiers, in this order: g0, ghi, plhi, gmid. Each `nlres_<R>_s1_linear` then `nlres_<R>_s3_mlp64`. Then class oracles for all 40 worlds. Then the report script
(`artifacts/full_study/nlres_report/build_nlres_report.py`, written and committed before Phase-2 starts, run once after all tags are complete; no interim method comparisons).

Abort / stop criteria (any one stops the experiment and is reported; none may be answered by changing the protocol):
- A1 any of G1-G10 fails.
- A2 no (s, m) in the family grid, or no g in the g grid, meets §3, or the placebo check fails.
- A3 S0 fails (g0 does not reproduce Phase-1 within 0.10 pt).
- A4 in any (R, tier) block, more than 30% of an arm's worlds are `all_collapsed`: stop that tier, finish the other blocks, report the tier as failed. The canary result is informational only, but an all-collapsed canary at g0 converts CausE-cap-lin to "reported with collapse flags" before the lock, not after.
- A5 `finite = False` in more than 20% of an arm's trials, or any NaN selected value.
- A6 `failures.csv` non-empty after two `--skip-completed` resumes of a tag.
- A7 a manifest assertion in §4.1 fails after a run.
No early stopping on efficacy or futility; no looking at D(R) or DiD until every tag is complete.

## 9. Lock addendum (filled at P0.8, before any Phase-2 run)

```
version: NLRES-v1                git sha: ________  tree clean: yes
residual family: s = __, m = __  psi(t) = tanh(s (t^2-1))
g grid result table: path = artifacts/full_study/nlres_p0_cal/table.csv
g_hi = __   g_mid = __   placebo g = g_hi   (kind = bil)
seeds 100-104; datasets ml anime; n 100000; val 20000; bias high
phase-2 tags: nlres_{g0,gmid,ghi,plhi}_{s1_linear,s3_mlp64}
report script sha256: ________   protocol sha256: ________
canary result (anime 102 g0 CausE-cap-lin): collapsed yes/no
```

## 10. Reporting tables (all produced by the one report script)

| Table | Contents |
|---|---|
| T0 world audit | per (dataset, seed, R): scale, offset', g, kind, `T_log` (equals g0), `V_L`, `V*`, `reference_ctr`, `best_item_ctr`, `uniform_ctr`, logger ESS; fingerprints; G1-G10 status |
| T1 calibration | the Phase-0 oracle table: LH, NH, TH, R2_lin, R2_mlp per g and seed; the chosen row marked |
| T2 control | g0 vs Phase-1 S3 (seeds 100-102): value, delta |
| T3 gains | per (tier, arm, R, dataset): mean, sd, median, per-world list, n; also selected and best-of-20 |
| T4 contrasts | G(OPC), G(CausE-C), D(R), DiD, DiD_PL: estimate, 95% CI, sign count, per-world list; all-worlds and no-collapse versions |
| T5 headroom closed | gain/(V* - V_L) per arm, R |
| T6 class-restored | per the §5 map, with `n/a` and `>100` flags |
| T7 health | per arm, R: `collapsed`, `all_collapsed`, `finite` fraction, selected trial index, OPC q_hat val AUC / NLL, CausE val NLL vs base-rate NLL |
| T8 criteria | S0-S5 with pass / fail / inconclusive and the number behind each |
| F1 | gain vs g (with PL as a separate marker), per method and tier, per-world points |
| F2 | D(R) and the PL point |

## 11. Cost (measured from the Phase-1 logs, 4x RTX PRO 6000, `--max-workers 2`)

Phase-1 wall times: S1 (18 cells, OPC-lin + CausE-cap + BLOB) about 85 min including one OOM restart, S2 (6 cells, 2 arms) 16 min, S3 (6 cells, OPC-mlp + CausE-cap-mlp) 11 min,
class oracles (30 worlds, 540 fits) 38 min on one GPU. Two workers each hold one GPU, so GPU-hours = wall x 2.

Per-world cost, 20 trials, 100k rows: tier L (OPC-lin, CausE-lin, BLOB) about 9-10 GPU-min (BLOB is about 60%); tier M (OPC-mlp, CausE-mlp) about 3.7 GPU-min; oracles
(lin + blob + mlp) about 3.5 GPU-min including the 35 s world build. Total about 17 GPU-min per world.

| Stage | Worlds | GPU-hours | Wall at 2 workers |
|---|---|---|---|
| Phase-0 | 20 calibration worlds (oracles) 1.2; family R2 0.1; smoke 0.2; canary and reproduction 0.3 | about 2 (1-3) | about 1 h |
| Phase-2 full | 40 worlds x 17 min | about 11 (7-16) | about 5-6 h |
| Phase-2 lean (no g_mid, 3 seeds) | 18 worlds | about 5 | about 2.5 h |
| Optional n = 1M follow-up | 10 worlds, tier M only | about 6 | about 3 h |

Reserve 1.5x for reruns (anime CausE collapse, OOM restarts).
