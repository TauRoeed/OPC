# OPC Research Workplan

Master tracker for continuing OPC research. Code lives under `utils/`, `training/`, `BPR/`; narrative under `docs/`.

## Priority (current phase)

1. **Noise / SNR characterization** — measure, compare, link to correction magnitude.
2. **KuaiRec / KuaiRand-Pure** — loaders + BPR wiring (no full study runs yet).
3. Regimes, bias examples, paper contributions, reflection (docs scaffolding).

## Tracks

| Track | Status | Docs / code |
|-------|--------|-------------|
| Workplan + framework | done | this file |
| Noise / SNR | done (v1) | [noise_snr.md](noise_snr.md), `utils/noise_snr.py`, `utils/noise_levels.py`, `training/characterize_noise_snr.py` |
| Experimental regimes | done (catalog) | [regimes.md](regimes.md) |
| Additional datasets | loaders done | [datasets_kuai.md](datasets_kuai.md), `BPR/` Kuai loaders |
| Bias modeling | scaffold | [bias_examples.md](bias_examples.md), [bias_axis_trial.md](bias_axis_trial.md) |
| Paper contributions | updated bullets | [paper_outline.md](paper_outline.md) §1 |
| Theoretical analysis | deferred | optional; after SNR × complexity sweeps |
| Critical reflection | scaffold | [reflection.md](reflection.md) |
| Regime inference / rand_CTR | drafted | [regime_inference.md](regime_inference.md), `utils/rand_ctr_sample_size.py` |

## Next actions

- [x] Run `python -m training.characterize_noise_snr --datasets ml` → `artifacts/noise_snr/`.
- [x] Download KuaiRec / KuaiRand-Pure once; generate BPR artifacts.
- [ ] Bias trial 1 — min val @ 1M train — [bias_axis_trial.md](bias_axis_trial.md); tag `bias_min_val_tr1m_t20_s5`
- [ ] Bias trial 2 — noise grid @ ≥0.5M train, fixed val — tag `bias_axis_comp_tr500k_l5_t20_s5`
- [ ] Real-world noise captions — [bias_examples.md](bias_examples.md)
- [ ] Test H1 — see [h1_experiment.md](h1_experiment.md); run `./scripts/run_h1_study.sh`
- [ ] Plot OPC lift vs measured SNR (reuse existing `artifacts/full_study` + new `snr` in `run_meta.json`).
- [ ] Fill paper Results placeholders from hurtlog / reward-model ablations.
- [ ] Design theory/complexity × SNR experiment grid (later).

## Out of scope (this phase)

- Full study sweeps on Kuai.
- New noise generators beyond characterization.
- Formal theorems.
