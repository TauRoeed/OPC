# BLOB reference code: the audit environment, the TensorFlow fixture and the Table 3 reproduction

These scripts run against the authors' release, never inside this repository's environment.
- **Release:** criteo-research/blob at e15cb38616a4b29a8ae8b8828975cd98dc62ea50.
- **Paper:** Sakhi, Bonner, Rohde and Vasile, *BLOB: A Probabilistic Model for Recommendation that Combines Organic
  and Bandit Signals*, KDD 2020, arXiv 2008.12504.

The audit and the model are described in `docs/blob_controlled_integration.md` §1–§2.

## Environment

The release targets TensorFlow 1.x and Python 3.6. Steps:
1. Create a Python 3.6 virtual environment.
2. Install `requirements-py36.txt` there.
3. Install recogym 0.1.3.0 with `--no-deps`. Its own pins include `intel-*` packages that are not available; the
   standard numpy, scipy and scikit-learn of the file replace them.
4. Install the CPU torch 1.10.2 wheel (used by the logistic-regression agent).

```bash
git clone https://github.com/criteo-research/blob ~/code/BLOB && git -C ~/code/BLOB checkout e15cb38
```

## `make_tf_fixture.py`: the reference for the PyTorch port

This script writes `tests/fixtures/blob_tf_reference.npz`, which `tests/test_blob_tf_reference.py` replays against
`models/blob.py`.
- It executes the bandit layer's graph-building block of `models/models_organic_bandit.py` verbatim. The block runs
  from `K, P = self.K, self.P` to the variable initializer; the lines are extracted from the source, not retyped.
- It runs on a small problem: P = 12, K = 4, 200 rows, batches of 64, 3 epochs.
- It records each step's noise, the losses, and the initial and final variables, for five cases: MNQ and NQ as
  released, MNQ without the Ψ normalization, and NQ and MNQ with wider priors.
- The provenance (commit, the file's sha256, a clean `models/` and the TensorFlow version) goes into the fixture.

```bash
cd ~/code/BLOB && <py36 env>/bin/python /home/noamk/code/OPC/scripts/blob_reference/make_tf_fixture.py
```

## `repro_table3.py`: the published Table 3 experiment, one repetition

This script is a copy of the authors' `simulate_abtest_with_bandit.py`.
- It is restricted to one repetition and to five agents: BLO, BLOB-MNQ, BLOB-NQ, logistic regression and random.
- The agents, their arguments and the evaluation calls are unchanged.
- It uses the README's Table 3 arguments: P = 100, K = 20, 1,000 bandit and 20,000 organic sessions, 1,000 organic
  and 800 bandit epochs, 4,000 scored users, flips 0 and 50.
- It must sit in the release's root, next to `models/` and `utils/`. `QUICK=1` runs a smoke test only.

```bash
cp repro_table3.py ~/code/BLOB/ && cd ~/code/BLOB && <py36 env>/bin/python repro_table3.py 100 0,50
```

The output goes to `results/repro_table3_P100.csv`. The run of 2026-10-05 is copied to
`artifacts/blob_reference/repro_table3_P100.csv`, and its comparison with the paper is in
`docs/blob_controlled_integration.md` §2.
