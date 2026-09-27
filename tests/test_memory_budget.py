"""Memory-aware worker planning (scheduling only; no GPU needed)."""

import pytest

import training.memory_budget as mb
from training.memory_budget import (
    CONTEXT_BYTES,
    PEAK_BATCH_MATRICES,
    SAFETY_FRACTION,
    catalog_shape,
    dense_qhat_copies,
    describe_plan,
    estimate_worker_peak_bytes,
    max_train_batch,
    plan_worker_groups,
    workers_per_device,
    workers_that_fit,
)

GB = 1024**3
ML_SHAPE = (6_040, 3_533)
MSD_SHAPE = (1_019_318, 42_053)
ANIME_SHAPE = (73_417, 10_803)
KUAIRAND_SHAPE = (27_111, 7_579)
# The Stage 2 study budget: train sizes 5k/25k/100k (batch up to 8,192), reward model fit on each size's own
# training rows and cross-fitted by user in 5 folds.
STAGE2 = {"train_sizes": [5_000, 25_000, 100_000], "reward_data": "train", "crossfit_folds": 5}
MSD_1M = {"train_sizes": [1_000_000]}  # q_hat too large to materialize: the training step is the whole peak
BIG_WORKSTATION = [95 * GB] * 4  # four 96 GB cards, ~1 GB each in use


def test_max_batch_follows_the_trainers_choices():
    assert max_train_batch({"train_sizes": [5_000]}) == 2048
    assert max_train_batch({"train_sizes": [5_000, 10_000_000]}) == 327_680
    assert max_train_batch(STAGE2) == 8_192
    assert max_train_batch(MSD_1M) == 32_768  # the schedule's choices at 1M: 8,192 / 16,384 / 32,768
    # --optuna-batch-sizes replaces the schedule's choices (trainer_trials: cli_optuna_batches)
    assert max_train_batch({"train_sizes": [5_000], "optuna_batch_sizes": [65_536]}) == 65_536
    assert max_train_batch(dict(MSD_1M, optuna_batch_sizes=[4_096, 8_192])) == 8_192
    assert max_train_batch(dict(MSD_1M, optuna_batch_sizes=[8_192], batch_size=16_384)) == 16_384


def test_estimates_cover_measured_peaks():
    # Measured on an RTX 6000 Ada: ml 10M train ~20.5 GB, msd 1M train (batch 16,384) ~16.7 GB, anime Stage 2 with
    # 2 workers at most 23,192 MiB in all (1 s samples; 777 MiB idle), ≈ 10.9 GiB each; the 4-worker spill ≈ 12.5 GB each.
    assert estimate_worker_peak_bytes({"train_sizes": [10_000_000]}, ML_SHAPE) > 20.5 * GB
    assert estimate_worker_peak_bytes(dict(MSD_1M, optuna_batch_sizes=[16_384]), MSD_SHAPE) > 16.7 * GB
    anime = estimate_worker_peak_bytes(STAGE2, ANIME_SHAPE)
    assert 2 * anime > (23_192 - 777) * 1024**2 and anime > 12.5e9
    small = estimate_worker_peak_bytes({"train_sizes": [5_000]}, ML_SHAPE)
    assert small < 2 * GB


def test_dense_qhat_copies_follow_the_reward_budget(monkeypatch):
    assert dense_qhat_copies(STAGE2, ANIME_SHAPE) == 3  # shared lookup + out-of-fold matrix + the refit's transient copy
    assert dense_qhat_copies(dict(STAGE2, crossfit_folds=0), ANIME_SHAPE) == 2
    assert dense_qhat_copies({"reward_data": "external", "crossfit_folds": 0}, ANIME_SHAPE) == 1
    assert dense_qhat_copies({"train_sizes": [5_000]}, ANIME_SHAPE) == 1  # the worker's defaults: external, no cross-fitting
    assert dense_qhat_copies(STAGE2, MSD_SHAPE) == 0  # above the dense limit: q_hat stays in linear form
    monkeypatch.setenv("OPC_QHAT_MATERIALIZE_MAX_GB", "2")  # the trainer's limit: anime's 2.95 GiB q_hat now lazy
    assert dense_qhat_copies(STAGE2, ANIME_SHAPE) == 0


def test_anime_estimate_is_the_calibrated_peak():
    """Training step at batch 8,192 plus three dense q_hat copies plus the per-process context: 12.3 GiB."""
    q_hat = 73_417 * 10_803 * 4
    peak = estimate_worker_peak_bytes(STAGE2, ANIME_SHAPE)
    assert peak == 6 * 8_192 * 10_803 * 4 + 3 * q_hat + CONTEXT_BYTES
    assert 12.3 * GB <= peak <= 12.4 * GB
    assert peak - estimate_worker_peak_bytes(dict(STAGE2, crossfit_folds=0), ANIME_SHAPE) == q_hat


def test_batch_dominated_conditions_are_sized_by_their_batch():
    """With a lazy q_hat the estimate is the batch term alone (its constant is calibrated on measured peaks, allocator
    included): no further margin, and restricting the batch choices lets more workers fit."""
    default = estimate_worker_peak_bytes(MSD_1M, MSD_SHAPE)
    assert default == PEAK_BATCH_MATRICES * 32_768 * 42_053 * 4 + CONTEXT_BYTES  # 32.3 GiB
    small = dict(MSD_1M, optuna_batch_sizes=[8_192])
    assert estimate_worker_peak_bytes(small, MSD_SHAPE) == PEAK_BATCH_MATRICES * 8_192 * 42_053 * 4 + CONTEXT_BYTES
    assert _workers(MSD_SHAPE, 47 * GB, cfg=MSD_1M) == 1
    assert _workers(MSD_SHAPE, 47 * GB, cfg=small) == 3
    assert _workers(MSD_SHAPE, 95 * GB, cfg=MSD_1M, max_workers=64) == 2


def _workers(shape, free_bytes, max_workers=4, n_configs=6, cfg=STAGE2):
    cfgs = [dict(cfg, run_key=f"c{i}") for i in range(n_configs)]
    plan = plan_worker_groups(cfgs, max_workers=max_workers, capacities=[int(free_bytes)], shape_fn=lambda c: shape)
    assert len(plan) == 1 and len(plan[0][1]) == n_configs
    return plan[0][0]


@pytest.mark.parametrize(
    "shape,free,expected",
    [
        (ANIME_SHAPE, 47 * GB, 2),  # was 4: 4 × ~12.5 GB spilled into shared memory on the 48 GB card
        (ANIME_SHAPE, 47_574 * 1024**2, 2),  # the card's free memory as queried (idle, 2026-09-27)
        (KUAIRAND_SHAPE, 47 * GB, 4),  # measured ~16 GB for 4 workers: no spill; --max-workers binds
        (KUAIRAND_SHAPE, 47_574 * 1024**2, 4),
        (ML_SHAPE, 47 * GB, 4),
        (ANIME_SHAPE, 24 * GB, 1),
    ],
)
def test_stage2_worker_counts(shape, free, expected):
    assert _workers(shape, free) == expected


def test_anime_on_a_larger_card():
    assert _workers(ANIME_SHAPE, 80 * GB) > 2
    assert _workers(ANIME_SHAPE, 80 * GB, max_workers=8) > 2
    assert _workers(ANIME_SHAPE, 80 * GB, max_workers=8) == int(SAFETY_FRACTION * 80 * GB // estimate_worker_peak_bytes(STAGE2, ANIME_SHAPE))


def test_budget_changes_the_count():
    """The count follows the configured reward-model budget, not the dataset: without cross-fitting anime holds one
    q_hat copy fewer, and with the external reward model (no refit per size) one fewer again."""
    assert _workers(ANIME_SHAPE, 47 * GB, max_workers=8) == 2
    assert _workers(ANIME_SHAPE, 47 * GB, max_workers=8, cfg=dict(STAGE2, crossfit_folds=0)) == 3
    assert _workers(ANIME_SHAPE, 47 * GB, max_workers=8, cfg=dict(STAGE2, crossfit_folds=0, reward_data="external")) == 5


def test_stage2_plan_mixes_datasets(monkeypatch):
    """The three-dataset Stage 2 launch on the 48 GB card: ml and kuairand run 4 at a time, then anime 2."""
    shapes = {"ml": (6_038, 3_533), "kuairand": KUAIRAND_SHAPE, "anime": ANIME_SHAPE}
    monkeypatch.setattr(mb, "catalog_shape", lambda emb_dir, ds: shapes[ds])
    cfgs = [dict(STAGE2, run_key=f"{ds}{i}", dataset_name=ds, emb_dir="emb") for ds in shapes for i in range(12)]
    plan = plan_worker_groups(cfgs, max_workers=4, capacities=[47 * GB], n_slots=1)
    assert [(w, sorted({c["dataset_name"] for c in g})) for w, g in plan] == [(4, ["kuairand", "ml"]), (2, ["anime"])]
    text = describe_plan(plan, "gpu", [47 * GB], 4, n_slots=1)
    assert "gpu 0: 47.0 free, 35.2 usable (GiB; safety fraction 0.75); requested --max-workers 4" in text
    anime = next(line for line in text.splitlines() if "anime" in line)
    assert "3 dense q_hat copies" in anime and "est. peak 12.3 GiB/worker" in anime
    assert "fit per device [2]" in anime and "12 condition(s) -> 2 concurrent worker(s) (per device [2])" in anime
    kuairand = next(line for line in text.splitlines() if "kuairand" in line)
    assert "fit per device [6]" in kuairand and "-> 4 concurrent worker(s)" in kuairand


def test_a_four_gpu_96gb_workstation(monkeypatch):
    """The same code on four 96 GB cards: every device takes floor(0.75 × free / peak) workers, and --max-workers
    caps the total."""
    shapes = {"anime": ANIME_SHAPE, "kuairand": KUAIRAND_SHAPE, "msd": MSD_SHAPE}
    monkeypatch.setattr(mb, "catalog_shape", lambda emb_dir, ds: shapes[ds])
    cfgs = ([dict(STAGE2, run_key=f"anime{i}", dataset_name="anime", emb_dir="e") for i in range(40)]
            + [dict(STAGE2, run_key=f"kuairand{i}", dataset_name="kuairand", emb_dir="e") for i in range(60)]
            + [dict(MSD_1M, run_key=f"msd{i}", dataset_name="msd", emb_dir="e") for i in range(20)])
    plan = plan_worker_groups(cfgs, max_workers=64, capacities=BIG_WORKSTATION, n_slots=4)
    got = {g[0]["dataset_name"]: w for w, g in plan}
    assert got == {"kuairand": 52, "anime": 20, "msd": 8}  # 13, 5 and 2 per card
    text = describe_plan(plan, "gpu", BIG_WORKSTATION, 64, n_slots=4)
    assert "gpu 3: 95.0 free, 71.2 usable" in text and "(per device [5, 5, 5, 5])" in text
    capped = plan_worker_groups(cfgs[40:100], max_workers=32, capacities=BIG_WORKSTATION, n_slots=4)
    assert [w for w, _ in capped] == [32]
    assert workers_per_device(32, 4) == [8, 8, 8, 8]


def test_workers_fit_round_robin_over_devices():
    assert workers_that_fit(10 * GB, [48 * GB]) == 3  # floor(0.75 * 48 / 10)
    assert workers_that_fit(10 * GB, [48 * GB, 24 * GB]) == 3  # round-robin: 2 on the 48 GB, 1 on the 24 GB (its 2nd won't fit)
    assert workers_that_fit(100 * GB, [48 * GB]) == 1  # never below one worker
    assert workers_per_device(3, 2) == [2, 1]


def test_plan_groups_by_size_and_respects_max_workers():
    small = [{"run_key": f"small{i}", "train_sizes": [5_000]} for i in range(40)]
    big = [{"run_key": "big", "train_sizes": [10_000_000]}]
    cfgs = small[:20] + big + small[20:]
    plan = plan_worker_groups(cfgs, max_workers=30, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    small_fit = workers_that_fit(estimate_worker_peak_bytes(small[0], ML_SHAPE), [48 * GB])
    assert 1 < small_fit < 30  # memory, not --max-workers, is the binding limit here
    assert [w for w, _ in plan] == [small_fit, 1]
    assert [c["run_key"] for c in plan[0][1]] == [c["run_key"] for c in small]
    assert [c["run_key"] for c in plan[1][1]] == ["big"]
    plan = plan_worker_groups(cfgs, max_workers=3, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    assert [w for w, _ in plan] == [3, 1]  # --max-workers stays the upper bound
    plan = plan_worker_groups(small[:2], max_workers=30, capacities=[48 * GB], shape_fn=lambda c: ML_SHAPE)
    assert [w for w, _ in plan] == [2]  # never more workers than conditions


def test_unknown_capacity_keeps_max_workers():
    cfgs = [{"run_key": "a", "train_sizes": [5_000]}]
    assert plan_worker_groups(cfgs, max_workers=7, capacities=None) == [(7, cfgs)]
    assert "capacity unknown" in describe_plan([(7, cfgs)], "gpu", None, 7)


def test_extra_gpu_slots_fall_back_to_device_zero():
    # --num-gpus 2 on a 1-GPU machine: slot 1 maps to device 0, so it holds all workers.
    assert workers_that_fit(10 * GB, [48 * GB], n_slots=2) == 3
    assert workers_per_device(3, 1, n_slots=2) == [3]
    # Two real devices, three slots: device 0 gets slots 0 and 2 (3 workers there, 2 on device 1).
    assert workers_that_fit(10 * GB, [48 * GB, 48 * GB], n_slots=3) == 5
    assert workers_per_device(5, 2, n_slots=3) == [3, 2]


def test_multi_gpu_plan_is_per_device():
    """Two cards: each holds floor(0.75 × its free memory / peak) anime workers; the log names each device."""
    cfgs = [dict(STAGE2, run_key=f"c{i}", dataset_name="anime") for i in range(8)]
    caps = [47 * GB, 24 * GB]
    plan = plan_worker_groups(cfgs, max_workers=8, capacities=caps, n_slots=2, shape_fn=lambda c: ANIME_SHAPE)
    assert [w for w, _ in plan] == [3]  # 2 + 1: the 24 GB card fits one, so round-robin stops at its second
    text = describe_plan(plan, "gpu", caps, 8, n_slots=2, shape_fn=lambda c: ANIME_SHAPE)
    assert "gpu 1: 24.0 free, 18.0 usable" in text and "fit per device [2, 1]" in text and "(per device [2, 1])" in text
    plan = plan_worker_groups(cfgs, max_workers=8, capacities=[47 * GB, 47 * GB], n_slots=2, shape_fn=lambda c: ANIME_SHAPE)
    assert [w for w, _ in plan] == [4]


def test_parallel_runner_uses_the_plan(monkeypatch, capsys):
    """run_full_study_parallel runs each group with the planned worker count and prints the plan."""
    import training.run_full_study_parallel as par

    shapes = {"kuairand": KUAIRAND_SHAPE, "anime": ANIME_SHAPE}
    monkeypatch.setattr(mb, "catalog_shape", lambda emb_dir, ds: shapes[ds])
    monkeypatch.setattr(par, "device_capacities", lambda num_gpus: ("gpu", [47 * GB]))
    calls = []
    monkeypatch.setattr(par, "_run_configs_with_oom_backoff",
                        lambda cfgs, *, max_workers, **kw: calls.append((max_workers, [c["dataset_name"] for c in cfgs])) or [])
    cfgs = [dict(STAGE2, run_key=f"{ds}{i}", dataset_name=ds, emb_dir="emb") for ds in ("anime", "kuairand") for i in range(6)]
    failures = par._run_with_memory_cap(cfgs, max_workers=4, min_workers=1, num_gpus=1, memory_cap=True,
                                        fail_fast=False, oom_backoff=True)
    assert failures == []
    assert calls == [(4, ["kuairand"] * 6), (2, ["anime"] * 6)]
    out = capsys.readouterr().out
    assert "anime (73,417 × 10,803; 3 dense q_hat copies): est. peak 12.3 GiB/worker" in out
    # two cards, --num-gpus 3: slot 2 falls back to device 0, which then hosts 3 of the 4 kuairand workers
    monkeypatch.setattr(par, "device_capacities", lambda num_gpus: ("gpu", [47 * GB, 47 * GB]))
    calls.clear()
    par._run_with_memory_cap(cfgs[6:], max_workers=4, min_workers=1, num_gpus=3, memory_cap=True, fail_fast=False,
                             oom_backoff=True)
    assert calls == [(4, ["kuairand"] * 6)]
    assert "-> 4 concurrent worker(s) (per device [3, 1])" in capsys.readouterr().out


def _dense_held(lookup) -> int:
    from training.trainer_trials import CrossFitScoresLookup

    if isinstance(lookup, CrossFitScoresLookup):
        return int(lookup.full_lookup.q_hat_all is not None) + int(lookup._dense is not None)
    return int(lookup.q_hat_all is not None)


@pytest.mark.parametrize(
    "reward_data,folds,limit_gb,steady,peak",
    [("train", 2, None, 2, 3), ("train", 0, None, 1, 2), ("external", 0, None, 1, 1), ("train", 2, "0", 0, 0)],
)
def test_copies_are_what_the_trainer_holds(tmp_path, monkeypatch, reward_data, folds, limit_gb, steady, peak):
    """The planner's dense q_hat count is what a real (toy) study run holds. During training, the loss's lookup holds
    two with cross-fitting (the shared lookup and the out-of-fold matrix), one without, none above the dense limit.
    At its peak, while the next train size's lookup is built, the run holds one more when q_hat is refit per size.
    On CUDA each holder is its own device copy."""
    import gc
    import weakref

    from test_reproducibility import _toy_embeddings

    import training.trainer_trials as tt
    from training.run_full_study import _run_condition

    if limit_gb is not None:
        monkeypatch.setenv("OPC_QHAT_MATERIALIZE_MAX_GB", limit_gb)
    _toy_embeddings(tmp_path)
    held, real_train = [], tt.train

    def spy(model, loader, scores, **kw):
        held.append(_dense_held(scores))
        return real_train(model, loader, scores, **kw)

    live, alive_at_build = weakref.WeakSet(), []

    def track(cls, dense):
        real_init = cls.__init__

        def init(self, *a, **kw):
            real_init(self, *a, **kw)
            if dense(self):
                live.add(self)
            gc.collect()  # count only what is still referenced
            alive_at_build.append(len(live))

        monkeypatch.setattr(cls, "__init__", init)

    monkeypatch.setattr(tt, "train", spy)
    track(tt.RegressionScoresLookup, lambda s: s.q_hat_all is not None)
    track(tt.CrossFitScoresLookup, lambda s: s._dense is not None)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000, 2000],
                   n_trials=1, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                   policy_reward_mode="exact", policy_reward_mc_sim=8, slim=True, shared_regression_size=2000,
                   methods=("opc", "dm"), run_dir=run_dir, reward_data=reward_data, crossfit_folds=folds)
    cfg = {"train_sizes": [1000, 2000], "reward_data": reward_data, "crossfit_folds": folds}
    assert len(held) == 4 and set(held) == {steady}  # two arms × two train sizes × one trial
    assert max(alive_at_build) == peak == dense_qhat_copies(cfg, catalog_shape(tmp_path, "toy"))
