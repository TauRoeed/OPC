"""Memory-aware worker planning (scheduling only; no GPU needed)."""

import pytest

import training.memory_budget as mb
from training.memory_budget import (
    ALLOCATOR_RESERVE,
    CONTEXT_BYTES,
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


def test_max_batch_follows_optuna_schedule():
    assert max_train_batch({"train_sizes": [5_000]}) == 2048
    assert max_train_batch({"train_sizes": [5_000, 10_000_000]}) == 327_680
    assert max_train_batch({"train_sizes": [5_000], "optuna_batch_sizes": [65_536]}) == 65_536
    assert max_train_batch(STAGE2) == 8_192


def test_estimates_cover_measured_peaks():
    # Measured on an RTX 6000 Ada: ml 10M train ~20.5 GB, msd 1M train ~16.7 GB, anime (Stage 2) ≈ 12.5 GB.
    assert estimate_worker_peak_bytes({"train_sizes": [10_000_000]}, ML_SHAPE) > 20.5 * GB
    assert estimate_worker_peak_bytes({"train_sizes": [1_000_000]}, MSD_SHAPE) > 16.7 * GB
    assert estimate_worker_peak_bytes(STAGE2, ANIME_SHAPE) >= 12.5 * GB
    small = estimate_worker_peak_bytes({"train_sizes": [5_000]}, ML_SHAPE)
    assert small < 2 * GB


def test_dense_qhat_copies_follow_the_reward_budget(monkeypatch):
    assert dense_qhat_copies(STAGE2, ANIME_SHAPE) == 2  # shared lookup + out-of-fold matrix
    assert dense_qhat_copies(dict(STAGE2, crossfit_folds=0), ANIME_SHAPE) == 1
    assert dense_qhat_copies({"reward_data": "external", "crossfit_folds": 0}, ANIME_SHAPE) == 1
    assert dense_qhat_copies({"train_sizes": [5_000]}, ANIME_SHAPE) == 1  # the worker's default: no cross-fitting
    assert dense_qhat_copies(STAGE2, MSD_SHAPE) == 0  # above the dense limit: q_hat stays in linear form
    monkeypatch.setenv("OPC_QHAT_MATERIALIZE_MAX_GB", "2")  # the trainer's limit: anime's 2.95 GiB q_hat now lazy
    assert dense_qhat_copies(STAGE2, ANIME_SHAPE) == 0


def test_anime_estimate_is_the_calibrated_peak():
    """Training step at batch 8,192 plus two dense q_hat copies (7.9 GiB live), with the allocator reserve and the
    per-process context: ≈ 12.5 GiB, the observed per-worker peak (4 workers demanded ~50 GB and spilled)."""
    live = 6 * 8_192 * 10_803 * 4 + 2 * 73_417 * 10_803 * 4
    peak = estimate_worker_peak_bytes(STAGE2, ANIME_SHAPE)
    assert peak == int(ALLOCATOR_RESERVE * live + CONTEXT_BYTES)
    assert 12.5 * GB <= peak <= 13.0 * GB
    # without cross-fitting one q_hat copy fewer; with a lazy q_hat none
    one = estimate_worker_peak_bytes(dict(STAGE2, crossfit_folds=0), ANIME_SHAPE)
    assert abs((peak - one) - ALLOCATOR_RESERVE * 73_417 * 10_803 * 4) <= 1


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
    """The count follows the configured reward-model budget, not the dataset: without cross-fitting anime holds
    one q_hat copy (more workers fit), and with a lazy q_hat none."""
    assert _workers(ANIME_SHAPE, 47 * GB, max_workers=8, cfg=dict(STAGE2, crossfit_folds=0)) == 4
    assert _workers(ANIME_SHAPE, 47 * GB, max_workers=8) == 2


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
    assert "2 dense q_hat copies" in anime and "est. peak 12.5 GiB/worker" in anime
    assert "fit per device [2]" in anime and "12 condition(s) -> 2 concurrent worker(s) (per device [2])" in anime
    kuairand = next(line for line in text.splitlines() if "kuairand" in line)
    assert "fit per device [6]" in kuairand and "-> 4 concurrent worker(s)" in kuairand


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
    assert "anime (73,417 × 10,803; 2 dense q_hat copies): est. peak 12.5 GiB/worker" in out
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


@pytest.mark.parametrize("folds,limit_gb", [(2, None), (0, None), (2, "0"), (0, "0")])
def test_copies_are_what_the_trainer_holds(tmp_path, monkeypatch, folds, limit_gb):
    """The planner's dense q_hat count is what the training loss's lookup holds in a real (toy) study run:
    two with cross-fitting (the shared lookup and the out-of-fold matrix), one without, none when q_hat is
    above the dense-materialize limit."""
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

    monkeypatch.setattr(tt, "train", spy)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _run_condition(dataset_name="toy", emb_dir=tmp_path, bias="medium", ctr=0.05, seed=0, train_sizes=[1000, 2000],
                   n_trials=1, batch_size=None, val_size=1000, val_frac=0.15, val_min=1000, val_max=None,
                   policy_reward_mode="exact", policy_reward_mc_sim=8, slim=True, shared_regression_size=2000,
                   methods=("opc", "dm"), run_dir=run_dir, reward_data="train", crossfit_folds=folds)
    cfg = {"train_sizes": [1000, 2000], "reward_data": "train", "crossfit_folds": folds}
    assert len(held) == 4  # two arms × two train sizes × one trial
    assert set(held) == {dense_qhat_copies(cfg, catalog_shape(tmp_path, "toy"))}
