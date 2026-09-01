"""Unit tests for OOM detection and worker backoff in run_full_study_parallel."""

import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from training.run_full_study_parallel import (
    _is_oom_like,
    _run_configs_with_oom_backoff,
)


def _cfg(key: str) -> dict:
    return {"run_key": key}


def _thread_pool_factory(workers, mp_ctx, worker_slot, num_gpus):
    return ThreadPoolExecutor(max_workers=workers)


class TestIsOomLike(unittest.TestCase):
    def test_memory_error(self):
        self.assertTrue(_is_oom_like(MemoryError("host OOM")))

    def test_cuda_runtime_error(self):
        self.assertTrue(
            _is_oom_like(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
        )

    def test_outofmemory_type_name(self):
        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(_is_oom_like(OutOfMemoryError("device-side alloc failed")))

    def test_broken_process_pool(self):
        self.assertTrue(
            _is_oom_like(
                BrokenProcessPool(
                    "A process in the process pool was terminated abruptly"
                )
            )
        )

    def test_killed_message(self):
        self.assertTrue(_is_oom_like(RuntimeError("Worker process killed by SIGKILL")))

    def test_non_oom_errors(self):
        self.assertFalse(_is_oom_like(ValueError("bad hyperparameter")))
        self.assertFalse(_is_oom_like(RuntimeError("shape mismatch in loss")))
        self.assertFalse(_is_oom_like(KeyError("missing run_key")))


class TestRunConfigsWithOomBackoff(unittest.TestCase):
    def test_oom_jobs_retried_and_workers_shrink(self):
        attempts: dict[str, int] = {}
        worker_sizes: list[int] = []

        def execute(cfg):
            key = cfg["run_key"]
            attempts[key] = attempts.get(key, 0) + 1
            if key == "ok":
                return None
            if key == "bad":
                raise ValueError("logic bug")
            if key == "oom":
                if attempts[key] == 1:
                    raise RuntimeError("CUDA out of memory")
                return None
            raise AssertionError(f"unexpected {key}")

        def factory(workers, mp_ctx, worker_slot, num_gpus):
            worker_sizes.append(workers)
            return ThreadPoolExecutor(max_workers=workers)

        failures = _run_configs_with_oom_backoff(
            [_cfg("ok"), _cfg("oom"), _cfg("bad")],
            max_workers=3,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=True,
            execute_fn=execute,
            executor_factory=factory,
        )

        self.assertEqual(attempts["ok"], 1)
        self.assertEqual(attempts["oom"], 2)
        self.assertEqual(attempts["bad"], 1)
        self.assertEqual(worker_sizes, [3, 2])
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["run_key"], "bad")
        self.assertIn("ValueError", failures[0]["error"])

    def test_non_oom_failure_not_retried(self):
        attempts = {"x": 0}

        def execute(cfg):
            attempts["x"] += 1
            raise RuntimeError("shape mismatch")

        failures = _run_configs_with_oom_backoff(
            [_cfg("x")],
            max_workers=2,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=True,
            execute_fn=execute,
            executor_factory=_thread_pool_factory,
        )

        self.assertEqual(attempts["x"], 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["run_key"], "x")

    def test_fails_at_min_workers_floor(self):
        worker_sizes: list[int] = []

        def execute(cfg):
            raise MemoryError("cannot allocate")

        def factory(workers, mp_ctx, worker_slot, num_gpus):
            worker_sizes.append(workers)
            return ThreadPoolExecutor(max_workers=workers)

        failures = _run_configs_with_oom_backoff(
            [_cfg("a"), _cfg("b")],
            max_workers=3,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=True,
            execute_fn=execute,
            executor_factory=factory,
        )

        self.assertEqual(worker_sizes, [3, 2, 1])
        self.assertEqual({f["run_key"] for f in failures}, {"a", "b"})
        self.assertTrue(all("MemoryError" in f["error"] for f in failures))

    def test_succeeds_after_retry_at_min_workers(self):
        worker_sizes: list[int] = []

        def execute(cfg):
            if worker_sizes[-1] > 1:
                raise RuntimeError("CUDA out of memory")
            return None

        def factory(workers, mp_ctx, worker_slot, num_gpus):
            worker_sizes.append(workers)
            return ThreadPoolExecutor(max_workers=workers)

        failures = _run_configs_with_oom_backoff(
            [_cfg("job")],
            max_workers=3,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=True,
            execute_fn=execute,
            executor_factory=factory,
        )

        self.assertEqual(failures, [])
        self.assertEqual(worker_sizes, [3, 2, 1])

    def test_oom_backoff_disabled_records_failure(self):
        attempts = 0

        def execute(cfg):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("CUDA out of memory")

        failures = _run_configs_with_oom_backoff(
            [_cfg("oom")],
            max_workers=2,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=False,
            execute_fn=execute,
            executor_factory=_thread_pool_factory,
        )

        self.assertEqual(attempts, 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["run_key"], "oom")

    def test_broken_process_pool_from_result_retries(self):
        attempts = 0
        worker_sizes: list[int] = []

        def execute(cfg):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise BrokenProcessPool("terminated abruptly")
            return None

        def factory(workers, mp_ctx, worker_slot, num_gpus):
            worker_sizes.append(workers)
            return ThreadPoolExecutor(max_workers=workers)

        failures = _run_configs_with_oom_backoff(
            [_cfg("job")],
            max_workers=2,
            min_workers=1,
            num_gpus=1,
            fail_fast=False,
            oom_backoff=True,
            execute_fn=execute,
            executor_factory=factory,
        )

        self.assertEqual(failures, [])
        self.assertEqual(attempts, 2)
        self.assertEqual(worker_sizes, [2, 1])

    def test_broken_process_pool_outer_requues_unfinished(self):
        """Outer BrokenProcessPool handler re-queues unfinished + OOM'd jobs."""

        def as_completed_raises_bpp(futures):
            for fut in futures:
                if fut.done():
                    yield fut
            raise BrokenProcessPool(
                "A process in the process pool was terminated abruptly"
            )

        waves: list[int] = []
        execute_calls: list[str] = []

        def execute(cfg):
            execute_calls.append(cfg["run_key"])
            return None

        class BrokenPoolOuterExecutor:
            def __init__(self):
                self._done_ok = Future()
                self._done_ok.set_result(None)
                self._unfinished = Future()
                self._boom = Future()
                self._boom.set_exception(BrokenProcessPool("pool dead"))

            def submit(self, fn, cfg):
                return {
                    "done": self._done_ok,
                    "unfinished": self._unfinished,
                    "boom": self._boom,
                }[cfg["run_key"]]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def factory(workers, mp_ctx, worker_slot, num_gpus):
            waves.append(workers)
            if len(waves) == 1:
                return BrokenPoolOuterExecutor()
            return ThreadPoolExecutor(max_workers=workers)

        with mock.patch(
            "training.run_full_study_parallel.as_completed",
            as_completed_raises_bpp,
        ):
            failures = _run_configs_with_oom_backoff(
                [_cfg("done"), _cfg("unfinished"), _cfg("boom")],
                max_workers=3,
                min_workers=1,
                num_gpus=1,
                fail_fast=False,
                oom_backoff=True,
                execute_fn=execute,
                executor_factory=factory,
            )

        self.assertEqual(failures, [])
        self.assertEqual(waves[0], 3)
        self.assertEqual(waves[1], 2)
        self.assertEqual(set(execute_calls), {"unfinished", "boom"})

    def test_fail_fast_reraises_non_oom(self):
        def execute(cfg):
            raise ValueError("fatal")

        with self.assertRaisesRegex(ValueError, "fatal"):
            _run_configs_with_oom_backoff(
                [_cfg("x")],
                max_workers=2,
                min_workers=1,
                num_gpus=1,
                fail_fast=True,
                oom_backoff=True,
                execute_fn=execute,
                executor_factory=_thread_pool_factory,
            )


if __name__ == "__main__":
    unittest.main()
