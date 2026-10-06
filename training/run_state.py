"""The state of a study's condition folders: which rows a request expects, whether a folder already holds them, and
how a run's rows join the folder's files (training/run_full_study.py and training/run_full_study_parallel.py).

A condition folder accumulates rows from any number of invocations of the runners. Every summary row carries
``arm_config_key``, a hash of the settings that determine its arm's results (``arm_config``). For a request, an arm is
complete in a folder when every label it writes (``arm_labels``: one per method, or one per CausE prediction and rho,
or per BLOB family and prior variant) has a summary row for every requested train size with the request's key. A run
executes only the incomplete arms and replaces exactly the rows it produces, keyed by (label, train size), in every
file of the folder; all other rows stay. A row of a requested label and size that holds a different key is a conflict:
the runners stop before running anything unless ``--no-skip-completed`` asks to replace it. Rows written before the key
existed carry none and count as matching.

The per-arm trial and run logs (``<arm>_trials_long.csv``, ``<arm>_runs_long.csv``) are appended one train size at a
time while an arm trains; an arm's rows for the sizes it is about to run are removed first (``reset_arm_logs``), so a
rerun or a resumed condition never repeats them. ``trials_long.csv`` and ``runs_long.csv`` are rebuilt from every
arm's logs (``rebuild_long_logs``). The stable identity of a logged trial is ``TRIAL_KEY`` (of a run, ``RUN_KEY``); the
analysis loaders read logs through ``dedupe_trials`` / ``dedupe_runs``, which keep the last copy, so files written
before these rules (one run holds duplicated, identical trials: docs/handoff_20261006.md §5) load correctly.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# arms trained by training/trainer_trials.py (one summary label each, the method's name) and their log-file prefixes
OPC_FAMILY_ARMS = {"opc": "opc", "no_propensity": "no_prop", "dm": "dm", "tempered_logger": "tempered_logger",
                   "shared_likelihood": "shared_likelihood", "shared_iw_likelihood": "shared_iw_likelihood",
                   "shared_iw_likelihood_clip10": "shared_iw_likelihood_clip10", "shared_opc": "shared_opc"}
TRIAL_KEY = ("method", "train_size", "run", "trial_number")
RUN_KEY = ("method", "train_size", "run")
ROW_KEY = ("method", "train_size")  # summary rows and the per-label trial files of the CausE and BLOB arms
CONFIG_KEY_COLUMN = "arm_config_key"
PAIRED_COLUMN = "opc_vs_noprop_pct"  # training.metrics_utils.add_paired_method_pct_columns
# settings that determine every arm's results; the folder name holds dataset, bias, CTR, seed, the non-default world
# options, the reward-model data, cross-fitting and a fixed validation size
COMMON_SETTINGS = ("sampler", "stage", "val_size", "val_frac", "val_min", "val_max", "shared_regression_size",
                   "logging_uniform_mix", "world_options", "reward_data", "crossfit_folds", "deterministic",
                   "cpu_threads")
# ... and those of the arms trained by trainer_trials (one set for all four: conservative for the simpler arms)
OPC_FAMILY_SETTINGS = ("policy_loss_types", "opc_gradient", "train_weights", "sn_scope", "select_weights",
                       "policy_transform", "learn_logit_scale", "post_temper", "search_space", "optuna_selection",
                       "optuna_batch_sizes", "batch_size", "reward_model", "reward_features", "slim",
                       "policy_reward_mode", "policy_reward_mc_sim")
# arm options that do not change a label's results: hardware, which other labels run, logging and saved files
CAUSE_UNKEYED = ("device", "rhos", "variants", "n_trials", "policy_dir")
BLOB_UNKEYED = ("device", "variants", "families", "n_trials", "policy_dir", "pick_diagnostics")


# ------------------------------------------------------------------------------------------------- what a request expects
def arm_labels(method: str, cause_options: dict | None = None, blob_options: dict | None = None) -> tuple[str, ...]:
    """The summary labels one arm writes: the method itself, or one per CausE prediction and rho
    (``cause_method_label``), or one per BLOB family and prior variant (``blob_method_label``)."""
    if method in OPC_FAMILY_ARMS:
        return (method,)
    if method == "cause":
        from models.cause import CAUSE_PREDICTIONS
        from training.cause_trials import CAUSE_DEFAULTS, FAMILY_PREDICTIONS, cause_method_label

        o = {**CAUSE_DEFAULTS, **(cause_options or {})}
        family = str(o["family"])
        if family == "native":
            preds = [p for v in o["variants"] for p, (pv, _side) in CAUSE_PREDICTIONS.items() if pv == v]
        else:
            preds = list(FAMILY_PREDICTIONS)
        return tuple(cause_method_label(p, r, family) for r in o["rhos"] for p in preds)
    if method == "blob":
        from training.blob_trials import BLOB_DEFAULTS, blob_method_label

        o = {**BLOB_DEFAULTS, **(blob_options or {})}
        return tuple(blob_method_label(f, v) for v in o["variants"] for f in o["families"])
    raise ValueError(f"unknown study method {method!r}")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def arm_config(method: str, cfg: dict) -> dict:
    """The settings of a condition config (as the runners build it) that determine one arm's results."""
    from training.trainer_trials import resolve_search_space

    out = {k: cfg.get(k) for k in COMMON_SETTINGS}
    n_trials = int(cfg["n_trials"])
    if method in OPC_FAMILY_ARMS:
        out.update({k: cfg.get(k) for k in OPC_FAMILY_SETTINGS})
        out["search_space"] = resolve_search_space(cfg.get("search_space"))
        if method.startswith("shared_"):  # the source anchor's candidate strengths (docs/shared_objective_study.md)
            from models.shared_objectives import LAMBDA_GRID

            out["shared"] = {"lambdas": [float(x) for x in LAMBDA_GRID], **(cfg.get("shared_options") or {})}
    elif method == "cause":
        from training.cause_trials import CAUSE_DEFAULTS

        o = {**CAUSE_DEFAULTS, **(cfg.get("cause_options") or {})}
        n_trials = int(o.get("n_trials") or n_trials)
        out["cause"] = {k: v for k, v in o.items() if k not in CAUSE_UNKEYED}
    elif method == "blob":
        from training.blob_trials import BLOB_DEFAULTS

        o = {**BLOB_DEFAULTS, **(cfg.get("blob_options") or {})}
        n_trials = int(o.get("n_trials") or n_trials)
        out["blob"] = {k: v for k, v in o.items() if k not in BLOB_UNKEYED}
    else:
        raise ValueError(f"unknown study method {method!r}")
    out["arm"] = method
    out["n_trials"] = n_trials
    return _jsonable(out)


def config_key(config: dict) -> str:
    """A short, stable hash of an arm configuration (prefixed so CSV readers keep it a string)."""
    text = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return "cfg-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------------------------- reading and comparing rows
def read_csv_or_none(path: Path, **kw) -> pd.DataFrame | None:
    """The CSV at ``path``; None when it does not exist, an empty frame when it is empty. Other read errors raise:
    a file that cannot be read must not be treated as missing (its rows would be dropped). Floats are parsed exactly
    (``round_trip``): the runners rewrite these files, and pandas' default parser can move a value by one unit in
    the last place, so kept rows would change on every merge."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False, float_precision="round_trip", **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_summary(path: Path) -> pd.DataFrame | None:
    return read_csv_or_none(path, dtype={CONFIG_KEY_COLUMN: str})


def _norm(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if np.isfinite(v) and float(v).is_integer() else float(v)
    return str(v)


def _key_tuples(df: pd.DataFrame, key) -> list[tuple]:
    return [tuple(_norm(v) for v in row) for row in df[list(key)].itertuples(index=False, name=None)]


def replace_rows(existing: pd.DataFrame | None, new: pd.DataFrame | None, key=ROW_KEY) -> pd.DataFrame:
    """``existing`` without the rows whose ``key`` occurs in ``new``, followed by ``new``: the rows a run produces
    replace that run's earlier rows of the same key, and every other row stays."""
    if new is None or new.empty:
        return existing if existing is not None else pd.DataFrame()
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    missing = [c for c in key if c not in existing.columns or c not in new.columns]
    if missing:
        raise ValueError(f"cannot merge rows without the key columns {missing}")
    produced = set(_key_tuples(new, key))
    keep = [k not in produced for k in _key_tuples(existing, key)]
    return pd.concat([existing[keep], new], ignore_index=True, sort=False)


def dedupe_rows(df: pd.DataFrame, key) -> pd.DataFrame:
    """One row per ``key`` (the columns of it that exist; ``method`` is required), keeping the last copy: a rerun's."""
    cols = [c for c in key if c in df.columns]
    if df.empty or "method" not in cols or len(cols) < 2:
        return df
    return df.drop_duplicates(cols, keep="last")


def dedupe_trials(df: pd.DataFrame) -> pd.DataFrame:
    return dedupe_rows(df, TRIAL_KEY)


def dedupe_runs(df: pd.DataFrame) -> pd.DataFrame:
    return dedupe_rows(df, RUN_KEY)


def read_trials_long(path: Path, **kw) -> pd.DataFrame:
    """A trial log with one row per trial (``TRIAL_KEY``, the last copy kept). ``usecols`` gains the key columns."""
    usecols = kw.pop("usecols", None)
    if usecols is not None and not callable(usecols):
        usecols = list(dict.fromkeys([*usecols, *TRIAL_KEY]))
        kw["usecols"] = lambda c, wanted=frozenset(usecols): c in wanted
    elif callable(usecols):
        kw["usecols"] = lambda c, f=usecols: f(c) or c in TRIAL_KEY
    return dedupe_trials(pd.read_csv(path, **kw))


# ------------------------------------------------------------------------------------------------------ status of a folder
def arm_status(summary: pd.DataFrame | None, labels, train_sizes, key: str) -> tuple[str, list]:
    """``("complete", [])``, ``("pending", [(label, size) missing])`` or ``("conflict", [(label, size, keys)])``."""
    if summary is None or summary.empty or "method" not in summary.columns or "train_size" not in summary.columns:
        return "pending", [(label, int(n)) for label in labels for n in train_sizes]
    have: dict[tuple, set] = {}
    keys = summary[CONFIG_KEY_COLUMN] if CONFIG_KEY_COLUMN in summary.columns else pd.Series([None] * len(summary))
    for (label, n), k in zip(_key_tuples(summary, ROW_KEY), keys):
        have.setdefault((label, n), set())
        if isinstance(k, str) and k:
            have[(label, n)].add(k)
    missing, conflicts = [], []
    for label in labels:
        for n in train_sizes:
            found = have.get((str(label), int(n)))
            if found is None:
                missing.append((label, int(n)))
            elif found and found != {key}:
                conflicts.append((label, int(n), sorted(found)))
    if conflicts:
        return "conflict", conflicts
    return ("pending", missing) if missing else ("complete", [])


def condition_plan(cfg: dict, requested, *, skip_completed: bool) -> dict:
    """For one condition config: ``pending`` (the arms to run), ``complete`` (skipped), ``conflicts`` (arms whose rows
    were made with other settings; with ``skip_completed`` they block the run, otherwise they are replaced),
    ``detail`` (a readable description of each conflict) and ``options`` (``cause_options`` / ``blob_options``
    narrowed to the missing labels: a pending CausE arm runs only the rhos and predictions it lacks, a BLOB arm only
    the prior variants and families it lacks, so completed rows stay as they are)."""
    run_dir = Path(cfg["run_dir"])
    summary = read_summary(run_dir / "summary_metrics.csv")
    meta = _read_json(run_dir / "run_meta.json") or {}
    sizes = [int(n) for n in cfg["train_sizes"]]
    out = {"pending": [], "complete": [], "conflicts": [], "detail": [], "options": {}}
    for method in requested:
        labels = arm_labels(method, cfg.get("cause_options"), cfg.get("blob_options"))
        config = arm_config(method, cfg)
        key = config_key(config)
        status, info = arm_status(summary, labels, sizes, key)
        if status == "complete" and skip_completed:
            out["complete"].append(method)
            continue
        if status == "conflict":
            out["conflicts"].append(method)
            for label, n, found in info:
                diff = _config_diff((meta.get("arm_configs") or {}).get(found[0]), config)
                out["detail"].append(f"{run_dir.name}: {label} at train size {n} holds {', '.join(found)}, the request "
                                     f"is {key}" + (f" (differs in {diff})" if diff else ""))
        elif status == "pending" and skip_completed:
            missing = {label for label, _n in info}
            if missing != set(labels):  # some labels are done: run only the groups that lack one
                out["options"].update(_narrowed_options(method, cfg, missing))
        out["pending"].append(method)
    return out


def _narrowed_options(method: str, cfg: dict, missing: set) -> dict:
    """The arm's options restricted to the groups of labels that hold a missing one: CausE rhos and native
    variants (a variant's predictions come from one model), BLOB prior variants and families. Labels are trained
    independently across these groups (tests/test_resume_idempotence.py), so a restricted run reproduces them."""
    if method == "cause":
        from training.cause_trials import CAUSE_DEFAULTS

        given = dict(cfg.get("cause_options") or {})
        full = {**CAUSE_DEFAULTS, **given}
        hit = lambda **o: bool(missing & set(arm_labels("cause", {**full, **o})))
        narrowed = {"rhos": [r for r in full["rhos"] if hit(rhos=[r])]}
        if str(full["family"]) == "native":
            narrowed["variants"] = [v for v in full["variants"] if hit(variants=[v])]
        return {"cause_options": {**given, **narrowed}}
    if method == "blob":
        from training.blob_trials import BLOB_DEFAULTS

        given = dict(cfg.get("blob_options") or {})
        full = {**BLOB_DEFAULTS, **given}
        hit = lambda **o: bool(missing & set(arm_labels("blob", blob_options={**full, **o})))
        return {"blob_options": {**given, "variants": [v for v in full["variants"] if hit(variants=[v])],
                                 "families": [f for f in full["families"] if hit(families=[f])]}}
    return {}


def _config_diff(old: dict | None, new: dict) -> str:
    if not old:
        return ""
    keys = sorted(set(old) | set(new))
    return ", ".join(f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in keys if old.get(k) != new.get(k))


# ----------------------------------------------------------------------------------------------------- writing the folder
def atomic_write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    """Write ``df`` to a temporary file next to ``path`` and rename it into place, so a reader or an interrupted run
    never sees half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_write_json(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_csv(path: Path, df: pd.DataFrame) -> None:
    """Append rows to a CSV log. When the new rows' columns differ from the file's header, the file is rewritten with
    the union of the columns (an append under a different header would shift every value into the wrong column)."""
    if path is None or df is None or df.empty:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    if header == [str(c) for c in df.columns]:
        df.to_csv(path, mode="a", header=False, index=False)
        return
    atomic_write_csv(pd.concat([read_csv_or_none(path), df], ignore_index=True, sort=False), path)


def arm_log_paths(run_dir: Path, method: str) -> dict:
    prefix = OPC_FAMILY_ARMS[method]
    run_dir = Path(run_dir)
    return {"trials": run_dir / f"{prefix}_trials_long.csv", "runs": run_dir / f"{prefix}_runs_long.csv"}


def reset_arm_logs(log_paths: dict, method: str, train_sizes) -> None:
    """Before an arm trains: remove its rows for these train sizes from its trial and run logs (left by an earlier or
    interrupted attempt), so the attempt's appends are the only copy."""
    sizes = {int(n) for n in train_sizes}
    for path in log_paths.values():
        df = read_csv_or_none(path)
        if df is None:
            continue
        if df.empty:
            Path(path).unlink()
            continue
        if "method" not in df.columns or "train_size" not in df.columns:
            continue
        drop = (df["method"].astype(str) == str(method)) & pd.to_numeric(df["train_size"], errors="coerce").isin(sizes)
        if not drop.any():
            continue
        rest = df[~drop]
        if rest.empty:
            Path(path).unlink()
        else:
            atomic_write_csv(rest, path)


def rebuild_long_logs(run_dir: Path, summary: pd.DataFrame | None = None) -> None:
    """``trials_long.csv`` and ``runs_long.csv`` from every arm's logs in the folder, one row per trial / run. With
    ``summary``, only the (label, train size) pairs it holds: an interrupted arm's partial trials stay out until the
    arm completes."""
    run_dir = Path(run_dir)
    done = set(_key_tuples(summary, ROW_KEY)) if summary is not None and not summary.empty else None
    for kind, key in (("trials", TRIAL_KEY), ("runs", RUN_KEY)):
        frames = [read_csv_or_none(arm_log_paths(run_dir, m)[kind]) for m in OPC_FAMILY_ARMS]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            continue
        df = dedupe_rows(pd.concat(frames, ignore_index=True, sort=False), key)
        if done is not None:
            df = df[[k in done for k in _key_tuples(df, ROW_KEY)]]
        atomic_write_csv(df, run_dir / f"{kind}_long.csv")


def merge_summary(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """The folder's summary after a run: its rows replaced, all others kept, the paired OPC / no-propensity column
    recomputed over the merged rows."""
    from training.metrics_utils import add_paired_method_pct_columns

    strip = lambda df: None if df is None else df.drop(columns=[PAIRED_COLUMN], errors="ignore")
    merged = replace_rows(strip(existing), strip(new), ROW_KEY)
    if {"opc", "no_propensity"}.issubset(set(merged.get("method", pd.Series(dtype=str)).astype(str))):
        merged = add_paired_method_pct_columns(merged)
    return merged


def merge_run_meta(old: dict | None, new: dict, labels: dict, configs: dict) -> dict:
    """``run_meta.json`` after a run: the condition fields of the latest run; ``study_methods`` every arm the folder
    holds; ``cause`` / ``blob`` the latest options given for them; ``labels`` each label's arm, configuration key, code
    commit and time; ``arm_configs`` the configuration of each key in use."""
    out = dict(new)
    old = old or {}
    out["study_methods"] = list(dict.fromkeys([*(old.get("study_methods") or []), *(new.get("study_methods") or [])]))
    for k in ("cause", "blob", "shared"):
        if out.get(k) is None and old.get(k) is not None:
            out[k] = old[k]
    merged_labels = {**(old.get("labels") or {}), **labels}
    merged_configs = {**(old.get("arm_configs") or {}), **configs}
    used = {v.get("config_key") for v in merged_labels.values()}
    out["labels"] = merged_labels
    out["arm_configs"] = {k: v for k, v in merged_configs.items() if k in used}
    return out


@contextmanager
def condition_lock(run_dir: Path):
    """Exclusive use of a condition folder (an advisory lock; a no-op where ``fcntl`` is unavailable): two invocations
    that reach the same folder run it one after the other, and the second sees the first's rows."""
    try:
        import fcntl
    except ImportError:  # not POSIX
        yield
        return
    path = Path(run_dir) / ".condition.lock"
    with open(path, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except OSError:  # a filesystem without locks
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def record_invocation(out_dir: Path, record: dict) -> None:
    """Append one line per runner invocation to ``run_invocations.jsonl`` (``run_manifest.json`` holds the latest)."""
    with open(Path(out_dir) / "run_invocations.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(record)) + "\n")
