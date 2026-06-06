"""tqdm wrappers for chunked user×action matrix evaluation."""

from __future__ import annotations

from tqdm import tqdm

# Show bars when the virtual matrix is at least this many cells.
MATRIX_PROGRESS_MIN_CELLS = 1_000_000


def _use_progress(show_progress: bool, n_rows: int, n_actions: int) -> bool:
    if not show_progress:
        return False
    return int(n_rows) * int(n_actions) >= MATRIX_PROGRESS_MIN_CELLS


def iter_user_action_blocks(
    n_rows: int,
    n_actions: int,
    user_chunk: int,
    action_chunk: int,
    *,
    desc: str = "matrix eval",
    show_progress: bool = False,
):
    """Yield (us, ue, a0, a1) over a virtual (n_rows × n_actions) grid in blocks."""
    use = _use_progress(show_progress, n_rows, n_actions)
    user_starts = range(0, int(n_rows), int(user_chunk))
    if use:
        user_starts = tqdm(
            user_starts,
            desc=f"{desc} | users",
            unit="u-blk",
            leave=True,
        )
    for us in user_starts:
        ue = min(int(n_rows), int(us) + int(user_chunk))
        action_starts = range(0, int(n_actions), int(action_chunk))
        if use:
            action_starts = tqdm(
                action_starts,
                desc=f"{desc} | actions",
                unit="a-blk",
                leave=False,
            )
        for a0 in action_starts:
            a1 = min(int(n_actions), int(a0) + int(action_chunk))
            yield int(us), int(ue), int(a0), int(a1)


def iter_action_blocks(
    n_actions: int,
    action_chunk: int,
    *,
    desc: str = "matrix eval",
    show_progress: bool = False,
    n_rows: int = 1,
):
    """Yield (a0, a1) over actions; use n_rows for progress threshold only."""
    use = _use_progress(show_progress, n_rows, n_actions)
    action_starts = range(0, int(n_actions), int(action_chunk))
    if use:
        action_starts = tqdm(
            action_starts,
            desc=f"{desc} | actions",
            unit="a-blk",
            leave=True,
        )
    for a0 in action_starts:
        a1 = min(int(n_actions), int(a0) + int(action_chunk))
        yield int(a0), int(a1)
