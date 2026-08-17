"""Shared noise level / axis tables for study + SNR characterization."""

from __future__ import annotations

NOISE_LEVEL_COMBINED: dict[str, tuple[float, float, float]] = {
    "low": (0.05, 0.05, 0.0),
    "medium": (0.10, 0.15, 0.05),
    "high": (0.20, 0.25, 0.10),
    # Harder logging damage (more GT wiped by noise templates).
    "extreme": (0.35, 0.40, 0.20),
    "brutal": (0.50, 0.50, 0.30),
}

NOISE_LEVEL_PER_AXIS: dict[str, dict[str, float]] = {
    "context": {
        "low": 0.05,
        "medium": 0.10,
        "high": 0.20,
        "extreme": 0.35,
        "brutal": 0.50,
    },
    "action": {
        "low": 0.05,
        "medium": 0.15,
        "high": 0.25,
        "extreme": 0.40,
        "brutal": 0.50,
    },
    "metadata": {
        "low": 0.0,
        "medium": 0.05,
        "high": 0.10,
        "extreme": 0.20,
        "brutal": 0.30,
    },
}

VALID_NOISE_AXES = ("combined", "context", "action", "metadata")
VALID_NOISE_LEVELS = tuple(NOISE_LEVEL_COMBINED.keys())


def noise_level_to_eps(level: str) -> tuple[float, float, float]:
    if level not in NOISE_LEVEL_COMBINED:
        raise ValueError(f"Unsupported noise level '{level}'")
    return NOISE_LEVEL_COMBINED[level]


def noise_eps(level: str, axis: str) -> tuple[float, float, float]:
    """Return (eps1, eps2, eps_meta) for the requested axis at a given level.

    - axis="combined" matches the legacy bundled mapping.
    - axis in {"context", "action", "metadata"} perturbs only that axis.
    """
    if axis not in VALID_NOISE_AXES:
        raise ValueError(f"Unsupported noise axis '{axis}'")
    if axis == "combined":
        return noise_level_to_eps(level)
    table = NOISE_LEVEL_PER_AXIS[axis]
    if level not in table:
        raise ValueError(f"Unsupported noise level '{level}' for axis '{axis}'")
    mag = float(table[level])
    if axis == "context":
        return (mag, 0.0, 0.0)
    if axis == "action":
        return (0.0, mag, 0.0)
    return (0.0, 0.0, mag)
