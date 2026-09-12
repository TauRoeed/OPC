"""Shared noise level / axis / component tables for study + SNR characterization.

Two independent knobs
--------------------
``noise_axis`` (where)
    Which embedding catalog is corrupted:
    - ``context`` — user factors ``our_x`` only
    - ``action`` — item factors ``our_a`` only
    - ``combined`` — both
    - ``metadata`` — *legacy alias*: both catalogs, metadata **component** only
      (prefer ``--noise-axes combined --noise-components metadata``)

``noise_component`` (what)
    Which mixture term is active (magnitudes from the level table):
    - ``linear`` / ``general`` — global linear warp + Gaussian (``eps1``)
    - ``cluster`` — k-means / random cluster templates (``eps2``)
    - ``metadata`` — side-info projection (``eps_meta``)
    - ``combined`` — all three terms at that level

Legacy bug (fixed)
------------------
Older ``noise_eps(level, axis)`` mapped ``context`` → linear-only and
``action`` → cluster-only, then applied those ε to **both** user and item.
That confused "where" with "what". Callers should now pass axis + component.
"""

from __future__ import annotations

NOISE_LEVEL_COMBINED: dict[str, tuple[float, float, float]] = {
    "low": (0.05, 0.05, 0.0),
    "medium": (0.10, 0.15, 0.05),
    "high": (0.20, 0.25, 0.10),
    # Harder logging damage (more GT wiped by noise templates).
    "extreme": (0.35, 0.40, 0.20),
    "brutal": (0.50, 0.50, 0.30),
    # Beyond brutal — most of the clean signal replaced by noise templates.
    "catastrophic": (0.70, 0.70, 0.45),
}

# Per-component magnitudes at each level (same numbers as the combined table columns).
NOISE_LEVEL_PER_COMPONENT: dict[str, dict[str, float]] = {
    "linear": {
        "low": 0.05,
        "medium": 0.10,
        "high": 0.20,
        "extreme": 0.35,
        "brutal": 0.50,
        "catastrophic": 0.70,
    },
    "cluster": {
        "low": 0.05,
        "medium": 0.15,
        "high": 0.25,
        "extreme": 0.40,
        "brutal": 0.50,
        "catastrophic": 0.70,
    },
    "metadata": {
        "low": 0.0,
        "medium": 0.05,
        "high": 0.10,
        "extreme": 0.20,
        "brutal": 0.30,
        "catastrophic": 0.45,
    },
}

# Back-compat alias used by older docs / SNR scripts.
NOISE_LEVEL_PER_AXIS = {
    "context": NOISE_LEVEL_PER_COMPONENT["linear"],
    "action": NOISE_LEVEL_PER_COMPONENT["cluster"],
    "metadata": NOISE_LEVEL_PER_COMPONENT["metadata"],
}

VALID_NOISE_AXES = ("combined", "context", "action", "metadata")
VALID_NOISE_COMPONENTS = ("combined", "linear", "general", "cluster", "metadata")
VALID_NOISE_LEVELS = tuple(NOISE_LEVEL_COMBINED.keys())

# Aliases: talk name → code name
_COMPONENT_ALIASES = {
    "general": "linear",
    "linear": "linear",
    "cluster": "cluster",
    "metadata": "metadata",
    "combined": "combined",
}


def normalize_noise_component(component: str) -> str:
    key = str(component).lower().strip()
    if key not in _COMPONENT_ALIASES:
        raise ValueError(
            f"Unsupported noise component '{component}'. "
            f"Expected one of {VALID_NOISE_COMPONENTS}"
        )
    return _COMPONENT_ALIASES[key]


def noise_level_to_eps(level: str) -> tuple[float, float, float]:
    if level not in NOISE_LEVEL_COMBINED:
        raise ValueError(f"Unsupported noise level '{level}'")
    return NOISE_LEVEL_COMBINED[level]


def noise_target_sides(axis: str) -> tuple[bool, bool]:
    """Return ``(apply_user, apply_item)`` for a noise axis.

    - ``context``: users only
    - ``action``: items only
    - ``combined`` / ``metadata`` (legacy): both catalogs
    """
    if axis not in VALID_NOISE_AXES:
        raise ValueError(f"Unsupported noise axis '{axis}'")
    if axis == "context":
        return True, False
    if axis == "action":
        return False, True
    return True, True


def resolve_noise_spec(
    level: str,
    axis: str = "combined",
    component: str = "combined",
) -> dict:
    """Resolve level × axis × component into ε and target sides.

    Returns dict with keys:
      ``eps1``, ``eps2``, ``eps_meta``, ``apply_user``, ``apply_item``,
      ``axis``, ``component``, ``level``.
    """
    if axis not in VALID_NOISE_AXES:
        raise ValueError(f"Unsupported noise axis '{axis}'")
    if level not in NOISE_LEVEL_COMBINED:
        raise ValueError(f"Unsupported noise level '{level}'")

    comp = normalize_noise_component(component)
    # Legacy: ``axis=metadata`` means metadata component on both catalogs.
    if axis == "metadata" and comp == "combined":
        comp = "metadata"

    apply_user, apply_item = noise_target_sides(axis)
    eps1, eps2, eps_meta = noise_level_to_eps(level)

    if comp == "linear":
        eps1, eps2, eps_meta = float(eps1), 0.0, 0.0
    elif comp == "cluster":
        eps1, eps2, eps_meta = 0.0, float(eps2), 0.0
    elif comp == "metadata":
        eps1, eps2, eps_meta = 0.0, 0.0, float(eps_meta)
    # combined: keep all three from the level table

    return {
        "level": level,
        "axis": axis,
        "component": comp,
        "eps1": float(eps1),
        "eps2": float(eps2),
        "eps_meta": float(eps_meta),
        "apply_user": bool(apply_user),
        "apply_item": bool(apply_item),
    }


def noise_eps(
    level: str,
    axis: str = "combined",
    component: str = "combined",
) -> tuple[float, float, float]:
    """Return ``(eps1, eps2, eps_meta)`` for level × axis × component.

    ``axis`` alone no longer remaps context→linear / action→cluster.
    Use ``component`` to isolate a mixture term and ``axis`` for where it lands
    (via :func:`noise_target_sides` / :func:`resolve_noise_spec`).
    """
    spec = resolve_noise_spec(level, axis=axis, component=component)
    return spec["eps1"], spec["eps2"], spec["eps_meta"]
