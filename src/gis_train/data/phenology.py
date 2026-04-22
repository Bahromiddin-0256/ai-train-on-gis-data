"""Phenological acquisition-window constants for Uzbekistan crop classification.

Internal representation stores window boundaries as (month, day) integer tuples
so the data are year-agnostic.  Call ``get_stack_windows(year)`` or
``get_crop_windows(crop, year)`` to obtain YYYY-MM-DD strings for a specific
growing season.

The module-level ``CROP_OPTIMAL_WINDOWS`` and ``STACK_WINDOWS`` exports use
year 2025 to match the existing ``configs/data/uzbekistan_s2.yaml`` date range.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Internal year-agnostic tables
# Each window is ((start_month, start_day), (end_month, end_day)).
# ---------------------------------------------------------------------------

_CROP_WINDOWS_MD: dict[str, list[tuple[tuple[int, int], tuple[int, int]]]] = {
    "wheat":       [((4,  1), (5, 31))],
    "cotton":      [((7, 15), (8, 31))],
    "rice":        [((7,  1), (8, 31))],
    "maize":       [((7,  1), (8, 15))],
    "grapes":      [((7,  1), (8, 31))],
    "melons":      [((6,  1), (7, 31))],
    "stone_fruit": [((4, 15), (5, 31))],
    "fallow":      [((3, 15), (4, 15))],
}

# Six-window temporal stack covering the full Uzbekistan growing season.
_STACK_WINDOWS_MD: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((3, 15), (4, 15)),   # baseline
    ((4, 15), (5, 31)),   # wheat peak
    ((6,  1), (6, 30)),   # transition
    ((7, 15), (8, 31)),   # summer peak
    ((9,  1), (9, 30)),   # late season
    ((10, 1), (11, 15)),  # senescence
]

# Uzbek crop-name aliases → canonical English name.
_ALIASES: dict[str, str] = {
    "bugdoy": "wheat",   # Uzbek: winter wheat
    "paxta":  "cotton",  # Uzbek: cotton
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _fmt(year: int, month: int, day: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def get_crop_windows(crop: str, year: int) -> list[tuple[str, str]]:
    """Return the optimal acquisition windows for *crop* in the given *year*.

    Parameters
    ----------
    crop:
        Canonical English crop name (e.g. ``"wheat"``) or a recognised Uzbek
        alias (``"bugdoy"``, ``"paxta"``).  Case-insensitive.
    year:
        Four-digit calendar year for the growing season (e.g. 2025).

    Returns
    -------
    list[tuple[str, str]]
        Each element is a ``(date_start, date_end)`` pair of ISO date strings.

    Raises
    ------
    KeyError
        If *crop* is not found in the registry after alias resolution.
    """
    resolved = _ALIASES.get(crop.lower(), crop.lower())
    if resolved not in _CROP_WINDOWS_MD:
        known = sorted(set(_CROP_WINDOWS_MD) | set(_ALIASES))
        raise KeyError(f"unknown crop {crop!r}; known names: {known}")
    return [
        (_fmt(year, s[0], s[1]), _fmt(year, e[0], e[1]))
        for s, e in _CROP_WINDOWS_MD[resolved]
    ]


def get_stack_windows(year: int) -> list[tuple[str, str]]:
    """Return the six-window phenological stack for *year*.

    Covers the full Uzbekistan growing season from the bare-soil baseline
    (mid-March) through post-harvest senescence (mid-November).

    Parameters
    ----------
    year:
        Four-digit calendar year for the growing season (e.g. 2025).

    Returns
    -------
    list[tuple[str, str]]
        Six ``(date_start, date_end)`` pairs in chronological order.
    """
    return [
        (_fmt(year, s[0], s[1]), _fmt(year, e[0], e[1]))
        for s, e in _STACK_WINDOWS_MD
    ]


def format_windows_cli(windows: list[tuple[str, str]]) -> str:
    """Serialise *windows* to the CLI string format expected by build_dataset.py.

    Parameters
    ----------
    windows:
        List of ``(date_start, date_end)`` ISO date string pairs.

    Returns
    -------
    str
        Comma-separated ``"YYYY-MM-DD:YYYY-MM-DD"`` pairs, e.g.
        ``"2025-03-15:2025-04-15,2025-04-15:2025-05-31"``.
    """
    return ",".join(f"{s}:{e}" for s, e in windows)


# ---------------------------------------------------------------------------
# Convenience constants for year 2025 (matches uzbekistan_s2.yaml date range)
# ---------------------------------------------------------------------------

CROP_OPTIMAL_WINDOWS: dict[str, list[tuple[str, str]]] = {
    crop: get_crop_windows(crop, 2025) for crop in _CROP_WINDOWS_MD
}

STACK_WINDOWS: list[tuple[str, str]] = get_stack_windows(2025)

__all__ = [
    "CROP_OPTIMAL_WINDOWS",
    "STACK_WINDOWS",
    "format_windows_cli",
    "get_crop_windows",
    "get_stack_windows",
]
