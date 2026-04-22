"""Unit tests for ``gis_train.data.phenology``."""

from __future__ import annotations

import re

import pytest

from gis_train.data.phenology import (
    CROP_OPTIMAL_WINDOWS,
    STACK_WINDOWS,
    format_windows_cli,
    get_crop_windows,
    get_stack_windows,
)

# ---------------------------------------------------------------------------
# get_stack_windows
# ---------------------------------------------------------------------------


def test_get_stack_windows_returns_six_pairs() -> None:
    assert len(get_stack_windows(2025)) == 6


def test_get_stack_windows_all_start_with_year() -> None:
    for start, end in get_stack_windows(2025):
        assert start.startswith("2025-")
        assert end.startswith("2025-")


def test_get_stack_windows_different_years_differ() -> None:
    w2024 = get_stack_windows(2024)
    w2025 = get_stack_windows(2025)
    assert w2024[0][0].startswith("2024-")
    assert w2025[0][0].startswith("2025-")
    assert w2024 != w2025


def test_get_stack_windows_first_window_is_baseline() -> None:
    assert get_stack_windows(2025)[0] == ("2025-03-15", "2025-04-15")


def test_get_stack_windows_last_window_is_senescence() -> None:
    assert get_stack_windows(2025)[-1] == ("2025-10-01", "2025-11-15")


# ---------------------------------------------------------------------------
# get_crop_windows
# ---------------------------------------------------------------------------


def test_get_crop_windows_wheat_2025() -> None:
    assert get_crop_windows("wheat", 2025) == [("2025-04-01", "2025-05-31")]


def test_get_crop_windows_cotton_2025() -> None:
    assert get_crop_windows("cotton", 2025) == [("2025-07-15", "2025-08-31")]


def test_get_crop_windows_uzbek_alias_bugdoy() -> None:
    assert get_crop_windows("bugdoy", 2025) == get_crop_windows("wheat", 2025)


def test_get_crop_windows_uzbek_alias_paxta() -> None:
    assert get_crop_windows("paxta", 2025) == get_crop_windows("cotton", 2025)


def test_get_crop_windows_case_insensitive() -> None:
    assert get_crop_windows("BUGDOY", 2025) == get_crop_windows("wheat", 2025)
    assert get_crop_windows("Paxta", 2025) == get_crop_windows("cotton", 2025)
    assert get_crop_windows("Wheat", 2025) == get_crop_windows("wheat", 2025)


def test_get_crop_windows_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown crop"):
        get_crop_windows("banana", 2025)


def test_get_crop_windows_year_changes_date_prefix() -> None:
    assert get_crop_windows("wheat", 2024)[0][0].startswith("2024-")
    assert get_crop_windows("wheat", 2026)[0][0].startswith("2026-")


def test_get_crop_windows_all_crops_have_at_least_one_window() -> None:
    for crop in ["wheat", "cotton", "rice", "maize", "grapes", "melons", "stone_fruit", "fallow"]:
        assert len(get_crop_windows(crop, 2025)) >= 1


# ---------------------------------------------------------------------------
# format_windows_cli
# ---------------------------------------------------------------------------


def test_format_windows_cli_single_window() -> None:
    assert format_windows_cli([("2025-04-01", "2025-05-31")]) == "2025-04-01:2025-05-31"


def test_format_windows_cli_multiple_windows() -> None:
    windows = [("2025-04-01", "2025-05-31"), ("2025-06-01", "2025-07-31")]
    assert format_windows_cli(windows) == "2025-04-01:2025-05-31,2025-06-01:2025-07-31"


def test_format_windows_cli_roundtrips_stack_windows() -> None:
    windows = get_stack_windows(2025)
    cli_str = format_windows_cli(windows)
    pairs = [tuple(p.split(":")) for p in cli_str.split(",")]
    assert pairs == windows


def test_format_windows_cli_no_spaces() -> None:
    assert " " not in format_windows_cli(get_stack_windows(2025))


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_stack_windows_constant_equals_get_stack_windows_2025() -> None:
    assert get_stack_windows(2025) == STACK_WINDOWS


def test_stack_windows_has_six_elements() -> None:
    assert len(STACK_WINDOWS) == 6


def test_crop_optimal_windows_wheat_matches_helper() -> None:
    assert CROP_OPTIMAL_WINDOWS["wheat"] == get_crop_windows("wheat", 2025)


def test_crop_optimal_windows_cotton_matches_helper() -> None:
    assert CROP_OPTIMAL_WINDOWS["cotton"] == get_crop_windows("cotton", 2025)


def test_crop_optimal_windows_has_all_canonical_crops() -> None:
    expected = {"wheat", "cotton", "rice", "maize", "grapes", "melons", "stone_fruit", "fallow"}
    assert set(CROP_OPTIMAL_WINDOWS.keys()) == expected


def test_crop_optimal_windows_no_uzbek_alias_keys() -> None:
    assert "bugdoy" not in CROP_OPTIMAL_WINDOWS
    assert "paxta" not in CROP_OPTIMAL_WINDOWS


def test_all_window_dates_are_iso_format() -> None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for start, end in STACK_WINDOWS:
        assert pattern.match(start), f"{start!r} is not ISO format"
        assert pattern.match(end), f"{end!r} is not ISO format"
    for windows in CROP_OPTIMAL_WINDOWS.values():
        for start, end in windows:
            assert pattern.match(start)
            assert pattern.match(end)
