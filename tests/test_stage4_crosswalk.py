"""Stage 4 verification: state harmonisation across sources.

An unmatched record is a silently dropped observation, so these tests assert
that every spelling observed in real data resolves, and that the match rate is
reported rather than assumed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hsre.monitoring import ledger
from hsre.transform.nigeria_states import (
    CANONICAL_STATES,
    coverage_by_state,
    harmonise,
    normalise,
)


def test_canonical_set_is_36_states_plus_fct():
    assert len(CANONICAL_STATES) == 37
    assert "FCT" in CANONICAL_STATES


def test_acled_spellings_resolve():
    """Every ADMIN1 value observed in the real ACLED Nigeria extract."""
    observed = [
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
        "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
        "Enugu", "Federal Capital Territory", "Gombe", "Imo", "Jigawa",
        "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos",
        "Nassarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau",
        "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
    ]
    assert len(observed) == 37
    for value in observed:
        assert normalise(value) is not None, f"ACLED value unresolved: {value}"


def test_ucdp_spellings_resolve():
    """Every adm_1 value observed in the real UCDP Nigeria extract."""
    observed = [
        "Borno state", "Plateau state", "Benue state", "Yobe state",
        "Adamawa state", "Taraba state", "Kaduna state", "Lagos state",
        "Rivers state", "Imo state", "Nasarawa state", "Delta state",
        "Anambra state", "Edo state", "Kano state", "Ogun state",
        "Enugu state", "Bauchi state", "Kogi state", "Bayelsa state",
        "Ebonyi state", "Osun state", "Oyo state", "Sokoto state",
        "Abia state", "Akwa Ibom state", "Gombe state", "Ondo state",
        "Kwara state", "Niger state", "Zamfara state",
        "Federal Capital territory", "Jigawa state", "Cross River state",
        "Kebbi state", "Katsina state", "Ekiti state",
    ]
    for value in observed:
        assert normalise(value) is not None, f"UCDP value unresolved: {value}"


def test_nigeria_watch_spellings_resolve():
    """State list observed in the Nigeria Watch search interface."""
    observed = [
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa",
        "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti",
        "Enugu", "FCT (Abuja)", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano",
        "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger",
        "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto",
        "Taraba", "Yobe", "Zamfara",
    ]
    for value in observed:
        assert normalise(value) is not None, f"Nigeria Watch value unresolved: {value}"


def test_the_three_sources_agree_on_nasarawa():
    """ACLED writes Nassarawa, UCDP writes Nasarawa state, Nigeria Watch
    writes Nasarawa. All three are the same locality."""
    assert normalise("Nassarawa") == normalise("Nasarawa state") == normalise("Nasarawa")


def test_the_three_sources_agree_on_fct():
    assert (
        normalise("Federal Capital Territory")
        == normalise("Federal Capital territory")
        == normalise("FCT (Abuja)")
        == "FCT"
    )


def test_dissolved_state_is_excluded_deliberately():
    """Gongola was dissolved in 1991. It is mapped to None explicitly so it
    is not counted as an unmatched record."""
    assert normalise("Gongola state") is None


def test_unknown_value_returns_none():
    assert normalise("Atlantis") is None
    assert normalise("") is None
    assert normalise(None) is None


def test_case_and_whitespace_are_tolerated():
    assert normalise("  BORNO  ") == "Borno"
    assert normalise("lagos state") == "Lagos"


def test_harmonise_reports_match_rate(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    frame = pd.DataFrame({"ADMIN1": ["Borno", "Nassarawa", "Atlantis", "Lagos"]})
    out, rate = harmonise(frame, "ADMIN1", "acled")
    assert rate == 0.75
    assert len(out) == 3
    assert set(out["state"]) == {"Borno", "Nasarawa", "Lagos"}


def test_low_match_rate_is_flagged_in_the_ledger(tmp_path, monkeypatch):
    """Crosswalk decay must be visible. An unmatched record is a dropped
    observation, not a neutral event."""
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", path)
    frame = pd.DataFrame({"ADMIN1": ["Borno", "Atlantis", "Narnia", "Oz"]})
    harmonise(frame, "ADMIN1", "acled", min_match_rate=0.95)
    entries = [e for e in ledger.read_ledger(path) if e.get("stage") == "transform"]
    assert entries[-1]["status"] == ledger.STATUS_QUARANTINED
    assert entries[-1]["match_rate"] == 0.25
    assert "Atlantis" in entries[-1]["unmatched_values"]


def test_coverage_reports_states_with_no_events():
    """A state absent from a source is the finding when comparing coverage,
    so it must appear as zero rather than vanish."""
    frame = pd.DataFrame({"state": ["Borno"] * 5 + ["Lagos"] * 2})
    counts = coverage_by_state(frame)
    assert len(counts) == 37
    assert counts["Borno"] == 5
    assert counts["Katsina"] == 0
