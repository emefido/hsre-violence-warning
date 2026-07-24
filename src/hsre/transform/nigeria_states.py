"""Nigerian state harmonisation.

Every outcome source spells Nigerian states differently. Observed variants:

    ACLED          "Nassarawa",  "Federal Capital Territory"
    UCDP GED       "Nasarawa state",  "Federal Capital territory"
    Nigeria Watch  "Nasarawa",  "FCT (Abuja)"

An unmatched record is a silently dropped observation, so the match rate is
emitted as a health metric rather than assumed. Where it falls below the floor
in config/geographies.yml, the affected localities are excluded from alerting
rather than scored on partial data.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from hsre.monitoring import ledger

# Canonical set: 36 states plus the Federal Capital Territory.
CANONICAL_STATES = [
    "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue",
    "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT",
    "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
    "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo",
    "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara",
]

# Variants that normalisation alone cannot resolve. Keys are already
# lowercased and stripped of the " state" suffix.
ALIASES = {
    "nassarawa": "Nasarawa",
    "nasarawa": "Nasarawa",
    "federal capital territory": "FCT",
    "federal capital": "FCT",
    "fct (abuja)": "FCT",
    "fct abuja": "FCT",
    "abuja": "FCT",
    "akwa-ibom": "Akwa Ibom",
    "cross-river": "Cross River",
    "akwaibom": "Akwa Ibom",
    "crossriver": "Cross River",
    # Dissolved in 1991 into Adamawa and Taraba. Outside the study window,
    # mapped explicitly so it is not counted as an unmatched record.
    "gongola": None,
}

_LOOKUP = {name.lower(): name for name in CANONICAL_STATES}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalise(value: object) -> str | None:
    """Map one source-specific state name to its canonical form.

    Returns None for values that cannot be resolved, including deliberate
    exclusions such as dissolved states.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = _strip_accents(str(value)).strip().lower()
    if not text:
        return None

    # Drop a trailing "state" or "province" qualifier.
    text = re.sub(r"\s+(state|province)$", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in ALIASES:
        return ALIASES[text]
    return _LOOKUP.get(text)


def harmonise(
    frame: pd.DataFrame,
    column: str,
    source_name: str,
    min_match_rate: float = 0.95,
    target_column: str = "state",
) -> tuple[pd.DataFrame, float]:
    """Add a canonical state column and report the match rate.

    Returns the frame with unmatched rows removed and the match rate, which is
    recorded in the ledger so that crosswalk decay is visible rather than
    silent.
    """
    working = frame.copy()
    working[target_column] = working[column].map(normalise)

    total = len(working)
    matched = int(working[target_column].notna().sum())
    rate = matched / total if total else 0.0

    unmatched = (
        working.loc[working[target_column].isna(), column]
        .astype("string")
        .value_counts()
        .head(20)
        .to_dict()
    )

    status = ledger.STATUS_OK if rate >= min_match_rate else ledger.STATUS_QUARANTINED
    ledger.record(
        stage="transform",
        status=status,
        source=source_name,
        crosswalk="nigeria_state",
        rows_in=total,
        rows_matched=matched,
        match_rate=round(rate, 4),
        min_match_rate=min_match_rate,
        unmatched_values=unmatched,
    )

    return working.loc[working[target_column].notna()].copy(), rate


def coverage_by_state(
    frame: pd.DataFrame, state_column: str = "state"
) -> pd.Series:
    """Event counts per canonical state, including states with none.

    States absent from a source are reported as zero rather than omitted,
    because an absent state is the finding when comparing source coverage.
    """
    counts = frame[state_column].value_counts()
    return counts.reindex(CANONICAL_STATES, fill_value=0).sort_values(ascending=False)
