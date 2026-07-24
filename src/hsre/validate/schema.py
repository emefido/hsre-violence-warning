"""Schema validation and quarantine.

Records that violate a source's declared schema are quarantined rather than
coerced. Sources change their formats without notice, and a pipeline that
quietly accepts a changed format produces wrong answers with full confidence.

Validation returns a report rather than raising, because a partial failure is
normal: some records are malformed while the rest are usable. The caller
decides whether the surviving share meets the completeness threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from hsre.config import REPO_ROOT, Source
from hsre.monitoring import ledger

QUARANTINE_ROOT = REPO_ROOT / "data" / "interim" / "quarantine"

# Reasons a record can be quarantined. Recorded per record so that the failure
# taxonomy in the manuscript can be populated from real counts.
REASON_MISSING_COLUMN = "missing_column"
REASON_TYPE_COERCION = "type_coercion_failed"
REASON_NULL_REQUIRED = "null_in_required_field"
REASON_OUT_OF_RANGE = "value_out_of_range"
REASON_DUPLICATE = "duplicate_record"


class SchemaDrift(RuntimeError):
    """Raised when declared columns are absent from the payload entirely.

    This is distinct from individual bad records. If a required column has
    vanished, the source has changed its format and a human must look at it
    before any data is admitted.
    """

    def __init__(self, source: str, missing: list[str]):
        self.source = source
        self.missing = missing
        super().__init__(
            f"source '{source}' is missing declared columns: {sorted(missing)}"
        )


@dataclass
class ValidationReport:
    source: str
    rows_in: int
    rows_out: int
    quarantined: int
    reasons: dict[str, int] = field(default_factory=dict)
    quarantine_path: Path | None = None

    @property
    def pass_rate(self) -> float:
        if self.rows_in == 0:
            return 0.0
        return self.rows_out / self.rows_in

    @property
    def status(self) -> str:
        if self.rows_out == 0:
            return ledger.STATUS_QUARANTINED
        return ledger.STATUS_OK


_TYPE_CHECKS = {
    "int": lambda s: pd.to_numeric(s, errors="coerce").astype("Float64"),
    "float": lambda s: pd.to_numeric(s, errors="coerce").astype("Float64"),
    "str": lambda s: s.astype("string"),
    "date": lambda s: pd.to_datetime(s, errors="coerce", format="mixed"),
}

# Coordinate bounds. Out-of-range geography is a common defect in event data
# and silently produces localities that do not exist.
_RANGE_CHECKS = {
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
    "ActionGeo_Lat": (-90.0, 90.0),
    "ActionGeo_Long": (-180.0, 180.0),
}


def _coerce_column(series: pd.Series, declared: str) -> tuple[pd.Series, pd.Series]:
    """Return the coerced series and a boolean mask of failures."""
    checker = _TYPE_CHECKS.get(declared)
    if checker is None:
        return series, pd.Series(False, index=series.index)
    coerced = checker(series)
    # A value that was present but became null has failed coercion.
    failed = coerced.isna() & series.notna()
    return coerced, failed


def validate(
    frame: pd.DataFrame,
    source: Source,
    required_non_null: list[str] | None = None,
    dedupe_on: list[str] | None = None,
    quarantine_root: Path | None = None,
    write_quarantine: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate a frame against its declared schema.

    Returns the surviving records and a report. Raises SchemaDrift only when
    declared columns are absent altogether, which requires human intervention.
    """
    declared = source.schema
    missing = [col for col in declared if col not in frame.columns]
    if missing:
        ledger.record(
            stage="validate",
            status=ledger.STATUS_SCHEMA_DRIFT,
            source=source.name,
            missing_columns=missing,
        )
        raise SchemaDrift(source.name, missing)

    rows_in = len(frame)
    working = frame.copy()
    reasons: dict[str, int] = {}
    # Accumulates rows to quarantine, with the reason attached.
    bad_mask = pd.Series(False, index=working.index)
    reason_col = pd.Series("", index=working.index, dtype="object")

    def mark(mask: pd.Series, reason: str) -> None:
        nonlocal bad_mask
        newly = mask & ~bad_mask
        count = int(newly.sum())
        if count:
            reasons[reason] = reasons.get(reason, 0) + count
            reason_col.loc[newly] = reason
            bad_mask = bad_mask | newly

    for column, declared_type in declared.items():
        coerced, failed = _coerce_column(working[column], declared_type)
        working[column] = coerced
        mark(failed, REASON_TYPE_COERCION)

    for column, (low, high) in _RANGE_CHECKS.items():
        if column in working.columns:
            numeric = pd.to_numeric(working[column], errors="coerce")
            out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))
            mark(out_of_range, REASON_OUT_OF_RANGE)

    for column in required_non_null or []:
        if column in working.columns:
            mark(working[column].isna(), REASON_NULL_REQUIRED)

    if dedupe_on:
        present = [c for c in dedupe_on if c in working.columns]
        if present:
            mark(working.duplicated(subset=present, keep="first"), REASON_DUPLICATE)

    quarantined = working.loc[bad_mask].copy()
    surviving = working.loc[~bad_mask].copy()

    quarantine_path: Path | None = None
    if write_quarantine and not quarantined.empty:
        quarantined["quarantine_reason"] = reason_col.loc[bad_mask]
        root = quarantine_root or QUARANTINE_ROOT
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        quarantine_path = root / source.name / f"{source.name}_{stamp}.csv"
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        quarantined.to_csv(quarantine_path, index=False)

    report = ValidationReport(
        source=source.name,
        rows_in=rows_in,
        rows_out=len(surviving),
        quarantined=len(quarantined),
        reasons=reasons,
        quarantine_path=quarantine_path,
    )

    ledger.record(
        stage="validate",
        status=report.status,
        source=source.name,
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        quarantined=report.quarantined,
        pass_rate=round(report.pass_rate, 4),
        reasons=report.reasons,
        quarantine_path=str(quarantine_path) if quarantine_path else None,
    )
    return surviving, report


def check_volume(
    source: Source,
    observed_rows: int,
    trailing_median: float | None,
) -> tuple[str, dict[str, Any]]:
    """Compare observed volume against the source's trailing baseline.

    This is the thin-source check. A source reporting materially less than its
    recent norm has not failed, so nothing raises, but the distinction between
    a quiet source and a quiet place must be recorded or it is lost.
    """
    if trailing_median is None or trailing_median <= 0:
        detail = {"reason": "no trailing baseline available", "observed": observed_rows}
        ledger.record(
            stage="validate",
            status=ledger.STATUS_OK,
            source=source.name,
            volume_check="skipped",
            **detail,
        )
        return ledger.STATUS_OK, detail

    ratio = observed_rows / trailing_median
    status = ledger.STATUS_THIN if ratio < source.volume_floor else ledger.STATUS_OK
    detail = {
        "observed": observed_rows,
        "trailing_median": trailing_median,
        "ratio": round(ratio, 4),
        "floor": source.volume_floor,
    }
    ledger.record(
        stage="validate",
        status=status,
        source=source.name,
        volume_check="applied",
        **detail,
    )
    return status, detail


def trailing_median_rows(
    source_name: str,
    window: int = 12,
    path: Path | None = None,
) -> float | None:
    """Median row count from the last `window` successful ingest runs."""
    counts = [
        entry["row_count"]
        for entry in ledger.read_ledger(path)
        if entry.get("stage") == "ingest"
        and entry.get("source") == source_name
        and entry.get("status") == ledger.STATUS_OK
        and isinstance(entry.get("row_count"), (int, float))
    ]
    if not counts:
        return None
    recent = counts[-window:]
    return float(pd.Series(recent).median())
