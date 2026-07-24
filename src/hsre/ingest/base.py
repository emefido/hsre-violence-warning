"""Base source adapter.

Every source inherits from SourceAdapter so that retry policy, the
visible-failure rule and checksummed immutable storage are written once and
cannot be skipped by an individual adapter.

Two rules govern this layer and both come from the manuscript.

First, adapters must fail visibly. A successful network response containing an
empty or truncated payload is not success. An adapter that returns silently on
an empty response produces a quiet week that is indistinguishable from a quiet
place, and in conflict settings that confusion fails in a dangerous direction.

Second, the raw landing zone is immutable. Files are stored under
source/retrieval-date paths with a SHA256 checksum. Transformation code reads
raw data and writes elsewhere; it never overwrites the original. This is what
permits reproduction when a source later revises its historical records.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from hsre.config import REPO_ROOT, Source
from hsre.monitoring import ledger

RAW_ROOT = REPO_ROOT / "data" / "raw"

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_TIMEOUT_S = 60


class NonRetryable(RuntimeError):
    """Errors that retrying cannot fix, such as a missing credential.

    Retrying these wastes time and, where a source enforces a request quota,
    spends budget on requests that were always going to fail.
    """


class SourceDown(RuntimeError):
    """Raised when an adapter exhausts its retries.

    Distinct from an empty result. A source that is down is a known unknown;
    the pipeline continues on remaining sources and marks the affected
    forecasts degraded.
    """

    def __init__(self, source: str, attempts: int, last_error: str):
        self.source = source
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"source '{source}' failed after {attempts} attempts: {last_error}"
        )


class EmptyPayload(SourceDown):
    """Raised when a request succeeds but returns nothing usable.

    Subclasses SourceDown deliberately: an empty payload is a failure, not a
    quiet period, and must be handled identically by callers.
    """

    def __init__(self, source: str, detail: str = "payload contained no records"):
        RuntimeError.__init__(self, f"source '{source}': {detail}")
        self.source = source
        self.attempts = 0
        self.last_error = detail


@dataclass
class FetchResult:
    """Outcome of one adapter run."""

    source: str
    status: str
    row_count: int
    raw_path: Path | None = None
    checksum: str | None = None
    retrieval_time: str = field(default_factory=ledger.utc_now)
    attempts: int = 1
    duration_s: float = 0.0
    window_start: date | None = None
    window_end: date | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == ledger.STATUS_OK


class SourceAdapter(ABC):
    """One adapter per source.

    Subclasses implement `fetch_payload` and `count_records`. Everything else,
    including retry, storage, checksumming and ledger emission, is handled
    here.
    """

    def __init__(
        self,
        source: Source,
        raw_root: Path | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        sleep_fn=time.sleep,
    ):
        self.source = source
        self.raw_root = raw_root or RAW_ROOT
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.timeout_s = timeout_s
        self._sleep = sleep_fn

    @property
    def name(self) -> str:
        return self.source.name

    @abstractmethod
    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        """Retrieve the raw bytes for the requested window.

        Implementations should raise on transport failure so that retry
        applies. They must not swallow errors and return empty bytes.
        """

    @abstractmethod
    def count_records(self, payload: bytes) -> int:
        """Number of records in the payload, used for the emptiness check."""

    def file_extension(self) -> str:
        return "json" if self.source.route == "api" else "csv"

    def raw_path_for(self, retrieval: datetime) -> Path:
        """Immutable, date-partitioned destination for one retrieval."""
        stamp = retrieval.strftime("%Y%m%dT%H%M%SZ")
        return (
            self.raw_root
            / self.name
            / retrieval.strftime("%Y-%m-%d")
            / f"{self.name}_{stamp}.{self.file_extension()}"
        )

    def _attempt_with_retry(
        self, window_start: date | None, window_end: date | None
    ) -> tuple[bytes, int]:
        """Retry with exponential backoff. Raises SourceDown on exhaustion."""
        last_error = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            try:
                payload = self.fetch_payload(window_start, window_end)
            except NonRetryable:
                # Credential and configuration errors are not transient.
                raise
            except Exception as exc:  # noqa: BLE001 - transport errors vary by route
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_base_s * (2 ** (attempt - 1)))
                    continue
                raise SourceDown(self.name, attempt, last_error) from exc
            return payload, attempt
        raise SourceDown(self.name, self.max_attempts, last_error)

    def run(
        self,
        window_start: date | None = None,
        window_end: date | None = None,
    ) -> FetchResult:
        """Fetch, verify non-emptiness, store immutably, and record health."""
        started = time.monotonic()
        retrieval = datetime.now(timezone.utc)

        try:
            payload, attempts = self._attempt_with_retry(window_start, window_end)
        except SourceDown as exc:
            result = FetchResult(
                source=self.name,
                status=ledger.STATUS_DOWN,
                row_count=0,
                attempts=exc.attempts,
                duration_s=round(time.monotonic() - started, 3),
                window_start=window_start,
                window_end=window_end,
                detail=exc.last_error,
            )
            self._emit(result)
            raise

        row_count = self.count_records(payload)
        if row_count == 0:
            result = FetchResult(
                source=self.name,
                status=ledger.STATUS_DOWN,
                row_count=0,
                attempts=attempts,
                duration_s=round(time.monotonic() - started, 3),
                window_start=window_start,
                window_end=window_end,
                detail="empty payload treated as failure, not as absence of events",
            )
            self._emit(result)
            raise EmptyPayload(self.name)

        raw_path = self.raw_path_for(retrieval)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable raw file: {raw_path}"
            )
        raw_path.write_bytes(payload)

        result = FetchResult(
            source=self.name,
            status=ledger.STATUS_OK,
            row_count=row_count,
            raw_path=raw_path,
            checksum=ledger.file_checksum(raw_path),
            retrieval_time=retrieval.isoformat(),
            attempts=attempts,
            duration_s=round(time.monotonic() - started, 3),
            window_start=window_start,
            window_end=window_end,
        )
        self._emit(result)
        return result

    def _emit(self, result: FetchResult) -> dict[str, Any]:
        return ledger.record(
            stage="ingest",
            status=result.status,
            source=result.source,
            row_count=result.row_count,
            attempts=result.attempts,
            duration_s=result.duration_s,
            raw_path=str(result.raw_path) if result.raw_path else None,
            checksum=result.checksum,
            window_start=result.window_start,
            window_end=result.window_end,
            detail=result.detail,
        )
