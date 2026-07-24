"""Concrete source adapters.

Three routes, matching the `route` field in config/sources.yml.

  api     paginated JSON endpoints (UCDP, Census ACS, VIEWS)
  bulk    scheduled file downloads (GDELT, NIBRS, WISQARS, WorldPop)
  export  operator-supplied files for sources with no supported API
          (Nigeria Watch, Afrobarometer)

The export route exists because not every authoritative source publishes a
machine interface. Pretending otherwise would produce a pipeline that cannot
run. What the architecture requires is not that every route is automated, but
that every route is monitored, versioned and checksummed identically. An
operator-supplied file is registered through the same ledger as an API pull.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from hsre.config import Source
from hsre.ingest.base import EmptyPayload, NonRetryable, SourceAdapter

USER_AGENT = "hsre-violence-warning/0.1 (academic research)"


class MissingToken(NonRetryable):
    """Raised when a source declares a token but the environment lacks it.

    Raised before any request is made, so a misconfigured credential does not
    burn requests against the daily quota.
    """

    def __init__(self, source: str, env_var: str):
        super().__init__(
            f"source '{source}' requires an access token in ${env_var}. "
            f"Set it in .env or export it before running."
        )


class ApiAdapter(SourceAdapter):
    """Paginated JSON API source.

    Pagination is driven by the TotalPages value in the response rather than
    by continuing until an empty page. UCDP returns a server error rather than
    an empty set when a page falls outside the bounds of the result, so
    iterating blind would turn the end of the data into a spurious failure.
    """

    page_size = 1000
    max_pages = 2000

    def build_params(
        self, window_start: date | None, window_end: date | None, page: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"pagesize": self.page_size, "page": page}
        if self.source.country_id is not None:
            params["Country"] = self.source.country_id
        # Both filters operate on date_end.
        if window_start:
            params["StartDate"] = window_start.isoformat()
        if window_end:
            params["EndDate"] = window_end.isoformat()
        return params

    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        token_env = self.source.route_setting("token_env")
        if token_env:
            token = os.environ.get(token_env)
            if not token:
                raise MissingToken(self.name, token_env)
            header_name = self.source.route_setting("auth_header") or "Authorization"
            headers[header_name] = token
        return headers

    def extract_records(self, page_body: Any) -> list[dict[str, Any]]:
        if isinstance(page_body, list):
            return page_body
        for key in ("Result", "result", "data", "results"):
            value = page_body.get(key)
            if isinstance(value, list):
                return value
        return []

    def total_pages(self, page_body: Any) -> int | None:
        if isinstance(page_body, dict):
            for key in ("TotalPages", "total_pages"):
                value = page_body.get(key)
                if isinstance(value, int):
                    return value
        return None

    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        records: list[dict[str, Any]] = []
        page = 0
        declared_pages: int | None = None

        while page < self.max_pages:
            response = requests.get(
                self.source.versioned_url,
                params=self.build_params(window_start, window_end, page),
                headers=self.headers(),
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            body = response.json()

            batch = self.extract_records(body)
            records.extend(batch)

            if declared_pages is None:
                declared_pages = self.total_pages(body)

            # Stop on the authoritative page count where the API supplies one,
            # rather than probing past the end of the result set.
            if declared_pages is not None:
                if page + 1 >= declared_pages:
                    break
            elif not batch:
                break
            page += 1

        return json.dumps(records).encode("utf-8")

    def count_records(self, payload: bytes) -> int:
        return len(json.loads(payload.decode("utf-8")))


class BulkAdapter(SourceAdapter):
    """Scheduled file download.

    Handles plain CSV and zipped CSV, since several official releases ship
    zipped. A zip containing no CSV member is a failure rather than an empty
    result.
    """

    def __init__(self, source: Source, url_template: str | None = None, **kwargs):
        super().__init__(source, **kwargs)
        self.url_template = url_template

    def build_url(self, window_start: date | None, window_end: date | None) -> str:
        base = self.source.versioned_url
        if self.url_template and window_start:
            return self.url_template.format(
                base=base,
                date=window_start.strftime("%Y%m%d"),
                year=window_start.year,
            )
        return base

    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        response = requests.get(
            self.build_url(window_start, window_end),
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        content = response.content
        if content[:2] == b"PK":
            return self._extract_csv_from_zip(content)
        return content

    def _extract_csv_from_zip(self, content: bytes) -> bytes:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith((".csv", ".txt"))]
            if not members:
                raise ValueError("zip archive contained no CSV or TXT member")
            return archive.read(members[0])

    def count_records(self, payload: bytes) -> int:
        """Count data rows, respecting quoted fields.

        A naive line count overstates the total whenever a field contains an
        embedded newline. UCDP GED carries full source-article text, so
        multi-line values are routine and the inflated count would corrupt the
        trailing baseline that the thin-source check depends on.
        """
        return _count_csv_rows(payload)


class ExportAdapter(SourceAdapter):
    """Operator-supplied file for sources with no supported API.

    The file is placed in an inbox directory and registered here. It receives
    the same checksum, immutable storage and ledger treatment as an automated
    pull, so provenance is identical even though acquisition is manual.
    """

    def __init__(self, source: Source, inbox: Path, **kwargs):
        super().__init__(source, **kwargs)
        self.inbox = Path(inbox)

    def _locate(self) -> Path:
        if not self.inbox.exists():
            raise FileNotFoundError(f"inbox does not exist: {self.inbox}")
        candidates = sorted(
            [
                p
                for p in self.inbox.iterdir()
                if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".sav"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"no export file found in {self.inbox}")
        return candidates[0]

    def file_extension(self) -> str:
        return "csv"

    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        path = self._locate()
        if path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
            return frame.to_csv(index=False).encode("utf-8")
        return path.read_bytes()

    def count_records(self, payload: bytes) -> int:
        return _count_csv_rows(payload)


def _count_csv_rows(payload: bytes) -> int:
    """Number of data rows in a CSV payload, excluding the header.

    Uses a real CSV reader so that quoted fields containing commas or
    newlines are counted as one row rather than several.
    """
    if not payload.strip():
        return 0
    text = payload.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        next(reader)  # header
    except StopIteration:
        return 0
    return sum(1 for row in reader if any(field.strip() for field in row))


def filter_to_country(frame: pd.DataFrame, country_id: int | None) -> pd.DataFrame:
    """Restrict a global bulk download to one country.

    The API applies this server-side. The bulk archive is global, so the same
    restriction is applied after download to keep the two routes comparable.
    """
    if country_id is None or "country_id" not in frame.columns:
        return frame
    codes = pd.to_numeric(frame["country_id"], errors="coerce")
    return frame.loc[codes == country_id].copy()


def load_frame(result_path: Path, source: Source) -> pd.DataFrame:
    """Read a stored raw file back into a frame for validation."""
    if result_path.suffix == ".json":
        records = json.loads(result_path.read_text(encoding="utf-8"))
        return pd.DataFrame.from_records(records)
    return pd.read_csv(result_path, low_memory=False)


ADAPTER_BY_ROUTE = {
    "api": ApiAdapter,
    "bulk": BulkAdapter,
    "export": ExportAdapter,
}

# Sources whose API contract differs enough to need a dedicated adapter.
# ACLED uses OAuth2 password grant rather than a static token header.
ADAPTER_BY_SOURCE = {
    "acled": {"api": "AcledApiAdapter", "bulk": "AcledBulkAdapter"},
}


def build_adapter(source: Source, **kwargs) -> SourceAdapter:
    """Instantiate the adapter matching a source's active route."""
    route = source.effective_route

    special = ADAPTER_BY_SOURCE.get(source.name, {}).get(route)
    if special:
        from hsre.ingest import acled as acled_mod

        return getattr(acled_mod, special)(source, **kwargs)

    cls = ADAPTER_BY_ROUTE.get(route)
    if cls is None:
        raise ValueError(f"no adapter for route {route!r}")
    if cls is ExportAdapter and "inbox" not in kwargs:
        raise ValueError(f"export source '{source.name}' requires an inbox path")
    return cls(source, **kwargs)
