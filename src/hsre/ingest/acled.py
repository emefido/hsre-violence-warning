"""ACLED adapter.

ACLED is the primary Nigerian outcome source. Its inclusion criteria are wider
than UCDP's organised-violence threshold, and the difference is not marginal:
in the observed 2016-2024 extract, Katsina rises from 13 UCDP events to 1,504
in ACLED, Zamfara from 27 to 1,684, and Borno's share of all Nigerian events
falls from 56.5% to 14.4%. The gap is almost entirely non-insurgency violence,
which is what this study is about.

Authentication is OAuth2 password grant. A short-lived bearer token is
obtained from the token endpoint and sent on each data request. Tokens are
cached on disk so that a normal run does not re-authenticate needlessly.

Two acquisition routes, as with UCDP:

  api    full disaggregated events, including admin2 (LGA), actor names and
         event narratives. Requires credentials.
  bulk   the weekly aggregated regional file, which is already at
         week x admin1 x event type and needs no credentials. Sufficient for
         a state-week panel but carries no LGA, actors or narratives.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from hsre.config import REPO_ROOT, Source
from hsre.ingest.base import NonRetryable, SourceAdapter

TOKEN_URL = "https://acleddata.com/oauth/token"
TOKEN_CACHE = REPO_ROOT / "data" / "metadata" / ".acled_token.json"

# Refresh a little before nominal expiry so a long paginated pull does not
# fail halfway through on an expired token.
TOKEN_REFRESH_MARGIN_S = 300


class AcledAuthError(NonRetryable):
    """Credential or token failure. Not transient, so it is not retried."""


def _read_cached_token(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    expires_at = cached.get("expires_at", 0)
    if time.time() < expires_at - TOKEN_REFRESH_MARGIN_S:
        return cached.get("access_token")
    return None


def _write_cached_token(path: Path, token: str, expires_in: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "expires_at": time.time() + expires_in,
                "obtained": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    # The token is a credential. Keep it out of other users' reach.
    try:
        path.chmod(0o600)
    except OSError:
        pass


def get_access_token(
    username: str | None = None,
    password: str | None = None,
    cache_path: Path | None = None,
    timeout_s: int = 30,
) -> str:
    """Obtain a bearer token, using the disk cache when it is still valid."""
    cache = cache_path or TOKEN_CACHE
    cached = _read_cached_token(cache)
    if cached:
        return cached

    user = username or os.environ.get("ACLED_USERNAME")
    secret = password or os.environ.get("ACLED_PASSWORD")
    if not user or not secret:
        raise AcledAuthError(
            "ACLED credentials missing. Set ACLED_USERNAME and ACLED_PASSWORD "
            "in .env. Register free at https://acleddata.com/register"
        )

    response = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": user,
            "password": secret,
            "grant_type": "password",
            "client_id": "acled",
        },
        timeout=timeout_s,
    )
    if response.status_code != 200:
        raise AcledAuthError(
            f"ACLED token request failed with {response.status_code}. "
            f"Check ACLED_USERNAME and ACLED_PASSWORD."
        )

    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise AcledAuthError("ACLED token response contained no access_token")
    _write_cached_token(cache, token, int(payload.get("expires_in", 3600)))
    return token


class AcledApiAdapter(SourceAdapter):
    """Paginated ACLED event data.

    Pagination uses page and limit. ACLED returns fewer rows than the limit on
    the final page, and an empty data array past the end, so iteration stops on
    a short page rather than probing for an error.
    """

    page_limit = 5000
    max_pages = 400

    def __init__(self, source: Source, token: str | None = None, **kwargs):
        super().__init__(source, **kwargs)
        self._token = token

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = get_access_token()
        return self._token

    def file_extension(self) -> str:
        return "json"

    def build_params(
        self, window_start: date | None, window_end: date | None, page: int
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "_format": "json",
            "limit": self.page_limit,
            "page": page,
        }
        country = self.source.route_setting("country_name")
        if country:
            params["country"] = country

        # event_date filters use the _where suffix to change the comparison
        # from equality to a range.
        if window_start:
            params["event_date"] = window_start.isoformat()
            params["event_date_where"] = ">="
        if window_start and window_end:
            # ACLED accepts a pipe-separated pair with BETWEEN.
            params["event_date"] = f"{window_start.isoformat()}|{window_end.isoformat()}"
            params["event_date_where"] = "BETWEEN"

        fields = self.source.route_setting("fields")
        if fields:
            params["fields"] = "|".join(fields)
        return params

    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        records: list[dict[str, Any]] = []
        page = 1
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        while page <= self.max_pages:
            response = requests.get(
                self.source.route_setting("base_url"),
                params=self.build_params(window_start, window_end, page),
                headers=headers,
                timeout=self.timeout_s,
            )
            if response.status_code in (401, 403):
                raise AcledAuthError(
                    f"ACLED rejected the request with {response.status_code}. "
                    "The token may have expired or the account may lack access."
                )
            response.raise_for_status()
            body = response.json()

            if not body.get("success", True):
                messages = body.get("messages") or ["no message supplied"]
                raise RuntimeError(f"ACLED reported failure: {messages}")

            batch = body.get("data") or []
            records.extend(batch)

            # A short page is the last page. ACLED returns an error rather than
            # an empty set for pages far beyond the data.
            if len(batch) < self.page_limit:
                break
            page += 1

        return json.dumps(records).encode("utf-8")

    def count_records(self, payload: bytes) -> int:
        return len(json.loads(payload.decode("utf-8")))


class AcledBulkAdapter(SourceAdapter):
    """Weekly aggregated regional file.

    Already at week x admin1 x event type, so it needs no aggregation for a
    state-week panel. Carries no admin2, actors or narratives, so LGA analysis
    and actor-based youth identification require the API route.
    """

    def file_extension(self) -> str:
        return "csv"

    def fetch_payload(
        self, window_start: date | None, window_end: date | None
    ) -> bytes:
        response = requests.get(
            self.source.route_setting("base_url"),
            headers={"User-Agent": "hsre-violence-warning/0.1 (academic research)"},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.content

    def count_records(self, payload: bytes) -> int:
        from hsre.ingest.adapters import _count_csv_rows

        return _count_csv_rows(payload)
