"""Stage 2 verification: adapters, retry, visible failure, schema quarantine.

These run against fakes rather than live endpoints. That is deliberate. The
behaviour that matters here is what happens when a source misbehaves, and
misbehaviour cannot be summoned on demand from a real API.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from hsre.config import Source, load_config
from hsre.ingest import adapters
from hsre.ingest.base import EmptyPayload, FetchResult, SourceAdapter, SourceDown
from hsre.monitoring import ledger
from hsre.validate import schema


@pytest.fixture
def fake_source() -> Source:
    return Source(
        name="fake_source",
        setting="nigeria",
        role="outcome",
        route="api",
        cadence="monthly",
        freshness_slo_hours=720,
        volume_floor=0.80,
        required=True,
        base_url="https://example.invalid/api",
        schema={"id": "int", "date": "date", "state": "str", "deaths": "int"},
    )


class ScriptedAdapter(SourceAdapter):
    """Adapter driven by a scripted sequence of outcomes."""

    def __init__(self, source, script, **kwargs):
        super().__init__(source, sleep_fn=lambda _: None, **kwargs)
        self.script = list(script)
        self.calls = 0

    def fetch_payload(self, window_start, window_end):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def count_records(self, payload):
        return len(json.loads(payload.decode("utf-8")))


def _payload(n: int) -> bytes:
    return json.dumps([{"id": i} for i in range(n)]).encode("utf-8")


def test_successful_fetch_stores_and_checksums(fake_source, tmp_path):
    adapter = ScriptedAdapter(fake_source, [_payload(5)], raw_root=tmp_path)
    result = adapter.run()
    assert result.ok
    assert result.row_count == 5
    assert result.raw_path.exists()
    assert len(result.checksum) == 64
    assert adapter.calls == 1


def test_retry_recovers_from_transient_failure(fake_source, tmp_path):
    script = [ConnectionError("timeout"), ConnectionError("timeout"), _payload(3)]
    adapter = ScriptedAdapter(fake_source, script, raw_root=tmp_path)
    result = adapter.run()
    assert result.ok
    assert result.attempts == 3
    assert adapter.calls == 3


def test_exhausted_retries_raise_source_down(fake_source, tmp_path):
    script = [ConnectionError("down")] * 4
    adapter = ScriptedAdapter(fake_source, script, raw_root=tmp_path)
    with pytest.raises(SourceDown) as excinfo:
        adapter.run()
    assert excinfo.value.attempts == 4


def test_empty_payload_is_a_failure_not_a_quiet_week(fake_source, tmp_path):
    """The central rule of this layer. A successful response containing no
    records must not be recorded as an absence of violence."""
    adapter = ScriptedAdapter(fake_source, [_payload(0)], raw_root=tmp_path)
    with pytest.raises(EmptyPayload):
        adapter.run()
    # Nothing was written, because nothing was valid.
    assert not list(tmp_path.rglob("*.json"))


def test_failure_is_recorded_in_the_ledger(fake_source, tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(ledger, "LEDGER_PATH", ledger_path)
    adapter = ScriptedAdapter(fake_source, [_payload(0)], raw_root=tmp_path)
    with pytest.raises(EmptyPayload):
        adapter.run()
    entries = list(ledger.read_ledger(ledger_path))
    assert entries
    assert entries[-1]["status"] == ledger.STATUS_DOWN
    assert "empty payload" in entries[-1]["detail"]


def test_raw_files_are_immutable(fake_source, tmp_path, monkeypatch):
    """A second run landing on the same path must refuse rather than
    overwrite. Reproduction depends on raw retrievals never being mutated."""
    adapter = ScriptedAdapter(
        fake_source, [_payload(2), _payload(9)], raw_root=tmp_path
    )
    first = adapter.run()
    assert first.ok
    original_bytes = first.raw_path.read_bytes()

    # Freeze the destination path so the second run collides with the first.
    monkeypatch.setattr(adapter, "raw_path_for", lambda retrieval: first.raw_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        adapter.run()

    # The original retrieval is untouched.
    assert first.raw_path.read_bytes() == original_bytes


def test_schema_validation_passes_clean_data(fake_source):
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2020-01-01", "2020-01-08", "2020-01-15"],
            "state": ["Kaduna", "Borno", "Zamfara"],
            "deaths": [2, 0, 5],
        }
    )
    surviving, report = validate_no_write(frame, fake_source)
    assert report.rows_out == 3
    assert report.quarantined == 0
    assert report.pass_rate == 1.0


def validate_no_write(frame, source, **kwargs):
    return schema.validate(frame, source, write_quarantine=False, **kwargs)


def test_missing_column_raises_schema_drift(fake_source):
    frame = pd.DataFrame({"id": [1], "date": ["2020-01-01"], "state": ["Kano"]})
    with pytest.raises(schema.SchemaDrift) as excinfo:
        validate_no_write(frame, fake_source)
    assert "deaths" in excinfo.value.missing


def test_bad_records_are_quarantined_not_coerced(fake_source):
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2020-01-01", "not-a-date", "2020-01-15"],
            "state": ["Kaduna", "Borno", "Zamfara"],
            "deaths": [2, 1, "unknown"],
        }
    )
    surviving, report = validate_no_write(frame, fake_source)
    assert report.rows_in == 3
    assert report.quarantined == 2
    assert report.rows_out == 1
    assert schema.REASON_TYPE_COERCION in report.reasons


def test_quarantine_file_is_written_with_reasons(fake_source, tmp_path):
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2020-01-01", "bad"],
            "state": ["Kaduna", "Borno"],
            "deaths": [2, 1],
        }
    )
    surviving, report = schema.validate(
        frame, fake_source, quarantine_root=tmp_path, write_quarantine=True
    )
    assert report.quarantine_path.exists()
    written = pd.read_csv(report.quarantine_path)
    assert "quarantine_reason" in written.columns
    assert len(written) == 1


def test_out_of_range_coordinates_are_caught():
    source = Source(
        name="geo_source",
        setting="nigeria",
        role="outcome",
        route="api",
        cadence="monthly",
        freshness_slo_hours=720,
        volume_floor=0.8,
        required=True,
        base_url=None,
        schema={"id": "int", "latitude": "float", "longitude": "float"},
    )
    frame = pd.DataFrame(
        {"id": [1, 2, 3], "latitude": [9.0, 200.0, 11.0], "longitude": [7.0, 8.0, 400.0]}
    )
    surviving, report = validate_no_write(frame, source)
    assert report.quarantined == 2
    assert report.reasons[schema.REASON_OUT_OF_RANGE] == 2


def test_duplicates_are_quarantined(fake_source):
    frame = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "date": ["2020-01-01", "2020-01-01", "2020-01-08"],
            "state": ["Kaduna", "Kaduna", "Borno"],
            "deaths": [2, 2, 1],
        }
    )
    surviving, report = validate_no_write(frame, fake_source, dedupe_on=["id"])
    assert report.quarantined == 1
    assert report.reasons[schema.REASON_DUPLICATE] == 1


def test_null_in_required_field_is_quarantined(fake_source):
    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2020-01-01", "2020-01-08"],
            "state": ["Kaduna", None],
            "deaths": [2, 1],
        }
    )
    surviving, report = validate_no_write(frame, fake_source, required_non_null=["state"])
    assert report.quarantined == 1
    assert report.reasons[schema.REASON_NULL_REQUIRED] == 1


def test_thin_source_is_flagged_without_failing(fake_source, tmp_path, monkeypatch):
    """A source reporting materially less than its norm has not failed, but
    the distinction between a quiet source and a quiet place must survive."""
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    status, detail = schema.check_volume(fake_source, observed_rows=40, trailing_median=100.0)
    assert status == ledger.STATUS_THIN
    assert detail["ratio"] == 0.4


def test_normal_volume_is_not_flagged(fake_source, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    status, _ = schema.check_volume(fake_source, observed_rows=95, trailing_median=100.0)
    assert status == ledger.STATUS_OK


def test_volume_check_skipped_without_baseline(fake_source, tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    status, detail = schema.check_volume(fake_source, observed_rows=10, trailing_median=None)
    assert status == ledger.STATUS_OK
    assert "no trailing baseline" in detail["reason"]


def test_trailing_median_from_ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rows = [
        {"stage": "ingest", "source": "s", "status": "ok", "row_count": 100},
        {"stage": "ingest", "source": "s", "status": "ok", "row_count": 120},
        {"stage": "ingest", "source": "s", "status": "down", "row_count": 0},
        {"stage": "ingest", "source": "s", "status": "ok", "row_count": 110},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert schema.trailing_median_rows("s", path=path) == 110.0


def test_bulk_adapter_counts_csv_rows(fake_source):
    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)
    payload = b"id,state\n1,Kaduna\n2,Borno\n3,Kano\n"
    assert adapter.count_records(payload) == 3


def test_bulk_adapter_handles_empty_file(fake_source):
    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)
    assert adapter.count_records(b"") == 0
    assert adapter.count_records(b"id,state\n") == 0


def test_build_adapter_matches_route():
    config = load_config()
    # UCDP defaults to the bulk route, which needs no credential.
    ucdp = config.sources["ucdp_ged"]
    assert isinstance(adapters.build_adapter(ucdp), adapters.BulkAdapter)
    # Switching the active route switches the adapter.
    assert isinstance(adapters.build_adapter(_ucdp_api_variant()), adapters.ApiAdapter)
    gdelt = config.sources["gdelt_events"]
    assert isinstance(adapters.build_adapter(gdelt), adapters.BulkAdapter)


def test_export_adapter_requires_inbox():
    config = load_config()
    nw = config.sources["nigeria_watch"]
    with pytest.raises(ValueError, match="requires an inbox"):
        adapters.build_adapter(nw)


def test_export_adapter_reads_latest_file(fake_source, tmp_path):
    source = Source(**{**fake_source.__dict__, "route": "export"})
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "export.csv").write_text("id,state\n1,Kaduna\n2,Borno\n", encoding="utf-8")
    adapter = adapters.ExportAdapter(source, inbox=inbox, raw_root=tmp_path / "raw")
    result = adapter.run()
    assert result.ok
    assert result.row_count == 2


def test_export_adapter_fails_on_empty_inbox(fake_source, tmp_path):
    source = Source(**{**fake_source.__dict__, "route": "export"})
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    adapter = adapters.ExportAdapter(
        source, inbox=inbox, raw_root=tmp_path / "raw", sleep_fn=lambda _: None
    )
    with pytest.raises(SourceDown):
        adapter.run()


def test_every_configured_source_can_build_an_adapter(tmp_path):
    """No source in the registry may lack a working adapter."""
    config = load_config()
    for name, source in config.sources.items():
        kwargs = {"inbox": tmp_path} if source.route == "export" else {}
        adapter = adapters.build_adapter(source, **kwargs)
        assert adapter.name == name


# --- UCDP contract tests -------------------------------------------------
# Written against the response envelope documented at https://ucdp.uu.se/apidocs/


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _ucdp_api_variant():
    """UCDP pinned to its API route, for tests of API-specific behaviour.

    The repository defaults to the bulk route because it needs no credential,
    so tests of the token path must select the API route explicitly.
    """
    config = load_config()
    source = config.sources["ucdp_ged"]
    return Source(**{**source.__dict__, "active_route": "api"})


def test_ucdp_url_includes_pinned_version():
    """A versioned call is guaranteed to return the same data indefinitely,
    which is what makes the retrieval reproducible."""
    ucdp = _ucdp_api_variant()
    assert ucdp.version == "26.1"
    assert ucdp.versioned_url == "https://ucdpapi.pcr.uu.se/api/gedevents/26.1"


def test_source_without_version_keeps_base_url():
    config = load_config()
    acs = config.sources["acs"]
    assert acs.versioned_url == acs.base_url


def test_ucdp_uses_custom_token_header(monkeypatch):
    adapter = adapters.ApiAdapter(_ucdp_api_variant())
    monkeypatch.setenv("UCDP_API_TOKEN", "test-token-value")
    headers = adapter.headers()
    assert headers["x-ucdp-access-token"] == "test-token-value"
    assert "Authorization" not in headers


def test_missing_token_raises_before_any_request(monkeypatch):
    """Fail before spending requests against the daily quota."""
    adapter = adapters.ApiAdapter(_ucdp_api_variant())
    monkeypatch.delenv("UCDP_API_TOKEN", raising=False)
    with pytest.raises(adapters.MissingToken, match="UCDP_API_TOKEN"):
        adapter.headers()


def test_ucdp_filters_to_nigeria_country_code():
    adapter = adapters.ApiAdapter(_ucdp_api_variant())
    params = adapter.build_params(date(2024, 1, 1), date(2024, 3, 31), page=0)
    assert params["Country"] == 475
    assert params["StartDate"] == "2024-01-01"
    assert params["EndDate"] == "2024-03-31"


def test_extract_records_reads_result_envelope():
    adapter = adapters.ApiAdapter(_ucdp_api_variant())
    body = {"TotalCount": 2, "TotalPages": 1, "Result": [{"id": 4}, {"id": 5}]}
    assert len(adapter.extract_records(body)) == 2
    assert adapter.total_pages(body) == 1


def test_pagination_stops_on_total_pages(monkeypatch):
    """UCDP returns a server error rather than an empty set beyond the last
    page, so pagination must respect TotalPages instead of probing."""
    adapter = adapters.ApiAdapter(_ucdp_api_variant())
    monkeypatch.setenv("UCDP_API_TOKEN", "t")

    calls = []

    def fake_get(url, params, headers, timeout):
        page = params["page"]
        calls.append(page)
        if page >= 3:
            raise AssertionError(f"paged past the declared bound: page {page}")
        return FakeResponse(
            {"TotalCount": 6, "TotalPages": 3, "Result": [{"id": page * 2}, {"id": page * 2 + 1}]}
        )

    monkeypatch.setattr(adapters.requests, "get", fake_get)
    payload = adapter.fetch_payload(None, None)
    assert calls == [0, 1, 2]
    assert adapter.count_records(payload) == 6


def test_pagination_falls_back_to_empty_batch(monkeypatch):
    """Sources that do not declare a page count stop on an empty batch."""
    config = load_config()
    adapter = adapters.ApiAdapter(config.sources["acs"])

    pages = [[{"a": 1}], [{"a": 2}], []]

    def fake_get(url, params, headers, timeout):
        return FakeResponse(pages[params["page"]])

    monkeypatch.setattr(adapters.requests, "get", fake_get)
    payload = adapter.fetch_payload(None, None)
    assert adapter.count_records(payload) == 2


def test_missing_token_does_not_retry(monkeypatch):
    """A missing credential is not transient. Retrying it wastes quota."""
    adapter = adapters.ApiAdapter(_ucdp_api_variant(), sleep_fn=lambda _: None)
    monkeypatch.delenv("UCDP_API_TOKEN", raising=False)

    attempts = []
    original = adapter.headers

    def counting_headers():
        attempts.append(1)
        return original()

    monkeypatch.setattr(adapter, "headers", counting_headers)
    with pytest.raises(adapters.MissingToken):
        adapter.run()
    assert len(attempts) == 1, "credential error should not be retried"


def test_ucdp_defaults_to_credential_free_bulk_route():
    """The pipeline must not be blocked on credential issuance when the
    maintainer publishes the same data as an open download."""
    ucdp = load_config().sources["ucdp_ged"]
    assert ucdp.effective_route == "bulk"
    assert ucdp.route_setting("token_env") is None
    assert ucdp.versioned_url.endswith(".zip")


def test_bulk_url_does_not_get_a_version_segment():
    """A version segment belongs on a REST path, not on a file download."""
    ucdp = load_config().sources["ucdp_ged"]
    assert "/26.1" not in ucdp.versioned_url
    assert ucdp.version == "26.1"


def test_both_routes_declare_the_same_version():
    """Routes must be comparable. A version mismatch would make the
    cross-check between them meaningless."""
    ucdp = load_config().sources["ucdp_ged"]
    api = Source(**{**ucdp.__dict__, "active_route": "api"})
    assert api.version == ucdp.version == "26.1"


def test_bulk_adapter_extracts_csv_from_zip(fake_source, tmp_path):
    import io
    import zipfile

    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("GEDEvent_v26_1.csv", "id,country\n1,Nigeria\n2,Nigeria\n")
    payload = adapter._extract_csv_from_zip(buffer.getvalue())
    assert adapter.count_records(payload) == 2


def test_bulk_adapter_rejects_zip_without_csv(fake_source):
    import io
    import zipfile

    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.pdf", "not data")
    with pytest.raises(ValueError, match="no CSV"):
        adapter._extract_csv_from_zip(buffer.getvalue())


def test_country_filter_applied_after_bulk_download():
    """The bulk archive is global, so the Nigeria filter runs post-download."""
    from hsre.ingest.adapters import filter_to_country

    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "country_id": [475, 501, 475, 615],
            "country": ["Nigeria", "Kenya", "Nigeria", "Algeria"],
        }
    )
    filtered = filter_to_country(frame, country_id=475)
    assert len(filtered) == 2
    assert set(filtered["country"]) == {"Nigeria"}


def test_country_filter_is_a_no_op_without_a_code():
    from hsre.ingest.adapters import filter_to_country

    frame = pd.DataFrame({"id": [1, 2], "country_id": [475, 501]})
    assert len(filter_to_country(frame, country_id=None)) == 2


def test_row_count_respects_embedded_newlines(fake_source):
    """UCDP carries full source-article text, so quoted multi-line fields are
    routine. A naive line count would inflate the trailing baseline."""
    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)
    payload = (
        b'id,country,source_article\n'
        b'1,Nigeria,"Reuters 2024-01-01, headline\nspanning two lines"\n'
        b'2,Nigeria,"AFP, single line"\n'
    )
    assert adapter.count_records(payload) == 2


def test_row_count_respects_quoted_commas(fake_source):
    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)
    payload = b'id,note\n1,"a, b, c"\n2,"d, e"\n'
    assert adapter.count_records(payload) == 2


def test_row_count_ignores_trailing_blank_lines(fake_source):
    source = Source(**{**fake_source.__dict__, "route": "bulk"})
    adapter = adapters.BulkAdapter(source)
    assert adapter.count_records(b"id,x\n1,a\n2,b\n\n\n") == 2


def test_inspect_finds_latest_retrieval(tmp_path, monkeypatch):
    from hsre.ingest import inspect as inspect_mod

    monkeypatch.setattr(inspect_mod, "RAW_ROOT", tmp_path)
    root = tmp_path / "demo" / "2024-01-01"
    root.mkdir(parents=True)
    older = root / "demo_old.csv"
    newer = root / "demo_new.csv"
    older.write_text("id\n1\n", encoding="utf-8")
    newer.write_text("id\n2\n", encoding="utf-8")
    import os, time
    os.utime(older, (time.time() - 500, time.time() - 500))
    assert inspect_mod.latest_retrieval("demo") == newer


def test_inspect_returns_none_without_retrieval(tmp_path, monkeypatch):
    from hsre.ingest import inspect as inspect_mod

    monkeypatch.setattr(inspect_mod, "RAW_ROOT", tmp_path)
    assert inspect_mod.latest_retrieval("never_fetched") is None


# --- outcome definition diagnostics -------------------------------------


def _sparse_panel():
    """Two localities: one active, one nearly silent. Mirrors the observed
    Nigerian distribution where one state holds half of all events."""
    weeks = pd.date_range("2016-01-04", periods=60, freq="W-MON")
    rows = []
    for i, week in enumerate(weeks):
        rows.append({"adm_1": "Borno state", "week": week,
                     "events": 3 if i % 2 == 0 else 5,
                     "deaths": 10})
        rows.append({"adm_1": "Ekiti state", "week": week,
                     "events": 1 if i == 30 else 0,
                     "deaths": 2 if i == 30 else 0})
    return pd.DataFrame(rows)


def test_zero_baseline_makes_single_events_escalations():
    """The failure the floor exists to prevent."""
    from hsre.validate.outcome_check import escalation_outcome

    panel = _sparse_panel()
    without_floor = escalation_outcome(
        panel, "adm_1", horizon_weeks=2, baseline_weeks=12,
        percentile=0.90, min_events=1,
    )
    quiet = without_floor.loc[without_floor["adm_1"] == "Ekiti state"]
    assert quiet["escalation"].sum() >= 1, "single event should flag without a floor"


def test_event_floor_suppresses_single_event_escalations():
    from hsre.validate.outcome_check import escalation_outcome

    panel = _sparse_panel()
    with_floor = escalation_outcome(
        panel, "adm_1", horizon_weeks=2, baseline_weeks=12,
        percentile=0.90, min_events=2,
    )
    quiet = with_floor.loc[with_floor["adm_1"] == "Ekiti state"]
    assert quiet["escalation"].sum() == 0, "floor should suppress lone events"


def test_escalation_requires_a_lethal_event():
    """The Nigerian primary outcome is lethal or serious violence."""
    from hsre.validate.outcome_check import escalation_outcome

    weeks = pd.date_range("2016-01-04", periods=40, freq="W-MON")
    panel = pd.DataFrame(
        [{"adm_1": "Kano state", "week": w, "events": 3, "deaths": 0} for w in weeks]
    )
    out = escalation_outcome(
        panel, "adm_1", horizon_weeks=2, baseline_weeks=12,
        percentile=0.90, min_events=2,
    )
    assert out["escalation"].sum() == 0, "non-lethal weeks must not count"


def test_outcome_uses_only_past_information_for_baseline():
    """The trailing baseline must not see the week it labels."""
    from hsre.validate.outcome_check import escalation_outcome

    panel = _sparse_panel()
    out = escalation_outcome(
        panel, "adm_1", horizon_weeks=2, baseline_weeks=12,
        percentile=0.90, min_events=2,
    )
    # The first baseline_weeks rows per locality cannot be labelled.
    borno = out.loc[out["adm_1"] == "Borno state"].reset_index(drop=True)
    assert borno.loc[:11, "escalation"].isna().all()


def test_panel_grid_is_complete():
    """A week with no event is a real zero the model must be able to see."""
    from hsre.validate.outcome_check import build_locality_weeks

    frame = pd.DataFrame(
        {
            "id": [1, 2],
            "adm_1": ["Borno state", "Ekiti state"],
            "date_start": ["2016-03-07", "2016-06-06"],
            "best": [4, 1],
        }
    )
    panel = build_locality_weeks(
        frame, "adm_1", "date_start",
        pd.Timestamp("2016-01-04"), pd.Timestamp("2016-12-26"),
    )
    assert panel["adm_1"].nunique() == 2
    assert len(panel) == 2 * panel["week"].nunique()
    assert (panel["events"] == 0).sum() > 0


# --- source coverage comparison -----------------------------------------


def test_state_names_normalise_across_sources():
    """UCDP writes 'Borno state', Nigeria Watch writes 'Borno'. They must
    compare as the same locality."""
    from hsre.validate.source_coverage import normalise_state

    ucdp = pd.Series(["Borno state", "Federal Capital territory", "Kano state"])
    watch = pd.Series(["Borno", "FCT", "KANO"])
    assert list(normalise_state(ucdp)) == list(normalise_state(watch))


def test_coverage_comparison_flags_invisible_states():
    """A state with violence in one source and none in the other is the
    finding, not a defect to be silently dropped."""
    from hsre.validate.source_coverage import compare

    primary = pd.DataFrame({"nw": [5000, 3000, 900]}, index=["Lagos", "Kano", "Borno"])
    secondary = pd.DataFrame({"ucdp": [180, 0, 4000]}, index=["Lagos", "Kano", "Borno"])
    merged = compare(primary, secondary, "nw", "ucdp")
    invisible = merged.loc[merged["ucdp"] == 0]
    assert list(invisible.index) == ["Kano"]


def test_coverage_ratio_is_computed():
    from hsre.validate.source_coverage import compare

    primary = pd.DataFrame({"nw": [1000]}, index=["Lagos"])
    secondary = pd.DataFrame({"ucdp": [100]}, index=["Lagos"])
    merged = compare(primary, secondary, "nw", "ucdp")
    assert merged.loc["Lagos", "ratio"] == 10.0
