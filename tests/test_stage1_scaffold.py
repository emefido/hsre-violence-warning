"""Stage 1 verification.

These tests prove the scaffold holds together: configuration parses and
validates, the analytical decisions in thresholds.yml are internally
consistent, and the run ledger records and reads back health entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hsre.config import (
    Config,
    ConfigError,
    load_config,
    load_geographies,
    load_sources,
    load_thresholds,
)
from hsre.monitoring import ledger


def test_sources_load_and_validate():
    sources = load_sources()
    assert sources, "no sources defined"
    # Each country panel needs at least one required outcome source.
    for setting in ("nigeria", "usa"):
        outcomes = [
            s
            for s in sources.values()
            if s.setting in (setting, "both")
            and s.role in {"outcome", "outcome_secondary"}
        ]
        assert outcomes, f"no outcome source for {setting}"


def test_every_source_declares_a_schema():
    for name, source in load_sources().items():
        assert source.schema, f"{name} has no schema contract"


def test_views_is_benchmark_only():
    """VIEWS must never enter the feature matrix."""
    sources = load_sources()
    assert sources["views"].is_benchmark
    config = Config(sources, load_thresholds(), load_geographies())
    assert "views" not in config.sources_for("nigeria")


def test_thresholds_load_and_validate():
    thresholds = load_thresholds()
    assert thresholds["evaluation"]["primary_metric"] == "average_precision"
    assert thresholds["outcome"]["horizon_days"] == 14


def test_error_budget_has_two_constraints():
    """A false alarm and a missed death are not symmetric, so the budget
    carries both a burden ceiling and a recall floor."""
    budget = load_thresholds()["alert_budget"]
    assert "max_review_burden_per_week" in budget
    assert "min_severe_recall" in budget


def test_geographies_load_and_validate():
    geo = load_geographies()
    assert geo["temporal"]["unit"] == "week"
    assert geo["nigeria"]["primary_unit"] == "state"
    assert geo["usa"]["primary_unit"] == "county"


def test_config_assembles():
    config = load_config()
    assert config.seed == 20260724
    assert "fbi_nibrs" in config.required_sources("usa")


def test_nigeria_watch_is_the_primary_nigerian_outcome():
    """UCDP codes organised political violence and omits most cultism,
    communal and criminal killing. Relying on it alone would make the
    Nigerian panel an insurgency study rather than a youth and community
    violence study."""
    config = load_config()
    assert "nigeria_watch" in config.required_sources("nigeria")
    assert config.sources["nigeria_watch"].role == "outcome"
    assert config.sources["ucdp_ged"].role == "outcome_secondary"
    assert not config.sources["ucdp_ged"].required


def test_bad_percentile_is_rejected(tmp_path, monkeypatch):
    """Validation must fail loudly rather than accept an impossible value."""
    import hsre.config as cfg

    bad = tmp_path / "thresholds.yml"
    bad.write_text(
        "study_period:\n"
        "  start: 2016-01-01\n"
        "  train_end: 2021-12-31\n"
        "  validation_end: 2022-12-31\n"
        "  test_start: 2023-01-01\n"
        "  end: 2024-12-31\n"
        "outcome:\n"
        "  percentile_cut: 1.5\n"
        "  nigeria: {combine: or}\n"
        "  usa: {combine: and}\n"
        "alert_budget:\n"
        "  k_percentiles: [0.01]\n"
        "  min_severe_recall: 0.7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    with pytest.raises(ConfigError, match="percentile_cut"):
        cfg.load_thresholds()


def test_out_of_order_study_period_is_rejected(tmp_path, monkeypatch):
    import hsre.config as cfg

    bad = tmp_path / "thresholds.yml"
    bad.write_text(
        "study_period:\n"
        "  start: 2016-01-01\n"
        "  train_end: 2023-12-31\n"
        "  validation_end: 2022-12-31\n"
        "  test_start: 2021-01-01\n"
        "  end: 2024-12-31\n"
        "outcome:\n"
        "  percentile_cut: 0.9\n"
        "  nigeria: {combine: or}\n"
        "  usa: {combine: and}\n"
        "alert_budget:\n"
        "  k_percentiles: [0.01]\n"
        "  min_severe_recall: 0.7\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    with pytest.raises(ConfigError, match="ordered"):
        cfg.load_thresholds()


def test_ledger_records_and_reads_back(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = ledger.record("ingest", ledger.STATUS_OK, source="ucdp_ged", row_count=12)
    assert entry["stage"] == "ingest"
    assert entry["status"] == "ok"

    # Write directly to a temp ledger to avoid depending on repo state.
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    read_back = list(ledger.read_ledger(path))
    assert len(read_back) == 1
    assert read_back[0]["source"] == "ucdp_ged"


def test_ledger_rejects_unknown_status():
    with pytest.raises(ValueError, match="unknown status"):
        ledger.record("ingest", "probably_fine")


def test_thin_status_exists():
    """The silent-degradation case must be representable, since a source
    reporting less looks identical to a place becoming peaceful."""
    assert ledger.STATUS_THIN in ledger.VALID_STATUSES


def test_pipeline_success_rate_on_empty_ledger(tmp_path):
    """An unobserved pipeline is not a healthy one."""
    assert ledger.pipeline_success_rate(path=tmp_path / "absent.jsonl") == 0.0


def test_pipeline_success_rate_computes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    rows = [
        {"timestamp": "t", "stage": "ingest", "status": "ok"},
        {"timestamp": "t", "stage": "ingest", "status": "down"},
        {"timestamp": "t", "stage": "ingest", "status": "ok"},
        {"timestamp": "t", "stage": "panel", "status": "ok"},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    assert ledger.pipeline_success_rate("ingest", path) == pytest.approx(2 / 3)
    assert ledger.pipeline_success_rate(path=path) == pytest.approx(3 / 4)


def test_checksum_is_stable(tmp_path):
    target = tmp_path / "sample.csv"
    target.write_text("a,b\n1,2\n", encoding="utf-8")
    first = ledger.file_checksum(target)
    second = ledger.file_checksum(target)
    assert first == second
    assert len(first) == 64
