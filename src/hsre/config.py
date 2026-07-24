"""Configuration loading and validation.

Every stage of the pipeline reads its settings from here rather than holding
literals. This keeps the analytical decisions in config/thresholds.yml, where
they can be frozen before test-set analysis and diffed afterwards.

Validation runs at load time so that a misconfiguration fails now rather than
at model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

VALID_ROUTES = {"api", "bulk", "export"}
VALID_ROLES = {
    "outcome",
    "outcome_secondary",
    "signal",
    "structural",
    "validation",
    "benchmark",
}
VALID_SETTINGS = {"nigeria", "usa", "both"}


class ConfigError(ValueError):
    """Raised when configuration is internally inconsistent."""


@dataclass(frozen=True)
class Source:
    name: str
    setting: str
    role: str
    route: str
    cadence: str
    freshness_slo_hours: int
    volume_floor: float
    required: bool
    base_url: str | None
    schema: dict[str, str]
    notes: str | None = None
    # API-specific. Present only for sources that need them.
    version: str | None = None
    country_id: int | None = None
    auth_header: str | None = None
    token_env: str | None = None
    daily_request_quota: int | None = None
    # Sources reachable by more than one route declare them here. The active
    # route is chosen in config so the pipeline is never blocked on credential
    # issuance when an open alternative exists.
    active_route: str | None = None
    routes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_benchmark(self) -> bool:
        """Benchmark sources must never enter the feature matrix."""
        return self.role == "benchmark"

    @property
    def effective_route(self) -> str:
        """The route actually used for retrieval."""
        return self.active_route or self.route

    def route_setting(self, key: str, default: Any = None) -> Any:
        """Read a setting from the active route, falling back to the source."""
        block = self.routes.get(self.effective_route, {})
        if key in block:
            return block[key]
        return getattr(self, key, default)

    @property
    def versioned_url(self) -> str | None:
        """Base URL with the pinned version appended where one is declared.

        UCDP requires a version segment in every call and guarantees that a
        versioned call returns the same data indefinitely, which is what makes
        the retrieval reproducible.
        """
        base = self.route_setting("base_url")
        if base is None:
            return None
        # A version segment belongs on a REST path, not on a file download.
        if self.version and self.effective_route == "api":
            return f"{base.rstrip('/')}/{self.version}"
        return base


def _read_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ConfigError(f"{name} did not parse to a mapping")
    return loaded


def load_sources() -> dict[str, Source]:
    raw = _read_yaml("sources.yml")
    sources: dict[str, Source] = {}
    for name, spec in raw.items():
        missing = {
            "setting",
            "role",
            "route",
            "cadence",
            "freshness_slo_hours",
            "volume_floor",
            "required",
            "schema",
        } - set(spec)
        if missing:
            raise ConfigError(f"source '{name}' missing keys: {sorted(missing)}")
        if spec["route"] not in VALID_ROUTES:
            raise ConfigError(f"source '{name}' has unknown route {spec['route']!r}")
        if spec["role"] not in VALID_ROLES:
            raise ConfigError(f"source '{name}' has unknown role {spec['role']!r}")
        if spec["setting"] not in VALID_SETTINGS:
            raise ConfigError(f"source '{name}' has unknown setting {spec['setting']!r}")
        floor = float(spec["volume_floor"])
        if not 0.0 < floor <= 1.0:
            raise ConfigError(f"source '{name}' volume_floor must be in (0, 1]")
        if not spec["schema"]:
            raise ConfigError(f"source '{name}' has an empty schema contract")
        sources[name] = Source(
            name=name,
            setting=spec["setting"],
            role=spec["role"],
            route=spec["route"],
            cadence=spec["cadence"],
            freshness_slo_hours=int(spec["freshness_slo_hours"]),
            volume_floor=floor,
            required=bool(spec["required"]),
            base_url=spec.get("base_url"),
            schema=dict(spec["schema"]),
            notes=spec.get("notes"),
            version=str(spec["version"]) if spec.get("version") else None,
            country_id=spec.get("country_id"),
            auth_header=spec.get("auth_header"),
            token_env=spec.get("token_env"),
            daily_request_quota=spec.get("daily_request_quota"),
            active_route=spec.get("active_route"),
            routes=dict(spec.get("routes", {})),
        )
    if not sources:
        raise ConfigError("sources.yml defined no sources")
    return sources


def _as_date(value: Any, label: str) -> date:
    if isinstance(value, date):
        return value
    raise ConfigError(f"{label} must be a date, got {value!r}")


def load_thresholds() -> dict[str, Any]:
    raw = _read_yaml("thresholds.yml")

    period = raw.get("study_period", {})
    start = _as_date(period.get("start"), "study_period.start")
    train_end = _as_date(period.get("train_end"), "study_period.train_end")
    validation_end = _as_date(period.get("validation_end"), "study_period.validation_end")
    test_start = _as_date(period.get("test_start"), "study_period.test_start")
    end = _as_date(period.get("end"), "study_period.end")

    ordered = [start, train_end, validation_end, test_start, end]
    if ordered != sorted(ordered):
        raise ConfigError(
            "study_period dates must be ordered: "
            "start <= train_end <= validation_end <= test_start <= end"
        )

    budget = raw.get("alert_budget", {})
    pcts = budget.get("k_percentiles", [])
    if not pcts or any(not 0 < p < 1 for p in pcts):
        raise ConfigError("alert_budget.k_percentiles must all lie in (0, 1)")
    if not 0 < float(budget.get("min_severe_recall", 0)) <= 1:
        raise ConfigError("alert_budget.min_severe_recall must lie in (0, 1]")

    outcome = raw.get("outcome", {})
    cut = float(outcome.get("percentile_cut", 0))
    if not 0 < cut < 1:
        raise ConfigError("outcome.percentile_cut must lie in (0, 1)")
    for setting in ("nigeria", "usa"):
        combine = outcome.get(setting, {}).get("combine")
        if combine not in {"and", "or"}:
            raise ConfigError(f"outcome.{setting}.combine must be 'and' or 'or'")

    return raw


def load_geographies() -> dict[str, Any]:
    raw = _read_yaml("geographies.yml")
    for setting in ("nigeria", "usa"):
        block = raw.get(setting)
        if not block:
            raise ConfigError(f"geographies.yml missing '{setting}' block")
        rate = float(block.get("min_match_rate", 0))
        if not 0 < rate <= 1:
            raise ConfigError(f"{setting}.min_match_rate must lie in (0, 1]")
    if raw.get("temporal", {}).get("unit") != "week":
        raise ConfigError("temporal.unit must be 'week' for the locality-week panel")
    return raw


@dataclass(frozen=True)
class Config:
    sources: dict[str, Source]
    thresholds: dict[str, Any]
    geographies: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.thresholds["evaluation"]["random_seed"])

    def sources_for(self, setting: str) -> dict[str, Source]:
        """Sources feeding a given country panel, benchmarks excluded."""
        return {
            name: src
            for name, src in self.sources.items()
            if src.setting in (setting, "both") and not src.is_benchmark
        }

    def required_sources(self, setting: str) -> list[str]:
        return sorted(
            name for name, src in self.sources_for(setting).items() if src.required
        )


def load_config() -> Config:
    return Config(
        sources=load_sources(),
        thresholds=load_thresholds(),
        geographies=load_geographies(),
    )
