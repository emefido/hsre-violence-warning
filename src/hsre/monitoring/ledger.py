"""Run ledger.

Every pipeline stage records what it did here. The ledger is not a debugging
convenience: the manuscript reports pipeline success rate, data freshness and
source completeness as results, and those numbers come from this file.

One JSON object per line, appended, never rewritten. Immutability matters
because a run that silently overwrites its own history cannot be audited.
"""

from __future__ import annotations

import json
import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from hsre.config import REPO_ROOT

LEDGER_PATH = REPO_ROOT / "logs" / "run_ledger.jsonl"

# Health status values. "thin" is the dangerous case: the source did not fail,
# it simply reported less, which in conflict settings is easily confused with a
# quiet place. See the manuscript on failure handling.
STATUS_OK = "ok"
STATUS_THIN = "thin"
STATUS_DOWN = "down"
STATUS_SCHEMA_DRIFT = "schema_drift"
STATUS_QUARANTINED = "quarantined"

VALID_STATUSES = {
    STATUS_OK,
    STATUS_THIN,
    STATUS_DOWN,
    STATUS_SCHEMA_DRIFT,
    STATUS_QUARANTINED,
}


def _git_revision() -> str:
    """Code version for the run. Unknown outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_checksum(path: Path) -> str:
    """SHA256 of a file, for the immutable raw landing zone."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "git_revision": _git_revision(),
    }


def record(
    stage: str,
    status: str,
    source: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Append one entry to the ledger and return it.

    stage   pipeline stage name, e.g. "ingest", "validate", "panel"
    status  one of VALID_STATUSES
    source  source name where the entry concerns a single source
    fields  arbitrary additional metrics, e.g. row_count, duration_s
    """
    if status not in VALID_STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {sorted(VALID_STATUSES)}")

    entry: dict[str, Any] = {
        "timestamp": utc_now(),
        "stage": stage,
        "status": status,
        "git_revision": _git_revision(),
    }
    if source is not None:
        entry["source"] = source
    entry.update(fields)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
    return entry


def read_ledger(path: Path | None = None) -> Iterator[dict[str, Any]]:
    """Yield ledger entries oldest first. Empty if the ledger does not exist."""
    target = path or LEDGER_PATH
    if not target.exists():
        return
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pipeline_success_rate(stage: str | None = None, path: Path | None = None) -> float:
    """Share of recorded runs that completed without a failure status.

    This is the service-level indicator reported in the manuscript. Returns
    0.0 when the ledger holds no matching entries, since an unobserved
    pipeline is not a healthy one.
    """
    entries = [e for e in read_ledger(path) if stage is None or e.get("stage") == stage]
    if not entries:
        return 0.0
    ok = sum(1 for e in entries if e["status"] == STATUS_OK)
    return ok / len(entries)


def clear_ledger(path: Path | None = None) -> None:
    """Remove the ledger. Used by tests only, never by pipeline stages."""
    target = path or LEDGER_PATH
    if target.exists():
        os.remove(target)
