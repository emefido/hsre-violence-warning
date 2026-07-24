"""Dependency declaration checks.

A package that is present in the development environment but absent from
pyproject.toml works for the author and fails for everyone replicating the
study. These tests catch that before publication rather than after.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from hsre.config import REPO_ROOT

# Import name where it differs from the distribution name.
IMPORT_NAMES = {
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
}


def _declared_dependencies() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    raw = config["project"]["dependencies"]
    names = []
    for entry in raw:
        for separator in (">=", "==", "<=", "~=", ">", "<"):
            if separator in entry:
                entry = entry.split(separator)[0]
                break
        names.append(entry.strip())
    return names


def test_every_declared_dependency_imports():
    for name in _declared_dependencies():
        module = IMPORT_NAMES.get(name, name.replace("-", "_"))
        importlib.import_module(module)


def test_xlsx_reader_is_declared():
    """The ACLED aggregated file ships as xlsx, so pandas needs openpyxl.
    Without it the pipeline fails at the first data load."""
    assert "openpyxl" in _declared_dependencies()


def test_pandas_can_actually_read_xlsx(tmp_path):
    """Declaring the dependency is not the same as it working."""
    import pandas as pd

    target = tmp_path / "sample.xlsx"
    pd.DataFrame({"COUNTRY": ["Nigeria"], "EVENTS": [1]}).to_excel(target, index=False)
    frame = pd.read_excel(target)
    assert frame.loc[0, "COUNTRY"] == "Nigeria"


def test_modules_imported_in_source_are_declared():
    """Third-party imports appearing in src/ must be declared."""
    declared = {IMPORT_NAMES.get(n, n.replace("-", "_")) for n in _declared_dependencies()}
    stdlib_and_local = {
        "abc", "argparse", "csv", "dataclasses", "datetime", "hashlib", "io",
        "json", "os", "pathlib", "platform", "re", "shutil", "socket",
        "subprocess", "sys", "time", "tomllib", "typing", "unicodedata",
        "zipfile", "hsre", "__future__", "collections", "functools", "math",
    }
    missing: set[str] = set()

    for path in (REPO_ROOT / "src").rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("import "):
                module = line.split()[1].split(".")[0]
            elif line.startswith("from ") and " import " in line:
                module = line.split()[1].split(".")[0]
            else:
                continue
            if module in stdlib_and_local or module in declared:
                continue
            missing.add(module)

    assert not missing, f"undeclared third-party imports in src/: {sorted(missing)}"


def test_no_compiled_extension_dependencies():
    """The pipeline avoids libraries requiring system-level compiled runtimes.

    LightGBM was removed in favour of scikit-learn's HistGradientBoosting:
    the same histogram-based algorithm, but it links no external OpenMP
    runtime. On macOS the LightGBM wheel installs successfully and then fails
    at import, which would break replication on a reviewer's machine for
    reasons unrelated to the research.
    """
    declared = _declared_dependencies()
    assert "lightgbm" not in declared, (
        "lightgbm requires an OpenMP runtime that macOS does not ship"
    )
