"""Static conformance evidence for the typed engine boundary.

Runs ``mypy --strict`` over ``tests/typing/``: valid adapters must produce zero diagnostics, and
every ``# E: <code>`` marker in ``conformance_bad.py`` must produce exactly that diagnostic —
a missing, extra, or changed error fails. mypy is a declared development dependency, so this
gate cannot silently skip.
"""

from __future__ import annotations

import re
from pathlib import Path

import mypy.api
import pytest

TYPING_DIR = Path(__file__).parent / "typing"
MARKER = re.compile(r"#\s*E:\s*([a-z-]+)\s*$")
DIAGNOSTIC = re.compile(r"^(?P<file>[^:]+):(?P<line>\d+):(?:\d+:)? error: .*\[(?P<code>[a-z-]+)\]$")


@pytest.fixture(scope="module")
def mypy_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("mypy-cache")


def _run_mypy(cache: Path, *files: Path) -> tuple[str, int]:
    stdout, stderr, status = mypy.api.run(
        [
            "--strict",
            "--no-error-summary",
            "--show-error-codes",
            "--hide-error-context",
            "--no-pretty",
            "--cache-dir",
            str(cache),
            *(str(path) for path in files),
        ]
    )
    return stdout + stderr, status


def _diagnostics(output: str, filename: str) -> set[tuple[int, str]]:
    found: set[tuple[int, str]] = set()
    for line in output.splitlines():
        match = DIAGNOSTIC.match(line)
        if match and Path(match["file"]).name == filename:
            found.add((int(match["line"]), match["code"]))
    return found


def _expected(path: Path) -> set[tuple[int, str]]:
    expected: set[tuple[int, str]] = set()
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = MARKER.search(text)
        if match:
            expected.add((number, match[1]))
    return expected


def test_valid_host_adapters_are_accepted(mypy_cache):
    output, status = _run_mypy(mypy_cache, TYPING_DIR / "conformance_ok.py")
    assert status == 0, output


def test_invalid_host_adapters_are_rejected_with_exactly_the_marked_diagnostics(mypy_cache):
    bad = TYPING_DIR / "conformance_bad.py"
    expected = _expected(bad)
    assert len(expected) >= 12  # the fixture itself must not quietly lose its cases
    output, status = _run_mypy(mypy_cache, TYPING_DIR / "conformance_ok.py", bad)
    assert status == 1, output
    assert _diagnostics(output, "conformance_ok.py") == set(), output
    assert _diagnostics(output, "conformance_bad.py") == expected, output
