"""Enforce the coverage gate from a coverage.py JSON report (statement + branch).

The measured 0.19.0 baseline was 87% total with ``service.py`` at 73% and its check runner never
executed. Floors now sit just under the 0.20.0 measurement, and the modules that make
authorization, dispatch, and wire-trust decisions carry their own floors so a regression there
cannot hide inside a healthy total. No module is excluded. A coverage percentage is not proof of
safety: the load-bearing evidence is the negative-path tests these floors keep executing.

Usage: ``python scripts/check_coverage.py coverage.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

TOTAL_FLOOR = 95.0
MODULE_FLOORS: dict[str, float] = {
    "slimx_agent/engine.py": 98.0,
    "slimx_agent/policies.py": 98.0,
    "slimx_agent/http_tools.py": 100.0,
    "slimx_agent/http_store.py": 98.0,
    "slimx_agent/host_client.py": 95.0,
    "slimx_agent/service.py": 93.0,
    "slimx_agent/planning.py": 97.0,
}


def failures(
    report: dict[str, Any],
    *,
    total_floor: float = TOTAL_FLOOR,
    module_floors: dict[str, float] = MODULE_FLOORS,
) -> list[str]:
    """Every floor the report misses, as human-readable lines."""
    found: list[str] = []
    total = float(report["totals"]["percent_covered"])
    if total < total_floor:
        found.append(f"total coverage {total:.2f}% is below the {total_floor:.2f}% floor")
    files = report.get("files", {})
    for path, floor in module_floors.items():
        entry = files.get(path)
        if entry is None:
            found.append(f"{path} is missing from the coverage report")
            continue
        covered = float(entry["summary"]["percent_covered"])
        if covered < floor:
            found.append(f"{path} coverage {covered:.2f}% is below its {floor:.2f}% floor")
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_coverage.py <coverage.json>", file=sys.stderr)
        return 2
    report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    found = failures(report)
    for failure in found:
        print(f"coverage gate: {failure}", file=sys.stderr)
    if not found:
        print(f"coverage gate passed: {float(report['totals']['percent_covered']):.2f}% total")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
