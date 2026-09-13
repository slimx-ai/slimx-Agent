"""Verify that every representation of the package version agrees.

``slimx_agent/__init__.py::__version__`` is the single maintained source. Checked against it:

- ``pyproject.toml`` must declare the version dynamic and derive it from that attribute;
- the newest ``CHANGELOG.md`` release heading;
- the installed distribution metadata, when the package is installed;
- the standalone service's ``GET /health`` version, when the ``service`` extra is installed.

Exit status 1 names every disagreement. Run from anywhere: ``python scripts/check_version.py``.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_SHAPE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?$")
CHANGELOG_HEADING = re.compile(r"^## \[?(?P<version>\d+\.\d+\.\d+\S*?)\]?(?:\s|$)", re.MULTILINE)


def source_version(root: Path = ROOT) -> str:
    """The literal ``__version__`` assignment, read without importing the package."""
    tree = ast.parse((root / "slimx_agent" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise ValueError("slimx_agent/__init__.py has no literal __version__ assignment")


def problems(
    root: Path = ROOT, *, installed: str | None = None, health: str | None = None
) -> list[str]:
    """Every disagreement between the maintained source and the other representations."""
    found: list[str] = []
    version = source_version(root)
    if not VERSION_SHAPE.match(version):
        found.append(f"__version__ {version!r} is not a plain release version")

    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project", {})
    if "version" in project:
        found.append("pyproject.toml declares a static [project] version; it must be dynamic")
    if "version" not in project.get("dynamic", []):
        found.append("pyproject.toml does not list 'version' in [project] dynamic")
    dynamic = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version")
    if dynamic != {"attr": "slimx_agent.__version__"}:
        found.append("[tool.setuptools.dynamic] version must be {attr = 'slimx_agent.__version__'}")

    changelog = root / "CHANGELOG.md"
    heading = (
        CHANGELOG_HEADING.search(changelog.read_text(encoding="utf-8"))
        if changelog.exists()
        else None
    )
    if heading is None:
        found.append("CHANGELOG.md has no '## <version>' release heading")
    elif heading["version"] != version:
        found.append(f"CHANGELOG.md newest heading is {heading['version']}, not {version}")

    if installed is not None and installed != version:
        found.append(f"installed distribution metadata is {installed}, not {version}")
    if health is not None and health != version:
        found.append(f"service /health reports {health}, not {version}")
    return found


def _installed_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("slimx-agent")
    except PackageNotFoundError:
        return None


def _health_version() -> str | None:
    try:
        from fastapi.testclient import TestClient

        from slimx_agent.service import create_app
    except ImportError:
        return None
    return str(TestClient(create_app()).get("/health").json()["version"])


def main() -> int:
    installed = _installed_version()
    health = _health_version()
    found = problems(installed=installed, health=health)
    for problem in found:
        print(f"version drift: {problem}", file=sys.stderr)
    if not found:
        checked = ["source", "pyproject", "changelog"]
        checked += ["installed metadata"] if installed is not None else []
        checked += ["health"] if health is not None else []
        print(f"slimx-agent {source_version()} agrees across: {', '.join(checked)}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
