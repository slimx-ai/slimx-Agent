"""Release identity: one version everywhere, and meaningful negative evidence for each gate.

The distribution build/install proof itself runs as a CI step (``scripts/verify_distribution.py``)
because it needs fresh virtual environments; its inspection logic is tested here against
deliberately broken synthetic artifacts.
"""

from __future__ import annotations

import importlib.util
import io
import zipfile
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

import slimx_agent

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check_version = _script("check_version")
check_coverage = _script("check_coverage")
verify_distribution = _script("verify_distribution")


def test_every_version_representation_agrees_in_this_checkout():
    from slimx_agent.service import create_app

    health = TestClient(create_app()).get("/health").json()["version"]
    assert check_version.problems(installed=version("slimx-agent"), health=health) == []
    assert health == slimx_agent.__version__ == check_version.source_version()


def test_the_typing_marker_ships_with_the_package():
    assert (Path(slimx_agent.__file__).parent / "py.typed").is_file()


def _synthetic_repo(tmp_path: Path, *, version_literal: str = "1.2.3") -> Path:
    (tmp_path / "slimx_agent").mkdir()
    (tmp_path / "slimx_agent" / "__init__.py").write_text(f'__version__ = "{version_literal}"\n')
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "slimx-agent"\ndynamic = ["version"]\n'
        '[tool.setuptools.dynamic]\nversion = { attr = "slimx_agent.__version__" }\n'
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## 1.2.3 — candidate\n\n## 1.2.2\n")
    return tmp_path


def test_the_version_checker_accepts_a_coherent_repository(tmp_path):
    repo = _synthetic_repo(tmp_path)
    assert check_version.problems(repo, installed="1.2.3", health="1.2.3") == []


@pytest.mark.parametrize(
    ("mutate", "installed", "health", "needle"),
    [
        (
            lambda repo: (repo / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n'),
            None,
            None,
            "static",
        ),
        (lambda repo: (repo / "CHANGELOG.md").write_text("## 1.2.2\n"), None, None, "CHANGELOG"),
        (
            lambda repo: (repo / "CHANGELOG.md").write_text("no headings\n"),
            None,
            None,
            "no '## <version>'",
        ),
        (lambda repo: None, "1.2.2", None, "installed distribution metadata"),
        (lambda repo: None, None, "0.19.0", "/health"),
        (
            lambda repo: (repo / "pyproject.toml").write_text(
                '[project]\ndynamic = ["version"]\n[tool.setuptools.dynamic]\n'
                'version = { attr = "slimx_agent.version.VERSION" }\n'
            ),
            None,
            None,
            "attr",
        ),
    ],
)
def test_the_version_checker_names_each_disagreement(tmp_path, mutate, installed, health, needle):
    repo = _synthetic_repo(tmp_path)
    mutate(repo)
    found = check_version.problems(repo, installed=installed, health=health)
    assert found and any(needle in problem for problem in found), found


def test_the_version_checker_refuses_a_non_literal_or_malformed_version(tmp_path):
    repo = _synthetic_repo(tmp_path, version_literal="v1")
    (repo / "CHANGELOG.md").write_text("## v1\n")
    assert any("plain release version" in problem for problem in check_version.problems(repo))
    (repo / "slimx_agent" / "__init__.py").write_text("__version__ = compute()\n")
    with pytest.raises(ValueError, match="literal __version__"):
        check_version.source_version(repo)


def _report(total: float, **modules: float) -> dict:
    return {
        "totals": {"percent_covered": total},
        "files": {
            path: {"summary": {"percent_covered": modules.get(Path(path).stem, 100.0)}}
            for path in check_coverage.MODULE_FLOORS
        },
    }


def test_the_coverage_gate_passes_a_report_at_its_floors():
    assert check_coverage.failures(_report(100.0)) == []


def test_the_coverage_gate_names_each_missed_floor():
    found = check_coverage.failures(_report(80.0, engine=50.0, http_tools=99.0))
    assert any("total coverage" in line for line in found)
    assert any("engine.py" in line for line in found)
    assert any("http_tools.py" in line for line in found)
    report = _report(100.0)
    del report["files"]["slimx_agent/service.py"]
    assert check_coverage.failures(report) == [
        "slimx_agent/service.py is missing from the coverage report"
    ]


def _wheel(
    tmp_path: Path,
    *,
    version: str = "1.2.3",
    drop: tuple[str, ...] = (),
    add: dict[str, str] | None = None,
) -> Path:
    dist_info = "slimx_agent-1.2.3.dist-info"
    files = {
        "slimx_agent/__init__.py": "",
        "slimx_agent/engine.py": "",
        "slimx_agent/py.typed": "",
        f"{dist_info}/licenses/LICENSE": "MIT",
        f"{dist_info}/METADATA": (
            f"Metadata-Version: 2.4\nName: slimx-agent\nVersion: {version}\n"
            "License-Expression: MIT\nRequires-Dist: pydantic>=2.7\n"
            'Requires-Dist: fastapi>=0.111; extra == "service"\n'
            'Requires-Dist: uvicorn>=0.30; extra == "service"\n'
            'Requires-Dist: httpx>=0.27; extra == "service"\n'
        ),
        **(add or {}),
    }
    path = tmp_path / "slimx_agent-1.2.3-py3-none-any.whl"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            if name not in drop:
                archive.writestr(name, content)
    path.write_bytes(buffer.getvalue())
    return path


MODULES = {"__init__.py", "engine.py"}


def test_the_wheel_inspection_accepts_a_complete_wheel(tmp_path):
    assert verify_distribution.wheel_problems(_wheel(tmp_path), "1.2.3", MODULES) == []


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"drop": ("slimx_agent/py.typed",)}, "py.typed"),
        ({"drop": ("slimx_agent-1.2.3.dist-info/licenses/LICENSE",)}, "LICENSE"),
        ({"drop": ("slimx_agent/engine.py",)}, "missing slimx_agent/engine.py"),
        ({"version": "1.2.2"}, "metadata version"),
        ({"add": {"tests/test_x.py": ""}}, "non-package path"),
    ],
)
def test_the_wheel_inspection_names_each_defect(tmp_path, kwargs, needle):
    found = verify_distribution.wheel_problems(_wheel(tmp_path, **kwargs), "1.2.3", MODULES)
    assert any(needle in problem for problem in found), found


def test_the_wheel_inspection_refuses_service_dependencies_outside_the_extra(tmp_path):
    dist_info = "slimx_agent-1.2.3.dist-info"
    wheel = _wheel(
        tmp_path,
        add={
            f"{dist_info}/METADATA": (
                "Metadata-Version: 2.4\nName: slimx-agent\nVersion: 1.2.3\n"
                "License-Expression: MIT\nRequires-Dist: pydantic>=2.7\n"
                "Requires-Dist: fastapi>=0.111\n"
            )
        },
    )
    found = verify_distribution.wheel_problems(wheel, "1.2.3", MODULES)
    assert any("fastapi is not confined" in problem for problem in found)
