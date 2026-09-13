"""Build the sdist and wheel, then prove the installed artifact works outside the source tree.

1. Build both distributions with ``python -m build`` into a temporary directory.
2. Inspect them: metadata version equals the source ``__version__``; ``py.typed``, the license
   file, and every package module ship; nothing from ``tests/`` or ``scripts/`` does; the core
   dependency is unconditional and the service dependencies sit behind the ``service`` extra.
3. Core-only install: a fresh virtual environment with the wheel and no extras. From a temporary
   working directory and in isolated mode (``python -I``: no current directory on ``sys.path``,
   no ``PYTHONPATH``, no user site), import every core module, confirm each resolves inside that
   environment, confirm the service extra is genuinely absent, and drive one in-memory run.
4. Service install: another fresh environment with ``wheel[service]``; ``GET /health`` must
   report the installed metadata version.

Network access is used only to install dependencies from the package index; no provider,
database, or host service is contacted. ``uv`` is used when available, else ``venv`` + ``pip``.
Usage: ``python scripts/verify_distribution.py [--keep]``. Prints a JSON evidence summary.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_version import source_version

CORE_MODULES = (
    "contracts",
    "engine",
    "host_client",
    "http_store",
    "http_tools",
    "planning",
    "policies",
    "runtime",
    "store",
    "tools",
)

CORE_SMOKE = r"""
import importlib, importlib.metadata, importlib.util, json, pathlib, sys
import slimx_agent
package = pathlib.Path(slimx_agent.__file__).resolve().parent
prefix = pathlib.Path(sys.prefix).resolve()
assert prefix in package.parents, f"imported from {package}, not the test environment {prefix}"
for name in MODULES:
    module = importlib.import_module("slimx_agent." + name)
    assert pathlib.Path(module.__file__).resolve().parent == package, name
assert importlib.metadata.version("slimx-agent") == slimx_agent.__version__
assert (package / "py.typed").is_file()
for name in slimx_agent.__all__:
    getattr(slimx_agent, name)
for extra in ("fastapi", "httpx", "uvicorn"):
    assert importlib.util.find_spec(extra) is None, f"{extra} is installed without the extra"
try:
    import slimx_agent.service
except ImportError:
    pass
else:
    raise AssertionError("slimx_agent.service imported without the service extra")

from slimx_agent import engine
from slimx_agent.runtime import RunProfile
from slimx_agent.store import UNSET
from slimx_agent.tools import ToolRegistry

class Run:
    def __init__(self):
        self.id, self.status, self.approval_policy = "r1", "planned", "auto_complete"
        self.auto_approve, self.allowed_tools_json = False, None

class Step:
    def __init__(self):
        self.id, self.type, self.title = "s1", "model_call", "step"
        self.status, self.requires_approval = "pending", False

class Store:
    handler_context = None
    def __init__(self):
        self.run, self.step, self.events = Run(), Step(), []
    def get_run(self, run_id):
        return self.run
    def get_steps(self, run_id):
        return [self.step]
    def get_step(self, step_id):
        return self.step
    def set_run_status(self, run, status):
        self.run.status = status
        return self.run
    def set_step_state(self, step_id, status, *, error=UNSET, output_refs=UNSET):
        self.step.status = status
        return self.step
    def rollback(self):
        pass
    def append_event(self, run_id, type, *, step_id=None, payload=None, commit=True):
        self.events.append({"sequence": len(self.events) + 1, "type": type})
    def next_sequence(self, run_id):
        return len(self.events) + 1
    def drained_events(self, run_id, after):
        return [event for event in self.events if event["sequence"] > after]

registry = ToolRegistry()
registry.register("model_call", lambda context, run, step, profile: {"ref": step.id})
store = Store()
final = engine.execute_run(store, registry, store.run, profile=RunProfile("ollama", "qwen3:8b"))
assert final.status == "completed", final.status
print(json.dumps({"version": slimx_agent.__version__, "package": str(package)}))
"""

SERVICE_SMOKE = r"""
import importlib.metadata, json, pathlib, sys
import slimx_agent
from fastapi.testclient import TestClient
from slimx_agent.service import create_app
assert pathlib.Path(sys.prefix).resolve() in pathlib.Path(slimx_agent.__file__).resolve().parents
body = TestClient(create_app()).get("/health").json()
assert body["version"] == importlib.metadata.version("slimx-agent"), body
assert (body["service"], body["mode"]) == ("slimx-agent", "standalone"), body
print(json.dumps(body))
"""


def source_modules(root: Path = ROOT) -> set[str]:
    return {path.name for path in (root / "slimx_agent").glob("*.py")}


def wheel_problems(wheel: Path, version: str, modules: set[str]) -> list[str]:
    """Everything wrong with a built wheel, as human-readable lines."""
    found: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        dist_info = f"slimx_agent-{version}.dist-info"
        if f"{dist_info}/METADATA" not in names:
            return [f"{wheel.name} has no {dist_info}/METADATA (version drift?)"]
        metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode("utf-8"))
    if metadata["Version"] != version:
        found.append(f"wheel metadata version {metadata['Version']} != source {version}")
    if metadata["License-Expression"] != "MIT":
        found.append("wheel metadata lacks License-Expression: MIT")
    if f"{dist_info}/licenses/LICENSE" not in names:
        found.append("wheel does not ship the LICENSE file")
    if "slimx_agent/py.typed" not in names:
        found.append("wheel does not ship the py.typed marker")
    shipped = {name.split("/", 1)[1] for name in names if name.startswith("slimx_agent/")}
    for module in sorted(modules - shipped):
        found.append(f"wheel is missing slimx_agent/{module}")
    for name in sorted(names):
        if name.startswith(("tests/", "scripts/")) or "/tests/" in name:
            found.append(f"wheel ships a non-package path: {name}")
    requirements = metadata.get_all("Requires-Dist") or []
    if not any(req.startswith("pydantic") and "extra ==" not in req for req in requirements):
        found.append("pydantic is not an unconditional requirement")
    for service_dependency in ("fastapi", "uvicorn", "httpx"):
        matching = [req for req in requirements if req.startswith(service_dependency)]
        # Other extras (dev) may repeat a service dependency; none may make it unconditional.
        if not any('extra == "service"' in req for req in matching):
            found.append(f"{service_dependency} is not declared by the service extra")
        if any("extra ==" not in req for req in matching):
            found.append(f"{service_dependency} is not confined to an extra (core would need it)")
    return found


def sdist_problems(sdist: Path, version: str) -> list[str]:
    found: list[str] = []
    prefix = f"slimx_agent-{version}/"
    with tarfile.open(sdist) as archive:
        names = set(archive.getnames())
    for required in ("PKG-INFO", "pyproject.toml", "LICENSE", "README.md", "slimx_agent/py.typed"):
        if prefix + required not in names:
            found.append(f"sdist is missing {required}")
    return found


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(target: Path, requirement: str) -> Path:
    """A fresh virtual environment at ``target`` with ``requirement`` installed; its python."""
    uv = shutil.which("uv")
    python = target / "bin" / "python"
    if uv:
        subprocess.run([uv, "venv", "--quiet", "--python", sys.executable, str(target)], check=True)
        subprocess.run(
            [uv, "pip", "install", "--quiet", "--python", str(python), requirement], check=True
        )
    else:
        subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", requirement], check=True)
    return python


def _smoke(python: Path, code: str, cwd: Path) -> str:
    result = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    return result.stdout.strip()


def main(argv: list[str]) -> int:
    keep = "--keep" in argv
    work = Path(tempfile.mkdtemp(prefix="slimx-agent-dist-"))
    try:
        version = source_version()
        dist = work / "dist"
        subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist), str(ROOT)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        (wheel,) = dist.glob("*.whl")
        (sdist,) = dist.glob("*.tar.gz")
        found = wheel_problems(wheel, version, source_modules()) + sdist_problems(sdist, version)
        if found:
            for problem in found:
                print(f"distribution problem: {problem}", file=sys.stderr)
            return 1
        outside = work / "outside-the-source-tree"
        outside.mkdir()
        core_python = _environment(work / "core-env", str(wheel))
        core = _smoke(core_python, f"MODULES = {CORE_MODULES!r}\n{CORE_SMOKE}", outside)
        service_python = _environment(work / "service-env", f"{wheel}[service]")
        service = _smoke(service_python, SERVICE_SMOKE, outside)
        print(
            json.dumps(
                {
                    "version": version,
                    "wheel": {"file": wheel.name, "sha256": _sha256(wheel)},
                    "sdist": {"file": sdist.name, "sha256": _sha256(sdist)},
                    "core_only_install": json.loads(core),
                    "service_install_health": json.loads(service),
                },
                indent=2,
            )
        )
        return 0
    finally:
        if keep:
            print(f"kept {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
