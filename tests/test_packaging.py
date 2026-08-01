"""Every hard third-party import used by required code must be declared in
setup.cfg's install_requires (see the missing typing_extensions dependency)."""

import ast
import configparser
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _third_party_imports_under(src_root):
    """Collect module-level import names under src_root.

    Only top-level statements of each module are inspected, so an import
    nested inside a function/class body -- e.g. the `matplotlib` import
    in ruptures/show/display.py, guarded by try/except ImportError -- is
    correctly treated as optional rather than required.
    """
    found = {}
    for path in src_root.rglob("*.py"):  # no third-party import in the .pyx sources
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found.setdefault(alias.name.split(".")[0], path)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.setdefault(node.module.split(".")[0], path)
    return found


def _install_requires_names():
    cfg_path = REPO_ROOT / "setup.cfg"
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path} not found")
    cfg = configparser.RawConfigParser()
    cfg.read(cfg_path)
    raw = cfg.get("options", "install_requires", fallback="")
    return {
        re.split(r"[<>=!~;\[]", line.strip(), maxsplit=1)[0].strip()
        for line in raw.splitlines()
        if line.strip()
    }


def test_install_requires_covers_runtime_imports():
    imports = _third_party_imports_under(REPO_ROOT / "src" / "ruptures")
    declared = _install_requires_names()
    missing = {
        name: str(path)
        for name, path in imports.items()
        if name not in sys.stdlib_module_names
        and name != "ruptures"
        and name not in declared
    }
    assert not missing, (
        "Modules imported by ruptures' required code but missing from "
        f"setup.cfg's install_requires: {missing}"
    )
