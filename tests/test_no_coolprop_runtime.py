"""Discipline test: the runtime package must not import CoolProp.

CoolProp is allowed in `scripts/` (one-time saturation-table generation) and
in `tests/` (validation against a reference). The library's value proposition
explicitly includes being CoolProp-free at runtime; this test makes that
mechanical so a stray import in `co2_eos/` is caught immediately.

We parse each file as an AST rather than grepping, so an unrelated mention
of "CoolProp" in a docstring or comment is not mistaken for an import.
"""

import ast
import pathlib


PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent / "co2_eos"


def _imports_coolprop(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "CoolProp" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "CoolProp":
                return True
    return False


def test_no_coolprop_in_runtime_package():
    offenders = []
    for py in sorted(PKG_ROOT.rglob("*.py")):
        tree = ast.parse(py.read_text(), filename=str(py))
        if _imports_coolprop(tree):
            offenders.append(py.relative_to(PKG_ROOT.parent))
    assert not offenders, (
        "Runtime package must not import CoolProp. Offenders:\n"
        + "\n".join(f"  {p}" for p in offenders)
    )
