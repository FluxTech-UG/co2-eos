"""Generic names hold no Span-Wagner content, and the docs state the current state.

Every generic name of the package follows ``DEFAULT_EOS`` (nist2026); Span-Wagner
is reached by its own name only.  These checks read the source the imported
``co2_eos`` was loaded from, so they test the checkout under test.
"""
import ast
import json
import re
from pathlib import Path

import jax.numpy as jnp
import pytest

import co2_eos
from co2_eos import backends

PKG = Path(co2_eos.__file__).resolve().parent
ROOT = PKG.parent


def _docstrings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [tree] + [n for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    return [d for d in (ast.get_docstring(n) for n in nodes) if d]


def test_span_wagner_content_lives_under_span_wagner_names():
    """A module importing Span-Wagner's coefficient tables is named for it."""
    for path in PKG.glob("*.py"):
        if path.stem.startswith("span_wagner"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "co2_eos.span_wagner":
                private = [a.name for a in node.names if a.name.startswith("_")]
                assert not private, f"{path.name} imports {private[:3]}"
    assert not (PKG / "helmholtz.py").exists()


def test_docs_state_the_current_state():
    pattern = re.compile(r"unchanged|no longer|Migrating to v0")
    for name in ("README.md", "CLAUDE.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        hits = [line for line in text.splitlines() if pattern.search(line)]
        assert not hits, f"{name}: {hits}"


def test_generic_modules_describe_every_backend():
    inversions = (PKG / "inversions.py").read_text(encoding="utf-8")
    assert not re.findall(r"\bsw\.", inversions)
    core_doc = ast.get_docstring(ast.parse((PKG / "core.py").read_text(encoding="utf-8")))
    assert "Span-Wagner" not in core_doc
    for doc in _docstrings(PKG / "saturation.py"):
        assert "Span-Wagner" not in doc
        assert "216.5" not in doc, doc


def test_inversion_temperature_ceiling_has_one_home():
    """The inversions' upper T bound is ``backends.INVERSION_T_MAX``, read by
    ``core`` and ``inversions``; the README states it."""
    assert backends.INVERSION_T_MAX == 800.0
    for name in ("core.py", "inversions.py"):
        assert "800.0" not in (PKG / name).read_text(encoding="utf-8"), name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "800 K" in readme


def test_launch_demo_runs_span_wagner_against_coolprop():
    """The notebook compares with CoolProp, whose CO2 is Span-Wagner: every
    co2_eos call in it names the backend."""
    nb = json.loads((ROOT / "examples" / "launch_demo.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert 'EOS = "span_wagner"' in code[0]
    calls = 0
    for src in code:
        for call in re.findall(r"co2\.state_from_PT\(([^()]*)\)", src):
            calls += 1
            assert "eos=EOS" in call, call
    assert calls >= 6


@pytest.mark.parametrize("eos", backends.CO2_EOS_NAMES)
def test_missing_tables_raise(eos, monkeypatch, tmp_path):
    """A missing seed or chi table raises at the first call that needs it,
    never a silent fallback; the build itself succeeds (the generators import
    the package to build the table)."""
    from co2_eos import core, transport
    monkeypatch.setattr(backends, "table_path",
                        lambda e, stem: tmp_path / f"{stem}_absent.npz")
    tr = transport._build(eos)
    with pytest.raises(FileNotFoundError, match="chi_ref_cheb"):
        tr._dpdrho_ref_reduced(jnp.asarray(1.0))
    monkeypatch.setattr(transport, "for_eos", lambda e: tr)
    kernels = core._build(eos)
    with pytest.raises(FileNotFoundError, match="seed_table"):
        kernels._seed_T(jnp.asarray(400.0), jnp.asarray(3.0e5))
