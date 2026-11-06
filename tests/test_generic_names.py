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


SPAN_WAGNER_MODULES = ("span_wagner", "span_wagner_kernel")
# The generic modules allowed to import a Span-Wagner module: ``backends`` (the
# registry that binds each backend's surface and kernel) and ``transport``
# (the correlations' private reducing parameters ``_CORR_*``).
SPAN_WAGNER_IMPORTERS = ("backends", "transport")


def _span_wagner_imports(source):
    """Every import of a Span-Wagner module in ``source``, as text, in any form:
    ``import co2_eos.span_wagner``, ``from co2_eos import span_wagner``,
    ``from co2_eos.span_wagner_kernel import x`` and their relative forms."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "co2_eos" and set(parts[1:2]) & set(SPAN_WAGNER_MODULES):
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".") if module else []
            inside = node.level > 0 or parts[:1] == ["co2_eos"]
            if not inside:
                continue
            sub = parts[1:2] if node.level == 0 else parts[:1]
            if set(sub) & set(SPAN_WAGNER_MODULES):
                hits.append(f"from {'.' * node.level}{module} import ...")
            elif not sub:
                for alias in node.names:
                    if alias.name in SPAN_WAGNER_MODULES:
                        hits.append(f"from {'.' * node.level}{module} import {alias.name}")
    return hits


@pytest.mark.parametrize("source", [
    "import co2_eos.span_wagner",
    "import co2_eos.span_wagner_kernel as k",
    "from co2_eos import span_wagner",
    "from co2_eos import backends, span_wagner_kernel as k",
    "from co2_eos.span_wagner import TC",
    "from co2_eos.span_wagner_kernel import _alpha_r",
    "from . import span_wagner",
    "from .span_wagner import TC",
    "def f():\n    from co2_eos import span_wagner\n",
])
def test_span_wagner_import_scan_sees_every_form(source):
    assert _span_wagner_imports(source), source


@pytest.mark.parametrize("source", [
    "from co2_eos import backends, nist2026",
    "from co2_eos.nist2026 import TC",
    "import span_wagner_api",
    "from . import saturation",
])
def test_span_wagner_import_scan_passes_other_imports(source):
    assert not _span_wagner_imports(source), source


def test_span_wagner_content_lives_under_span_wagner_names():
    """Only a module named for Span-Wagner, the registry ``backends`` and
    ``transport`` import a Span-Wagner module; every other module reaches it
    through ``backends`` by name."""
    offenders = {}
    for path in sorted(PKG.glob("*.py")):
        if path.stem.startswith("span_wagner") or path.stem in SPAN_WAGNER_IMPORTERS:
            continue
        hits = _span_wagner_imports(path.read_text(encoding="utf-8"))
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"generic modules import Span-Wagner: {offenders}"
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


@pytest.mark.parametrize("eos", backends.CO2_EOS_NAMES)
def test_missing_saturation_table_raises(eos, monkeypatch, tmp_path):
    """``inversions._build`` tolerates an absent saturation table (the table
    script imports the package to build one); every saturation lookup and
    every inversion that reads one then raises FileNotFoundError, eagerly and
    under jit, and the absent table is never cached as loaded."""
    import jax
    from co2_eos import inversions, saturation
    monkeypatch.setattr(saturation, "_TABLES", {})
    monkeypatch.setattr(backends, "table_path",
                        lambda e, stem: tmp_path / f"{stem}_absent.npz")
    inv = inversions._build(eos)
    assert eos not in saturation._TABLES
    sat = saturation.for_eos(eos)
    for name in saturation._LOOKUPS:
        arg = jnp.asarray(5.0e6 if name.endswith(("_P", "temperature", "enthalpies"))
                          else 280.0)
        with pytest.raises(FileNotFoundError, match="saturation_table"):
            getattr(sat, name)(arg)
    # The inversions that read the saturation curve; the (rho, u) inversions
    # read none and run without the table.
    T, P, h = jnp.asarray([280.0]), jnp.asarray([5.0e6]), jnp.asarray([2.5e5])
    auto = jnp.asarray([inversions.SUPERCRITICAL])
    calls = {
        "density_from_PT": lambda: inv.density_from_PT(T, P, auto),
        "jit density_from_PT": lambda: jax.jit(inv.density_from_PT)(T, P, auto),
        "grad density_from_PT": lambda: jax.grad(
            lambda t: inv.density_from_PT(t, P, auto).sum())(T),
        "state_from_Ph": lambda: inv.state_from_Ph(P, h, auto),
        "jvp state_from_Ph": lambda: jax.jvp(
            lambda p: inv.state_from_Ph(p, h, auto), (P,), (jnp.ones_like(P),)),
    }
    silent = {}
    for label, call in calls.items():
        try:
            out = call()
        except FileNotFoundError as err:
            assert "saturation_table" in str(err), (label, err)
        else:
            silent[label] = out
    assert not silent, f"returned with the saturation table absent: {silent}"
    assert eos not in saturation._TABLES
