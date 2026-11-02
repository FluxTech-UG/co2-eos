"""The Span-Wagner default is bit for bit the 0.5.0 package.

co2-eos 0.6.0 adds a second backend (``eos="nist2026"``) behind a static
switch whose default is ``"span_wagner"``.  Every public function called as
0.5.0 called it must return the same bits.  The reference is computed live,
not stored: XLA:CPU's fusions differ across CPUs, so a stored array would
fail on another machine for reasons unrelated to the code.  The 0.5.0 package
is extracted from git (``git archive``) and the same probe
(``sw_default_probe.py``) runs against it and against this checkout, each in
its own interpreter, and every array must be equal (NaN equal to NaN).

Skips when git or the 0.5.0 commit is not available (a shallow clone).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
PROBE = Path(__file__).resolve().parent / "sw_default_probe.py"
REF_COMMIT = "f54b86b6092d254c593863a2321e7c5ab1af9983"   # 0.5.0 + docs/MAP.md


def _run_probe(root, out):
    env = dict(os.environ, PYTHONPATH=str(root), JAX_ENABLE_X64="1")
    r = subprocess.run([sys.executable, "-P", str(PROBE), str(out)], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    # Not vacuous: each run imported the package under its own root, not the
    # editable install's.
    loaded = Path(r.stdout.split("co2_eos loaded from", 1)[1].strip())
    assert loaded.resolve().is_relative_to(Path(root).resolve()), loaded
    return dict(np.load(out))


def test_sw_default_bit_for_bit(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not on PATH")
    probe = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e",
                            REF_COMMIT + "^{commit}"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f"0.5.0 commit {REF_COMMIT[:7]} not in this clone")
    old = tmp_path / "v0_5_0"
    old.mkdir()
    archive = subprocess.run(["git", "-C", str(REPO), "archive", REF_COMMIT,
                              "co2_eos"], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive.stdout,
                   check=True)

    ref = _run_probe(old, tmp_path / "ref.npz")
    new = _run_probe(REPO, tmp_path / "new.npz")
    assert sorted(ref) == sorted(new)
    differ = [k for k in ref
              if ref[k].shape != new[k].shape
              or not np.array_equal(ref[k], new[k], equal_nan=True)]
    assert not differ, f"{len(differ)} of {len(ref)} arrays differ: {differ[:12]}"
