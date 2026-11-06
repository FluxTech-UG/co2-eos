"""The Span-Wagner backend, named with ``eos="span_wagner"``, is bit for bit 0.5.0.

0.5.0 had Span-Wagner as its only equation of state.  Every public function of
0.5.0, called here with ``eos="span_wagner"``, must return the bits 0.5.0
returned for the same call without the keyword.  The reference is computed
live, not stored: XLA:CPU's fusions differ across CPUs, so a stored array
would fail on another machine for reasons unrelated to the code.  The 0.5.0
package is extracted from git (``git archive``) and the same probe
(``span_wagner_probe.py``) runs against it and against this checkout, each in
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
PROBE = Path(__file__).resolve().parent / "span_wagner_probe.py"
REF_COMMIT = "f54b86b6092d254c593863a2321e7c5ab1af9983"   # 0.5.0 + docs/MAP.md


def _run_probe(root, out, *eos):
    env = dict(os.environ, PYTHONPATH=str(root), JAX_ENABLE_X64="1")
    # -P (3.11+) keeps the probe's own directory off sys.path; on 3.10 the
    # directory is tests/, which holds no co2_eos, and the loaded-from check
    # below proves the import came from `root` either way.
    safe_path = ["-P"] if sys.version_info >= (3, 11) else []
    r = subprocess.run([sys.executable, *safe_path, str(PROBE), str(out), *eos],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr
    # Not vacuous: each run imported the package under its own root, not the
    # editable install's.
    loaded = Path(r.stdout.split("co2_eos loaded from", 1)[1].strip())
    assert loaded.resolve().is_relative_to(Path(root).resolve()), loaded
    return dict(np.load(out))


@pytest.mark.slow
def test_span_wagner_bit_for_bit(tmp_path):
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
    new = _run_probe(REPO, tmp_path / "new.npz", "span_wagner")
    assert sorted(ref) == sorted(new)
    differ = [k for k in ref
              if ref[k].shape != new[k].shape
              or not np.array_equal(ref[k], new[k], equal_nan=True)]
    assert not differ, f"{len(differ)} of {len(ref)} arrays differ: {differ[:12]}"
