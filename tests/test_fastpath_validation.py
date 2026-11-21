"""The economized (ρ, u) fast path must match the independent oracle chain.

The reference is built in ``bench/validate_fastpath.py`` from the
``span_wagner`` autodiff oracle plus literal-formula transport, sharing no
code with the economized kernels.  This test runs the same comparison on a
smaller sample so CI keeps the fast path pinned at round-off accuracy.
"""

import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tests.span_wagner_api import co2 as co2_eos   # the Span-Wagner path, eos bound

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "validate_fastpath", _REPO / "bench" / "validate_fastpath.py")
vf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vf)


@pytest.fixture(scope="module")
def envelope():
    rho, u, T = vf.envelope_single_phase(1500, seed=23)
    return rho, u, T


def test_properties_vs_oracle(envelope):
    """Every property ≤ 5e-10 rel vs the independent chain (measured ~1e-10)."""
    rho, u, _ = envelope
    fast = co2_eos.properties_from_rho_u(rho, u)
    ref = jax.jit(jax.vmap(vf.reference_state))(rho, u)
    for q in vf.QUANTITIES:
        e = vf.rel(np.asarray(fast[q]), np.asarray(ref[q]), vf.FLOORS[q])
        assert e.max() < 5e-10, f"{q}: max rel {e.max():.3e}"
