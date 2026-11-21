"""The economized (ρ, u) fast path must match the independent oracle chain.

The reference is built in ``bench/validate_fastpath.py`` from the
``span_wagner`` autodiff oracle plus literal-formula transport, sharing no
code with the economized kernels.  This test runs the same comparison on a
smaller sample so CI keeps the fast path pinned at round-off accuracy.
"""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
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


def test_derivatives_vs_ift_oracle(envelope):
    """d(q)/d(ρ, u) by jax.grad ≤ 1e-9 against the implicit-function oracle.

    The fast path differentiates its fixed-iteration Newton through a hand-written
    custom_jvp (core._temperature_from_Du), and the conductivity's critical
    enhancement through two more (transport._xi_of_excess and
    _lambda_critical_shared); a wrong rule gives smooth, wrong gradients.  The
    reference assembles the derivatives from the implicit function theorem,
    dq/du|ρ = q_T / c_v and dq/dρ|u = q_ρ − q_T u_ρ / c_v, with every partial
    taken by autodiff of the Span-Wagner oracle and the literal transport formulas
    (bench/validate_fastpath.py), sharing no code with the fast path.  It is an
    exact identity on a second implementation of the same equations: it proves
    the derivatives are those of Span-Wagner and its transport, not that they are
    right, and it covers the single-phase envelope (290 to 350 K, the
    near-critical ring from T_c + 0.02 K), not the dome.  Temperature carries the
    inversion's rule alone, the conductivity all three; the other properties
    add compile time and no rule.  Reverse mode only: JAX transposes the same
    jvp rule for it, so forward mode exercises no other code.  The bar is the
    1e-9 the bench reports on its 800-state run.
    """
    rho, u, T = envelope
    T = np.asarray(T)
    corridor = np.where((T > 304.2) & (T < 310.0))[0]
    tight = np.where(T < 305.2)[0]
    # The bulk, the near-critical corridor, the tight approach to T_c, the state above T_c.
    idx = np.r_[0:4, corridor[-4:], tight[-4:], len(T) - 1]
    rho_d, u_d = rho[idx], u[idx]
    quantities = ("temperature", "thermal_conductivity")

    def fast(r, uu):
        st = co2_eos.state_from_Du(r, uu)
        return jnp.stack([st[q] for q in quantities])

    def ref(r, uu):
        return jnp.stack([jnp.stack(vf.reference_derivs(r, uu, q)) for q in quantities])

    got_r, got_u = (np.asarray(g) for g in
                    jax.jit(jax.vmap(jax.jacrev(fast, argnums=(0, 1))))(rho_d, u_d))
    want = np.asarray(jax.jit(jax.vmap(ref))(rho_d, u_d))   # (state, quantity, ρ|u)
    dr_ref, du_ref = want[..., 0], want[..., 1]
    scale_r = np.maximum(np.abs(dr_ref), np.abs(du_ref) * 1e-3 + 1e-30)
    scale_u = np.maximum(np.abs(du_ref), np.abs(dr_ref) * 1e-3 + 1e-30)
    er = np.abs(got_r - dr_ref) / scale_r
    eu = np.abs(got_u - du_ref) / scale_u
    for j, q in enumerate(quantities):
        assert er[:, j].max() < 1e-9 and eu[:, j].max() < 1e-9, \
            f"{q}: d/drho {er[:, j].max():.3e}, d/du {eu[:, j].max():.3e}"
