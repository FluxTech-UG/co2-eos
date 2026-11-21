"""Reverse-mode gradients (jax.grad, through the custom_jvp rule) of
`state_from_Du` match central finite differences to 1e-4 relative.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from tests.span_wagner_api import inversions   # the Span-Wagner instance

_density_from_PT = inversions._density_from_PT
_state_from_Du = inversions._state_from_Du
SUPERCRITICAL, LIQUID = inversions.SUPERCRITICAL, inversions.LIQUID


# ── Helpers ──────────────────────────────────────────────────────────────

def finite_diff(f, x, idx, rel_eps=1e-7):
    """Central finite difference of f with respect to x[idx].

    Uses a relative step size: eps = rel_eps * max(|x[idx]|, 1) to avoid
    truncation error when x[idx] is large (e.g. pressure in Pa).
    """
    val = float(x[idx])
    eps = rel_eps * max(abs(val), 1.0)
    x_plus = [xi.copy() if hasattr(xi, 'copy') else xi for xi in x]
    x_minus = [xi.copy() if hasattr(xi, 'copy') else xi for xi in x]
    x_plus[idx] = x_plus[idx] + eps
    x_minus[idx] = x_minus[idx] - eps
    f_plus = f(*x_plus)
    f_minus = f(*x_minus)
    if isinstance(f_plus, tuple):
        return tuple((fp - fm) / (2 * eps) for fp, fm in zip(f_plus, f_minus))
    return (f_plus - f_minus) / (2 * eps)


def assert_close(a, b, rtol=1e-4, atol=1e-8, name=""):
    """Assert two values are close, with helpful error message."""
    a, b = np.asarray(a), np.asarray(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.maximum(scale, atol)
    rel_err = np.abs(a - b) / scale
    assert np.all(rel_err < rtol), (
        f"{name}: max rel error {rel_err.max():.2e}, "
        f"got {a}, expected {b}"
    )


# ── Test points ──────────────────────────────────────────────────────────

# Supercritical CO2: T=320 K, P=10 MPa
T_SC = 320.0
P_SC = 10.0e6
HINT_SC = SUPERCRITICAL

# Liquid CO2: T=280 K, P=8 MPa
T_LIQ = 280.0
P_LIQ = 8.0e6
HINT_LIQ = LIQUID


# ═════════════════════════════════════════════════════════════════════════
# state_from_Du
# ═════════════════════════════════════════════════════════════════════════

class TestStateFromDu:
    """Forward and reverse AD through state_from_Du."""

    @pytest.fixture(params=[
        (T_SC, P_SC, HINT_SC, "supercritical"),
        (T_LIQ, P_LIQ, HINT_LIQ, "liquid"),
    ], ids=lambda p: p[3])
    def test_point(self, request):
        T, P, hint, _ = request.param
        T, P = jnp.float64(T), jnp.float64(P)
        rho = _density_from_PT(T, P, hint)
        u = sw._scalar_internal_energy(T, rho)
        return rho, jnp.float64(u), hint

    def test_vjp_matches_fd(self, test_point):
        rho, u, hint = test_point
        f = lambda rho, u: _state_from_Du(rho, u, hint)

        # VJP of T output w.r.t. both inputs
        grad_T_rho = jax.grad(lambda rho: f(rho, u)[0])(rho)
        grad_T_u = jax.grad(lambda u: f(rho, u)[0])(u)
        fd_rho = finite_diff(f, [rho, u], 0)
        fd_u = finite_diff(f, [rho, u], 1)
        assert_close(grad_T_rho, fd_rho[0], name="vjp dT/drho")
        assert_close(grad_T_u, fd_u[0], name="vjp dT/du")

        # VJP of P output
        grad_P_rho = jax.grad(lambda rho: f(rho, u)[1])(rho)
        grad_P_u = jax.grad(lambda u: f(rho, u)[1])(u)
        assert_close(grad_P_rho, fd_rho[1], name="vjp dP/drho")
        assert_close(grad_P_u, fd_u[1], name="vjp dP/du")
