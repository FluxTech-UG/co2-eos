"""Forward-mode (jvp) and reverse-mode (grad/vjp) gradients through every inversion.

Each inversion (`density_from_PT`, `state_from_Ph`, `temperature_from_Du`,
`state_from_Du`) defines a `custom_jvp` rule. JAX automatically transposes
the jvp to derive the vjp, so both modes should work. We verify:
  1. jax.jvp returns finite tangents
  2. jax.grad returns finite gradients (reverse mode via transposition)
  3. Tangent and gradient values match central finite differences to 1e-4 relative.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from co2_eos.inversions import (
    _density_from_PT,
    _state_from_Ph,
    _temperature_from_Du,
    _state_from_Du,
    SUPERCRITICAL,
    LIQUID,
    VAPOR,
)
from co2_eos import span_wagner as sw


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

# Vapor CO2: T=310 K, P=5 MPa
T_VAP = 310.0
P_VAP = 5.0e6
HINT_VAP = VAPOR


# ═════════════════════════════════════════════════════════════════════════
# density_from_PT
# ═════════════════════════════════════════════════════════════════════════

class TestDensityFromPT:
    """Forward and reverse AD through density_from_PT."""

    @pytest.fixture(params=[
        (T_SC, P_SC, HINT_SC, "supercritical"),
        (T_LIQ, P_LIQ, HINT_LIQ, "liquid"),
        (T_VAP, P_VAP, HINT_VAP, "vapor"),
    ], ids=lambda p: p[3])
    def test_point(self, request):
        T, P, hint, _ = request.param
        return jnp.float64(T), jnp.float64(P), hint

    def test_jvp_finite(self, test_point):
        T, P, hint = test_point
        primals, tangents = jax.jvp(
            lambda T, P: _density_from_PT(T, P, hint),
            (T, P), (1.0, 0.0)
        )
        assert jnp.isfinite(primals)
        assert jnp.isfinite(tangents)

    def test_grad_finite(self, test_point):
        T, P, hint = test_point
        grad_T = jax.grad(lambda T: _density_from_PT(T, P, hint))(T)
        grad_P = jax.grad(lambda P: _density_from_PT(T, P, hint))(P)
        assert jnp.isfinite(grad_T)
        assert jnp.isfinite(grad_P)

    def test_jvp_matches_fd(self, test_point):
        T, P, hint = test_point
        f = lambda T, P: _density_from_PT(T, P, hint)

        # JVP w.r.t. T
        _, drho_dT_jvp = jax.jvp(f, (T, P), (1.0, 0.0))
        drho_dT_fd = finite_diff(f, [T, P], 0)
        assert_close(drho_dT_jvp, drho_dT_fd, name="drho/dT")

        # JVP w.r.t. P
        _, drho_dP_jvp = jax.jvp(f, (T, P), (0.0, 1.0))
        drho_dP_fd = finite_diff(f, [T, P], 1)
        assert_close(drho_dP_jvp, drho_dP_fd, name="drho/dP")

    def test_vjp_matches_fd(self, test_point):
        T, P, hint = test_point
        f = lambda T, P: _density_from_PT(T, P, hint)

        grad_T = jax.grad(f, argnums=0)(T, P)
        grad_P = jax.grad(f, argnums=1)(T, P)
        drho_dT_fd = finite_diff(f, [T, P], 0)
        drho_dP_fd = finite_diff(f, [T, P], 1)
        assert_close(grad_T, drho_dT_fd, name="vjp drho/dT")
        assert_close(grad_P, drho_dP_fd, name="vjp drho/dP")


# ═════════════════════════════════════════════════════════════════════════
# state_from_Ph
# ═════════════════════════════════════════════════════════════════════════

class TestStateFromPh:
    """Forward and reverse AD through state_from_Ph."""

    @pytest.fixture(params=[
        (P_SC, HINT_SC, "supercritical"),
        (P_LIQ, HINT_LIQ, "liquid"),
        (P_VAP, HINT_VAP, "vapor"),
    ], ids=lambda p: p[2])
    def test_point(self, request):
        P, hint, _ = request.param
        P = jnp.float64(P)
        # Get a consistent (T, rho) and compute h from it
        rho = _density_from_PT(jnp.float64(T_SC if hint == HINT_SC else (T_LIQ if hint == HINT_LIQ else T_VAP)),
                               P, hint)
        T_ref = jnp.float64(T_SC if hint == HINT_SC else (T_LIQ if hint == HINT_LIQ else T_VAP))
        h = sw._scalar_enthalpy(T_ref, rho)
        return P, jnp.float64(h), hint

    def test_jvp_finite(self, test_point):
        P, h, hint = test_point
        primals, tangents = jax.jvp(
            lambda P, h: _state_from_Ph(P, h, hint),
            (P, h), (1.0, 0.0)
        )
        assert all(jnp.isfinite(p) for p in primals)
        assert all(jnp.isfinite(t) for t in tangents)

    def test_grad_finite(self, test_point):
        P, h, hint = test_point
        # grad of T output w.r.t. P
        grad_T_P = jax.grad(lambda P: _state_from_Ph(P, h, hint)[0])(P)
        # grad of rho output w.r.t. P
        grad_rho_P = jax.grad(lambda P: _state_from_Ph(P, h, hint)[1])(P)
        assert jnp.isfinite(grad_T_P)
        assert jnp.isfinite(grad_rho_P)

    def test_jvp_matches_fd(self, test_point):
        P, h, hint = test_point
        f = lambda P, h: _state_from_Ph(P, h, hint)

        # JVP w.r.t. P
        _, (dT_dP, drho_dP) = jax.jvp(f, (P, h), (1.0, 0.0))
        fd = finite_diff(f, [P, h], 0)
        assert_close(dT_dP, fd[0], name="dT/dP")
        assert_close(drho_dP, fd[1], name="drho/dP")

        # JVP w.r.t. h
        _, (dT_dh, drho_dh) = jax.jvp(f, (P, h), (0.0, 1.0))
        fd = finite_diff(f, [P, h], 1)
        assert_close(dT_dh, fd[0], name="dT/dh")
        assert_close(drho_dh, fd[1], name="drho/dh")

    def test_vjp_matches_fd(self, test_point):
        P, h, hint = test_point
        f = lambda P, h: _state_from_Ph(P, h, hint)

        # VJP of T output
        grad_T_P = jax.grad(lambda P: f(P, h)[0])(P)
        fd_T_P = finite_diff(f, [P, h], 0)[0]
        assert_close(grad_T_P, fd_T_P, name="vjp dT/dP")

        grad_T_h = jax.grad(lambda h: f(P, h)[0])(h)
        fd_T_h = finite_diff(f, [P, h], 1)[0]
        assert_close(grad_T_h, fd_T_h, name="vjp dT/dh")


# ═════════════════════════════════════════════════════════════════════════
# temperature_from_Du
# ═════════════════════════════════════════════════════════════════════════

class TestTemperatureFromDu:
    """Forward and reverse AD through temperature_from_Du."""

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

    def test_jvp_finite(self, test_point):
        rho, u, hint = test_point
        primals, tangents = jax.jvp(
            lambda rho, u: _temperature_from_Du(rho, u, hint),
            (rho, u), (1.0, 0.0)
        )
        assert jnp.isfinite(primals)
        assert jnp.isfinite(tangents)

    def test_grad_finite(self, test_point):
        rho, u, hint = test_point
        grad_rho = jax.grad(lambda rho: _temperature_from_Du(rho, u, hint))(rho)
        grad_u = jax.grad(lambda u: _temperature_from_Du(rho, u, hint))(u)
        assert jnp.isfinite(grad_rho)
        assert jnp.isfinite(grad_u)

    def test_jvp_matches_fd(self, test_point):
        rho, u, hint = test_point
        f = lambda rho, u: _temperature_from_Du(rho, u, hint)

        _, dT_drho = jax.jvp(f, (rho, u), (1.0, 0.0))
        assert_close(dT_drho, finite_diff(f, [rho, u], 0), name="dT/drho")

        _, dT_du = jax.jvp(f, (rho, u), (0.0, 1.0))
        assert_close(dT_du, finite_diff(f, [rho, u], 1), name="dT/du")

    def test_vjp_matches_fd(self, test_point):
        rho, u, hint = test_point
        f = lambda rho, u: _temperature_from_Du(rho, u, hint)

        grad_rho = jax.grad(f, argnums=0)(rho, u)
        grad_u = jax.grad(f, argnums=1)(rho, u)
        assert_close(grad_rho, finite_diff(f, [rho, u], 0), name="vjp dT/drho")
        assert_close(grad_u, finite_diff(f, [rho, u], 1), name="vjp dT/du")


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

    def test_jvp_finite(self, test_point):
        rho, u, hint = test_point
        primals, tangents = jax.jvp(
            lambda rho, u: _state_from_Du(rho, u, hint),
            (rho, u), (1.0, 0.0)
        )
        assert all(jnp.isfinite(p) for p in primals)
        assert all(jnp.isfinite(t) for t in tangents)

    def test_grad_finite(self, test_point):
        rho, u, hint = test_point
        # grad of T output w.r.t. rho
        grad_T_rho = jax.grad(lambda rho: _state_from_Du(rho, u, hint)[0])(rho)
        # grad of P output w.r.t. u
        grad_P_u = jax.grad(lambda u: _state_from_Du(rho, u, hint)[1])(u)
        assert jnp.isfinite(grad_T_rho)
        assert jnp.isfinite(grad_P_u)

    def test_jvp_matches_fd(self, test_point):
        rho, u, hint = test_point
        f = lambda rho, u: _state_from_Du(rho, u, hint)

        # JVP w.r.t. rho
        _, (dT, dP, dh) = jax.jvp(f, (rho, u), (1.0, 0.0))
        fd = finite_diff(f, [rho, u], 0)
        assert_close(dT, fd[0], name="dT/drho")
        assert_close(dP, fd[1], name="dP/drho")
        assert_close(dh, fd[2], name="dh/drho")

        # JVP w.r.t. u
        _, (dT, dP, dh) = jax.jvp(f, (rho, u), (0.0, 1.0))
        fd = finite_diff(f, [rho, u], 1)
        assert_close(dT, fd[0], name="dT/du")
        assert_close(dP, fd[1], name="dP/du")
        assert_close(dh, fd[2], name="dh/du")

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
