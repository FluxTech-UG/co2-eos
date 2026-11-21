"""Gradients through the inversions' custom_jvp rules match central finite differences.

Each inversion (ρ from (P, T), (T, ρ) from (P, h), T from (ρ, u)) carries a
hand-written implicit-function rule; a wrong rule gives smooth, plausible and
wrong gradients.  The reference is a central finite difference of the same
public function's primal value, which never runs the rule: a numerical
derivative, sturdier than another library's table but resting on the primal
being smooth over the step (a relative step of 1e-7; the solves converge to
about 1e-13, so the difference carries about 1e-6 relative noise, under the
1e-4 bar).  It stops holding across a kink: at the dome's edge in (P, h), and
at T_c itself, where (∂ρ/∂P)_T diverges.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
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


# ═════════════════════════════════════════════════════════════════════════
# density_from_PT and state_from_Ph, the public pressure paths
# ═════════════════════════════════════════════════════════════════════════

# (P [Pa], T [K], phase hint): supercritical, near-critical above T_c (1.3 K
# above, 1.6 % above p_c, where c_p is near its peak), compressed liquid,
# subcritical vapour.
PRESSURE_PATH_STATES = [
    (10.0e6, 320.0, ce.SUPERCRITICAL),
    (7.5e6, 305.5, ce.SUPERCRITICAL),
    (8.0e6, 280.0, ce.LIQUID),
    (3.0e6, 280.0, ce.VAPOR),
]


def _fd_columns(f, x, y, rel_eps=1e-7):
    """Central differences of the batched f(x, y) in x and in y, a relative step each."""
    ex, ey = rel_eps * jnp.abs(x), rel_eps * jnp.abs(y)
    dx = (np.asarray(f(x + ex, y)) - np.asarray(f(x - ex, y))) / (2 * np.asarray(ex))
    dy = (np.asarray(f(x, y + ey)) - np.asarray(f(x, y - ey))) / (2 * np.asarray(ey))
    return dx, dy


def test_pressure_paths_match_fd():
    """jax.grad of density_from_PT and state_from_Ph (T and ρ) against central differences.

    On the default backend: the rules are one code built over each backend
    (``inversions._build``), so a second backend adds compile time, not cover.
    The (P, h) path is also taken inside the dome at 5 MPa, where the rule
    differentiates the saturation splines instead of the 2-D solve.
    """
    P, T, hint = (jnp.asarray(c) for c in zip(*PRESSURE_PATH_STATES))
    rho_PT = jax.jit(jax.vmap(ce.density_from_PT))
    grad_PT = jax.jit(jax.vmap(jax.grad(ce.density_from_PT, argnums=(0, 1))))
    for g, fd, x in zip(grad_PT(P, T, hint),
                        _fd_columns(lambda P, T: rho_PT(P, T, hint), P, T), "PT"):
        assert_close(g, fd, name=f"dρ/d{x}")

    h = ce.properties_from_rho_T(rho_PT(P, T, hint), T)["enthalpy"]
    h_l, h_v = ce.saturation_enthalpies_P(5.0e6)
    P = jnp.append(P, 5.0e6)
    h = jnp.append(h, 0.5 * (h_l + h_v))
    hint = jnp.append(hint, ce.LIQUID)

    def T_rho(P, h, k):
        st = ce.state_from_Ph(P, h, k)
        return st["temperature"], st["density"]

    Ph = jax.jit(jax.vmap(T_rho))
    jac = jax.jit(jax.vmap(jax.jacrev(T_rho, argnums=(0, 1))))(P, h, hint)
    for i, out in enumerate(("T", "ρ")):
        fd = _fd_columns(lambda P, h: Ph(P, h, hint)[i], P, h)
        for g, d, x in zip(jac[i], fd, "Ph"):
            assert_close(g, d, name=f"d{out}/d{x}")
