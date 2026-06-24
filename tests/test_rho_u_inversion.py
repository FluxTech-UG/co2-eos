"""The (ρ, u) → T inversion and the fused properties_from_rho_u path.

Covers:
  * round-trip T → (ρ, u) → T ≤ 1e-8 K across the near-critical regime and the
    subcritical liquid/vapor branches the seed table spans;
  * the fused properties_from_rho_u dict matching properties(T, ρ) at the
    recovered T (i.e. fusion introduces no inconsistency);
  * forward (jvp) and reverse (grad/vjp) gradients of the inversion vs central
    finite differences.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as co2
from co2_eos import core
from co2_eos.span_wagner import TC, RHOC, R


def _u(T, rho):
    return jax.vmap(lambda t, r: core._u_and_cv(t, r)[0])(
        jnp.asarray(T, jnp.float64), jnp.asarray(rho, jnp.float64))


@pytest.mark.parametrize("label,Tlo,Thi,rlo,rhi", [
    ("regime", 300.0, 333.0, 170.0, 470.0),
    ("near_critical", 300.0, 310.0, 400.0, 470.0),
    ("supercritical", 305.0, 345.0, 60.0, 700.0),
    ("subcrit_liquid", 260.0, 300.0, 900.0, 1040.0),
    ("subcrit_vapor", 260.0, 303.0, 60.0, 120.0),
])
def test_roundtrip_under_1e8(label, Tlo, Thi, rlo, rhi):
    rng = np.random.default_rng(42)
    T = rng.uniform(Tlo, Thi, 4000)
    rho = rng.uniform(rlo, rhi, 4000)
    u = _u(T, rho)
    T_back = co2.temperature_from_rho_u(rho, u)
    err = np.max(np.abs(np.asarray(T_back) - T))
    assert err <= 1e-8, f"{label}: round-trip max |ΔT| = {err:.2e} K > 1e-8"


def test_fused_matches_properties():
    """properties_from_rho_u(ρ,u) == properties(T_recovered, ρ) elementwise."""
    rng = np.random.default_rng(1)
    T = rng.uniform(300.0, 333.0, 500)
    rho = rng.uniform(170.0, 470.0, 500)
    u = _u(T, rho)
    fused = co2.properties_from_rho_u(rho, u)
    direct = jax.vmap(co2.properties)(jnp.asarray(fused["temperature"]),
                                      jnp.asarray(rho))
    for k in direct:
        a = np.asarray(fused[k]); b = np.asarray(direct[k])
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
        assert rel.max() < 1e-9, f"{k}: fused vs direct rel err {rel.max():.2e}"
    # echoed constraints exact
    assert np.allclose(np.asarray(fused["density"]), rho, atol=0, rtol=0)
    assert np.allclose(np.asarray(fused["internal_energy"]), np.asarray(u),
                       atol=0, rtol=0)


# ── Gradients of the inversion (IFT custom_jvp) ─────────────────────────────

def _fd(f, x, i, eps=1e-6):
    e = eps * max(abs(float(x[i])), 1.0)
    xp = list(x); xm = list(x)
    xp[i] = xp[i] + e; xm[i] = xm[i] - e
    return (float(f(*xp)) - float(f(*xm))) / (2 * e)


@pytest.mark.parametrize("T0,rho0", [(313.0, 350.0), (305.0, 460.0),
                                     (330.0, 200.0), (280.0, 950.0)])
def test_inversion_gradients(T0, rho0):
    rho = jnp.float64(rho0)
    u = jnp.float64(float(core._u_and_cv(jnp.float64(T0), rho)[0]))
    f = lambda r, uu: core._temperature_from_Du(r, uu, 2)

    _, dr_jvp = jax.jvp(f, (rho, u), (1.0, 0.0))
    _, du_jvp = jax.jvp(f, (rho, u), (0.0, 1.0))
    dr_vjp = jax.grad(f, 0)(rho, u)
    du_vjp = jax.grad(f, 1)(rho, u)
    dr_fd = _fd(lambda r, uu: f(r, uu), [rho, u], 0)
    du_fd = _fd(lambda r, uu: f(r, uu), [rho, u], 1)

    for got in (dr_jvp, dr_vjp):
        assert abs(float(got) - dr_fd) < 1e-4 * max(abs(dr_fd), 1e-6)
    for got in (du_jvp, du_vjp):
        assert abs(float(got) - du_fd) < 1e-4 * max(abs(du_fd), 1e-6)


def test_properties_from_rho_u_differentiable():
    """grad through the fused path (e.g. dP/du) is finite and matches FD."""
    rho = jnp.float64(350.0)
    u = jnp.float64(float(core._u_and_cv(jnp.float64(313.0), rho)[0]))
    P_of_u = lambda uu: core._state_from_rho_u(rho, uu, 2)["pressure"]
    g = jax.grad(P_of_u)(u)
    fd = _fd(lambda uu: P_of_u(uu), [u], 0)
    assert np.isfinite(float(g))
    assert abs(float(g) - fd) < 1e-4 * max(abs(fd), 1e-6)
