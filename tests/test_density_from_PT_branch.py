"""
density_from_PT below TC must land on the stable branch under the auto hint.

Below TC the Span-Wagner isotherm is an analytic continuation through the dome
and loops between the spinodals; at 294 K it swings up to about 12 MPa near
550 kg/m³, so P(T, ρ) = P_target has roots inside the dome for any target below
that swing, each with ∂P/∂ρ > 0 and a zero residual. A solve seeded inside the
dome converges to one of them and reports nothing wrong. The regression that
exposed this: T = 294.15 K (0.02 K above the old `T < TC - 10` liquid guard),
P = 8.12 and 9.04 MPa, auto hint, returned 499 and 506 kg/m³ against the
compressed-liquid 821 and 836.

The contract tested here: under the auto hint the seed, the bisection bracket
and the branch guard all follow P against P_sat(T), so the result matches
CoolProp on the liquid side (P ≥ P_sat) and on the vapor side (P < P_sat) at
every subcritical temperature, through jit and vmap, and a root on the wrong
side of the dome is never returned as a number.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import co2_eos
from co2_eos import span_wagner as sw
from co2_eos import saturation as sat
from co2_eos.inversions import _on_requested_branch

CP = pytest.importorskip("CoolProp.CoolProp")

RTOL = 1e-8


def _ref(T, P):
    return CP.PropsSI("D", "T", float(T), "P", float(P), "CO2")


# ── The regression points ───────────────────────────────────────────────────

@pytest.mark.parametrize("T,P", [
    (294.15, 8.12e6),
    (294.15, 9.04e6),
    (294.15, 6.5e6),
    (296.15, 6.5e6),
])
def test_regression_points_match_liquid_hint_and_coolprop(T, P):
    rho_auto = float(co2_eos.density_from_PT(P, T, co2_eos.AUTO))
    rho_liq = float(co2_eos.density_from_PT(P, T, co2_eos.LIQUID))
    ref = _ref(T, P)
    assert rho_auto == pytest.approx(ref, rel=RTOL)
    assert rho_liq == pytest.approx(ref, rel=RTOL)


# ── Both sides of the dome, every subcritical temperature ───────────────────

def _subcritical_grid():
    for T in np.linspace(220.0, 304.0, 22):
        P_sat = CP.PropsSI("P", "T", T, "Q", 0, "CO2")
        for P in np.geomspace(P_sat * 1.0005, 30e6, 8):
            yield T, P, "liquid"
        for P in np.geomspace(0.1e6, P_sat * 0.9995, 5):
            yield T, P, "vapor"


@pytest.mark.parametrize("T,P,side", list(_subcritical_grid()))
def test_auto_hint_follows_P_sat_below_TC(T, P, side):
    rho = float(co2_eos.density_from_PT(P, T, co2_eos.AUTO))
    ref = _ref(T, P)
    assert np.isfinite(rho), f"NaN at T={T}, P={P} ({side})"
    assert rho == pytest.approx(ref, rel=RTOL)
    rho_l, rho_v = sat.saturation_densities(jnp.float64(T))
    if side == "liquid":
        assert rho >= float(rho_l) * (1.0 - 1e-4)
    else:
        assert rho <= float(rho_v) * (1.0 + 1e-4)


# ── jit and vmap through the public surface, mixed hints and sides ──────────

def test_jit_vmap_mixed_hints():
    P = jnp.array([8.12e6, 9.04e6, 6.5e6, 9.04e6, 12e6, 3.0e6])
    T = jnp.array([294.15, 294.15, 296.15, 320.0, 310.0, 290.0])
    hint = jnp.array([co2_eos.AUTO, co2_eos.AUTO, co2_eos.AUTO,
                      co2_eos.AUTO, co2_eos.LIQUID, co2_eos.AUTO])
    f = jax.jit(jax.vmap(co2_eos.density_from_PT, in_axes=(0, 0, 0)))
    rho = np.asarray(f(P, T, hint))
    ref = np.array([_ref(t, p) for p, t in zip(np.asarray(P), np.asarray(T))])
    np.testing.assert_allclose(rho, ref, rtol=RTOL)


def test_gradient_on_the_liquid_branch_near_the_old_hole():
    T, P = 294.15, 9.04e6
    g = float(jax.grad(lambda t: co2_eos.density_from_PT(P, t, co2_eos.AUTO))(T))
    ref = CP.PropsSI("d(D)/d(T)|P", "T", T, "P", P, "CO2")
    assert g == pytest.approx(ref, rel=1e-6)


# ── The branch guard itself ────────────────────────────────────────────────

def test_branch_guard_rejects_a_dome_root_under_auto_only():
    T, P = jnp.float64(294.15), jnp.float64(8.12e6)
    dome_root = jnp.float64(499.33)   # the old spurious answer
    liquid_root = jnp.float64(820.72)
    assert not bool(_on_requested_branch(T, P, dome_root, jnp.int32(co2_eos.AUTO)))
    assert bool(_on_requested_branch(T, P, liquid_root, jnp.int32(co2_eos.AUTO)))
    # explicit hints are the caller's assertion and are not checked
    assert bool(_on_requested_branch(T, P, dome_root, jnp.int32(co2_eos.LIQUID)))
    # above TC nothing is checked
    assert bool(_on_requested_branch(jnp.float64(sw.TC + 5.0), P, dome_root,
                                     jnp.int32(co2_eos.AUTO)))
