"""Tests for two-phase dome detection in state_from_Ph.

When (P, h) falls inside the saturation dome at subcritical pressures,
state_from_Ph should return saturation-boundary properties instead of
solving the single-phase Newton iteration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from co2_eos.inversions import (
    _state_from_Ph,
    state_from_Ph,
    LIQUID,
    VAPOR,
    SUPERCRITICAL,
)
from co2_eos import span_wagner as sw
from co2_eos import saturation as sat


P_TEST = 7.1e6  # Pa — subcritical, well inside dome range


@pytest.fixture
def dome_data():
    """Precompute saturation data at P_TEST."""
    T_sat = float(sat.saturation_temperature(jnp.float64(P_TEST)))
    h_l, h_v = sat.saturation_enthalpies(jnp.float64(P_TEST))
    rho_l, rho_v = sat.saturation_densities_P(jnp.float64(P_TEST))
    return {
        "T_sat": T_sat,
        "h_l": float(h_l),
        "h_v": float(h_v),
        "rho_l": float(rho_l),
        "rho_v": float(rho_v),
    }


class TestDomeSweepLiquid:
    """Enthalpy sweep at P=7.1 MPa with LIQUID hint."""

    def test_all_finite(self, dome_data):
        """All returned T and ρ are finite across h_l-50kJ to h_v+50kJ."""
        h_arr = jnp.linspace(
            dome_data["h_l"] - 50e3,
            dome_data["h_v"] + 50e3,
            100,
        )
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, LIQUID, dtype=jnp.int32)

        T_out, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        assert jnp.all(jnp.isfinite(T_out)), f"NaN/Inf in T: {T_out}"
        assert jnp.all(jnp.isfinite(rho_out)), f"NaN/Inf in ρ: {rho_out}"

    def test_T_equals_Tsat_inside_dome(self, dome_data):
        """Inside dome, T should equal T_sat(P)."""
        h_arr = jnp.linspace(dome_data["h_l"], dome_data["h_v"], 50)
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, LIQUID, dtype=jnp.int32)

        T_out, _ = state_from_Ph(P_arr, h_arr, hint_arr)
        np.testing.assert_allclose(
            np.array(T_out), dome_data["T_sat"], atol=1e-6,
            err_msg="T != T_sat inside dome",
        )

    def test_rho_equals_rho_l_inside_dome(self, dome_data):
        """Inside dome with LIQUID hint, ρ should equal ρ_l(P)."""
        h_arr = jnp.linspace(dome_data["h_l"], dome_data["h_v"], 50)
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, LIQUID, dtype=jnp.int32)

        _, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        np.testing.assert_allclose(
            np.array(rho_out), dome_data["rho_l"], rtol=1e-10,
            err_msg="ρ != ρ_l inside dome with LIQUID hint",
        )


class TestDomeSweepVapor:
    """Enthalpy sweep at P=7.1 MPa with VAPOR hint."""

    def test_all_finite(self, dome_data):
        """All returned T and ρ are finite across h_l-50kJ to h_v+50kJ."""
        h_arr = jnp.linspace(
            dome_data["h_l"] - 50e3,
            dome_data["h_v"] + 50e3,
            100,
        )
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, VAPOR, dtype=jnp.int32)

        T_out, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        assert jnp.all(jnp.isfinite(T_out)), f"NaN/Inf in T: {T_out}"
        assert jnp.all(jnp.isfinite(rho_out)), f"NaN/Inf in ρ: {rho_out}"

    def test_rho_equals_rho_v_inside_dome(self, dome_data):
        """Inside dome with VAPOR hint, ρ should equal ρ_v(P)."""
        h_arr = jnp.linspace(dome_data["h_l"], dome_data["h_v"], 50)
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, VAPOR, dtype=jnp.int32)

        _, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        np.testing.assert_allclose(
            np.array(rho_out), dome_data["rho_v"], rtol=1e-10,
            err_msg="ρ != ρ_v inside dome with VAPOR hint",
        )


class TestDomeCpPositive:
    """Cp computed from dome-returned (T, ρ) must be positive."""

    def test_cp_positive_liquid(self, dome_data):
        """Cp > 0 across entire sweep with LIQUID hint."""
        h_arr = jnp.linspace(
            dome_data["h_l"] - 50e3,
            dome_data["h_v"] + 50e3,
            100,
        )
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, LIQUID, dtype=jnp.int32)

        T_out, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        cp_vals = jax.vmap(sw._scalar_cp)(T_out, rho_out)
        assert jnp.all(jnp.isfinite(cp_vals)), f"Non-finite Cp: {cp_vals}"
        assert jnp.all(cp_vals > 0), f"Negative Cp found: {cp_vals[cp_vals <= 0]}"

    def test_cp_positive_vapor(self, dome_data):
        """Cp > 0 across entire sweep with VAPOR hint."""
        h_arr = jnp.linspace(
            dome_data["h_l"] - 50e3,
            dome_data["h_v"] + 50e3,
            100,
        )
        P_arr = jnp.full_like(h_arr, P_TEST)
        hint_arr = jnp.full(h_arr.shape, VAPOR, dtype=jnp.int32)

        T_out, rho_out = state_from_Ph(P_arr, h_arr, hint_arr)
        cp_vals = jax.vmap(sw._scalar_cp)(T_out, rho_out)
        assert jnp.all(jnp.isfinite(cp_vals)), f"Non-finite Cp: {cp_vals}"
        assert jnp.all(cp_vals > 0), f"Negative Cp found: {cp_vals[cp_vals <= 0]}"


class TestDomeGradients:
    """Gradients through state_from_Ph at a point inside the dome."""

    def test_grad_finite_inside_dome(self, dome_data):
        """jax.grad returns finite values for a point inside the dome."""
        h_mid = 0.5 * (dome_data["h_l"] + dome_data["h_v"])
        P = jnp.float64(P_TEST)
        h = jnp.float64(h_mid)

        # grad of T w.r.t. P
        grad_T_P = jax.grad(lambda P: _state_from_Ph(P, h, LIQUID)[0])(P)
        assert jnp.isfinite(grad_T_P), f"dT/dP not finite: {grad_T_P}"

        # grad of T w.r.t. h (should be ~0 inside dome)
        grad_T_h = jax.grad(lambda h: _state_from_Ph(P, h, LIQUID)[0])(h)
        assert jnp.isfinite(grad_T_h), f"dT/dh not finite: {grad_T_h}"
        assert jnp.abs(grad_T_h) < 1e-10, f"dT/dh should be ~0 inside dome, got {grad_T_h}"

        # grad of rho w.r.t. P
        grad_rho_P = jax.grad(lambda P: _state_from_Ph(P, h, LIQUID)[1])(P)
        assert jnp.isfinite(grad_rho_P), f"dρ/dP not finite: {grad_rho_P}"

        # grad of rho w.r.t. h (should be ~0 inside dome)
        grad_rho_h = jax.grad(lambda h: _state_from_Ph(P, h, LIQUID)[1])(h)
        assert jnp.isfinite(grad_rho_h), f"dρ/dh not finite: {grad_rho_h}"
        assert jnp.abs(grad_rho_h) < 1e-10, f"dρ/dh should be ~0 inside dome, got {grad_rho_h}"

    def test_jvp_finite_inside_dome(self, dome_data):
        """JVP returns finite tangents for a point inside the dome."""
        h_mid = 0.5 * (dome_data["h_l"] + dome_data["h_v"])
        P = jnp.float64(P_TEST)
        h = jnp.float64(h_mid)

        primals, tangents = jax.jvp(
            lambda P, h: _state_from_Ph(P, h, LIQUID),
            (P, h), (1.0, 0.0),
        )
        assert all(jnp.isfinite(p) for p in primals)
        assert all(jnp.isfinite(t) for t in tangents)


class TestDomeSupercriticalBypass:
    """At supercritical P, dome detection should not activate."""

    def test_supercritical_unchanged(self):
        """state_from_Ph at P > PC should behave exactly as before."""
        P = jnp.float64(10.0e6)  # supercritical
        T_ref = 320.0
        rho_ref = sw._scalar_pressure  # dummy — compute h from known state
        from co2_eos.inversions import _density_from_PT
        rho = _density_from_PT(jnp.float64(T_ref), P, SUPERCRITICAL)
        h = sw._scalar_enthalpy(jnp.float64(T_ref), rho)

        T_out, rho_out = _state_from_Ph(P, h, SUPERCRITICAL)
        np.testing.assert_allclose(float(T_out), T_ref, atol=1e-6)
        np.testing.assert_allclose(float(rho_out), float(rho), rtol=1e-8)
