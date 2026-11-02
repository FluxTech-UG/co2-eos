"""The nist2026 saturation table, its auxiliary-equation check and the dome.

``saturation_table_nist2026.npz`` is a Maxwell construction on the NIST
surface (scripts/generate_saturation_table.py --eos nist2026).  Checks:

* the table against NIST IR 8608 Eqs. 16 to 18 (pp. 32 to 33), which the
  report says reproduce the equation's saturation boundary within 0.004 %
  (p_s) and 0.01 % (ρ_L, ρ_V, except very near T_c).  Measured: p_s within
  0.0040 % from 220 K to 301 K, a smooth fit residual of about ±0.001 %; at
  the triple point Eq. 16 reads 0.517 983 MPa where the equation gives the
  printed p_t = 0.517 95 MPa (test_nist2026_verification), so the 0.0064 %
  there is Eq. 16's edge, not the surface's.  ρ_L and ρ_V within 0.01 % below
  300 K (0.027 % and 0.13 % in the last 0.5 K);
* the table's round trip: the spline between knots against a direct Maxwell
  solve on the surface, and the T- and P-indexed splines against each other;
* the dome detection of state_from_Ph on the nist2026 surface.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import nist2026 as nb
from co2_eos import nist2026_data as nd

E = "nist2026"


def _aux(T, a, t):
    x = 1.0 - T / nd.TC
    return sum(ai * x ** ti for ai, ti in zip(a, t))


def test_auxiliary_equations_against_table():
    T = np.linspace(220.0, 300.0, 161)
    Tj = jnp.asarray(T)
    ps = np.asarray(ce.saturation_pressure(Tj, eos=E))
    rl, rv = (np.asarray(x) for x in ce.saturation_densities(Tj, eos=E))
    ps_aux = nd.PC_TABLE * np.exp(nd.TC / T * _aux(T, nd.AUX_PS_A, nd.AUX_PS_T))
    rl_aux = nb.RHOC * np.exp(_aux(T, nd.AUX_RHOL_A, nd.AUX_RHOL_T))
    rv_aux = nb.RHOC * np.exp(_aux(T, nd.AUX_RHOV_A, nd.AUX_RHOV_T))
    assert np.max(np.abs(ps_aux / ps - 1.0)) < nd.AUX_PS_TOL
    assert np.max(np.abs(rl_aux / rl - 1.0)) < nd.AUX_RHO_TOL
    assert np.max(np.abs(rv_aux / rv - 1.0)) < nd.AUX_RHO_TOL


@pytest.mark.parametrize("T", [230.37, 263.11, 291.9, 303.55, 304.1])
def test_table_round_trip_against_maxwell(T):
    from tests.test_nist2026_verification import _maxwell
    rl0, rv0 = (float(x) for x in ce.saturation_densities(T, eos=E))
    rl, rv = _maxwell(T, rl0, rv0)
    p = float(nb._scalar_pressure(T, rl))
    assert abs(float(ce.saturation_pressure(T, eos=E)) / p - 1.0) < 1e-8
    assert abs(rl0 / rl - 1.0) < 1e-6 and abs(rv0 / rv - 1.0) < 1e-6
    # The P-indexed splines invert the T-indexed ones.
    assert abs(float(ce.saturation_temperature(p, eos=E)) - T) < 1e-6
    hl, hv = ce.saturation_enthalpies_P(p, eos=E)
    assert abs(float(hl) - float(nb._scalar_enthalpy(T, rl))) < 0.05
    assert abs(float(hv) - float(nb._scalar_enthalpy(T, rv))) < 0.05


P_DOME = 7.1e6


def test_dome_detection_on_nist2026():
    T_sat = float(ce.saturation_temperature(P_DOME, eos=E))
    h_l, h_v = (float(x) for x in ce.saturation_enthalpies_P(P_DOME, eos=E))
    for hint in (ce.LIQUID, ce.VAPOR, ce.AUTO):
        for frac in (0.1, 0.5, 0.9):
            st = ce.state_from_Ph(P_DOME, h_l + frac * (h_v - h_l), hint, eos=E)
            assert abs(float(st["temperature"]) - T_sat) < 1e-6
            assert np.isfinite(float(st["density"]))
    # Outside the dome on either side: a single-phase solve that reproduces
    # (P, h) on the surface.
    for h in (h_l - 30e3, h_v + 30e3):
        st = ce.state_from_Ph(P_DOME, h, eos=E)
        T, rho = float(st["temperature"]), float(st["density"])
        assert abs(float(nb._scalar_pressure(T, rho)) / P_DOME - 1.0) < 1e-9
        assert abs(float(nb._scalar_enthalpy(T, rho)) - h) < 1e-4
    # Supercritical pressure bypasses the dome.
    st = ce.state_from_Ph(9.1e6, 3.6e5, eos=E)
    assert float(st["temperature"]) > nd.TC


def test_dome_gradients_finite_on_nist2026():
    h_l, h_v = (float(x) for x in ce.saturation_enthalpies_P(P_DOME, eos=E))
    g = jax.grad(lambda p, h: ce.state_from_Ph(p, h, eos=E)["temperature"],
                 argnums=(0, 1))(P_DOME, 0.5 * (h_l + h_v))
    assert all(np.isfinite(float(x)) for x in g)
