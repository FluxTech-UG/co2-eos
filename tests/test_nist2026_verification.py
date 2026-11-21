"""nist2026 reproduces the values NIST IR 8608 computes from its equation.

The report prints no table of computed states for implementers.  The values
it does print that come from the equation itself are the check values here,
each to the digits printed (half a unit in the last printed digit):

  * Table 1, p. 2: p_c = 7.377 03 MPa at (T_c, ρ_c); p_t = 0.517 95 MPa, the
    saturation pressure at T_t = 216.5909 K;
  * p. 5: the reference state of n°6 and n°7, saturated liquid at 273.15 K
    with h = 200 kJ/kg and s = 1 kJ/(kg·K).

The saturation states are solved here by a Maxwell construction (equal p and
g) on the equation itself, so no table's spline error enters.
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from co2_eos import nist2026 as nb
from co2_eos import nist2026_data as nd


def _maxwell(T, rho_l0, rho_v0, iters=40):
    """Coexisting (ρ_l, ρ_v) at T: Newton on (ln ρ_l, ln ρ_v)."""
    def F(x):
        rl, rv = jnp.exp(x[0]), jnp.exp(x[1])
        return jnp.array([
            (nb._scalar_pressure(T, rl) - nb._scalar_pressure(T, rv)) / nb.PC,
            (nb._scalar_gibbs_energy(T, rl) - nb._scalar_gibbs_energy(T, rv))
            / (nb.R * T)])
    J = jax.jacfwd(F)
    x = jnp.log(jnp.array([rho_l0, rho_v0]))
    for _ in range(iters):
        x = x - jnp.linalg.solve(J(x), F(x))
    assert float(jnp.max(jnp.abs(F(x)))) < 1e-13
    return float(jnp.exp(x[0])), float(jnp.exp(x[1]))


def _printed(value, printed, half_unit):
    assert abs(value - printed) <= half_unit, (value, printed, half_unit)


def test_critical_pressure_table1():
    _printed(float(nb._scalar_pressure(nb.TC, nb.RHOC)), nd.PC_TABLE, 0.005e3)


def test_reference_state_saturated_liquid_273K():
    rl, _ = _maxwell(nd.REF_T, 927.0, 97.0)
    h = float(nb._scalar_enthalpy(nd.REF_T, rl))
    s = float(nb._scalar_entropy(nd.REF_T, rl))
    # n°6 and n°7 are given to 17 digits: the zeros hold to round-off.
    assert abs(h - nd.REF_H_LIQ) < 1e-6, h
    assert abs(s - nd.REF_S_LIQ) < 1e-9, s


def test_triple_point_pressure_table1():
    rl, _ = _maxwell(nd.T_TRIPLE, 1178.0, 14.0)
    _printed(float(nb._scalar_pressure(nd.T_TRIPLE, rl)), nd.P_TRIPLE_TABLE, 0.005e3)
