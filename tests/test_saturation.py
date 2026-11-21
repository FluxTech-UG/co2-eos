"""The shipped saturation tables hold phase equilibrium, up to the critical point.

Each CO₂ backend ships ``saturation_table<suffix>.npz``, a Maxwell construction on
its own surface (scripts/generate_saturation_table.py --eos <name>).  The
reference here is a thermodynamic identity, the sturdiest kind this suite has:
two coexisting phases have equal pressure and equal Gibbs energy at the same T.
A direct Newton solve of those two equations on the surface gives (ρ_l, ρ_v),
and the table's splines between knots must meet it; the T- and P-indexed
splines must invert each other.

What it leans on: the surface itself.  It proves the tables belong to the
equation the package evaluates, not that the equation is right (the printed
NIST values and CoolProp do that).  Where it stops holding: at T_c the two
phases merge and the Newton system is singular, so the nearest state checked is
304.12 K, 0.008 K below T_c = 304.1282 K, where the coexisting densities still
differ by about 10 % and the splines are at their steepest.
"""
import jax
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import backends

from tests.test_nist2026_verification import _maxwell


@pytest.mark.parametrize("eos", ce.CO2_EOS_NAMES)
@pytest.mark.parametrize("T", [230.37, 263.11, 291.9, 303.55, 304.1, 304.12])
def test_table_round_trip_against_maxwell(eos, T):
    surf = backends.get(eos).surface
    rl0, rv0 = (float(x) for x in ce.saturation_densities(T, eos=eos))
    rl, rv = _maxwell(T, rl0, rv0, surf=surf)
    p = float(surf._scalar_pressure(T, rl))
    assert abs(float(ce.saturation_pressure(T, eos=eos)) / p - 1.0) < 1e-8
    assert abs(rl0 / rl - 1.0) < 1e-6 and abs(rv0 / rv - 1.0) < 1e-6
    # The P-indexed splines invert the T-indexed ones.
    assert abs(float(ce.saturation_temperature(p, eos=eos)) - T) < 1e-6
    hl, hv = ce.saturation_enthalpies_P(p, eos=eos)
    assert abs(float(hl) - float(surf._scalar_enthalpy(T, rl))) < 0.05
    assert abs(float(hv) - float(surf._scalar_enthalpy(T, rv))) < 0.05
