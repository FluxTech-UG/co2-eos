"""The shipped nist2026 saturation table holds phase equilibrium.

``saturation_table_nist2026.npz`` is a Maxwell construction on the NIST
surface (scripts/generate_saturation_table.py --eos nist2026).  Its splines
between knots meet a direct Maxwell solve (equal p and g) on the surface, and
the T- and P-indexed splines invert each other.
"""
import jax
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import nist2026 as nb

E = "nist2026"


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
