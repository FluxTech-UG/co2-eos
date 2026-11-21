"""The reference-gas backends (air, helium, argon) behind the ``eos=`` keyword.

The values meet the stated tolerances against the stored CoolProp reference
(tests/data/reference_gas_values.csv, generated once by
scripts/generate_reference_gas_values.py, never compared by hash), and the
transport meets the check values Lemmon and Jacobsen (2004) print.
"""

import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import backends

GASES = ("air", "helium", "argon")
VALUES = Path(__file__).parent / "data" / "reference_gas_values.csv"

# The stated tolerances of the route: c_p and c_v 0.2 %, speed of sound 0.1 %.
TOL = {"cp": 2e-3, "cv": 2e-3, "w": 1e-3}


def _table():
    with VALUES.open() as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    out = {}
    for gas in GASES:
        sel = [r for r in rows if r["gas"] == gas]
        out[gas] = {k: np.array([float(r[k]) for r in sel]) for k in sel[0]
                    if k not in ("gas", "tag")}
        out[gas]["tag"] = np.array([r["tag"] for r in sel])
    return out


TABLE = _table()


def _model(gas):
    t = TABLE[gas]
    rho = np.asarray(jax.vmap(lambda p, T: ce.density_from_PT(p, T, eos=gas))(t["P_Pa"], t["T_K"]))
    return rho, ce.properties_from_rho_T(jnp.asarray(rho), jnp.asarray(t["T_K"]), eos=gas)


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("key,field", [("cp", "cp"), ("cv", "cv"), ("w", "speed_of_sound")])
def test_thermo_against_reference(gas, key, field):
    _, st = _model(gas)
    err = np.abs(np.asarray(st[field]) / TABLE[gas][key] - 1.0)
    i = int(np.argmax(err))
    assert err[i] < TOL[key], (
        f"{gas} {key}: {err[i]:.2e} at {TABLE[gas]['T_K'][i]} K, {TABLE[gas]['P_Pa'][i]} Pa")


@pytest.mark.parametrize("gas,table", [
    # (T [K], ρ [mol/dm³], η [μPa·s], λ [mW/(m·K)], share): Lemmon and Jacobsen (2004) Table V,
    # p. 28, the rows in the temperature range.  The dense rows lie past P_MAX (air 133 and 124
    # bar, argon 114 and 119 bar), as a fuller check of the same terms.  Their printed λ carries
    # the critical enhancement this backend leaves out; `share` is its fraction of the printed
    # λ (Eqs. 7-10, read once from CoolProp's conductivity contributions), zero on air at 300 K,
    # above T_ref = 265 K (Table I), and at zero density.
    ("air", [(300.0, 0.0, 18.5230, 26.3529, 0.0), (200.0, 10.0, 21.1392, 35.3185, 3.807e-2),
             (300.0, 5.0, 21.3241, 32.6062, 0.0)]),
    ("argon", [(300.0, 0.0, 22.7241, 17.8042, 0.0), (200.0, 10.0, 25.5662, 26.1377, 6.430e-2),
               (300.0, 5.0, 26.3706, 23.2302, 9.891e-4)]),
])
def test_transport_against_the_published_check_values(gas, table):
    surf = backends.get(gas).surface
    for T, rho_molar, eta, lam, share in table:
        rho = max(rho_molar * 1e3, 1e-12) * surf.M
        assert float(ce.viscosity(T, rho, eos=gas)) * 1e6 == pytest.approx(eta, rel=5e-5)
        ours = float(ce.thermal_conductivity(T, rho, eos=gas)) * 1e3
        assert ours / lam == pytest.approx(1.0 - share, abs=5e-5), (gas, T, rho_molar)
