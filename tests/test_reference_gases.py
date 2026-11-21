"""The reference-gas backends (air, helium, argon) behind the ``eos=`` keyword.

Three references, none of them an identity; each is a table from outside the
code, less sturdy than a conservation law:

* The stored CoolProp values (tests/data/reference_gas_values.csv, CoolProp
  7.2.0, generated once by scripts/generate_reference_gas_values.py, never
  compared by hash).  CoolProp evaluates the full published equations of state
  (air Lemmon et al. 2000, argon Tegeler et al. 1999, helium Ortiz-Vega et al.,
  the REFPROP 10 coefficients); this backend keeps only their virial terms, so
  the comparison checks the truncation and the transcription against the bars
  ``co2_eos/reference_gases_data.py`` states: density 0.05 % to ``P_MAX``, c_p
  and c_v 0.2 %, speed of sound 0.1 %.  It stops holding past ``P_MAX``, where
  the series is not claimed.  Viscosity and conductivity are not compared here:
  the printed check values below reach the same terms at higher density, and
  CoolProp's helium transport is another correlation (Arp et al.; Hands and Arp).
* The check values Lemmon and Jacobsen (2004) print for air and argon transport.
* Helium's dilute transport against the ab initio table it is fitted to (Hurly
  and Mehl 2007, Table 5, pp. 87-88; their Eq. 37, p. 90, puts the expanded
  uncertainty (k = 2) of η and λ at 0.0002 + 0.005 K/T, about 0.02 % from 200 to
  1000 K).  The table is printed to six digits (a half unit of the last one is
  1.1e-6 to 4.2e-6 relative), so the fit is held to 5e-6, far inside the 0.02 %;
  the table, not the fit, carries the physics.
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
from co2_eos import reference_gases_data

GASES = ("air", "helium", "argon")
VALUES = Path(__file__).parent / "data" / "reference_gas_values.csv"

# The stated bars of the route (co2_eos/reference_gases_data.py): density 0.05 %,
# c_p and c_v 0.2 %, speed of sound 0.1 %.
TOL = {"rho": 5e-4, "cp": 2e-3, "cv": 2e-3, "w": 1e-3}


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
@pytest.mark.parametrize("key,field", [
    ("rho", "density"), ("cp", "cp"), ("cv", "cv"), ("w", "speed_of_sound")])
def test_against_the_stored_coolprop_values(gas, key, field):
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


def test_helium_transport_against_its_ab_initio_table():
    """Helium's η and λ, the cubic fit in ln T, meet every row of Hurly and Mehl's table."""
    for T, eta, lam in reference_gases_data.HELIUM_DILUTE_TABLE:
        assert float(ce.viscosity(T, 1.0, eos="helium")) * 1e6 == pytest.approx(eta, rel=5e-6), T
        assert float(ce.thermal_conductivity(T, 1.0, eos="helium")) * 1e3 == pytest.approx(
            lam, rel=5e-6), T
