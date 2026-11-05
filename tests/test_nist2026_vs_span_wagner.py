"""nist2026 against Span-Wagner away from the critical point.

At matched (P, T) (each backend inverts for its own ρ, the way Span-Wagner's
uncertainties are stated) over 230 to 800 K and 0.05 to 100 MPa, single
phase, outside the box |T − T_c| ≤ 5 K and |ρ − ρ_c| ≤ 20 % ρ_c, off the
saturation line (|P/P_sat − 1| > 0.2 %) and clear of the melting line
(T ≥ 240 K above 30 MPa).  Two surfaces each within an uncertainty u of the
truth can differ by 2u, so the bands are twice the stated uncertainty:

* T ≤ 523 K and P ≤ 30 MPa, Span-Wagner's stated region (SW 1996, its
  Figs. 38 to 40; NIST IR 8608 Sec. 8 keeps them for the new equation):
  density 0.05 %, c_p 1.5 %, w 1 % at most; bands 0.1 %, 3 %, 2 %;
* beyond it (to 800 K, 100 MPa) no single stated value: density 1.2 %, the
  ±0.6 % scatter of the data above 500 K about both equations in NIST IR 8608
  Fig. 13 (p. 18); c_p and w the same 3 % and 2 %.

The same bands hold against CoolProp's Span-Wagner (test-only).  Enthalpy
and entropy share an origin (the IIR reference state, saturated liquid at
273.15 K) on both backends; their differences elsewhere are the surfaces'.

Measured (2026-11-02; maximum, at its state):
  stated region: density 0.082 %, c_p 0.83 % (290 K, 5.25 MPa, vapour by
  saturation), w 0.86 %;
  beyond: density 0.69 % (750 K, 100 MPa), c_p 0.77 %, w 1.18 % (750 K,
  85.6 MPa); |Δh| 892 J/kg (800 K, 53.8 MPa), |Δs| 1.29 J/(kg·K)
  (490 K, 39.4 MPa).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce

TC, RHOC = 304.1282, 467.6
T1 = np.linspace(230.0, 800.0, 39)
P1 = np.geomspace(0.05e6, 100e6, 31)
TT, PP = (a.ravel() for a in np.meshgrid(T1, P1, indexing="ij"))
STATED = (TT <= 523.0) & (PP <= 30e6)
BANDS = {  # (stated region, beyond)
    "density": (0.1e-2, 1.2e-2),
    "cp": (3e-2, 3e-2),
    "speed_of_sound": (2e-2, 2e-2),
}


def _states(eos):
    rho = jax.jit(jax.vmap(lambda p, t: ce.density_from_PT(p, t, eos=eos)))(
        jnp.asarray(PP), jnp.asarray(TT))
    st = ce.properties_from_rho_T(rho, jnp.asarray(TT), eos=eos)
    psat = ce.saturation_pressure(jnp.asarray(np.minimum(TT, TC - 0.01)), eos=eos)
    return {k: np.asarray(v) for k, v in st.items()}, np.asarray(psat)


@pytest.fixture(scope="module")
def both():
    sw, psat = _states("span_wagner")
    nb, _ = _states("nist2026")
    rho = sw["density"]
    keep = (((np.abs(TT - TC) > 5.0) | (np.abs(rho / RHOC - 1.0) > 0.2))
            & ((TT >= TC) | (np.abs(PP / psat - 1.0) > 2e-3))
            & ~((TT < 240.0) & (PP > 30e6))
            & np.isfinite(rho) & np.isfinite(nb["density"]))
    return sw, nb, keep


def _check(ref, nb, keep, key):
    for region, band in ((STATED, BANDS[key][0]), (~STATED, BANDS[key][1])):
        m = keep & region
        dev = np.abs(nb[key][m] / ref[key][m] - 1.0)
        i = int(dev.argmax())
        assert dev[i] < band, (key, dev[i], TT[m][i], PP[m][i])


@pytest.mark.parametrize("key", sorted(BANDS))
def test_against_span_wagner(both, key):
    sw, nb, keep = both
    assert keep.sum() > 900
    _check(sw, nb, keep, key)


def test_reference_state_shared():
    for eos in ce.CO2_EOS_NAMES:
        T = 273.15
        rl, _ = ce.saturation_densities(T, eos=eos)
        st = ce.properties(T, rl, eos=eos)
        assert abs(float(st["enthalpy"]) - 200e3) < 20.0, eos
        assert abs(float(st["entropy"]) - 1e3) < 0.1, eos


def test_enthalpy_entropy_differences_bounded(both):
    sw, nb, keep = both
    assert np.max(np.abs(nb["enthalpy"][keep] - sw["enthalpy"][keep])) < 1.5e3
    assert np.max(np.abs(nb["entropy"][keep] - sw["entropy"][keep])) < 2.0


def test_against_coolprop(both):
    CP = pytest.importorskip("CoolProp.CoolProp")
    _, nb, keep = both
    idx = np.where(keep)[0][::3]
    ref = {"density": [], "cp": [], "speed_of_sound": []}
    for i in idx:
        ref["density"].append(CP.PropsSI("D", "T", TT[i], "P", PP[i], "CO2"))
        ref["cp"].append(CP.PropsSI("C", "T", TT[i], "P", PP[i], "CO2"))
        ref["speed_of_sound"].append(CP.PropsSI("A", "T", TT[i], "P", PP[i], "CO2"))
    sub = {k: np.asarray(v) for k, v in ref.items()}
    nbs = {k: nb[k][idx] for k in ref}
    for key in ref:
        for region, band in ((STATED[idx], BANDS[key][0]),
                             (~STATED[idx], BANDS[key][1])):
            dev = np.abs(nbs[key][region] / sub[key][region] - 1.0)
            assert dev.max() < band, (key, dev.max())
