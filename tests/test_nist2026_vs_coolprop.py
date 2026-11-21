"""nist2026, the default backend, against CoolProp in Span-Wagner's stated region.

The reference is a table from another library, the least sturdy kind this suite
has: CoolProp's CO₂ is Span and Wagner (1996), a different equation fitted to
much the same data, so the comparison catches a wrong coefficient or term in
nist2026 (it moves a property by far more than the two equations differ), not an
error both equations share.

The bands are twice the stated uncertainty, since two surfaces each within u of
the truth can differ by 2u.  Span and Wagner (1996, abstract, p. 1509) state, up
to 30 MPa and 523 K: density 0.03 % to 0.05 %, speed of sound 0.03 % to 1 %,
isobaric heat capacity 0.15 % to 1.5 %; NIST IR 8608 (Sec. 8, p. 34) keeps those
density and c_p uncertainties for its equation.  Bands: density 0.1 %, w 2 %,
c_p 3 %.  Beyond 523 K or 30 MPa no single value is stated, and no state there
is compared.

Where it stops holding: near the critical point neither source states an
uncertainty (Span and Wagner, Figs. 38 to 40, region G: "difficult to
estimate"), and there the two equations do differ (NIST IR 8608 Sec. 5:
Span-Wagner's nonanalytic terms make its derivatives oscillate), so the box
|T − T_c| ≤ 5 K, |ρ − ρ_c| ≤ 20 % ρ_c is left out, as are states within 0.2 % of
the saturation pressure (a branch choice, not a property).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

CP = pytest.importorskip("CoolProp.CoolProp")

import co2_eos as ce

E = "nist2026"
TC, RHOC = ce.critical_point(eos=E)[0], ce.critical_point(eos=E)[2]
T1 = np.linspace(230.0, 523.0, 16)
# From 0.2 MPa: density_from_PT solves no density below 1 kg/m³ (its bracket's
# floor, inversions._BISECT_LO) and returns NaN there, 0.1 MPa at 523 K.
P1 = np.geomspace(0.2e6, 30e6, 14)
TT, PP = (a.ravel() for a in np.meshgrid(T1, P1, indexing="ij"))
BANDS = {"density": ("D", 1e-3), "cp": ("C", 3e-2), "speed_of_sound": ("A", 2e-2)}


def test_against_coolprop_in_the_stated_region():
    rho = jax.jit(jax.vmap(lambda p, t: ce.density_from_PT(p, t, eos=E)))(
        jnp.asarray(PP), jnp.asarray(TT))
    st = {k: np.asarray(v) for k, v in
          ce.properties_from_rho_T(rho, jnp.asarray(TT), eos=E).items()}
    psat = np.asarray(ce.saturation_pressure(jnp.asarray(np.minimum(TT, TC - 0.01)), eos=E))
    rho = np.asarray(rho)
    keep = (((np.abs(TT - TC) > 5.0) | (np.abs(rho / RHOC - 1.0) > 0.2))
            & ((TT >= TC) | (np.abs(PP / psat - 1.0) > 2e-3)))
    assert np.all(np.isfinite(rho[keep])) and keep.sum() > 180, int(keep.sum())
    for key, (cp_key, band) in BANDS.items():
        ref = np.array([CP.PropsSI(cp_key, "T", t, "P", p, "CO2")
                        for t, p in zip(TT[keep], PP[keep])])
        dev = np.abs(st[key][keep] / ref - 1.0)
        i = int(dev.argmax())
        assert dev[i] < band, (key, dev[i], TT[keep][i], PP[keep][i])
