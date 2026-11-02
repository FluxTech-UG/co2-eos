"""The near-critical regression the nist2026 backend exists for.

Span-Wagner's nonanalytic terms (span_wagner.py terms 40 to 42) make its
derivatives oscillate near the critical point: SW 1996 warns that
(∂c_v/∂T)_ρ oscillates for about 420 to 550 kg/m³ within 15 K of T_c, and
NIST IR 8608 Sec. 5 (Figs. 25 and 26, pp. 28 to 29) shows an unphysical dip
in c_p at T_c + 0.1 K and in κ_T at 304.3 K within about 10 % of ρ_c.  The
NIST equation has no nonanalytic terms and is smooth there.

Measured on this implementation (2026-11-02):

* (∂c_v/∂T)_ρ from T_c + 0.05 K to T_c + 15 K (2000 log-spaced points) on the
  isochores 420 to 550 kg/m³ in steps of 5: nist2026 is finite, negative and
  monotone in T on every one, the critical isochore included.  Span-Wagner's
  oscillates on the isochores 505 to 550 kg/m³: two turning points between
  T − T_c ≈ 5.6 and 7.9 K, a wiggle of amplitude up to 0.75 J/(kg·K²) (at
  550 kg/m³, on a slope of about −19 J/(kg·K²)).  On the critical isochore
  itself Span-Wagner's (∂c_v/∂T)_ρ is monotone at this resolution, so the
  oscillation is asserted where it exists, not on ρ_c.
* c_p at T_c + 0.1 K over 0.85 to 1.15 ρ_c: nist2026 has one peak,
  114.4 kJ/(mol·K) at 462 kg/m³ (Fig. 25 fig-read: about 114 at 10.5 mol/L);
  Span-Wagner has humps of 87.0 and 86.3 at 457 and 479 kg/m³ around a dip
  of 82.2 at 467.6 (fig-read: 87, 86 and 82).
* κ_T at 304.3 K over the same densities: nist2026 has one peak, 69.7 MPa⁻¹
  at 462 kg/m³ (Fig. 26 fig-read: about 70); Span-Wagner has humps of 59.4
  and 57.4 around a dip of 56.2 MPa⁻¹ (fig-read: 59 and 57).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import backends, core

TC = 304.1282
T_ISO = TC + np.geomspace(0.05, 15.0, 2000)
RHO_ISO = np.arange(420.0, 551.0, 5.0)
M_NIST = backends.get("nist2026").surface.M


def _dcv_dT(eos):
    c = core.for_eos(eos)
    g = jax.grad(lambda T, r: c._thermo(T, r)[0][1])
    f = jax.jit(jax.vmap(jax.vmap(g, (0, None)), (None, 0)))
    return np.asarray(f(jnp.asarray(T_ISO), jnp.asarray(RHO_ISO)))


def _turning_points(y):
    s = np.sign(np.diff(y))
    return np.where(s[1:] != s[:-1])[0] + 1


def test_dcv_dT_monotone_and_finite_on_nist2026():
    D = _dcv_dT("nist2026")
    assert np.all(np.isfinite(D))
    assert np.all(D < 0.0)
    for r, d in zip(RHO_ISO, D):
        assert len(_turning_points(d)) == 0, f"(∂c_v/∂T) turns at ρ = {r}"
    rc = backends.get("nist2026").surface.RHOC
    d_c = np.asarray(jax.vmap(jax.grad(
        lambda T: core.for_eos("nist2026")._thermo(T, rc)[0][1]))(
            jnp.asarray(T_ISO)))
    assert np.all(np.isfinite(d_c)) and len(_turning_points(d_c)) == 0


def test_dcv_dT_oscillates_on_span_wagner():
    D = _dcv_dT("span_wagner")
    oscillating = [r for r, d in zip(RHO_ISO, D) if len(_turning_points(d))]
    assert oscillating and min(oscillating) <= 505.0 and max(oscillating) == 550.0
    d = D[-1]                                    # 550 kg/m³
    tp = _turning_points(d)
    assert len(tp) == 2
    assert 5.0 < T_ISO[tp[0]] - TC < T_ISO[tp[1]] - TC < 8.5
    amplitude = abs(d[tp[0]] - d[tp[1]])
    assert 0.5 < amplitude < 1.0, amplitude


def _scan(eos, T, key):
    rc = backends.get(eos).surface.RHOC
    r = jnp.asarray(np.linspace(0.85, 1.15, 1201) * rc)
    st = ce.properties_from_rho_T(r, jnp.full(r.shape, T), eos=eos)
    if key == "kappa_T":
        y = np.asarray(1.0 / (r * st["dP_drho_T"])) * 1e6       # MPa⁻¹
    else:
        y = np.asarray(st["cp"]) * M_NIST * 1e-3                # kJ/(mol·K)
    return np.asarray(r), y


@pytest.mark.parametrize("T,key,peak,at", [
    (TC + 0.1, "cp", 114.4, 462.0),
    (304.3, "kappa_T", 69.7, 462.0),
])
def test_single_smooth_peak_on_nist2026(T, key, peak, at):
    r, y = _scan("nist2026", T, key)
    assert np.all(np.isfinite(y)) and np.all(y > 0.0)
    tp = _turning_points(y)
    assert len(tp) == 1
    assert abs(y[tp[0]] - peak) < 0.1 and abs(r[tp[0]] - at) < 1.0


@pytest.mark.parametrize("T,key", [(TC + 0.1, "cp"), (304.3, "kappa_T")])
def test_span_wagner_dip(T, key):
    _, y = _scan("span_wagner", T, key)
    tp = _turning_points(y)
    assert len(tp) == 3                          # hump, dip, hump
    hump1, dip, hump2 = y[tp]
    assert dip < hump1 and dip < hump2
