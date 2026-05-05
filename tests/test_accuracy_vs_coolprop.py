"""Headline accuracy test against CoolProp.

Both libraries implement the same Span-Wagner (1996) reference equations
for thermodynamics and the same reference correlations for transport
(Laesecke-Muzny 2017 viscosity, Huber et al. 2016 thermal conductivity).
Density and viscosity therefore agree at machine precision (~1e-13) across
all three regions covered here. Quantities that depend on second derivatives
(``cp``, ``speed_of_sound``) or include a critical-enhancement term
(``thermal_conductivity``) are looser at points where ∂q/∂ρ is large,
because the inversion's ~1e-13 ρ-residual amplifies through that derivative.
The per-region per-quantity tolerances below are tightened to ~2-3× the
empirical max so a real regression still trips the test.

Grid covers three regions, each chosen to avoid dome-edge conditioning:

  - Supercritical:        T ∈ [305, 360] K, P ∈ [8, 20] MPa
  - Subcritical liquid:   T ∈ [240, 304] K, P ≥ 1.1·P_sat(T)
  - Subcritical vapor:    T ∈ [240, 304] K, P ≤ 0.9·P_sat(T)

The 10 % offset from the saturation curve in the subcritical regions keeps
us cleanly off the dome boundary, where CoolProp can switch branches and
small numerical disagreements would dominate the comparison.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

CP = pytest.importorskip("CoolProp.CoolProp")

import co2_eos as co2


# State-dict key → CoolProp PropsSI key
_CP_KEYS = {
    "density":              "D",
    "cp":                   "C",
    "speed_of_sound":       "A",
    "viscosity":            "V",
    "thermal_conductivity": "L",
}


# ── Per-(region, quantity) tolerances ──────────────────────────────────────
# Default rtol is 1e-12 (machine precision for double-precision Span-Wagner).
# Where a (region, quantity) cell needs to be looser, it gets its own entry
# with the empirical worst case in the comment. The relaxed value is ~2-3×
# the observed max, never just padded — so a real regression still trips.

_DEFAULT_RTOL = 1e-12

_RTOL_OVERRIDES = {
    # Supercritical ─ Widom-line region (T~316 K, P~10 MPa).
    # cp, w, and λ all peak near the Widom line. The inversion's ~1e-13
    # ρ-residual amplifies through the locally large dq/dρ.
    ("supercritical", "cp"):                    1e-7,   # observed max 3.7e-8
    ("supercritical", "speed_of_sound"):        2e-8,   # observed max 7.6e-9
    ("supercritical", "thermal_conductivity"):  2e-8,   # observed max 6.1e-9 (Huber critical-enhancement term is ρ-sensitive here)

    # Subcritical liquid ─ at 1.1·P_sat the compressibility is still high
    # near T~278 K, so derived quantities pick up some inversion sensitivity.
    ("subcritical_liquid", "cp"):                   5e-9,   # observed max 1.5e-9
    ("subcritical_liquid", "speed_of_sound"):       5e-9,   # observed max 1.3e-9
    ("subcritical_liquid", "thermal_conductivity"): 2e-10,  # observed max 4.8e-11

    # Subcritical vapor ─ at 0.9·P_sat near T~278 K, similar story.
    ("subcritical_vapor", "cp"):                    3e-9,   # observed max 9.4e-10
    ("subcritical_vapor", "speed_of_sound"):        1e-9,   # observed max 2.8e-10
    ("subcritical_vapor", "thermal_conductivity"): 3e-11,  # observed max 8.8e-12
}


# ── Grid construction ──────────────────────────────────────────────────────

def _supercritical_grid():
    Ts = np.linspace(305.0, 360.0, 6)        # K
    Ps = np.linspace(8.0e6, 20.0e6, 6)       # Pa
    return [(float(T), float(P), co2.SUPERCRITICAL) for T in Ts for P in Ps]


def _subcritical_liquid_grid():
    Ts = np.linspace(240.0, 303.0, 6)
    P_offsets = (1.1, 1.5, 5.0)              # multiples of P_sat(T)
    return [
        (float(T), k * float(co2.saturation_pressure(jnp.float64(T))), co2.LIQUID)
        for T in Ts for k in P_offsets
    ]


def _subcritical_vapor_grid():
    Ts = np.linspace(240.0, 303.0, 6)
    P_offsets = (0.9, 0.5, 0.1)              # fractions of P_sat(T)
    return [
        (float(T), k * float(co2.saturation_pressure(jnp.float64(T))), co2.VAPOR)
        for T in Ts for k in P_offsets
    ]


REGIONS = {
    "supercritical":      _supercritical_grid(),
    "subcritical_liquid": _subcritical_liquid_grid(),
    "subcritical_vapor":  _subcritical_vapor_grid(),
}


# ── Per-region cache: compute every state once, share across quantity tests ─

@pytest.fixture(scope="module", params=list(REGIONS.keys()))
def region_data(request):
    """Compute co2_eos and CoolProp values once per region."""
    region = request.param
    points = REGIONS[region]

    co2_values = {q: np.zeros(len(points)) for q in _CP_KEYS}
    cp_values  = {q: np.zeros(len(points)) for q in _CP_KEYS}

    for i, (T, P, hint) in enumerate(points):
        state = co2.state_from_PT(P=P, T=T, phase_hint=hint)
        for q, cp_key in _CP_KEYS.items():
            co2_values[q][i] = float(state[q])
            cp_values[q][i]  = CP.PropsSI(cp_key, "T", T, "P", P, "CO2")

    return {
        "region": region,
        "points": points,
        "co2": co2_values,
        "cp":  cp_values,
    }


@pytest.mark.parametrize("quantity", list(_CP_KEYS.keys()))
def test_matches_coolprop(region_data, quantity):
    """Per (region, quantity), every grid point agrees with CoolProp to rtol."""
    region = region_data["region"]
    rtol = _RTOL_OVERRIDES.get((region, quantity), _DEFAULT_RTOL)

    co2_arr = region_data["co2"][quantity]
    cp_arr  = region_data["cp"][quantity]
    rel_err = np.abs(co2_arr - cp_arr) / np.abs(cp_arr)

    worst = int(np.argmax(rel_err))
    T_w, P_w, hint_w = region_data["points"][worst]

    assert rel_err.max() < rtol, (
        f"{region} / {quantity}: "
        f"max rel err = {rel_err.max():.3e} > {rtol:.0e} "
        f"at (T={T_w:.3f} K, P={P_w/1e6:.4f} MPa, hint={hint_w})\n"
        f"  co2_eos  = {co2_arr[worst]:.15e}\n"
        f"  coolprop = {cp_arr[worst]:.15e}"
    )
