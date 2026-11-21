"""Bulk viscosity ζ = ζ_vib + ζ_crit: the dilute limit against Cramer (2012)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tests.span_wagner_api import co2, core, transport as tr   # Span-Wagner, eos bound

_thermo = core._thermo

def _parts(T, rho):
    """(ζ_vib, ζ_crit, τ₁₁, ξ) at scalar (T, ρ) through the shipped kernels."""
    T = jnp.float64(T)
    (_, cv, cp, w, *_), (ar_d, ar_dd, _, _, a0_tt) = _thermo(T, rho)
    mu = tr._scalar_viscosity(T, rho)
    xi, _ = tr._correlation_length_and_excess(T, rho, ar_d, ar_dd)
    zv, zc, t11 = tr._bulk_viscosity_parts(T, rho, mu, w, cp, cv, a0_tt, xi)
    return float(zv), float(zc), float(t11), float(xi)


def _rho(P, T):
    rho = co2.density_from_PT(P, T)
    assert np.isfinite(float(rho)), (P, T)
    return rho


@pytest.mark.parametrize("T", [300.0, 450.0, 500.0])
def test_dilute_limit_against_cramer(T):
    """1 atm: τ₁₁ against Cramer's p·τ_v = 0.19152 (T/800 K)^−1.353 Pa·s.

    Cramer fits Simpson et al.'s shock-tube data (296 to 1711 K); the
    collision model is fitted to independent gas-phase data, so agreement to
    15 % (2 % hot, 13 % at 300 K) checks both the constants and the hot-side
    extrapolation.  At 300 K the form's ζ/μ is held to Cramer's 3849 within
    the model's factor-1.5 form uncertainty.
    """
    P = 101325.0
    rho = _rho(P, T)
    zv, zc, t11, _ = _parts(T, rho)
    t_cramer = 0.19152 * (T / 800.0) ** -1.353 / P
    assert abs(t11 / t_cramer - 1.0) < 0.15
    assert zc < 1e-9 * zv
    if T == 300.0:
        ratio = (zv + zc) / float(co2.viscosity(T, rho))
        assert 3849.0 / 1.5 < ratio < 3849.0 * 1.5
