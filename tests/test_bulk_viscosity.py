"""Bulk viscosity ζ = ζ_vib + ζ_crit (0.5.0).

The reference table is the one computed for the 1DSim4 bulk-viscosity study
(2026-10-28) with CoolProp's HEOS properties fed into the same model:
Span-Wagner c, c_p, c_v and c_v⁰, Laesecke-Muzny μ, the Huber 2016
Olchowy-Sengers ξ.  co2-eos implements the same correlations and agrees with
CoolProp on those inputs to below 1e-8 (``test_accuracy_vs_coolprop``), so the
residual against the table is its printed rounding: three significant figures
(at most 0.5 % relative) for ζ_vib and τ₁₁, two for ζ_crit (at most 5 %).

Beyond the table: the dilute limit against Cramer (2012), ζ_crit = 0 where
ξ = 0, the zero-frequency guard ω·τ₁₁ ≪ 1 at the grid's top frequency, the
smooth mean-free-path switch, and finite jvp and grad through every branch.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as co2
from co2_eos import transport as tr
from co2_eos.core import _thermo

# (P [Pa], T [K], τ₁₁ [ns], ζ_vib [Pa·s], ζ_crit [Pa·s]); ζ_crit None where
# the table prints fewer than two significant figures.
TABLE = [
    (9.1e6, 296.0, 13.8, 0.252, 9.0e-5),
    (9.1e6, 305.0, 14.9, 0.158, 1.3e-4),
    (9.1e6, 314.0, 21.6, 0.0667, 1.9e-4),
    (9.1e6, 320.0, 30.5, 0.0688, 1.4e-5),
    (9.1e6, 350.0, 42.4, 0.0640, 2.3e-7),
    (9.1e6, 400.0, 44.6, 0.0508, 1.0e-8),
    (9.1e6, 450.0, 41.9, 0.0405, None),
    (9.1e6, 500.0, 37.8, 0.0324, 0.0),
    (7.5e6, 305.0, 27.5, 0.0368, 1.2e-3),
]
IDS = [f"{P / 1e6:g}MPa_{T:g}K" for P, T, *_ in TABLE]


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


@pytest.mark.parametrize("P,T,t11_ns,zvib,zcrit", TABLE, ids=IDS)
def test_table_states(P, T, t11_ns, zvib, zcrit):
    zv, zc, t11, _ = _parts(T, _rho(P, T))
    assert abs(t11 * 1e9 / t11_ns - 1.0) < 5e-3
    assert abs(zv / zvib - 1.0) < 5e-3
    if zcrit is None:
        return
    if zcrit == 0.0:
        assert zc == 0.0
    else:
        assert abs(zc / zcrit - 1.0) < 5e-2


@pytest.mark.parametrize("P,T,t11_ns,zvib,zcrit", TABLE, ids=IDS)
def test_public_entry_points_agree(P, T, t11_ns, zvib, zcrit):
    rho = _rho(P, T)
    zv, zc, _, _ = _parts(T, rho)
    z = float(co2.bulk_viscosity(T, rho))
    assert z == pytest.approx(zv + zc, rel=1e-13)
    full = co2.properties(T, rho)
    assert float(full["bulk_viscosity"]) == pytest.approx(z, rel=1e-13)
    assert float(full["bulk_viscosity_vibrational"]) == pytest.approx(zv, rel=1e-13)
    batched = co2.properties_from_rho_T(jnp.array([rho]), jnp.array([T]))
    assert float(batched["bulk_viscosity"][0]) == pytest.approx(z, rel=1e-13)
    assert float(tr.bulk_viscosity(jnp.array([T]), jnp.array([rho]))[0]) \
        == pytest.approx(z, rel=1e-13)


def test_every_full_state_carries_bulk_viscosity():
    rho = _rho(9.1e6, 314.0)
    u = co2.properties(314.0, rho)["internal_energy"]
    for state in (co2.state_from_PT(9.1e6, 314.0),
                  co2.state_from_Du(rho, u),
                  co2.properties_from_rho_u(jnp.array([rho]),
                                            jnp.array([u]))):
        assert "bulk_viscosity" in state
        assert np.all(np.asarray(state["bulk_viscosity"]) > 0.0)


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


def test_zeta_crit_vanishes_where_xi_does():
    rho = _rho(9.1e6, 500.0)
    zv, zc, _, xi = _parts(500.0, rho)
    assert xi == 0.0 and zc == 0.0
    assert float(co2.bulk_viscosity(500.0, rho)) == pytest.approx(zv, rel=1e-14)


@pytest.mark.parametrize("P,T,t11_ns,zvib,zcrit", TABLE, ids=IDS)
def test_zero_frequency_holds_at_grid_top(P, T, t11_ns, zvib, zcrit):
    """ω·τ₁₁ at 110 kHz (the N_x 2048 Nyquist on a 4 m loop) below 0.05."""
    _, _, t11, _ = _parts(T, _rho(P, T))
    assert 2.0 * np.pi * 110e3 * t11 < 0.05


def test_mean_free_path_switch_is_smooth_and_faithful():
    """The logistic switch tracks Lin & Trusler's hard switch at 896 kg/m³."""
    T = 300.0
    for rho in (100.0, 472.0, 722.0, 821.0):
        n = rho / tr.M * tr._NA
        lam_dil = 1.0 / (np.sqrt(2.0) * np.pi * tr._SIGMA_HS ** 2 * n)
        v = np.sqrt(8.0 * tr.R_MOLAR * T / (np.pi * tr.M))
        hard = lam_dil / (v * tr._P11_A * np.exp(T / tr._P11_T))
        smooth = float(tr._vibrational_relaxation_time(T, rho))
        assert abs(smooth / hard - 1.0) < 1e-4, rho
    # Continuous through the switch: a 0.1 kg/m³ step moves τ₁₁ by < 0.1 %.
    rhos = jnp.linspace(870.0, 920.0, 501)
    t = jax.vmap(lambda r: tr._vibrational_relaxation_time(T, r))(rhos)
    assert float(jnp.max(jnp.abs(jnp.diff(t)) / t[:-1])) < 1e-3


GRAD_STATES = [(296.0, 821.1), (314.0, 472.1), (305.0, 389.8),
               (500.0, 102.4), (300.0, 896.0), (300.0, 1.8)]


@pytest.mark.parametrize("T,rho", GRAD_STATES)
def test_jvp_and_grad_finite(T, rho):
    T, rho = jnp.float64(T), jnp.float64(rho)
    g = jax.grad(co2.bulk_viscosity, argnums=(0, 1))(T, rho)
    assert all(np.isfinite(float(x)) for x in g)
    _, tan = jax.jvp(co2.bulk_viscosity, (T, rho),
                     (jnp.float64(1.0), jnp.float64(1.0)))
    assert np.isfinite(float(tan))
    # The forward and reverse modes agree on the same directional derivative.
    assert float(tan) == pytest.approx(float(g[0] + g[1]), rel=1e-10)
    # The batched bundle differentiates too.
    gb = jax.grad(lambda r: co2.properties_from_rho_T(
        r[None], T[None])["bulk_viscosity"][0])(rho)
    assert np.isfinite(float(gb))


def test_conductivity_gradient_finite_where_xi_is_zero():
    """0.5.0 fix: λ's gradient was NaN where the correlation length is zero."""
    T, rho = jnp.float64(500.0), jnp.float64(102.4)
    g = jax.grad(co2.thermal_conductivity, argnums=(0, 1))(T, rho)
    assert all(np.isfinite(float(x)) for x in g)
