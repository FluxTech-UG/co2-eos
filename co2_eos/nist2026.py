"""NIST IR 8608 (2026) equation of state for CO₂: pure JAX, analytic derivatives.

Lemmon, Harvey and Hellmann, "Alternative Fundamental Equation of State for
Fluid Carbon Dioxide", NIST IR 8608 (2026), doi:10.6028/NIST.IR.8608.  A
24-term Helmholtz equation (5 polynomial, 7 exponential, 12 Gaussian
bell-shaped terms) with no nonanalytic terms, free of Span-Wagner's
near-critical oscillations in c_p, κ_T and (∂c_v/∂T)_ρ (report Sec. 5).

This module is the second backend of co2-eos, selected by ``eos="nist2026"``
on every public function.  It provides, under the same names and with the
same return shapes, what ``span_wagner`` and ``span_wagner_kernel`` provide for the
first backend:

  * the constants ``TC, RHOC, RHOC_MOLAR, R, R_MOLAR, M, PC, T_TRIPLE``;
  * the plain reduced Helmholtz energy ``alpha0``, ``alphar``, ``alpha``
    (written term by term, the reference the analytic derivatives are tested
    against with ``jax.grad``);
  * the analytic kernels ``ideal_derivs``, ``ideal_tau_only``,
    ``residual_derivs``, ``residual_tau_prep``, ``residual_tau_fast``,
    ``residual_tau_derivs``;
  * the scalar property functions ``_scalar_pressure``, ``_scalar_enthalpy``,
    ``_scalar_internal_energy``, ``_scalar_entropy``, ``_scalar_cv``,
    ``_scalar_cp``, ``_scalar_speed_of_sound``, ``_scalar_gibbs_energy``.

The coefficients live in ``nist2026_data`` (the single source).  None of the
Span-Wagner economizations in ``span_wagner_kernel`` hold here (τ-exponents are not
multiples of ¼, the exponential envelope carries a factor r, the Gaussian ε
differ per term), so the kernel is the generic form vectorized over the 24
terms: one ln τ, one ln δ, and per term one exp for τ^t δ^d and one for the
envelope.  The polynomial terms are the exponential form with r = 0 and the
Gaussian with η = β = 0, so all 24 share one formula and one exp evaluates
to exactly 1 where the envelope is absent.

Convention: τ = T_c/T, δ = ρ/ρ_c; derivatives are plain partials in τ and δ.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import nist2026_data as _d

NAME = "nist2026"

# ── Constants (Table 1, p. 2) ───────────────────────────────────────────────
TC = _d.TC
RHOC_MOLAR = _d.RHOC_MOLAR
M = _d.M
RHOC = RHOC_MOLAR * M          # kg/m³ ≈ 467.5965 (p. 3: 467.597)
R_MOLAR = _d.R_MOLAR
R = R_MOLAR / M                # J/(kg·K) ≈ 188.9241
PC = _d.PC_TABLE               # Pa; the equation reproduces it (tests)
T_TRIPLE = _d.T_TRIPLE

# ── Ideal part (Table 3, p. 5) ──────────────────────────────────────────────
_I_LOGTAU = _d.IDEAL_N0 - 1.0
_I_N = _d.IDEAL_N
_I_THETA = _d.IDEAL_M / _d.TC   # m°_i τ / T_c = θ_i τ
_I_N6 = _d.IDEAL_N6
_I_N7 = _d.IDEAL_N7

# ── Residual part (Table 4, p. 6), unified over the 24 terms ───────────────
_NT = _d.N_POL + _d.N_EXP + _d.N_GBS
_N = np.asarray(_d.RES_N, dtype=np.float64)
_T = np.asarray(_d.RES_T, dtype=np.float64)
_D = np.asarray(_d.RES_D, dtype=np.float64)
_L = np.zeros(_NT)
_R = np.zeros(_NT)
_ETA = np.zeros(_NT)
_BETA = np.zeros(_NT)
_GAMMA = np.zeros(_NT)
_EPS = np.zeros(_NT)
_e0, _g0 = _d.N_POL, _d.N_POL + _d.N_EXP
_L[_e0:_g0] = _d.EXP_L
_R[_e0:_g0] = _d.EXP_R
_ETA[_g0:] = _d.GBS_ETA
_BETA[_g0:] = _d.GBS_BETA
_GAMMA[_g0:] = _d.GBS_GAMMA
_EPS[_g0:] = _d.GBS_EPS
_RL = _R * _L                   # r·l, zero for the non-exponential terms
_RLL1 = _R * _L * (_L - 1.0)    # r·l·(l − 1)
_ETA2 = 2.0 * _ETA
_BETA2 = 2.0 * _BETA
_TT1 = _T * (_T - 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# The plain Helmholtz energy (the autodiff reference)
# ═══════════════════════════════════════════════════════════════════════════

def alpha0(tau, delta):
    """Ideal-gas reduced Helmholtz energy α°(τ, δ), Eq. (5)."""
    return (jnp.log(delta) + _I_N6 + _I_N7 * tau + _I_LOGTAU * jnp.log(tau)
            + jnp.sum(_I_N * jnp.log(1.0 - jnp.exp(-_I_THETA * tau))))


def alphar(tau, delta):
    """Residual reduced Helmholtz energy αʳ(τ, δ), Eq. (6), as three sums."""
    p = slice(0, _e0)
    e = slice(_e0, _g0)
    g = slice(_g0, _NT)
    pol = jnp.sum(_N[p] * delta ** _D[p] * tau ** _T[p])
    ex = jnp.sum(_N[e] * delta ** _D[e] * tau ** _T[e]
                 * jnp.exp(-_R[e] * delta ** _L[e]))
    gbs = jnp.sum(_N[g] * delta ** _D[g] * tau ** _T[g]
                  * jnp.exp(-_ETA[g] * (delta - _EPS[g]) ** 2
                            - _BETA[g] * (tau - _GAMMA[g]) ** 2))
    return pol + ex + gbs


def alpha(tau, delta):
    """Total reduced Helmholtz energy α(τ, δ) = α° + αʳ."""
    return alpha0(tau, delta) + alphar(tau, delta)


# ═══════════════════════════════════════════════════════════════════════════
# Analytic kernels (the names and return shapes of ``span_wagner_kernel``)
# ═══════════════════════════════════════════════════════════════════════════

def ideal_derivs(tau, delta):
    """Return (α°, α°_τ, α°_ττ) at scalar (τ, δ)."""
    e = jnp.exp(-_I_THETA * tau)
    ome = 1.0 - e
    a0 = (jnp.log(delta) + _I_N6 + _I_N7 * tau + _I_LOGTAU * jnp.log(tau)
          + jnp.sum(_I_N * jnp.log(ome)))
    a0_t = _I_N7 + _I_LOGTAU / tau + jnp.sum(_I_N * _I_THETA * e / ome)
    a0_tt = (-_I_LOGTAU / tau ** 2
             - jnp.sum(_I_N * _I_THETA ** 2 * e / ome ** 2))
    return a0, a0_t, a0_tt


def ideal_tau_only(tau):
    """Return (α°_τ, α°_ττ), independent of δ."""
    e = jnp.exp(-_I_THETA * tau)
    ome = 1.0 - e
    a0_t = _I_N7 + _I_LOGTAU / tau + jnp.sum(_I_N * _I_THETA * e / ome)
    a0_tt = (-_I_LOGTAU / tau ** 2
             - jnp.sum(_I_N * _I_THETA ** 2 * e / ome ** 2))
    return a0_t, a0_tt


def _delta_side(delta):
    """Per-term δ factors: (δ-coefficient n δ^d exp(−r δ^l − η(δ−ε)²), D1, D1')."""
    inv_d = 1.0 / delta
    ln_d = jnp.log(delta)
    dl = jnp.exp(_L * ln_d)                  # δ^l (1 where l = 0)
    dme = delta - _EPS
    coef = _N * jnp.exp(_D * ln_d - _R * dl - _ETA * dme * dme)
    D1 = (_D - _RL * dl) * inv_d - _ETA2 * dme
    D1p = (-_D - _RLL1 * dl) * inv_d * inv_d - _ETA2
    return coef, D1, D1p


def _tau_side(tau):
    """Per-term τ factors: (τ^t exp(−β(τ−γ)²), T1, T1')."""
    inv_t = 1.0 / tau
    tmg = tau - _GAMMA
    fac = jnp.exp(_T * jnp.log(tau) - _BETA * tmg * tmg)
    T1 = _T * inv_t - _BETA2 * tmg
    T1p = -_T * inv_t * inv_t - _BETA2
    return fac, T1, T1p


def residual_derivs(tau, delta):
    """Return (αʳ, αʳ_δ, αʳ_τ, αʳ_δδ, αʳ_ττ, αʳ_δτ) at scalar (τ, δ).

    Per term A = n δ^d τ^t exp(−r δ^l − η(δ−ε)² − β(τ−γ)²) with
    ∂ln A/∂δ = D1 = (d − r l δ^l)/δ − 2η(δ−ε),
    ∂D1/∂δ   = (−d − r l (l−1) δ^l)/δ² − 2η,
    ∂ln A/∂τ = T1 = t/τ − 2β(τ−γ),  ∂T1/∂τ = −t/τ² − 2β.
    """
    coef, D1, D1p = _delta_side(delta)
    fac, T1, T1p = _tau_side(tau)
    A = coef * fac
    ar = jnp.sum(A)
    ar_d = jnp.sum(A * D1)
    ar_t = jnp.sum(A * T1)
    ar_dd = jnp.sum(A * (D1 * D1 + D1p))
    ar_tt = jnp.sum(A * (T1 * T1 + T1p))
    ar_dt = jnp.sum(A * D1 * T1)
    return ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt


def residual_tau_prep(delta):
    """The δ-invariant state of the τ-derivative inner loop at fixed ρ."""
    coef, _, _ = _delta_side(delta)
    return coef


def residual_tau_fast(tau, dstate):
    """Return (αʳ_τ, αʳ_ττ) from the precomputed δ-state ``dstate``."""
    fac, T1, T1p = _tau_side(tau)
    A = dstate * fac
    return jnp.sum(A * T1), jnp.sum(A * (T1 * T1 + T1p))


def residual_tau_derivs(tau, delta):
    """Return (αʳ_τ, αʳ_ττ) at scalar (τ, δ)."""
    return residual_tau_fast(tau, residual_tau_prep(delta))


# ═══════════════════════════════════════════════════════════════════════════
# Scalar property functions (the names ``span_wagner`` gives them)
# ═══════════════════════════════════════════════════════════════════════════

def _scalar_pressure(T, rho):
    """Pressure [Pa] at scalar (T, ρ)."""
    _, ar_d, _, _, _, _ = residual_derivs(TC / T, rho / RHOC)
    return rho * R * T * (1.0 + rho / RHOC * ar_d)


def _scalar_cv(T, rho):
    """Isochoric heat capacity c_v [J/(kg·K)] at scalar (T, ρ)."""
    tau = TC / T
    _, ar_tt = residual_tau_derivs(tau, rho / RHOC)
    _, a0_tt = ideal_tau_only(tau)
    return -R * tau ** 2 * (a0_tt + ar_tt)


def _bracket_terms(T, rho):
    tau = TC / T
    delta = rho / RHOC
    _, ar_d, _, ar_dd, ar_tt, ar_dt = residual_derivs(tau, delta)
    _, a0_tt = ideal_tau_only(tau)
    sum_tt = a0_tt + ar_tt
    num = (1.0 + delta * ar_d - delta * tau * ar_dt) ** 2
    den = 1.0 + 2.0 * delta * ar_d + delta ** 2 * ar_dd
    return tau, sum_tt, num, den


def _scalar_cp(T, rho):
    """Isobaric heat capacity c_p [J/(kg·K)] at scalar (T, ρ)."""
    tau, sum_tt, num, den = _bracket_terms(T, rho)
    return -R * tau ** 2 * sum_tt + R * num / den


def _scalar_speed_of_sound(T, rho):
    """Speed of sound [m/s] at scalar (T, ρ)."""
    tau, sum_tt, num, den = _bracket_terms(T, rho)
    w_sq = R * T * (den - num / (tau ** 2 * sum_tt))
    return jnp.sqrt(jnp.maximum(w_sq, 0.0))


def _scalar_enthalpy(T, rho):
    """Specific enthalpy h [J/kg] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    _, ar_d, ar_t, _, _, _ = residual_derivs(tau, delta)
    a0_t, _ = ideal_tau_only(tau)
    return R * T * (tau * (a0_t + ar_t) + delta * ar_d + 1.0)


def _scalar_internal_energy(T, rho):
    """Specific internal energy u [J/kg] at scalar (T, ρ)."""
    tau = TC / T
    ar_t, _ = residual_tau_derivs(tau, rho / RHOC)
    a0_t, _ = ideal_tau_only(tau)
    return R * TC * (a0_t + ar_t)


def _scalar_entropy(T, rho):
    """Specific entropy s [J/(kg·K)] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    ar, _, ar_t, _, _, _ = residual_derivs(tau, delta)
    a0, a0_t, _ = ideal_derivs(tau, delta)
    return R * (tau * (a0_t + ar_t) - a0 - ar)


def _scalar_gibbs_energy(T, rho):
    """Specific Gibbs energy g = h − T s [J/kg] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    ar, ar_d, _, _, _, _ = residual_derivs(tau, delta)
    a0, _, _ = ideal_derivs(tau, delta)
    return R * T * (1.0 + delta * ar_d + a0 + ar)
