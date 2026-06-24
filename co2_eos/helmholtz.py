"""Span-Wagner (1996) reduced Helmholtz energy with hand-coded analytic
derivatives — pure JAX, one fused pass.

This is the numerical core of the redesigned EOS.  The original
``span_wagner.py`` obtains every derivative of α(τ,δ) through ``jax.grad``
(including nested grad for the second derivatives).  That is correct and
elegant but pays for it in the simulation hot path: each Newton iteration and
each property evaluation re-traces α and differentiates it several times.

Here we instead compute α and all the derivatives a property/inversion pass
needs — α_δ, α_τ, α_δδ, α_ττ, α_δτ — analytically, sharing the per-term
common subexpressions (the powers δ^d, τ^t and the exponential envelopes) so
each term contributes to every derivative by a cheap algebraic multiple of its
own value.  The functions are still pure JAX: JIT-compilable, vmappable and
differentiable, so higher-order AD (gradients of the inversions) keeps working.

Validated term-by-term against the ``jax.grad`` derivatives of
``span_wagner.alphar`` / ``alpha0`` to ~1e-10 (see ``tests/test_analytic_derivs.py``).

Convention: τ = Tc/T (reduced inverse temperature), δ = ρ/ρc (reduced
density).  Derivatives are plain partials w.r.t. τ and δ (not the δ·∂/∂δ
reduced form), matching the formulas in ``span_wagner``.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Pull the coefficient tables straight from span_wagner so there is a single
# source of truth for the constants.
from co2_eos.span_wagner import (
    _A0_A1, _A0_A2, _A0_LOGTAU, _A0_PE_N, _A0_PE_T,
    _AR_N, _AR_D, _AR_T, _AR_L,
    _AR_GAUSS_N, _AR_GAUSS_D, _AR_GAUSS_T,
    _AR_GAUSS_ETA, _AR_GAUSS_BETA, _AR_GAUSS_GAMMA, _AR_GAUSS_EPS,
    _NA_N, _NA_A, _NA_B, _NA_BETA, _NA_BIG_A, _NA_BIG_B, _NA_BIG_C, _NA_BIG_D,
)

# Reciprocal of (2β) for the non-analytic θ exponent, precomputed.
_NA_P = 1.0 / (2.0 * _NA_BETA)            # = 1/(2β)


# ═══════════════════════════════════════════════════════════════════════════
# Ideal-gas part α⁰ — τ derivatives (δ part is only ln δ, not needed here)
# ═══════════════════════════════════════════════════════════════════════════

def ideal_derivs(tau, delta):
    """Return (α⁰, α⁰_τ, α⁰_ττ) at scalar (τ, δ).

    α⁰ = a1 + a2·τ + L·ln τ + ln δ + Σ nk·ln(1 - exp(-θk·τ))
    Only the τ-derivatives and the value are used by the property/inversion
    formulas (the ideal δ-dependence cancels out of every measurable property
    except through the additive ln δ in the value, which entropy/Gibbs need).
    """
    L = _A0_LOGTAU
    e = jnp.exp(-_A0_PE_T * tau)            # exp(-θk τ)
    one_minus_e = 1.0 - e

    a0 = (_A0_A1 + _A0_A2 * tau + L * jnp.log(tau) + jnp.log(delta)
          + jnp.sum(_A0_PE_N * jnp.log(one_minus_e)))

    # d/dτ ln(1-e) = θ·e/(1-e)
    a0_t = (_A0_A2 + L / tau
            + jnp.sum(_A0_PE_N * _A0_PE_T * e / one_minus_e))

    # d²/dτ² ln(1-e) = -θ²·e/(1-e)²
    a0_tt = (-L / tau ** 2
             - jnp.sum(_A0_PE_N * _A0_PE_T ** 2 * e / one_minus_e ** 2))

    return a0, a0_t, a0_tt


def ideal_tau_only(tau):
    """Return (α⁰_τ, α⁰_ττ) — the only ideal quantities the Newton needs.

    Independent of δ, so cheaper than ``ideal_derivs`` for the inner loop.
    """
    L = _A0_LOGTAU
    e = jnp.exp(-_A0_PE_T * tau)
    one_minus_e = 1.0 - e
    a0_t = (_A0_A2 + L / tau
            + jnp.sum(_A0_PE_N * _A0_PE_T * e / one_minus_e))
    a0_tt = (-L / tau ** 2
             - jnp.sum(_A0_PE_N * _A0_PE_T ** 2 * e / one_minus_e ** 2))
    return a0_t, a0_tt


# ═══════════════════════════════════════════════════════════════════════════
# Residual part αʳ — full analytic derivative bundle (one fused pass)
# ═══════════════════════════════════════════════════════════════════════════

def residual_derivs(tau, delta):
    """Return (αʳ, αʳ_δ, αʳ_τ, αʳ_δδ, αʳ_ττ, αʳ_δτ) at scalar (τ, δ).

    All six are accumulated from the same per-term values, so the expensive
    transcendentals (pow, exp) are evaluated once per term.
    """
    inv_d = 1.0 / delta
    inv_d2 = inv_d * inv_d
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    # ── Polynomial + exponential terms ─────────────────────────────────────
    # value Tk = nk · δ^dk · τ^tk · Ek,  Ek = exp(-δ^lk) for lk>0 else 1
    dl = delta ** _AR_L                      # δ^l  (=1 when l=0)
    E = jnp.where(_AR_L > 0, jnp.exp(-dl), 1.0)
    Tk = _AR_N * delta ** _AR_D * tau ** _AR_T * E

    # δ log-derivative  D1 = d/δ - l·δ^(l-1) = d/δ - (l·δ^l)/δ
    D1 = _AR_D * inv_d - (_AR_L * dl) * inv_d
    # D1' = -d/δ² - l(l-1)·δ^(l-2) = -d/δ² - l(l-1)·δ^l/δ²
    D1p = -_AR_D * inv_d2 - (_AR_L * (_AR_L - 1.0) * dl) * inv_d2
    # τ log-derivative  t/τ ;  second  t(t-1)/τ²
    Tt = _AR_T * inv_t
    Ttt = _AR_T * (_AR_T - 1.0) * inv_t2

    p_a   = jnp.sum(Tk)
    p_d   = jnp.sum(Tk * D1)
    p_t   = jnp.sum(Tk * Tt)
    p_dd  = jnp.sum(Tk * (D1 * D1 + D1p))
    p_tt  = jnp.sum(Tk * Ttt)
    p_dt  = jnp.sum(Tk * D1 * Tt)

    # ── Gaussian bell-shaped terms ─────────────────────────────────────────
    dme = delta - _AR_GAUSS_EPS
    tmg = tau - _AR_GAUSS_GAMMA
    G = (_AR_GAUSS_N * delta ** _AR_GAUSS_D * tau ** _AR_GAUSS_T
         * jnp.exp(-_AR_GAUSS_ETA * dme ** 2 - _AR_GAUSS_BETA * tmg ** 2))

    Gd = _AR_GAUSS_D * inv_d - 2.0 * _AR_GAUSS_ETA * dme
    Gdp = -_AR_GAUSS_D * inv_d2 - 2.0 * _AR_GAUSS_ETA
    Gt = _AR_GAUSS_T * inv_t - 2.0 * _AR_GAUSS_BETA * tmg
    Gtp = -_AR_GAUSS_T * inv_t2 - 2.0 * _AR_GAUSS_BETA

    g_a  = jnp.sum(G)
    g_d  = jnp.sum(G * Gd)
    g_t  = jnp.sum(G * Gt)
    g_dd = jnp.sum(G * (Gd * Gd + Gdp))
    g_tt = jnp.sum(G * (Gt * Gt + Gtp))
    g_dt = jnp.sum(G * Gd * Gt)

    # ── Non-analytic terms ─────────────────────────────────────────────────
    # s = (δ-1)²;  θ = (1-τ) + A·s^p ;  Δ = θ² + B·s^a ;  Ψ = exp(-C s - D(τ-1)²)
    # term value W = n · Δ^b · δ · Ψ
    dm1 = delta - 1.0
    s = dm1 * dm1
    tm1 = tau - 1.0

    # powers of s (all exponents are > 0 in the regime, so s=0 → 0, finite)
    s_p   = s ** _NA_P                       # s^(1/2β)
    s_pm1 = s ** (_NA_P - 1.0)               # s^(1/2β - 1)
    s_a   = s ** _NA_A                        # s^a
    s_am1 = s ** (_NA_A - 1.0)               # s^(a-1)

    theta = tm1 * (-1.0) + _NA_BIG_A * s_p   # (1-τ) + A·s^p
    # θ derivatives
    th_t = -1.0
    th_d = 2.0 * _NA_BIG_A * _NA_P * dm1 * s_pm1          # 2 A p (δ-1) s^(p-1)
    th_dd = 2.0 * _NA_BIG_A * _NA_P * (2.0 * _NA_P - 1.0) * s_pm1  # 2 A p(2p-1) s^(p-1)

    Delta = theta * theta + _NA_BIG_B * s_a
    # Δ derivatives
    De_t  = -2.0 * theta                                  # 2θ·θ_τ, θ_τ=-1
    De_tt = 2.0                                           # 2θ_τ² + 2θ θ_ττ
    De_d  = 2.0 * theta * th_d + 2.0 * _NA_BIG_B * _NA_A * dm1 * s_am1
    De_dd = (2.0 * th_d * th_d + 2.0 * theta * th_dd
             + 2.0 * _NA_BIG_B * _NA_A * (2.0 * _NA_A - 1.0) * s_am1)
    De_dt = -2.0 * th_d                                   # ∂/∂τ(De_d) = 2θ_τ θ_δ

    # Δ^b and its derivatives (guard the base like span_wagner does)
    Db   = jnp.maximum(Delta, 1e-300)
    F   = Db ** _NA_B
    Fm1 = Db ** (_NA_B - 1.0)
    Fm2 = Db ** (_NA_B - 2.0)
    bb  = _NA_B
    F_d  = bb * Fm1 * De_d
    F_t  = bb * Fm1 * De_t
    F_dd = bb * (bb - 1.0) * Fm2 * De_d * De_d + bb * Fm1 * De_dd
    F_tt = bb * (bb - 1.0) * Fm2 * De_t * De_t + bb * Fm1 * De_tt
    F_dt = bb * (bb - 1.0) * Fm2 * De_d * De_t + bb * Fm1 * De_dt

    # Ψ and its derivatives
    Psi = jnp.exp(-_NA_BIG_C * s - _NA_BIG_D * tm1 * tm1)
    Ps_d  = -2.0 * _NA_BIG_C * dm1 * Psi
    Ps_t  = -2.0 * _NA_BIG_D * tm1 * Psi
    Ps_dd = (4.0 * _NA_BIG_C * _NA_BIG_C * s - 2.0 * _NA_BIG_C) * Psi
    Ps_tt = (4.0 * _NA_BIG_D * _NA_BIG_D * tm1 * tm1 - 2.0 * _NA_BIG_D) * Psi
    Ps_dt = 4.0 * _NA_BIG_C * _NA_BIG_D * dm1 * tm1 * Psi

    # W = F·δ·Ψ  (product of three factors, F2=δ)
    nW = _NA_N
    W   = F * delta * Psi
    # first
    W_d = F_d * delta * Psi + F * Psi + F * delta * Ps_d
    W_t = F_t * delta * Psi + F * delta * Ps_t
    # second
    W_dd = (F_dd * delta * Psi + 2.0 * F_d * Psi + 2.0 * F_d * delta * Ps_d
            + 2.0 * F * Ps_d + F * delta * Ps_dd)
    W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
    W_dt = (F_dt * delta * Psi + F_d * delta * Ps_t + F_t * Psi + F * Ps_t
            + F_t * delta * Ps_d + F * delta * Ps_dt)

    na_a  = jnp.sum(nW * W)
    na_d  = jnp.sum(nW * W_d)
    na_t  = jnp.sum(nW * W_t)
    na_dd = jnp.sum(nW * W_dd)
    na_tt = jnp.sum(nW * W_tt)
    na_dt = jnp.sum(nW * W_dt)

    ar    = p_a  + g_a  + na_a
    ar_d  = p_d  + g_d  + na_d
    ar_t  = p_t  + g_t  + na_t
    ar_dd = p_dd + g_dd + na_dd
    ar_tt = p_tt + g_tt + na_tt
    ar_dt = p_dt + g_dt + na_dt
    return ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt


def residual_tau_prep(delta):
    """Precompute the δ-invariant envelopes for the τ-derivative inner loop.

    At fixed ρ (the Newton solves T at fixed density) every δ-dependent power
    and exponential is loop-invariant.  We compute them once and feed them to
    ``residual_tau_fast`` so each Newton step only evaluates the τ-dependent
    transcendentals.
    """
    dl = delta ** _AR_L
    E = jnp.where(_AR_L > 0, jnp.exp(-dl), 1.0)
    Cpoly = _AR_N * delta ** _AR_D * E                     # poly δ-coefficients

    dme = delta - _AR_GAUSS_EPS
    Cg = (_AR_GAUSS_N * delta ** _AR_GAUSS_D
          * jnp.exp(-_AR_GAUSS_ETA * dme ** 2))            # gauss δ-coefficients

    dm1 = delta - 1.0
    s = dm1 * dm1
    s_p = s ** _NA_P
    Bsa = _NA_BIG_B * s ** _NA_A
    expCs = jnp.exp(-_NA_BIG_C * s)
    return Cpoly, Cg, s_p, Bsa, expCs, delta


def residual_tau_fast(tau, dstate):
    """Return (αʳ_τ, αʳ_ττ) from precomputed δ-state ``dstate``.

    Identical result to ``residual_tau_derivs(tau, delta)`` but only the
    τ-dependent transcendentals are evaluated per call.
    """
    Cpoly, Cg, s_p, Bsa, expCs, delta = dstate
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    tk = Cpoly * tau ** _AR_T
    p_t = jnp.sum(tk * (_AR_T * inv_t))
    p_tt = jnp.sum(tk * (_AR_T * (_AR_T - 1.0) * inv_t2))

    tmg = tau - _AR_GAUSS_GAMMA
    gk = Cg * tau ** _AR_GAUSS_T * jnp.exp(-_AR_GAUSS_BETA * tmg ** 2)
    Gt = _AR_GAUSS_T * inv_t - 2.0 * _AR_GAUSS_BETA * tmg
    Gtp = -_AR_GAUSS_T * inv_t2 - 2.0 * _AR_GAUSS_BETA
    g_t = jnp.sum(gk * Gt)
    g_tt = jnp.sum(gk * (Gt * Gt + Gtp))

    tm1 = tau - 1.0
    theta = -tm1 + _NA_BIG_A * s_p
    Delta = theta * theta + Bsa
    De_t = -2.0 * theta
    De_tt = 2.0
    Db = jnp.maximum(Delta, 1e-300)
    Fm1 = Db ** (_NA_B - 1.0)
    Fm2 = Db ** (_NA_B - 2.0)
    F = Db ** _NA_B
    bb = _NA_B
    F_t = bb * Fm1 * De_t
    F_tt = bb * (bb - 1.0) * Fm2 * De_t * De_t + bb * Fm1 * De_tt
    Psi = expCs * jnp.exp(-_NA_BIG_D * tm1 * tm1)
    Ps_t = -2.0 * _NA_BIG_D * tm1 * Psi
    Ps_tt = (4.0 * _NA_BIG_D * _NA_BIG_D * tm1 * tm1 - 2.0 * _NA_BIG_D) * Psi
    W_t = F_t * delta * Psi + F * delta * Ps_t
    W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
    na_t = jnp.sum(_NA_N * W_t)
    na_tt = jnp.sum(_NA_N * W_tt)

    return p_t + g_t + na_t, p_tt + g_tt + na_tt


def residual_tau_derivs(tau, delta):
    """Return (αʳ_τ, αʳ_ττ) — the residual quantities the Newton needs.

    Computes only the τ-derivatives; still touches the δ-dependent envelopes
    (which are loop-invariant at fixed ρ — XLA hoists them across an unrolled
    Newton).
    """
    inv_t = 1.0 / tau
    inv_t2 = inv_t * inv_t

    dl = delta ** _AR_L
    E = jnp.where(_AR_L > 0, jnp.exp(-dl), 1.0)
    Tk = _AR_N * delta ** _AR_D * tau ** _AR_T * E
    p_t  = jnp.sum(Tk * (_AR_T * inv_t))
    p_tt = jnp.sum(Tk * (_AR_T * (_AR_T - 1.0) * inv_t2))

    dme = delta - _AR_GAUSS_EPS
    tmg = tau - _AR_GAUSS_GAMMA
    G = (_AR_GAUSS_N * delta ** _AR_GAUSS_D * tau ** _AR_GAUSS_T
         * jnp.exp(-_AR_GAUSS_ETA * dme ** 2 - _AR_GAUSS_BETA * tmg ** 2))
    Gt = _AR_GAUSS_T * inv_t - 2.0 * _AR_GAUSS_BETA * tmg
    Gtp = -_AR_GAUSS_T * inv_t2 - 2.0 * _AR_GAUSS_BETA
    g_t  = jnp.sum(G * Gt)
    g_tt = jnp.sum(G * (Gt * Gt + Gtp))

    dm1 = delta - 1.0
    s = dm1 * dm1
    tm1 = tau - 1.0
    s_p = s ** _NA_P
    s_a = s ** _NA_A
    theta = -tm1 + _NA_BIG_A * s_p
    Delta = theta * theta + _NA_BIG_B * s_a
    De_t  = -2.0 * theta
    De_tt = 2.0
    Db  = jnp.maximum(Delta, 1e-300)
    Fm1 = Db ** (_NA_B - 1.0)
    Fm2 = Db ** (_NA_B - 2.0)
    F   = Db ** _NA_B
    bb = _NA_B
    F_t  = bb * Fm1 * De_t
    F_tt = bb * (bb - 1.0) * Fm2 * De_t * De_t + bb * Fm1 * De_tt
    Psi = jnp.exp(-_NA_BIG_C * s - _NA_BIG_D * tm1 * tm1)
    Ps_t  = -2.0 * _NA_BIG_D * tm1 * Psi
    Ps_tt = (4.0 * _NA_BIG_D * _NA_BIG_D * tm1 * tm1 - 2.0 * _NA_BIG_D) * Psi
    W_t  = F_t * delta * Psi + F * delta * Ps_t
    W_tt = F_tt * delta * Psi + 2.0 * F_t * delta * Ps_t + F * delta * Ps_tt
    na_t  = jnp.sum(_NA_N * W_t)
    na_tt = jnp.sum(_NA_N * W_tt)

    return p_t + g_t + na_t, p_tt + g_tt + na_tt
