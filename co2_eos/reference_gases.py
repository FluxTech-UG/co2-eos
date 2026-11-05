"""The reference gases (air, helium, argon): one closed-form Helmholtz energy each.

Each gas is a backend of ``co2_eos`` selected by ``eos="air"``, ``"helium"`` or
``"argon"``.  The Helmholtz energy per unit mass is a(ρ, T) = R T α(τ, δ) with

    α(τ, δ) = α⁰(τ, δ) + δ b(τ) + δ² c(τ),     τ = T_r/T, δ = ρ/ρ_r,

where α⁰ integrates a NASA-7 c_p⁰(T) polynomial, and b and c are the first two
coefficients of the δ-series of the gas's reference equation of state, summed
in closed form from its published residual terms (``reference_gases_data``).
b ρ_r⁻¹ is the second virial coefficient B(T) and 2 c ρ_r⁻² the third, C(T).
Every property is a derivative of α, read by the same fused code as the CO₂
backends (``core._build``), so the bundle's fields and its thermodynamic
identities are those of Span-Wagner by construction.

A reference gas has no saturation dome and no near-critical machinery: the
saturation lookups refuse it, and its transport carries no critical
enhancement.  Air and argon carry the dilute and residual transport of Lemmon and
Jacobsen (2004).  Helium's transport is dilute only, an ab initio fit, which holds
its 1 % bar to ``TRANSPORT.P_ONE_PERCENT`` (Swift's top state, 13.8 bar, with its 10 %
swing: worst μ 0.77 %, λ 0.90 %) and reaches 1.09 % at 20 bar, 1.33 % at 25 bar.  The
bulk viscosity is 0: exact for the dilute monatomic gases; for air the rotational part
is left out, an open item (README, "Reference gases").

The domain is ``T_RANGE`` and P ≤ ``P_MAX``, where the tolerances are measured
(tests/test_reference_gases.py).  Outside it the bundle is NaN, the solver
boundary's failure signal: a virial series past its range returns a smooth wrong
number otherwise.

This module provides per gas, under the names of the backend contract in
``backends``, the surface and kernel (``TC, RHOC, RHOC_MOLAR, R, R_MOLAR, M, PC``,
``alpha0, alphar, alpha``, the analytic derivative kernels and the scalar property
functions), and two namespaces ``transport`` and ``inversions`` with the
names ``core`` and the public API read from their CO₂ counterparts.

Convention: τ = T_r/T, δ = ρ/ρ_r; derivatives are plain partials in τ and δ.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import reference_gases_data as _data

# Fixed Newton iterations.  Measured 2026-11-05 over each gas's whole domain:
# (ρ, u) → T from the linear seed and (P, T) → ρ from the ideal-gas seed reach
# round-off in these counts (test_reference_gases round trips).
_NEWTON_ITERS_T = 4
_NEWTON_ITERS_RHO = 4
_NEWTON_ITERS_PH = 5
_SEED_T = 500.0     # K; where the linear u⁰(T) and h⁰(T) seeds are expanded
_NEWTON_T_CLIP = (100.0, 1500.0)   # K; c_v⁰ > 0 across it for all three gases
_EDGE_SLACK = 1e-9


def _delta_series(terms):
    """Coefficients of δ¹ and δ² in Σ n δ^d τ^t exp(−δ^l), aligned on the distinct t.

    exp(−δ^l) = 1 − δ^l + ...: a d = 1 term gives n τ^t to δ¹ and, when l = 1,
    −n τ^t to δ²; a d = 2 term gives n τ^t to δ².  Returns (t, n₁, n₂).
    """
    t = np.array(sorted({term[1] for term in terms}), dtype=np.float64)
    n1 = np.zeros_like(t)
    n2 = np.zeros_like(t)
    for n, ti, d, l in terms:
        i = int(np.searchsorted(t, ti))
        if d == 1:
            n1[i] += n
            if l == 1:
                n2[i] -= n
        elif d == 2:
            n2[i] += n
        else:
            raise ValueError(f"a residual term of order d = {d} is outside the δ² series")
    return t, n1, n2


def _make_transport(gas, RHOC, TC, in_domain):
    """Transport namespace of a gas: μ(T, ρ) and λ(T, ρ) from its published fits."""
    tr = gas.TRANSPORT

    def _vector_terms(table):
        a, t, d, l = (np.array(col, dtype=np.float64) for col in zip(*table))
        return a, t, d, l, (l > 0).astype(np.float64)

    def _density_part(T, rho, terms):
        """Σ N τ^t δ^d exp(−γ δ^l), the form of Lemmon and Jacobsen (2004), Eq. (3, 6)."""
        a, t, d, l, gamma = terms
        ln_tau = jnp.log(TC / T)
        ln_delta = jnp.log(rho / RHOC)
        return jnp.sum(a * jnp.exp(t * ln_tau + d * ln_delta - gamma * jnp.exp(l * ln_delta)))

    if tr.KIND == "lemmon_jacobsen":
        eta_res = _vector_terms(tr.ETA_RES)
        lam_res = _vector_terms(tr.LAMBDA_RES)
        b = np.array(tr.OMEGA_B, dtype=np.float64)
        prefactor = 0.0266958 * np.sqrt(tr.M_ETA) / tr.SIGMA_NM ** 2  # L&J Eq. (2)
        n_lam = np.array(tr.LAMBDA_DILUTE_N, dtype=np.float64)
        t_lam = np.array(tr.LAMBDA_DILUTE_T, dtype=np.float64)

        def _eta_dilute(T):
            """η⁰ [μPa·s]."""
            ln_t = jnp.log(T / tr.EPS_K)
            omega = jnp.exp(b[0] + ln_t * (b[1] + ln_t * (b[2] + ln_t * (b[3] + ln_t * b[4]))))
            return prefactor * jnp.sqrt(T) / omega

        def _lambda_dilute(T):
            """λ⁰ [mW/(m·K)]: N₁ η⁰ + Σ N_i τ^t_i."""
            return n_lam[0] * _eta_dilute(T) + jnp.sum(
                n_lam[1:] * jnp.exp(t_lam[1:] * jnp.log(TC / T)))

        def _viscosity(T, rho):
            return 1e-6 * (_eta_dilute(T) + _density_part(T, rho, eta_res))

        def _conductivity(T, rho):
            return 1e-3 * (_lambda_dilute(T) + _density_part(T, rho, lam_res))
    elif tr.KIND == "ab_initio_dilute":
        c_eta = np.array(tr.LN_ETA, dtype=np.float64)
        c_lam = np.array(tr.LN_LAMBDA, dtype=np.float64)

        def _cubic(c, x):
            return c[0] + x * (c[1] + x * (c[2] + x * c[3]))

        def _viscosity(T, rho):
            return 1e-6 * jnp.exp(_cubic(c_eta, jnp.log(T / _data.HELIUM_FIT_T0)))

        def _conductivity(T, rho):
            return 1e-3 * jnp.exp(_cubic(c_lam, jnp.log(T / _data.HELIUM_FIT_T0)))
    else:
        raise ValueError(f"unknown transport kind {tr.KIND!r} for {gas.NAME}")

    def _scalar_viscosity(T, rho):
        """Dynamic viscosity μ [Pa·s] at scalar (T, ρ); NaN outside T_RANGE."""
        return _viscosity(in_domain(T), rho)

    def _scalar_thermal_conductivity(T, rho):
        """Thermal conductivity λ [W/(m·K)] at scalar (T, ρ); NaN outside T_RANGE."""
        return _conductivity(in_domain(T), rho)

    def _thermal_conductivity_shared(T, rho, mu, ar_d, ar_dd, ar_tt, ar_dt, a0_tt):
        """λ [W/(m·K)]; a reference gas reads no EOS derivative (no enhancement)."""
        return _conductivity(T, rho)

    def _correlation_length_and_excess(T, rho, ar_d, ar_dd):
        """(ξ, Δχ̃) = (0, 0): the gases carry no near-critical machinery."""
        zero = jnp.zeros_like(T)
        return zero, zero

    def _bulk_viscosity_parts(T, rho, mu, w, cp, cv, a0_tt, xi):
        """(ζ_vib, ζ_crit, τ₁₁) = 0: see the module docstring."""
        zero = jnp.zeros_like(T)
        return zero, zero, zero

    def _scalar_bulk_viscosity(T, rho):
        """Bulk viscosity ζ = 0 [Pa·s]: see the module docstring."""
        return jnp.zeros_like(in_domain(T))

    return SimpleNamespace(
        eos=gas.NAME,
        _scalar_viscosity=_scalar_viscosity,
        _scalar_thermal_conductivity=_scalar_thermal_conductivity,
        _thermal_conductivity_shared=_thermal_conductivity_shared,
        _correlation_length_and_excess=_correlation_length_and_excess,
        _bulk_viscosity_parts=_bulk_viscosity_parts,
        _scalar_bulk_viscosity=_scalar_bulk_viscosity)


def _make(gas):
    """The surface, kernel, transport and inversions of one gas, as one namespace."""
    TC = gas.TC
    RHOC_MOLAR = gas.RHOC_MOLAR
    M = gas.M
    RHOC = RHOC_MOLAR * M
    R_MOLAR = gas.R_MOLAR
    R = R_MOLAR / M
    T_LO, T_HI = gas.T_RANGE
    P_MAX = gas.P_MAX
    # A state exactly on an edge stays in, past the round-off of a solve landing there.
    P_LIMIT = P_MAX * (1.0 + _EDGE_SLACK)
    T_LO = T_LO * (1.0 - _EDGE_SLACK)
    T_HI = T_HI * (1.0 + _EDGE_SLACK)

    a1, a2, a3, a4, a5, a6, a7 = gas.NASA7
    _T, _N1, _N2 = _delta_series(gas.RESIDUAL_TERMS)
    _TT1 = _T * (_T - 1.0)

    # ── The ideal part, from the NASA-7 c_p⁰/R ─────────────────────────────

    def _cp0_R(T):
        return a1 + T * (a2 + T * (a3 + T * (a4 + T * a5)))

    def _h0_RT(T):
        """h⁰/(R T)."""
        return (a1 + T * (a2 / 2.0 + T * (a3 / 3.0 + T * (a4 / 4.0 + T * a5 / 5.0)))
                + a6 / T)

    def _s0_R(T):
        """s⁰(T, P_REF)/R."""
        return (a1 * jnp.log(T)
                + T * (a2 + T * (a3 / 2.0 + T * (a4 / 3.0 + T * a5 / 4.0))) + a7)

    def alpha0(tau, delta):
        """α⁰(τ, δ) = ln(ρRT/P_REF) + h⁰/(RT) − 1 − s⁰/R."""
        T = TC / tau
        return (jnp.log(delta * RHOC * R * T / _data.P_REF) + _h0_RT(T) - 1.0 - _s0_R(T))

    def ideal_tau_only(tau):
        """(α⁰_τ, α⁰_ττ): τ α⁰_τ = u⁰/(RT) and −τ² α⁰_ττ = c_v⁰/R."""
        T = TC / tau
        return (_h0_RT(T) - 1.0) / tau, (1.0 - _cp0_R(T)) / tau ** 2

    def ideal_derivs(tau, delta):
        """Return (α⁰, α⁰_τ, α⁰_ττ) at scalar (τ, δ)."""
        a0_t, a0_tt = ideal_tau_only(tau)
        return alpha0(tau, delta), a0_t, a0_tt

    # ── The residual part, δ b(τ) + δ² c(τ) ────────────────────────────────

    def _b_c(tau):
        """(b, c, b_τ, c_τ, b_ττ, c_ττ): one exp per distinct τ-exponent."""
        p = jnp.exp(_T * jnp.log(tau))
        inv = 1.0 / tau
        p1 = _T * p * inv
        p2 = _TT1 * p * inv * inv
        return (jnp.dot(_N1, p), jnp.dot(_N2, p), jnp.dot(_N1, p1), jnp.dot(_N2, p1),
                jnp.dot(_N1, p2), jnp.dot(_N2, p2))

    def alphar(tau, delta):
        """αʳ(τ, δ) as the plain sum over the published terms kept in the series."""
        p = jnp.exp(_T * jnp.log(tau))
        return delta * jnp.dot(_N1, p) + delta ** 2 * jnp.dot(_N2, p)

    def alpha(tau, delta):
        """Total reduced Helmholtz energy α(τ, δ) = α⁰ + αʳ."""
        return alpha0(tau, delta) + alphar(tau, delta)

    def residual_derivs(tau, delta):
        """Return (αʳ, αʳ_δ, αʳ_τ, αʳ_δδ, αʳ_ττ, αʳ_δτ) at scalar (τ, δ)."""
        b, c, b_t, c_t, b_tt, c_tt = _b_c(tau)
        return (delta * (b + delta * c),
                b + 2.0 * delta * c,
                delta * (b_t + delta * c_t),
                2.0 * c,
                delta * (b_tt + delta * c_tt),
                b_t + 2.0 * delta * c_t)

    def residual_tau_prep(delta):
        """The δ-invariant state of the τ-derivative inner loop: δ itself."""
        return delta

    def residual_tau_fast(tau, dstate):
        """Return (αʳ_τ, αʳ_ττ) from the δ-state ``dstate``."""
        _, _, b_t, c_t, b_tt, c_tt = _b_c(tau)
        return (dstate * (b_t + dstate * c_t), dstate * (b_tt + dstate * c_tt))

    def residual_tau_derivs(tau, delta):
        """Return (αʳ_τ, αʳ_ττ) at scalar (τ, δ)."""
        return residual_tau_fast(tau, residual_tau_prep(delta))

    # ── The domain ─────────────────────────────────────────────────────────

    def in_domain(T):
        """T, or NaN outside T_RANGE."""
        return jnp.where((T >= T_LO) & (T <= T_HI), T, jnp.nan)

    def domain_sentinel(T, P):
        """0.0 inside the domain, NaN outside; added to every bundle field."""
        return jnp.where((T >= T_LO) & (T <= T_HI) & (P <= P_LIMIT), 0.0, jnp.nan)

    # ── The (ρ, u) → T seed ────────────────────────────────────────────────

    _u_seed = R * _SEED_T * (float(_h0_RT(_SEED_T)) - 1.0)
    _cv_seed = R * (float(_cp0_R(_SEED_T)) - 1.0)
    _h_seed = R * _SEED_T * float(_h0_RT(_SEED_T))
    _cp_seed = R * float(_cp0_R(_SEED_T))

    def seed_T(rho, u):
        """Initial T for the (ρ, u) inversion: the ideal u⁰(T) linear about _SEED_T."""
        return jnp.clip(_SEED_T + (u - _u_seed) / _cv_seed, *_NEWTON_T_CLIP)

    # ── Scalar property functions (the surface contract) ───────────────────

    def _scalar_pressure(T, rho):
        """Pressure [Pa] at scalar (T, ρ)."""
        _, ar_d, _, _, _, _ = residual_derivs(TC / T, rho / RHOC)
        return rho * R * T * (1.0 + rho / RHOC * ar_d)

    def _scalar_internal_energy(T, rho):
        """Specific internal energy u [J/kg] at scalar (T, ρ)."""
        tau = TC / T
        ar_t, _ = residual_tau_derivs(tau, rho / RHOC)
        a0_t, _ = ideal_tau_only(tau)
        return R * TC * (a0_t + ar_t)

    def _scalar_enthalpy(T, rho):
        """Specific enthalpy h [J/kg] at scalar (T, ρ)."""
        tau = TC / T
        delta = rho / RHOC
        _, ar_d, ar_t, _, _, _ = residual_derivs(tau, delta)
        a0_t, _ = ideal_tau_only(tau)
        return R * T * (tau * (a0_t + ar_t) + delta * ar_d + 1.0)

    def _scalar_entropy(T, rho):
        """Specific entropy s [J/(kg·K)] at scalar (T, ρ)."""
        tau = TC / T
        delta = rho / RHOC
        ar, _, ar_t, _, _, _ = residual_derivs(tau, delta)
        a0, a0_t, _ = ideal_derivs(tau, delta)
        return R * (tau * (a0_t + ar_t) - a0 - ar)

    # ── Inversions (fixed Newton, implicit-function derivatives) ───────────

    def _solve_density(T, P):
        tau = TC / T
        b, c, *_ = _b_c(tau)
        delta = P / (R * T * RHOC)
        for _ in range(_NEWTON_ITERS_RHO):
            d_ar_d = delta * (b + 2.0 * delta * c)
            f = delta * RHOC * R * T * (1.0 + d_ar_d) - P
            df = RHOC * R * T * (1.0 + 2.0 * d_ar_d + delta * delta * 2.0 * c)
            delta = delta - f / df
        return delta * RHOC

    @jax.custom_jvp
    def _density_raw(T, P):
        return _solve_density(T, P)

    @_density_raw.defjvp
    def _density_raw_jvp(primals, tangents):
        """dρ = (dP − (∂P/∂T)_ρ dT) / (∂P/∂ρ)_T."""
        T, P = primals
        dT, dP = tangents
        rho = _density_raw(T, P)
        tau = TC / T
        delta = rho / RHOC
        _, ar_d, _, ar_dd, _, ar_dt = residual_derivs(tau, delta)
        P_rho = R * T * (1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd)
        P_T = rho * R * (1.0 + delta * ar_d - delta * tau * ar_dt)
        return rho, (dP - P_T * dT) / P_rho

    def _density_from_PT(T, P, phase_hint):
        """ρ at scalar (T, P); ``phase_hint`` is accepted for API symmetry and unused."""
        return _density_raw(T, P) + domain_sentinel(T, P)

    def density_from_PT(T, P, phase_hint):
        """Named as the CO₂ inversions name it."""
        return _density_from_PT(T, P, phase_hint)

    def _h_cp(T, rho):
        """(h, c_p) at scalar (T, ρ), read through the fused kernel."""
        from co2_eos import core as _core
        (_, _, cp, _, h, *_), _ = _core.for_eos(gas.NAME)._thermo(T, rho)
        return h, cp

    @jax.custom_jvp
    def _temperature_from_Ph(P, h):
        T = jnp.clip(_SEED_T + (h - _h_seed) / _cp_seed, *_NEWTON_T_CLIP)
        for _ in range(_NEWTON_ITERS_PH):
            rho = _solve_density(T, P)
            h_T, cp = _h_cp(T, rho)
            T = jnp.clip(T - (h_T - h) / cp, *_NEWTON_T_CLIP)
        return T

    @_temperature_from_Ph.defjvp
    def _temperature_from_Ph_jvp(primals, tangents):
        """dT = (dh − (∂h/∂P)_T dP) / c_p, with (∂h/∂P)_T = (∂h/∂ρ)_T / (∂P/∂ρ)_T."""
        P, h = primals
        dP, dh = tangents
        T = _temperature_from_Ph(P, h)
        rho = _solve_density(T, P)
        tau = TC / T
        delta = rho / RHOC
        _, ar_d, _, ar_dd, _, ar_dt = residual_derivs(tau, delta)
        _, cp = _h_cp(T, rho)
        P_rho = R * T * (1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd)
        h_rho = R * T / RHOC * (ar_d + delta * ar_dd + tau * ar_dt)
        return T, (dh - h_rho / P_rho * dP) / cp

    def _state_from_Ph(P, h, phase_hint):
        """(T, ρ) at scalar (P, h); ``phase_hint`` is accepted and unused."""
        T = _temperature_from_Ph(P, h)
        return T, _density_from_PT(T, P, phase_hint)

    def state_from_Ph(P, h, phase_hint):
        """Named as the CO₂ inversions name it."""
        return _state_from_Ph(P, h, phase_hint)

    inversions = SimpleNamespace(
        eos=gas.NAME, _density_from_PT=_density_from_PT, density_from_PT=density_from_PT,
        _state_from_Ph=_state_from_Ph, state_from_Ph=state_from_Ph)

    return SimpleNamespace(
        NAME=gas.NAME, TC=TC, RHOC=RHOC, RHOC_MOLAR=RHOC_MOLAR, R=R, R_MOLAR=R_MOLAR, M=M,
        PC=gas.PC, T_RANGE=gas.T_RANGE, P_MAX=P_MAX,
        T_CLIP=_NEWTON_T_CLIP, NEWTON_ITERS=_NEWTON_ITERS_T,
        alpha0=alpha0, alphar=alphar, alpha=alpha,
        ideal_derivs=ideal_derivs, ideal_tau_only=ideal_tau_only,
        residual_derivs=residual_derivs, residual_tau_prep=residual_tau_prep,
        residual_tau_fast=residual_tau_fast, residual_tau_derivs=residual_tau_derivs,
        seed_T=seed_T, in_domain=in_domain, domain_sentinel=domain_sentinel,
        _scalar_pressure=_scalar_pressure, _scalar_enthalpy=_scalar_enthalpy,
        _scalar_internal_energy=_scalar_internal_energy, _scalar_entropy=_scalar_entropy,
        transport=_make_transport(gas, RHOC, TC, in_domain), inversions=inversions)


GASES = {name: _make(gas) for name, gas in _data.GASES.items()}
