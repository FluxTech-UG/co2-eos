"""
Thermodynamic inversions — pure JAX, JIT-compilable, vmappable,
with correct gradients via implicit differentiation (custom_jvp).

Defines custom_jvp rules so that both forward-mode (jvp) and reverse-mode
(vjp, via transposition) AD work through the Newton solvers.

- density_from_PT(T, P, phase_hint):  1D inversion  P(T,ρ) = P_target
- state_from_Ph(P, h, phase_hint):    2D inversion  [P(T,ρ), h(T,ρ)] = [P_target, h_target]
- temperature_from_Du(rho, u, phase_hint): 1D inversion  u(T,ρ) = u_target at fixed ρ
- state_from_Du(rho, u, phase_hint):  convenience wrapper returning (T, P, h)
"""

import jax
import jax.numpy as jnp

from co2_eos import span_wagner as sw
from co2_eos import saturation as sat

# Enable float64
jax.config.update("jax_enable_x64", True)

# Eagerly load the saturation table at module-import time (outside any trace
# context). If the table is loaded for the first time inside a JIT trace,
# the arrays it materialises end up tagged as tracers, breaking reverse-mode
# differentiation through density_from_PT. Loading it here pins them as
# concrete constants. If the file is missing we tolerate it — first
# inversion call will surface a clear FileNotFoundError.
try:
    sat._ensure_loaded()
except FileNotFoundError:
    pass

# Phase hint constants
LIQUID = 0
VAPOR = 1
SUPERCRITICAL = 2

# Newton solver parameters
MAX_ITER = 50
TOL = 1e-10


# ── Pressure and its density-derivative (scalar) ─────────────────────────

_dP_drho = jax.grad(sw._scalar_pressure, argnums=1)
_d2P_drho2 = jax.grad(jax.grad(sw._scalar_pressure, argnums=1), argnums=1)
_dP_dT = jax.grad(sw._scalar_pressure, argnums=0)
_dh_dT = jax.grad(sw._scalar_enthalpy, argnums=0)
_dh_drho = jax.grad(sw._scalar_enthalpy, argnums=1)
_du_dT = jax.grad(sw._scalar_internal_energy, argnums=0)
_du_drho = jax.grad(sw._scalar_internal_energy, argnums=1)


# ── Phase-aware initial guess ─────────────────────────────────────────────

def _initial_guess(T, P, phase_hint):
    """Return an initial density guess based on phase_hint.

    phase_hint: 0=liquid, 1=vapor, 2=supercritical/auto.
    Uses jnp.where for JIT traceability.
    """
    subcritical_T = T < sw.TC

    # T-based saturation densities — good for compressed liquid (rho_l at T)
    # and low-pressure vapor (rho_v at T)
    rho_l_T, rho_v_T = sat.saturation_densities(T)

    # Liquid: saturation liquid density at T (good for compressed liquid),
    # fall back to 2*RHOC for supercritical T
    liquid_guess = jnp.where(subcritical_T, rho_l_T, 2.0 * sw.RHOC)
    # Vapor: saturation vapor density at T, fall back to ideal gas
    ideal_gas_rho = P / (sw.R * T)
    vapor_guess = jnp.where(subcritical_T, rho_v_T, ideal_gas_rho)
    # Supercritical: pressure-aware guess near the critical region.
    # Near PC and TC, use RHOC. Otherwise, vapor-like (P < PC) uses ideal gas,
    # liquid-like (P > 1.5*PC) uses 2*RHOC.
    # When T is well below TC (liquid-like regardless of pressure), use
    # saturated liquid density at T — same logic as the LIQUID branch.
    near_critical = jnp.logical_and(
        jnp.abs(P - sw.PC) < 0.2 * sw.PC,
        jnp.abs(T - sw.TC) < 15.0,
    )
    clearly_liquid = T < sw.TC - 10.0
    sc_guess = jnp.where(
        clearly_liquid, rho_l_T,
        jnp.where(
            near_critical, sw.RHOC,
            jnp.where(P > 1.5 * sw.PC, 2.0 * sw.RHOC, ideal_gas_rho)
        )
    )

    rho0 = jnp.where(
        phase_hint == LIQUID, liquid_guess,
        jnp.where(phase_hint == VAPOR, vapor_guess, sc_guess)
    )
    return rho0


# ── Sign-preserving denominator guard ─────────────────────────────────────

def _safe_denom(x, eps=1e-30):
    """Guard a denominator: preserve sign for small values, fall back to +eps for NaN."""
    return jnp.where(
        jnp.isnan(x), eps,
        jnp.where(jnp.abs(x) > eps, x, jnp.where(x >= 0, eps, -eps))
    )


# ── Halley solver via lax.while_loop ─────────────────────────────────────

def _halley_body(state):
    """One Halley step (Householder order 2) for cubic convergence.

    ρ_{n+1} = ρ_n − 2·f·f' / (2·f'² − f·f'')
    """
    rho, T, P_target, i, converged = state
    f = sw._scalar_pressure(T, rho) - P_target
    fp = _dP_drho(T, rho)
    fpp = _d2P_drho2(T, rho)
    # Halley denominator: 2·f'² − f·f''
    halley_denom = 2.0 * fp * fp - f * fpp
    step = 2.0 * f * fp / _safe_denom(halley_denom)
    # Step damping: |Δρ/ρ| ≤ 50% per iteration
    step = jnp.clip(step, -0.5 * rho, 0.5 * rho)
    rho_new = rho - step
    # Clamp to stay positive
    rho_new = jnp.maximum(rho_new, 1.0)
    converged_new = jnp.logical_or(
        jnp.abs(f) < TOL,
        jnp.abs(step) < 1e-13 * jnp.maximum(jnp.abs(rho_new), 1.0)
    )
    return (rho_new, T, P_target, i + 1, converged_new)


def _halley_cond(state):
    """Continue while not converged and under max iterations."""
    rho, T, P_target, i, converged = state
    return jnp.logical_and(i < MAX_ITER, jnp.logical_not(converged))


# ── Bisection fallback via lax.while_loop ────────────────────────────────

_BISECT_LO = 1.0       # kg/m³ — lower bound
_BISECT_HI = 1200.0    # kg/m³ — upper bound
_BISECT_ITERS = 60      # ~10⁻¹⁸ relative precision


def _bisection_body(state):
    """One bisection step on f(ρ) = P(T,ρ) − P_target.

    Monotonicity assumption: P(T,ρ) is monotonically increasing in ρ for
    single-phase stable fluid (∂P/∂ρ > 0).  This does NOT hold inside the
    spinodal (unstable) region, but that's fine — the bisection is a safety
    net for near-critical supercritical states where Halley oscillates, not
    for subcritical dome-crossing (handled by dome detection in state_from_Ph).
    """
    lo, hi, T, P_target, i = state
    mid = 0.5 * (lo + hi)
    f_mid = sw._scalar_pressure(T, mid) - P_target
    # P(T,ρ) is monotonically increasing in ρ for single-phase fluid,
    # so f_mid > 0 means mid is too high.
    lo_new = jnp.where(f_mid < 0.0, mid, lo)
    hi_new = jnp.where(f_mid < 0.0, hi, mid)
    return (lo_new, hi_new, T, P_target, i + 1)


def _bisection_cond(state):
    lo, hi, T, P_target, i = state
    return jnp.logical_and(i < _BISECT_ITERS, (hi - lo) > 1e-14 * hi)


def _bisection_solve(T, P):
    """Bisection solve for ρ given (T, P). Scalar inputs. Bounds [1, 1200] kg/m³."""
    init_state = (_BISECT_LO, _BISECT_HI, T, P, jnp.int32(0))
    final_state = jax.lax.while_loop(_bisection_cond, _bisection_body, init_state)
    lo, hi, _, _, _ = final_state
    return 0.5 * (lo + hi)


def _solve_density(T, P, phase_hint):
    """Halley solve for ρ given (T, P, phase_hint), with bisection fallback. Scalar inputs."""
    rho0 = _initial_guess(T, P, phase_hint)
    init_state = (rho0, T, P, jnp.int32(0), jnp.bool_(False))
    final_state = jax.lax.while_loop(_halley_cond, _halley_body, init_state)
    rho_halley, _, _, _, converged = final_state

    # Bisection safety net — only runs when Halley did not converge
    def _bisect_branch(_):
        rho_bisect = _bisection_solve(T, P)
        bisect_residual = jnp.abs(sw._scalar_pressure(T, rho_bisect) - P)
        bisect_ok = bisect_residual < 1e-3 * jnp.maximum(jnp.abs(P), 1.0)
        return jnp.where(bisect_ok, rho_bisect, jnp.nan)

    return jax.lax.cond(
        converged,
        lambda _: rho_halley,
        _bisect_branch,
        None,
    )


# ── Implicit differentiation via custom_jvp ───────────────────────────────

@jax.custom_jvp
def _density_from_PT(T, P, phase_hint):
    """Find ρ such that P(T,ρ) = P_target. Scalar inputs.

    Args:
        T: Temperature [K] (scalar)
        P: Pressure [Pa] (scalar)
        phase_hint: 0=liquid, 1=vapor, 2=supercritical (integer scalar)

    Returns:
        rho: Density [kg/m³] (scalar)
    """
    return _solve_density(T, P, phase_hint)


@_density_from_PT.defjvp
def _density_from_PT_jvp(primals, tangents):
    """JVP via implicit function theorem.

    P(T, ρ) = P_target  →  (∂P/∂T)dT + (∂P/∂ρ)dρ = dP
    ⇒  dρ = (dP − (∂P/∂T)·dT) / (∂P/∂ρ)
    """
    T, P, phase_hint = primals
    dT, dP, _ = tangents

    rho_star = _density_from_PT(T, P, phase_hint)

    dP_drho_val = _dP_drho(T, rho_star)
    dP_dT_val = _dP_dT(T, rho_star)

    drho = (dP - dP_dT_val * dT) / _safe_denom(dP_drho_val)

    return rho_star, drho


# ── Public API ────────────────────────────────────────────────────────────

@jax.jit
def density_from_PT(T, P, phase_hint):
    """Find ρ [kg/m³] such that P(T,ρ) = P_target.

    All inputs broadcast: scalar or 1-D arrays of the same length.

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        phase_hint: 0=liquid, 1=vapor, 2=supercritical/auto

    Returns:
        rho: Density [kg/m³]
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)
    phase_hint = jnp.asarray(phase_hint, dtype=jnp.int32)
    return jax.vmap(_density_from_PT)(T, P, phase_hint)


# ═════════════════════════════════════════════════════════════════════════
# 2D inversion: state_from_Ph — find (T, ρ) given (P, h)
# ═════════════════════════════════════════════════════════════════════════

# ── Initial guess for (T, ρ) from (P, h) ──────────────────────────────

def _initial_guess_Ph(P, h, phase_hint):
    """Estimate (T, ρ) from (P, h) for the 2D Newton solver.

    Strategy:
    - Sub-critical P: use T_sat(P) as temperature estimate.
      Compare h to saturation enthalpies to decide phase.
    - Super-critical P: estimate T from ideal-gas enthalpy inversion
      (h ≈ cp0 * T → T ≈ h / cp0), clamped to a reasonable range.
    - Then call the 1D density solver to get ρ from (T_guess, P).
    """
    subcritical = P < sw.PC

    # Sub-critical path: T_sat from saturation table
    T_sat = sat.saturation_temperature(P)
    h_l, h_v = sat.saturation_enthalpies(P)

    # If h > h_v → superheated vapor, estimate T above T_sat
    # If h < h_l → compressed liquid, use T_sat as starting point
    # Otherwise → near saturation, use T_sat
    T_sub = jnp.where(
        h > h_v,
        T_sat + (h - h_v) / (sw.R * 3.5),  # rough Cp estimate
        jnp.where(h < h_l, T_sat - 1.0, T_sat)
    )

    # Super-critical path: anchor on pseudocritical temperature.
    # Near Pc, the Clausius-Clapeyron slope dT/dP ≈ 0.028 K/kPa ≈ 28 K/MPa.
    # This extrapolation gives T_pc ≈ Tc + (P - Pc) * slope, which tracks
    # the Cp peak (Widom line) much better than an ideal-gas estimate.
    dTdP_sat = 2.8e-5  # K/Pa — Clausius-Clapeyron slope near Tc
    T_pseudo = sw.TC + (P - sw.PC) * dTdP_sat

    # Ideal-gas fallback for pressures far above Pc where the pseudocritical
    # extrapolation becomes less relevant
    T_ideal = h / (sw.R * 5.0)

    # Use pseudocritical anchor when P < 2*Pc (near-critical), else ideal-gas
    T_super = jnp.where(P < 2.0 * sw.PC, T_pseudo, T_ideal)

    # Pseudocritical-aware override: when P is close to PC and h is in the
    # pseudocritical enthalpy band, start from TC.  The Widom-line Cp peak
    # causes the Newton iteration to oscillate wildly if the initial T guess
    # is on the wrong side; anchoring to TC lets the damped Newton converge.
    h_crit = sw._scalar_enthalpy(sw.TC, sw.RHOC)  # enthalpy at critical point
    near_pc = jnp.abs(P - sw.PC) < 0.5 * sw.PC
    h_in_band = jnp.abs(h - h_crit) < 100.0e3  # ±100 kJ/kg
    T_super = jnp.where(jnp.logical_and(near_pc, h_in_band), sw.TC, T_super)

    # Compressed-liquid override: when P > PC but h is well below the
    # critical-point enthalpy, the fluid is subcooled liquid.  Estimate T
    # from a rough liquid Cp (~2500 J/kg/K) below TC rather than anchoring
    # on the pseudocritical line which overshoots into the supercritical region.
    Cp_liq = 2500.0  # J/(kg·K) — rough subcooled CO₂ liquid heat capacity
    T_liq_est = sw.TC - (h_crit - h) / Cp_liq
    h_clearly_liquid = h < h_crit - 50.0e3  # h at least 50 kJ/kg below h_crit
    T_super = jnp.where(h_clearly_liquid, T_liq_est, T_super)

    T_super = jnp.clip(T_super, 250.0, 800.0)

    T0 = jnp.where(subcritical, T_sub, T_super)
    # Clamp T to physical range
    T0 = jnp.clip(T0, sw.T_TRIPLE, 800.0)

    rho0 = _solve_density(T0, P, phase_hint)
    return T0, rho0


# ── 2D Newton solver ──────────────────────────────────────────────────

def _newton_body_2d(state):
    """One 2D Newton step on F = [P(T,ρ) - P_target, h(T,ρ) - h_target]."""
    T, rho, P_target, h_target, i, converged = state

    # Residual
    fP = sw._scalar_pressure(T, rho) - P_target
    fh = sw._scalar_enthalpy(T, rho) - h_target

    # Jacobian: J = [[dP/dT, dP/drho], [dh/dT, dh/drho]]
    j00 = _dP_dT(T, rho)
    j01 = _dP_drho(T, rho)
    j10 = _dh_dT(T, rho)
    j11 = _dh_drho(T, rho)

    # Analytic 2×2 inverse: J^{-1} = (1/det) * [[j11, -j01], [-j10, j00]]
    det = j00 * j11 - j01 * j10

    dT = (j11 * fP - j01 * fh) / _safe_denom(det)
    drho = (-j10 * fP + j00 * fh) / _safe_denom(det)

    # Step damping: prevent overshooting across Cp peak near Widom line
    dT = jnp.clip(dT, -20.0, 20.0)
    drho = jnp.clip(drho, -0.5 * rho, 0.5 * rho)

    T_new = T - dT
    rho_new = rho - drho

    # Clamp to physical bounds
    T_new = jnp.maximum(T_new, sw.T_TRIPLE)
    rho_new = jnp.maximum(rho_new, 1.0)

    residual_small = jnp.maximum(jnp.abs(fP), jnp.abs(fh)) < TOL
    step_small = jnp.logical_and(
        jnp.abs(dT) < 1e-13 * jnp.maximum(jnp.abs(T_new), 1.0),
        jnp.abs(drho) < 1e-13 * jnp.maximum(jnp.abs(rho_new), 1.0)
    )
    converged_new = jnp.logical_or(residual_small, step_small)
    return (T_new, rho_new, P_target, h_target, i + 1, converged_new)


def _newton_cond_2d(state):
    """Continue while not converged and under max iterations."""
    T, rho, P_target, h_target, i, converged = state
    return jnp.logical_and(i < MAX_ITER, jnp.logical_not(converged))


def _solve_state_Ph(P, h, phase_hint):
    """2D Newton solve for (T, ρ) given (P, h, phase_hint). Scalar inputs."""
    T0, rho0 = _initial_guess_Ph(P, h, phase_hint)
    init_state = (T0, rho0, P, h, jnp.int32(0), jnp.bool_(False))
    final_state = jax.lax.while_loop(_newton_cond_2d, _newton_body_2d, init_state)
    T_out, rho_out, _, _, _, converged = final_state
    return (jnp.where(converged, T_out, jnp.nan),
            jnp.where(converged, rho_out, jnp.nan))


# ── Implicit differentiation via custom_jvp ──────────────────────────

@jax.custom_jvp
def _state_from_Ph(P, h, phase_hint):
    """Find (T, ρ) such that P(T,ρ) = P and h(T,ρ) = h. Scalar inputs.

    For subcritical P where h falls inside the two-phase dome (h_l ≤ h ≤ h_v),
    returns saturation-boundary properties instead of solving the single-phase
    Newton: T = T_sat(P), ρ = ρ_l(P) or ρ_v(P) depending on phase_hint.

    Args:
        P: Pressure [Pa] (scalar)
        h: Specific enthalpy [J/kg] (scalar)
        phase_hint: 0=liquid, 1=vapor, 2=supercritical (integer scalar)

    Returns:
        (T, rho): Temperature [K] and density [kg/m³] (scalars)
    """
    # --- Dome detection ---
    # Saturation lookups (always evaluated; JAX semantics)
    T_sat = sat.saturation_temperature(P)
    h_l, h_v = sat.saturation_enthalpies(P)
    rho_l, rho_v = sat.saturation_densities_P(P)

    in_dome = jnp.logical_and(P < sw.PC, jnp.logical_and(h >= h_l, h <= h_v))

    # Dome-branch density: respect phase_hint
    # LIQUID → ρ_l, VAPOR → ρ_v, SUPERCRITICAL → pick by enthalpy midpoint
    h_mid = 0.5 * (h_l + h_v)
    rho_dome = jnp.where(
        phase_hint == LIQUID, rho_l,
        jnp.where(phase_hint == VAPOR, rho_v,
                  jnp.where(h < h_mid, rho_l, rho_v))
    )

    # --- Single-phase path (Newton solver, unchanged) ---
    T_sp, rho_sp = _solve_state_Ph(P, h, phase_hint)

    # Merge: dome wins when inside
    T_out = jnp.where(in_dome, T_sat, T_sp)
    rho_out = jnp.where(in_dome, rho_dome, rho_sp)
    return T_out, rho_out


@_state_from_Ph.defjvp
def _state_from_Ph_jvp(primals, tangents):
    """JVP via implicit function theorem on F(T,ρ; P,h) = 0.

    Single-phase path:
        F = [P(T,ρ) - P_target, h(T,ρ) - h_target]
        J = [[∂P/∂T, ∂P/∂ρ], [∂h/∂T, ∂h/∂ρ]]
        [dT, dρ] = J⁻¹ · [dP, dh]

    Dome path:
        T = T_sat(P), so dT = (dT_sat/dP)·dP, dT/dh = 0
        ρ = ρ_branch(P), so dρ = (dρ_branch/dP)·dP, dρ/dh = 0
    """
    P, h, phase_hint = primals
    dP, dh, _ = tangents

    T_star, rho_star = _state_from_Ph(P, h, phase_hint)

    # --- Dome detection (same logic as primal) ---
    h_l, h_v = sat.saturation_enthalpies(P)
    in_dome = jnp.logical_and(P < sw.PC, jnp.logical_and(h >= h_l, h <= h_v))

    # --- Dome tangents via autodiff through saturation splines ---
    dTsat_dP = jax.grad(sat.saturation_temperature)(P)

    # Differentiate the selected density branch w.r.t. P
    def _rho_dome_of_P(P_):
        rho_l_, rho_v_ = sat.saturation_densities_P(P_)
        h_l_, h_v_ = sat.saturation_enthalpies(P_)
        h_mid_ = 0.5 * (h_l_ + h_v_)
        return jnp.where(
            phase_hint == LIQUID, rho_l_,
            jnp.where(phase_hint == VAPOR, rho_v_,
                      jnp.where(h < h_mid_, rho_l_, rho_v_))
        )
    drhodome_dP = jax.grad(_rho_dome_of_P)(P)

    dT_dome = dTsat_dP * dP
    drho_dome = drhodome_dP * dP

    # --- Single-phase tangents (implicit function theorem) ---
    j00 = _dP_dT(T_star, rho_star)
    j01 = _dP_drho(T_star, rho_star)
    j10 = _dh_dT(T_star, rho_star)
    j11 = _dh_drho(T_star, rho_star)

    det = j00 * j11 - j01 * j10
    det_safe = _safe_denom(det)

    # J⁻¹ = (1/det) * [[j11, -j01], [-j10, j00]]
    dT_sp = (j11 * dP - j01 * dh) / det_safe
    drho_sp = (-j10 * dP + j00 * dh) / det_safe

    # Merge tangents
    dT = jnp.where(in_dome, dT_dome, dT_sp)
    drho = jnp.where(in_dome, drho_dome, drho_sp)

    return (T_star, rho_star), (dT, drho)


# ── Public API ────────────────────────────────────────────────────────

@jax.jit
def state_from_Ph(P, h, phase_hint):
    """Find (T [K], ρ [kg/m³]) such that P(T,ρ) = P and h(T,ρ) = h.

    All inputs broadcast: scalar or 1-D arrays of the same length.

    Args:
        P: Pressure [Pa]
        h: Specific enthalpy [J/kg]
        phase_hint: 0=liquid, 1=vapor, 2=supercritical/auto

    Returns:
        (T, rho): Temperature [K] and density [kg/m³]
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    phase_hint = jnp.asarray(phase_hint, dtype=jnp.int32)
    return jax.vmap(_state_from_Ph)(P, h, phase_hint)


# ═════════════════════════════════════════════════════════════════════════
# 1D inversion: temperature_from_Du — find T given (ρ, u)
# ═════════════════════════════════════════════════════════════════════════

# ── Initial guess for T from (ρ, u) ──────────────────────────────────

def _initial_guess_Du(rho, u):
    """Estimate T from (ρ, u) using u ≈ Cv_approx * R * T.

    Cv_approx ~ 3.5 is a rough dimensionless heat capacity for CO₂.
    """
    T0 = u / (3.5 * sw.R)
    return jnp.clip(T0, sw.T_TRIPLE, 800.0)


# ── 1D Newton solver for T(ρ, u) ─────────────────────────────────────

def _newton_body_Du(state):
    """One Newton step: T_{n+1} = T_n - f(T)/f'(T) where f = u(T,ρ) - u_target."""
    T, rho, u_target, i, converged = state
    f = sw._scalar_internal_energy(T, rho) - u_target
    fp = _du_dT(T, rho)  # = Cv > 0
    step = f / _safe_denom(fp)
    T_new = T - step
    T_new = jnp.clip(T_new, sw.T_TRIPLE, 800.0)
    converged_new = jnp.logical_or(
        jnp.abs(f) < TOL,
        jnp.abs(step) < 1e-13 * jnp.maximum(jnp.abs(T_new), 1.0)
    )
    return (T_new, rho, u_target, i + 1, converged_new)


def _newton_cond_Du(state):
    """Continue while not converged and under max iterations."""
    T, rho, u_target, i, converged = state
    return jnp.logical_and(i < MAX_ITER, jnp.logical_not(converged))


def _solve_temperature_Du(rho, u, phase_hint):
    """Newton solve for T given (ρ, u, phase_hint). Scalar inputs."""
    T0 = _initial_guess_Du(rho, u)
    init_state = (T0, rho, u, jnp.int32(0), jnp.bool_(False))
    final_state = jax.lax.while_loop(_newton_cond_Du, _newton_body_Du, init_state)
    T_out, _, _, _, converged = final_state
    return jnp.where(converged, T_out, jnp.nan)


# ── Implicit differentiation via custom_jvp ──────────────────────────

@jax.custom_jvp
def _temperature_from_Du(rho, u, phase_hint):
    """Find T such that u(T, ρ) = u_target at fixed ρ. Scalar inputs.

    Args:
        rho: Density [kg/m³] (scalar)
        u: Specific internal energy [J/kg] (scalar)
        phase_hint: 0=liquid, 1=vapor, 2=supercritical (integer scalar)

    Returns:
        T: Temperature [K] (scalar)
    """
    return _solve_temperature_Du(rho, u, phase_hint)


@_temperature_from_Du.defjvp
def _temperature_from_Du_jvp(primals, tangents):
    """JVP via implicit function theorem.

    u(T, ρ) = u_target  →  (∂u/∂T)dT + (∂u/∂ρ)dρ = du
    ⇒  dT = (du − (∂u/∂ρ)·dρ) / (∂u/∂T)
    """
    rho, u, phase_hint = primals
    drho, du, _ = tangents

    T_star = _temperature_from_Du(rho, u, phase_hint)

    du_dT_val = _du_dT(T_star, rho)     # = Cv
    du_drho_val = _du_drho(T_star, rho)

    dT = (du - du_drho_val * drho) / _safe_denom(du_dT_val)

    return T_star, dT


# ── Public API ────────────────────────────────────────────────────────

@jax.jit
def temperature_from_Du(rho, u, phase_hint):
    """Find T [K] such that u(T, ρ) = u_target at fixed ρ.

    All inputs broadcast: scalar or 1-D arrays of the same length.

    Args:
        rho: Density [kg/m³]
        u: Specific internal energy [J/kg]
        phase_hint: 0=liquid, 1=vapor, 2=supercritical/auto

    Returns:
        T: Temperature [K]
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase_hint = jnp.asarray(phase_hint, dtype=jnp.int32)
    return jax.vmap(_temperature_from_Du)(rho, u, phase_hint)


# ── Convenience wrapper: state_from_Du ────────────────────────────────

@jax.custom_jvp
def _state_from_Du(rho, u, phase_hint):
    """Find (T, P, h) given (ρ, u). Scalar inputs.

    Solves for T via 1D Newton, then forward-evaluates P and h.

    Args:
        rho: Density [kg/m³] (scalar)
        u: Specific internal energy [J/kg] (scalar)
        phase_hint: 0=liquid, 1=vapor, 2=supercritical (integer scalar)

    Returns:
        (T, P, h): Temperature [K], Pressure [Pa], Enthalpy [J/kg]
    """
    T = _solve_temperature_Du(rho, u, phase_hint)
    P = sw._scalar_pressure(T, rho)
    h = sw._scalar_enthalpy(T, rho)
    return T, P, h


@_state_from_Du.defjvp
def _state_from_Du_jvp(primals, tangents):
    """JVP via implicit differentiation + chain rule.

    The implicit solve gives T*(ρ, u):
        dT = (du − (∂u/∂ρ)·dρ) / (∂u/∂T)

    Then chain rule for P and h:
        dP = (∂P/∂T)·dT + (∂P/∂ρ)·dρ
        dh = (∂h/∂T)·dT + (∂h/∂ρ)·dρ
    """
    rho, u, phase_hint = primals
    drho, du, _ = tangents

    T_star, P_star, h_star = _state_from_Du(rho, u, phase_hint)

    # IFT for T*: u(T,ρ) = u_target
    Cv = _du_dT(T_star, rho)
    du_drho_val = _du_drho(T_star, rho)
    dT = (du - du_drho_val * drho) / _safe_denom(Cv)

    # Chain rule for P and h
    dP = _dP_dT(T_star, rho) * dT + _dP_drho(T_star, rho) * drho
    dh = _dh_dT(T_star, rho) * dT + _dh_drho(T_star, rho) * drho

    return (T_star, P_star, h_star), (dT, dP, dh)


@jax.jit
def state_from_Du(rho, u, phase_hint):
    """Find (T [K], P [Pa], h [J/kg]) given (ρ, u) at fixed density.

    Solves for T via 1D Newton on u(T,ρ) = u_target, then forward-evaluates
    pressure and enthalpy.

    All inputs broadcast: scalar or 1-D arrays of the same length.

    Args:
        rho: Density [kg/m³]
        u: Specific internal energy [J/kg]
        phase_hint: 0=liquid, 1=vapor, 2=supercritical/auto

    Returns:
        (T, P, h): Temperature [K], Pressure [Pa], Enthalpy [J/kg]
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase_hint = jnp.asarray(phase_hint, dtype=jnp.int32)
    return jax.vmap(_state_from_Du)(rho, u, phase_hint)
