"""CO2 saturation curve lookups — pure JAX, JIT-compilable.

Loads a precomputed saturation table (cubic spline coefficients) generated
by scripts/generate_saturation_table.py from the Span-Wagner EOS.

All lookup functions use cubic spline interpolation and are fully
differentiable and JIT-compilable. They accept scalar or array inputs.

Usage:
    P = saturation_pressure(280.0)                    # eager
    P = jax.jit(saturation_pressure)(280.0)           # JIT
    dPdT = jax.grad(saturation_pressure)(280.0)       # differentiable
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

jax.config.update("jax_enable_x64", True)

_TABLE_PATH = Path(__file__).parent.parent / "data" / "saturation_table.npz"

# ── Table loading (lazy, once) ───────────────────────────────────────────
# Table is loaded outside JIT tracing to avoid tracer leaks.
# JAX arrays are stored as module globals; JIT captures them as constants.

_TABLE = {}


def _ensure_loaded():
    """Load saturation table on first use. Must be called outside JIT."""
    if _TABLE:
        return
    data = np.load(_TABLE_PATH)
    for k in data:
        _TABLE[k] = jnp.array(data[k])


# ── Cubic spline evaluation ─────────────────────────────────────────────

def _eval_spline(x, breaks, coeffs):
    """Evaluate cubic spline at x.

    Args:
        x: evaluation point(s), scalar or array
        breaks: knot positions, shape (N,), ascending
        coeffs: scipy CubicSpline coefficients, shape (4, N-1)
            S(x) = c[0]*(x-x_i)^3 + c[1]*(x-x_i)^2 + c[2]*(x-x_i) + c[3]

    Returns:
        Interpolated value(s), same shape as x
    """
    n = coeffs.shape[1]
    # Clamp x to valid range
    x_c = jnp.clip(x, breaks[0], breaks[-1])
    # Find interval index
    idx = jnp.clip(jnp.searchsorted(breaks, x_c, side='right') - 1, 0, n - 1)
    dx = x_c - breaks[idx]
    # Horner evaluation: ((c0*dx + c1)*dx + c2)*dx + c3
    return ((coeffs[0, idx] * dx + coeffs[1, idx]) * dx + coeffs[2, idx]) * dx + coeffs[3, idx]


# ── T-based lookups ─────────────────────────────────────────────────────

def saturation_pressure(T):
    """Saturation pressure P_sat [Pa] at temperature T [K].

    Valid for T in [216.592, 304.127] K (triple point to near-critical).
    JIT-compilable and differentiable.
    """
    _ensure_loaded()
    return _eval_spline(T, _TABLE['T_breaks'], _TABLE['P_sat_c'])


def saturation_densities(T):
    """Saturated liquid and vapor densities [kg/m3] at temperature T [K].

    Returns:
        (rho_l, rho_v) — liquid and vapor densities
    """
    _ensure_loaded()
    rho_l = _eval_spline(T, _TABLE['T_breaks'], _TABLE['rho_l_c'])
    rho_v = _eval_spline(T, _TABLE['T_breaks'], _TABLE['rho_v_c'])
    return rho_l, rho_v


def saturation_enthalpies_T(T):
    """Saturated liquid and vapor enthalpies [J/kg] at temperature T [K].

    Returns:
        (h_l, h_v)
    """
    _ensure_loaded()
    h_l = _eval_spline(T, _TABLE['T_breaks'], _TABLE['h_l_c'])
    h_v = _eval_spline(T, _TABLE['T_breaks'], _TABLE['h_v_c'])
    return h_l, h_v


def saturation_entropies(T):
    """Saturated liquid and vapor entropies [J/(kg*K)] at temperature T [K].

    Returns:
        (s_l, s_v)
    """
    _ensure_loaded()
    s_l = _eval_spline(T, _TABLE['T_breaks'], _TABLE['s_l_c'])
    s_v = _eval_spline(T, _TABLE['T_breaks'], _TABLE['s_v_c'])
    return s_l, s_v


# ── P-based lookups ─────────────────────────────────────────────────────

def saturation_temperature(P):
    """Saturation temperature T_sat [K] at pressure P [Pa].

    Valid for P in [P_triple, P_near_crit].
    """
    _ensure_loaded()
    return _eval_spline(P, _TABLE['P_breaks'], _TABLE['T_sat_c'])


def saturation_enthalpies(P):
    """Saturated liquid and vapor enthalpies [J/kg] at pressure P [Pa].

    Returns:
        (h_l, h_v)
    """
    _ensure_loaded()
    h_l = _eval_spline(P, _TABLE['P_breaks'], _TABLE['h_l_P_c'])
    h_v = _eval_spline(P, _TABLE['P_breaks'], _TABLE['h_v_P_c'])
    return h_l, h_v


def saturation_densities_P(P):
    """Saturated liquid and vapor densities [kg/m3] at pressure P [Pa].

    Returns:
        (rho_l, rho_v)
    """
    _ensure_loaded()
    rho_l = _eval_spline(P, _TABLE['P_breaks'], _TABLE['rho_l_P_c'])
    rho_v = _eval_spline(P, _TABLE['P_breaks'], _TABLE['rho_v_P_c'])
    return rho_l, rho_v
