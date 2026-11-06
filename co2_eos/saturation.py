"""CO2 saturation curve lookups: pure JAX, JIT-compilable.

Loads a precomputed saturation table (cubic spline coefficients) generated
by scripts/generate_saturation_table.py by a Maxwell construction on the
selected backend's surface: ``saturation_table.npz`` for Span-Wagner,
``saturation_table_nist2026.npz`` for ``eos="nist2026"``.  Every lookup takes
the static keyword ``eos`` with the default ``DEFAULT_EOS`` (nist2026), like
the other submodules' module-level names (``core``, ``inversions``,
``transport``) and the package's public names; Span-Wagner is named,
``eos="span_wagner"``.

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
from functools import partial
from types import SimpleNamespace

jax.config.update("jax_enable_x64", True)

from co2_eos import backends as _backends
from co2_eos.backends import DEFAULT_EOS

# ── Table loading (lazy, once) ───────────────────────────────────────────
# Table is loaded outside JIT tracing to avoid tracer leaks.
# JAX arrays are stored as module globals; JIT captures them as constants.

_TABLES = {}


def _ensure_loaded(eos=DEFAULT_EOS):
    """Load the backend's saturation table on first use. Call outside JIT."""
    if eos in _TABLES:
        return
    if not _backends.get(eos).has_dome:
        raise ValueError(
            f"eos {eos!r} is a reference gas: it has no saturation curve; "
            f"the saturation lookups serve {_backends.CO2_EOS_NAMES}")
    data = np.load(_backends.table_path(eos, "saturation_table"))
    _TABLES[eos] = {k: jnp.array(data[k]) for k in data}


def _table(eos):
    _ensure_loaded(eos)
    return _TABLES[eos]


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

def saturation_pressure(T, eos=DEFAULT_EOS):
    """Saturation pressure P_sat [Pa] at temperature T [K].

    Valid for T in [216.592, 304.127] K (triple point to near-critical).
    JIT-compilable and differentiable.
    """
    tab = _table(eos)
    return _eval_spline(T, tab['T_breaks'], tab['P_sat_c'])


def saturation_densities(T, eos=DEFAULT_EOS):
    """Saturated liquid and vapor densities [kg/m3] at temperature T [K].

    Returns:
        (rho_l, rho_v): liquid and vapor densities
    """
    tab = _table(eos)
    rho_l = _eval_spline(T, tab['T_breaks'], tab['rho_l_c'])
    rho_v = _eval_spline(T, tab['T_breaks'], tab['rho_v_c'])
    return rho_l, rho_v


def saturation_enthalpies_T(T, eos=DEFAULT_EOS):
    """Saturated liquid and vapor enthalpies [J/kg] at temperature T [K].

    Returns:
        (h_l, h_v)
    """
    tab = _table(eos)
    h_l = _eval_spline(T, tab['T_breaks'], tab['h_l_c'])
    h_v = _eval_spline(T, tab['T_breaks'], tab['h_v_c'])
    return h_l, h_v


def saturation_entropies(T, eos=DEFAULT_EOS):
    """Saturated liquid and vapor entropies [J/(kg*K)] at temperature T [K].

    Returns:
        (s_l, s_v)
    """
    tab = _table(eos)
    s_l = _eval_spline(T, tab['T_breaks'], tab['s_l_c'])
    s_v = _eval_spline(T, tab['T_breaks'], tab['s_v_c'])
    return s_l, s_v


# ── P-based lookups ─────────────────────────────────────────────────────

def saturation_temperature(P, eos=DEFAULT_EOS):
    """Saturation temperature T_sat [K] at pressure P [Pa].

    Valid for P in [P_triple, P_near_crit].
    """
    tab = _table(eos)
    return _eval_spline(P, tab['P_breaks'], tab['T_sat_c'])


def saturation_enthalpies(P, eos=DEFAULT_EOS):
    """Saturated liquid and vapor enthalpies [J/kg] at pressure P [Pa].

    Returns:
        (h_l, h_v)
    """
    tab = _table(eos)
    h_l = _eval_spline(P, tab['P_breaks'], tab['h_l_P_c'])
    h_v = _eval_spline(P, tab['P_breaks'], tab['h_v_P_c'])
    return h_l, h_v


def saturation_densities_P(P, eos=DEFAULT_EOS):
    """Saturated liquid and vapor densities [kg/m3] at pressure P [Pa].

    Returns:
        (rho_l, rho_v)
    """
    tab = _table(eos)
    rho_l = _eval_spline(P, tab['P_breaks'], tab['rho_l_P_c'])
    rho_v = _eval_spline(P, tab['P_breaks'], tab['rho_v_P_c'])
    return rho_l, rho_v


def saturation_entropies_P(P, eos=DEFAULT_EOS):
    """Saturated liquid and vapor entropies [J/(kg*K)] at pressure P [Pa].

    Returns:
        (s_l, s_v)
    """
    tab = _table(eos)
    s_l = _eval_spline(P, tab['P_breaks'], tab['s_l_P_c'])
    s_v = _eval_spline(P, tab['P_breaks'], tab['s_v_P_c'])
    return s_l, s_v


_LOOKUPS = ("saturation_pressure", "saturation_densities",
            "saturation_enthalpies_T", "saturation_entropies",
            "saturation_temperature", "saturation_enthalpies",
            "saturation_densities_P", "saturation_entropies_P")


def for_eos(eos):
    """The lookups bound to backend ``eos``, under their module names."""
    _backends.get(eos)
    ns = {name: partial(globals()[name], eos=eos) for name in _LOOKUPS}
    return SimpleNamespace(eos=eos, _ensure_loaded=partial(_ensure_loaded, eos),
                           **ns)
