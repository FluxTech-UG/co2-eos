"""CO2-EOS: Differentiable CO₂ thermodynamic properties in JAX.

Public API
==========

State functions (return a dict of thermodynamic + transport properties):

    properties(T, rho)              — full state from (T, ρ)
    state_from_PT(P, T)             — full state from (P, T)
    state_from_Ph(P, h)             — full state from (P, h)
    state_from_Du(rho, u)           — full state from (ρ, u)

Single-property inversion:

    density_from_PT(P, T)           — ρ from (P, T)

Saturation curve (cubic-spline interpolation of a precomputed Maxwell table):

    saturation_pressure(T)          → P_sat
    saturation_temperature(P)       → T_sat
    saturation_densities(T)         → (ρ_l, ρ_v)
    saturation_densities_P(P)       → (ρ_l, ρ_v)
    saturation_enthalpies(T)        → (h_l, h_v)
    saturation_enthalpies_P(P)      → (h_l, h_v)
    saturation_entropies(T)         → (s_l, s_v)
    saturation_entropies_P(P)       → (s_l, s_v)

Transport:

    viscosity(T, rho)               → μ  [Pa·s]
    thermal_conductivity(T, rho)    → λ  [W/(m·K)]

Phase hints (for state_from_PT, state_from_Ph, state_from_Du, density_from_PT):

    LIQUID = 0, VAPOR = 1, SUPERCRITICAL = 2 (alias: AUTO)

Convention
----------
Public state functions take their arguments in the order suggested by the
function name: ``state_from_PT(P, T)``, ``state_from_Ph(P, h)``,
``state_from_Du(ρ, u)``. ``properties`` and the transport correlations follow
the natural Helmholtz form ``(T, ρ)``.

All public functions are scalar in the inputs they accept (0-d arrays or
Python floats). For batches, wrap with ``jax.vmap``::

    states = jax.vmap(lambda T: co2.state_from_PT(P=8e6, T=T))(T_array)
"""

import jax
import jax.numpy as jnp

# Enable float64 for the entire library (also done by submodules).
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

# ── Phase hint constants ────────────────────────────────────────────────────
LIQUID = 0
VAPOR = 1
SUPERCRITICAL = 2
AUTO = SUPERCRITICAL  # alias: auto-pick in the dome / supercritical-style guess

# ── Internal modules ────────────────────────────────────────────────────────
from co2_eos import span_wagner as _sw
from co2_eos import transport as _tr
from co2_eos import saturation as _sat
from co2_eos import inversions as _inv

# ── Re-exports: saturation curve ────────────────────────────────────────────
# Naming convention: the T-input form is the unsuffixed name; the P-input
# form has a "_P" suffix. The internal saturation module uses inconsistent
# names — these aliases give the public surface a single coherent rule.
saturation_pressure = _sat.saturation_pressure
saturation_temperature = _sat.saturation_temperature
saturation_densities = _sat.saturation_densities
saturation_densities_P = _sat.saturation_densities_P
saturation_enthalpies = _sat.saturation_enthalpies_T
saturation_enthalpies_P = _sat.saturation_enthalpies
saturation_entropies = _sat.saturation_entropies
saturation_entropies_P = _sat.saturation_entropies_P

# ── Re-exports: transport ───────────────────────────────────────────────────
# Re-bound as scalar-input functions so they fit the rest of the public API.
# The internal transport module's `viscosity`/`thermal_conductivity` are
# pre-vmapped and would reject scalar inputs; here we expose the scalar form.
_scalar_viscosity = _tr._scalar_viscosity
_scalar_thermal_conductivity = _tr._scalar_thermal_conductivity


@jax.jit
def viscosity(T, rho):
    """Dynamic viscosity μ [Pa·s] at scalar (T, ρ).

    For batches: ``jax.vmap(viscosity)(T_array, rho_array)``.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _scalar_viscosity(T, rho)


@jax.jit
def thermal_conductivity(T, rho):
    """Thermal conductivity λ [W/(m·K)] at scalar (T, ρ).

    For batches: ``jax.vmap(thermal_conductivity)(T_array, rho_array)``.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _scalar_thermal_conductivity(T, rho)


# ── State dict builder (shared helper) ──────────────────────────────────────

def _state_dict(T, rho):
    """Compute every thermo + transport property at scalar (T, ρ).

    Reuses the shared-derivative bundle in span_wagner so the eight
    Helmholtz-derived properties cost a single autodiff pass.
    """
    P, Cv, Cp, w, h, u, s, g = _sw._scalar_all_properties_shared(T, rho)
    mu = _scalar_viscosity(T, rho)
    lam = _scalar_thermal_conductivity(T, rho)
    return {
        "temperature": T,
        "density": rho,
        "pressure": P,
        "cv": Cv,
        "cp": Cp,
        "speed_of_sound": w,
        "enthalpy": h,
        "internal_energy": u,
        "entropy": s,
        "gibbs_energy": g,
        "viscosity": mu,
        "thermal_conductivity": lam,
    }


# ── Public state functions ──────────────────────────────────────────────────

@jax.jit
def properties(T, rho):
    """Full thermodynamic + transport state at (T, ρ).

    Args:
        T: Temperature [K] (scalar)
        rho: Density [kg/m³] (scalar)

    Returns:
        dict with keys ``temperature``, ``density``, ``pressure``, ``cv``,
        ``cp``, ``speed_of_sound``, ``enthalpy``, ``internal_energy``,
        ``entropy``, ``gibbs_energy``, ``viscosity``, ``thermal_conductivity``.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _state_dict(T, rho)


@jax.jit
def state_from_PT(P, T, phase_hint=AUTO):
    """Full state at (P, T). Phase-aware Halley solve for ρ, then derive all.

    Args:
        P: Pressure [Pa] (scalar)
        T: Temperature [K] (scalar)
        phase_hint: LIQUID, VAPOR, or SUPERCRITICAL/AUTO (default AUTO).
            AUTO is an alias for SUPERCRITICAL. For points clearly in the
            subcritical liquid or vapor region, passing an explicit
            ``LIQUID`` or ``VAPOR`` hint gives a better initial guess and
            more robust convergence.

    Returns:
        State dict (see ``properties``). ``pressure`` echoes the input ``P``.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    rho = _inv._density_from_PT(T, P, phase)
    state = _state_dict(T, rho)
    state["pressure"] = P  # use the constraint exactly, regardless of solver residual
    return state


@jax.jit
def density_from_PT(P, T, phase_hint=AUTO):
    """Density [kg/m³] at (P, T). Phase-aware Halley solve.

    ``phase_hint`` defaults to AUTO (alias for SUPERCRITICAL). For points
    clearly in subcritical liquid or vapor regions, an explicit ``LIQUID``
    or ``VAPOR`` hint gives a better initial guess and more robust
    convergence.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    return _inv._density_from_PT(T, P, phase)


@jax.jit
def state_from_Ph(P, h, phase_hint=AUTO):
    """Full state at (P, h). Detects the two-phase dome at subcritical P
    and returns saturation-boundary properties; otherwise 2-D Newton on (T, ρ).

    Args:
        P: Pressure [Pa] (scalar)
        h: Specific enthalpy [J/kg] (scalar)
        phase_hint: LIQUID, VAPOR, or SUPERCRITICAL/AUTO (default AUTO).
            AUTO is an alias for SUPERCRITICAL; in the dome it picks ρ_l
            vs ρ_v by which side of the mean enthalpy ``h`` falls. For
            points clearly in the subcritical liquid or vapor region,
            passing an explicit ``LIQUID`` or ``VAPOR`` hint gives a
            better initial guess and more robust convergence.

    Returns:
        State dict (see ``properties``). ``pressure`` and ``enthalpy``
        echo the inputs.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    T, rho = _inv._state_from_Ph(P, h, phase)
    state = _state_dict(T, rho)
    state["pressure"] = P
    state["enthalpy"] = h
    return state


@jax.jit
def state_from_Du(rho, u, phase_hint=AUTO):
    """Full state at (ρ, u). 1-D Newton on u(T, ρ) at fixed ρ.

    Intended for simulation codes carrying (ρ, u) as conserved variables.

    Args:
        rho: Density [kg/m³] (scalar)
        u: Specific internal energy [J/kg] (scalar)
        phase_hint: accepted for API symmetry; not used by the (ρ, u) inversion.

    Returns:
        State dict (see ``properties``). ``density`` and ``internal_energy``
        echo the inputs.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    T = _inv._temperature_from_Du(rho, u, phase)
    state = _state_dict(T, rho)
    state["density"] = rho
    state["internal_energy"] = u
    return state


__all__ = [
    "__version__",
    "LIQUID", "VAPOR", "SUPERCRITICAL", "AUTO",
    "properties",
    "state_from_PT", "state_from_Ph", "state_from_Du",
    "density_from_PT",
    "saturation_pressure", "saturation_temperature",
    "saturation_densities", "saturation_densities_P",
    "saturation_enthalpies", "saturation_enthalpies_P",
    "saturation_entropies", "saturation_entropies_P",
    "viscosity", "thermal_conductivity",
]
