"""CO2-EOS: Differentiable CO₂ thermodynamic properties in JAX.

Pure-JAX Helmholtz equations of state for carbon dioxide with hand-coded
analytic α-derivatives.  JIT-compilable, ``vmap``-able, fully differentiable.

Five backends, selected by the keyword ``eos`` every public function takes (a
Python string, static under ``jax.jit``: one compiled kernel per backend).
Two are CO₂ equations of state::

    eos="nist2026"      Lemmon, Harvey & Hellmann, NIST IR 8608 (2026), the
                        default: 24 terms, no nonanalytic terms, free of
                        Span-Wagner's near-critical oscillations in c_p, κ_T,
                        (∂c_v/∂T)_ρ
    eos="span_wagner"   Span & Wagner (1996), bit for bit 0.5.0

``CO2_EOS_NAMES`` lists them.  The transport correlations keep their own
parameters on either backend; their critical parts read the selected
backend's thermodynamics.

Three reference gases are backends of the same keyword (``REFERENCE_GAS_NAMES``)::

    eos="air"           Lemmon et al. (2000) virial series, Lemmon & Jacobsen (2004)
    eos="helium"        Ortiz Vega et al. virial series, an ab initio dilute fit
    eos="argon"         Tegeler et al. (1999) virial series, Lemmon & Jacobsen (2004)

Each is one closed-form Helmholtz energy (``reference_gases``): every bundle
field below, no saturation curve (the lookups raise), 200 to 1000 K up to its
``P_MAX``, NaN outside.  ``EOS_NAMES`` lists all five.

Public API
==========

Simulation hot path: full state from the conserved variables (ρ, u).  These are
**batched / array-native**: pass 1-D arrays, get back a dict of 1-D arrays.  A
single fused pass solves T once and reuses the α-derivatives for every property::

    properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL) -> dict
    temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL) -> T   # T only

Full state from (ρ, T), for a code that evolves temperature: also batched, no
inversion (a direct evaluation of the Helmholtz surface)::

    properties_from_rho_T(rho, T) -> dict

State functions: **scalar** in their inputs (0-d arrays or Python floats); wrap
with ``jax.vmap`` for batches::

    properties(T, rho)                full state from (T, ρ)
    state_from_PT(P, T)               full state from (P, T)
    state_from_Ph(P, h)               full state from (P, h)
    state_from_Du(rho, u)             full state from (ρ, u)   (scalar form)
    density_from_PT(P, T)             ρ from (P, T)

Each full-state dict has keys: ``temperature`` [K], ``density`` [kg/m³],
``pressure`` [Pa], ``cv`` and ``cp`` [J/(kg·K)], ``speed_of_sound`` [m/s],
``enthalpy``, ``internal_energy`` and ``gibbs_energy`` [J/kg], ``entropy``
[J/(kg·K)], ``dP_dT_rho`` ((∂P/∂T)_ρ [Pa/K]), ``dP_drho_T`` ((∂P/∂ρ)_T
[Pa·m³/kg]), ``viscosity`` [Pa·s], ``thermal_conductivity`` [W/(m·K)],
``bulk_viscosity`` [Pa·s] (ζ = ζ_vib + ζ_crit) and
``bulk_viscosity_vibrational`` [Pa·s] (its vibrational part ζ_vib alone).

Saturation curve (cubic-spline interpolation of a precomputed Maxwell table)::

    saturation_pressure(T) / saturation_temperature(P)
    saturation_densities(T) / saturation_densities_P(P)
    saturation_enthalpies(T) / saturation_enthalpies_P(P)
    saturation_entropies(T) / saturation_entropies_P(P)

Transport (scalar; vmap for batches)::

    viscosity(T, rho)               → μ  [Pa·s]  (no EOS input)
    thermal_conductivity(T, rho)    → λ  [W/(m·K)]
    bulk_viscosity(T, rho)          → ζ  [Pa·s]  (vibrational + critical)

Every function above also takes ``eos=`` (default ``"nist2026"``, ``DEFAULT_EOS``).

Phase hints::

    LIQUID = 0, VAPOR = 1, SUPERCRITICAL = 2 (alias: AUTO)

See README "Migrating to v0.2" for the move from the old per-quantity calls to
the fused ``properties_from_rho_u``.
"""

from functools import partial

import jax
import jax.numpy as jnp

# Enable float64 for the entire library (also done by submodules).
jax.config.update("jax_enable_x64", True)

__version__ = "0.8.0"

# ── Phase hint constants ────────────────────────────────────────────────────
LIQUID = 0
VAPOR = 1
SUPERCRITICAL = 2
AUTO = SUPERCRITICAL  # alias: auto-pick in the dome / supercritical-style guess

# ── Internal modules ────────────────────────────────────────────────────────
from co2_eos.backends import (EOS_NAMES, CO2_EOS_NAMES, REFERENCE_GAS_NAMES,
                              DEFAULT_EOS, SPAN_WAGNER, NIST2026, AIR, HELIUM,
                              ARGON)
from co2_eos import span_wagner as _sw
from co2_eos import transport as _tr
from co2_eos import saturation as _sat
from co2_eos import inversions as _inv
from co2_eos import core as _core

# jit with the backend name static: one compiled kernel per backend.
_jit = partial(jax.jit, static_argnames=("eos",))

# ── Saturation curve: the submodule's lookups at the package default ────────
# ``saturation``'s own names and these both default to ``DEFAULT_EOS``.

def saturation_pressure(T, eos=DEFAULT_EOS):
    """Saturation pressure P_sat [Pa] at temperature T [K]."""
    return _sat.saturation_pressure(T, eos=eos)


def saturation_temperature(P, eos=DEFAULT_EOS):
    """Saturation temperature T_sat [K] at pressure P [Pa]."""
    return _sat.saturation_temperature(P, eos=eos)


def saturation_densities(T, eos=DEFAULT_EOS):
    """Saturated (rho_l, rho_v) [kg/m³] at temperature T [K]."""
    return _sat.saturation_densities(T, eos=eos)


def saturation_densities_P(P, eos=DEFAULT_EOS):
    """Saturated (rho_l, rho_v) [kg/m³] at pressure P [Pa]."""
    return _sat.saturation_densities_P(P, eos=eos)


def saturation_enthalpies(T, eos=DEFAULT_EOS):
    """Saturated (h_l, h_v) [J/kg] at temperature T [K]."""
    return _sat.saturation_enthalpies_T(T, eos=eos)


def saturation_enthalpies_P(P, eos=DEFAULT_EOS):
    """Saturated (h_l, h_v) [J/kg] at pressure P [Pa]."""
    return _sat.saturation_enthalpies(P, eos=eos)


def saturation_entropies(T, eos=DEFAULT_EOS):
    """Saturated (s_l, s_v) [J/(kg·K)] at temperature T [K]."""
    return _sat.saturation_entropies(T, eos=eos)


def saturation_entropies_P(P, eos=DEFAULT_EOS):
    """Saturated (s_l, s_v) [J/(kg·K)] at pressure P [Pa]."""
    return _sat.saturation_entropies_P(P, eos=eos)



# ── Transport (scalar form; vmap for batches) ───────────────────────────────

@_jit
def viscosity(T, rho, eos=DEFAULT_EOS):
    """Dynamic viscosity μ [Pa·s] at scalar (T, ρ). vmap for batches.

    Laesecke-Muzny reads no EOS property; ``eos`` is accepted (and checked)
    so every property function takes the same keyword.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _tr.for_eos(eos)._scalar_viscosity(T, rho)


@_jit
def thermal_conductivity(T, rho, eos=DEFAULT_EOS):
    """Thermal conductivity λ [W/(m·K)] at scalar (T, ρ). vmap for batches."""
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _tr.for_eos(eos)._scalar_thermal_conductivity(T, rho)


@_jit
def bulk_viscosity(T, rho, eos=DEFAULT_EOS):
    """Bulk viscosity ζ [Pa·s] at scalar (T, ρ). vmap for batches.

    ζ = ζ_vib + ζ_crit, the zero-frequency coefficient: vibrational relaxation
    (Lin & Trusler 2014, pure CO₂) plus the critical part (Onuki 2007,
    R_B = 0.03) on the Olchowy-Sengers correlation length the conductivity
    enhancement uses.  Valid while ω·τ₁₁ ≪ 1, single phase, 270 to 500 K.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _tr.for_eos(eos)._scalar_bulk_viscosity(T, rho)


# ═══════════════════════════════════════════════════════════════════════════
# Hot path: full state from conserved (ρ, u), batched / array-native
# ═══════════════════════════════════════════════════════════════════════════

@_jit
def properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL,
                          eos=DEFAULT_EOS):
    """Full thermodynamic + transport state from (ρ, u). Batched.

    The simulation hot path: solves T once from u(T,ρ)=u via the analytic-Cv
    fixed-iteration Newton, then derives every property reusing the
    α-derivatives.  ``rho`` and ``u`` are 1-D arrays of equal length; the result
    is a dict of 1-D arrays.  ``density`` and ``internal_energy`` echo the
    inputs exactly.

    Args:
        rho: Density [kg/m³], 1-D array.
        u: Specific internal energy [J/kg], 1-D array.
        phase_hint: accepted for API symmetry; the single-phase (ρ, u)
            inversion is phase-agnostic (u is monotone in T at fixed ρ on a
            single-phase branch).

    Note: this is a single-phase EOS inversion. A (ρ, u) state inside the
    two-phase dome (subcritical T, ρ between ρ_v and ρ_l) is not a single-phase
    state; the EOS returns its unstable single-phase extrapolation there. The
    near-critical supercritical operating regime has no dome.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.broadcast_to(jnp.asarray(phase_hint, dtype=jnp.int32),
                             jnp.shape(rho))
    return jax.vmap(_core.for_eos(eos)._state_from_rho_u)(rho, u, phase)


@_jit
def properties_from_rho_T(rho, T, eos=DEFAULT_EOS):
    """Full thermodynamic + transport state from (ρ, T). Batched.

    The array-native counterpart of the scalar ``properties(T, rho)`` (note the
    argument order: ρ first, matching ``properties_from_rho_u``).  A direct
    evaluation of the Helmholtz surface: no inversion, so no phase hint.  One
    analytic α-derivative pass per point feeds every property, including
    ``dP_dT_rho`` and ``dP_drho_T``.  ``rho`` and ``T`` are 1-D arrays of
    equal length; the result is a dict of 1-D float64 arrays.

    This is ``jax.vmap`` of the same scalar kernel ``properties`` uses, so the
    two agree bit for bit on Span-Wagner (on nist2026, whose kernel sums its
    24 terms as one vector reduction, to round-off); it exists as the
    documented batched entry point.

    Note: inside the two-phase dome (subcritical T, ρ between ρ_v and ρ_l) the
    EOS returns its unstable single-phase extrapolation (there
    ``dP_drho_T`` can be negative); the caller keeps its states out of the dome.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    if rho.ndim != 1 or T.shape != rho.shape:
        raise ValueError(
            f"properties_from_rho_T expects 1-D rho and T of equal shape; "
            f"got rho {rho.shape}, T {T.shape}")
    return jax.vmap(_core.for_eos(eos)._state_from_T_rho)(T, rho)


@_jit
def temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL,
                           eos=DEFAULT_EOS):
    """Temperature T [K] such that u(T,ρ)=u at fixed ρ. Batched.

    The lean inversion (no property derivation); use when only T is needed.
    Differentiable: correct jvp/vjp via the implicit function theorem.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.broadcast_to(jnp.asarray(phase_hint, dtype=jnp.int32),
                             jnp.shape(rho))
    return jax.vmap(_core.for_eos(eos)._temperature_from_Du)(rho, u,
                                                            phase)


# ═══════════════════════════════════════════════════════════════════════════
# State functions: scalar in their inputs (vmap for batches)
# ═══════════════════════════════════════════════════════════════════════════

@_jit
def properties(T, rho, eos=DEFAULT_EOS):
    """Full thermodynamic + transport state at scalar (T, ρ).

    Returns a dict (see module docstring for keys). vmap for batches.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    return _core.for_eos(eos)._state_from_T_rho(T, rho)


@_jit
def state_from_PT(P, T, phase_hint=AUTO, eos=DEFAULT_EOS):
    """Full state at (P, T). Phase-aware Halley solve for ρ, then derive all.

    ``pressure`` in the returned dict echoes the input ``P`` exactly.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    rho = _inv.for_eos(eos)._density_from_PT(T, P, phase)
    state = _core.for_eos(eos)._state_from_T_rho(T, rho)
    state["pressure"] = P
    return state


@_jit
def density_from_PT(P, T, phase_hint=AUTO, eos=DEFAULT_EOS):
    """Density [kg/m³] at scalar (P, T). Phase-aware Halley solve."""
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    return _inv.for_eos(eos)._density_from_PT(T, P, phase)


@_jit
def state_from_Ph(P, h, phase_hint=AUTO, eos=DEFAULT_EOS):
    """Full state at (P, h). Dome-aware at subcritical P; else 2-D Newton.

    ``pressure`` and ``enthalpy`` echo the inputs.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    T, rho = _inv.for_eos(eos)._state_from_Ph(P, h, phase)
    state = _core.for_eos(eos)._state_from_T_rho(T, rho)
    state["pressure"] = P
    state["enthalpy"] = h
    return state


@_jit
def state_from_Du(rho, u, phase_hint=AUTO, eos=DEFAULT_EOS):
    """Full state at scalar (ρ, u). Scalar form of ``properties_from_rho_u``.

    ``density`` and ``internal_energy`` echo the inputs.
    """
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u = jnp.asarray(u, dtype=jnp.float64)
    phase = jnp.asarray(phase_hint, dtype=jnp.int32)
    return _core.for_eos(eos)._state_from_rho_u(rho, u, phase)


__all__ = [
    "__version__",
    "EOS_NAMES", "CO2_EOS_NAMES", "REFERENCE_GAS_NAMES", "DEFAULT_EOS",
    "SPAN_WAGNER", "NIST2026", "AIR", "HELIUM", "ARGON",
    "LIQUID", "VAPOR", "SUPERCRITICAL", "AUTO",
    "properties_from_rho_u", "properties_from_rho_T", "temperature_from_rho_u",
    "properties",
    "state_from_PT", "state_from_Ph", "state_from_Du",
    "density_from_PT",
    "saturation_pressure", "saturation_temperature",
    "saturation_densities", "saturation_densities_P",
    "saturation_enthalpies", "saturation_enthalpies_P",
    "saturation_entropies", "saturation_entropies_P",
    "viscosity", "thermal_conductivity", "bulk_viscosity",
]
