"""Run every Python snippet from README.md and print results.

Use this to regression-check that the README's advertised public API still
works end-to-end. Each block from the README is reproduced verbatim (or as
close as possible) and its outputs are validated against expected ranges.

Run::

    python examples/readme_check.py

Exits 0 on success, 1 on the first failure.
"""

import math
import sys

import jax
import jax.numpy as jnp

import co2_eos as co2


def _check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}{('  — ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def block_properties_from_T_rho():
    print("\n# Properties from (T, ρ)")
    state = co2.properties(T=310.0, rho=350.0)
    P = float(state["pressure"])
    cp = float(state["cp"])
    w = float(state["speed_of_sound"])
    print(f"  pressure        = {P:.4e} Pa")
    print(f"  cp              = {cp:.4e} J/(kg·K)")
    print(f"  speed_of_sound  = {w:.4e} m/s")
    _check("pressure is finite and positive", math.isfinite(P) and P > 0)
    _check("cp is finite and positive", math.isfinite(cp) and cp > 0)
    _check("speed_of_sound is finite and positive", math.isfinite(w) and w > 0)


def block_state_from_PT():
    print("\n# State from (P, T) — phase-aware, robust near the critical point")
    state = co2.state_from_PT(P=8e6, T=310.0)
    rho = float(state["density"])
    P = float(state["pressure"])
    print(f"  density   = {rho:.3f} kg/m³")
    print(f"  pressure  = {P:.4e} Pa")
    # CoolProp reference at (P=8 MPa, T=310 K): 327.7120900180151 kg/m³
    # (gas-like side of the Widom line — supercritical but low-density).
    _check("density agrees with CoolProp at (8 MPa, 310 K) to 1e-6 rel.",
           abs(rho - 327.7120900180151) / 327.7120900180151 < 1e-6,
           f"got {rho:.6f}, expected ~327.712")
    _check("pressure echoes the input (8e6 Pa)",
           abs(P - 8e6) < 1.0,
           f"got {P:.3e}")


def block_jax_grad():
    print("\n# Differentiate anything — e.g. ∂ρ/∂T at constant P")
    drho_dT = jax.grad(lambda T: co2.state_from_PT(P=8e6, T=T)["density"])(310.0)
    val = float(drho_dT)
    print(f"  drho/dT |_(P=8e6, T=310) = {val}")
    _check("jax.grad returned a finite scalar", math.isfinite(val))
    # Near the Widom line at (8 MPa, 310 K), CO2 expands strongly with T:
    # ∂ρ/∂T at constant P should be a large negative number (≈ -8 kg/(m³·K)).
    _check("∂ρ/∂T is negative (gas-like density decreases with T)", val < 0)


def block_jax_vmap():
    print("\n# Vectorise across conditions")
    T_array = jnp.linspace(280, 340, 100)
    states = jax.vmap(lambda T: co2.state_from_PT(P=8e6, T=T))(T_array)
    rho_arr = states["density"]
    P_arr = states["pressure"]
    print(f"  density.shape   = {rho_arr.shape}")
    print(f"  pressure.shape  = {P_arr.shape}")
    print(f"  density[0],[-1] = {float(rho_arr[0]):.3f}, {float(rho_arr[-1]):.3f}")
    _check("density is a length-100 array", rho_arr.shape == (100,))
    _check("pressure is a length-100 array", P_arr.shape == (100,))
    _check("all densities finite and positive",
           bool(jnp.all(jnp.isfinite(rho_arr))) and bool(jnp.all(rho_arr > 0)))


def block_dir_check():
    print("\n# `dir(co2_eos)` exposes every function named in README's 'What's included'")
    advertised = [
        "properties",
        "state_from_PT", "state_from_Ph", "state_from_Du",
        "saturation_pressure", "saturation_temperature",
        "saturation_densities", "saturation_densities_P",
        "saturation_enthalpies", "saturation_enthalpies_P",
        "saturation_entropies", "saturation_entropies_P",
        "viscosity", "thermal_conductivity",
    ]
    names = set(dir(co2))
    for f in advertised:
        _check(f"co2_eos exposes `{f}`", f in names)


def main():
    print("Running README.md code blocks against installed co2_eos…")
    block_properties_from_T_rho()
    block_state_from_PT()
    block_jax_grad()
    block_jax_vmap()
    block_dir_check()
    print("\nAll README snippets ran end-to-end with finite, in-range results.")


if __name__ == "__main__":
    main()
