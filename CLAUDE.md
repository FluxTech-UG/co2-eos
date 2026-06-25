# co2-eos

Differentiable CO₂ thermodynamic properties in JAX — a pure-JAX implementation of the
Span-Wagner (1996) equation of state with hand-coded analytic α-derivatives. Every
property is JIT-compilable, `vmap`-vectorisable, and differentiable with `jax.grad`.
This is a **package**: the simulation and modelling repos import it; it imports nothing
back. Public and open-source (Apache-2.0).

## Where the code is

The file map is generated: read **`docs/MAP.md`** (produced by `repo-outline`) for the
current symbol-level inventory. The shape:

- `co2_eos/` — the importable package. `__init__.py` is the **public API surface**: the
  hot path `properties_from_rho_u(ρ, u)`, the scalar `state_from_Du`, `properties(T, ρ)`,
  `state_from_PT`, `state_from_Ph`, `density_from_PT`, `viscosity`,
  `thermal_conductivity`, and the phase constants. Internals: `span_wagner.py` (the EOS
  and analytic α-derivatives), `helmholtz.py`, `core.py` (the fused (ρ, u)
  primitive-recovery kernel), `inversions.py`, `saturation.py`, `transport.py`, and the
  precomputed saturation/seed tables in `co2_eos/data/`.
- `tests/` — the validation suite; it is the package's contract (see Critical rules).
- `bench/` — profiling and validation studies; `scripts/` — table generation;
  `examples/` — the launch demo.

## Critical rules

**The public API surface is the contract — keep it stable and single-sourced.**
Consumers import this package, so a breaking change to a public name or signature
ripples into every simulation. Canonical values (the Span-Wagner coefficients, the
critical point) live in exactly one place and every other site derives from them.

**Every property derives from the Helmholtz free energy A(T, ρ).** Pressure, heat
capacities, speed of sound, and the rest come from the hand-coded analytic
α-derivatives. They are validated against `jax.grad` to < 3e-12; if they diverge, the
derivatives are wrong, not the tolerance.

**CoolProp is a test-only dependency — never reachable from an import of the package.**
It generates validation data and the offline tables; shipped code never calls it.
`tests/test_no_coolprop_runtime.py` guards this.

**Preserve differentiability, JIT, and vmap.** The code is pure JAX: no Python-side
data-dependent branching that breaks tracing, no NumPy on the traced path. Use `jax.lax`
control flow for anything that depends on a traced value, so `jit`, `vmap`, `grad`, and
`custom_jvp` keep working end-to-end.

**Accuracy is validated against CoolProp across the full valid range.** The test suite
(`tests/test_accuracy_vs_coolprop.py`, `test_analytic_derivs.py`,
`test_inversion_gradients.py`, `test_dome_detection.py`) is the acceptance bar; a change
is done when it is green.
