# co2-eos

Differentiable CO₂ thermodynamic properties in JAX: pure-JAX Helmholtz equations of state
(NIST IR 8608 of 2026, the default, and Span-Wagner 1996) with hand-coded analytic
α-derivatives, and three reference gases (air, helium, argon) on the same keyword. Every
property is JIT-compilable, `vmap`-vectorisable, and differentiable with `jax.grad`.
This is a **package**: the simulation and modelling repos import it; it imports nothing
back. Public and open-source (Apache-2.0).

## Where the code is

The file map is generated: read **`docs/MAP.md`** (produced by `repo-outline`) for the
current symbol-level inventory. The shape:

- `co2_eos/`: the importable package. `__init__.py` is the **public API surface**: the
  hot path `properties_from_rho_u(ρ, u)`, the scalar `state_from_Du`, `properties(T, ρ)`,
  `state_from_PT`, `state_from_Ph`, `density_from_PT`, `viscosity`,
  `thermal_conductivity`, `bulk_viscosity`, `critical_point`, `triple_point`, and the phase
  constants. Internals: `span_wagner.py` (the Span-Wagner EOS and its autodiff properties),
  `span_wagner_kernel.py` (its analytic α-derivatives), `core.py` (the fused (ρ, u)
  primitive-recovery kernel), `inversions.py`, `saturation.py`, `transport.py`, and the
  precomputed saturation/seed tables in `co2_eos/data/`.
- **Five backends, one static switch.** Every public function takes `eos=`, a Python string
  static under `jax.jit`: `"nist2026"` (the default, `DEFAULT_EOS` in `backends.py`; NIST
  IR 8608, Lemmon, Harvey and Hellmann 2026: `nist2026_data.py` holds its constants, cited
  by page, `nist2026.py` its α, analytic kernels and property functions), `"span_wagner"`
  (`span_wagner.py` and `span_wagner_kernel.py`), and the reference gases `"air"`, `"helium"`, `"argon"`
  (`reference_gases_data.py`, every coefficient cited by page; `reference_gases.py`, one
  virial Helmholtz form, no saturation dome, NaN outside `T_RANGE` and `P_MAX`).
  `CO2_EOS_NAMES` and `REFERENCE_GAS_NAMES` are the two groups, `EOS_NAMES` all five; code
  that walks the CO₂ backends reads `CO2_EOS_NAMES`.
  `backends.py` is the registry; `core`, `inversions`, `saturation` and the
  EOS-coupled half of `transport` are built once per backend by a factory over the same
  code. Every generic name follows `DEFAULT_EOS`: the package's public names in
  `__init__.py` and every submodule's module-level names (`core._thermo`,
  `inversions.state_from_Ph`, `transport.thermal_conductivity`,
  `saturation.saturation_pressure`, ...) are the nist2026 instance. Span-Wagner is reached
  by its own name only: `eos="span_wagner"`, a submodule's `for_eos("span_wagner")`, the
  `span_wagner` module and its analytic kernel `span_wagner_kernel`. A module holding
  Span-Wagner content carries the name; the transport
  correlations' Span-Wagner reducing parameters are private (`transport._CORR_*`). Each backend carries its own R,
  M, ρ_c and tables (`<stem>_nist2026.npz`, from the scripts' `--eos`). `eos="span_wagner"`
  is bit for bit 0.5.0; keep it so until a release says otherwise. Tests that validate
  Span-Wagner reach it through `tests/span_wagner_api.py`; a bench names the backend it runs.
- `tests/`: the validation suite, each test against a reference from outside the code (see
  Tests).
- `bench/`: profiling and validation studies; `scripts/`: table generation;
  `examples/`: the launch demo.

## Tests

A test here checks the package against a reference from outside its own code (a value
NIST IR 8608 prints, CoolProp, a published table, an exact identity, `jax.grad`, finite
differences) where a wrong result would otherwise look fine. A failure that shows itself
in use (an error, a crash, a missing file) gets no test. One guard sits beside them:
`tests/test_public_text.py` holds the public text free of the private working setup's
names, because a published leak cannot be taken back.

- **`make test`**: the suite on 8 xdist workers (`--dist loadfile`), about 26 s on the
  laptop. Needs only the `[test]` extra.
- **`make gate`**: the commit gate, run as `gate <repo path>`: the same run as `make test`.
  `gate` fails a run longer than `GATE_BUDGET_S`, which lives in the `Makefile`.

CI runs `pytest tests/ -v`.

## Critical rules

**The public API surface is the contract: keep it stable and single-sourced.**
Consumers import this package, so a breaking change to a public name or signature
ripples into every simulation. Canonical values (the Span-Wagner coefficients, the
critical point) live in exactly one place and every other site derives from them.

**Every property derives from the Helmholtz free energy A(T, ρ).** Pressure, heat
capacities, speed of sound, and the rest come from the hand-coded analytic
α-derivatives. On nist2026 they are validated against `jax.grad` by
`tests/test_nist2026_derivs.py` (relative error below 1e-9 for the residual part and 1e-10
for the ideal part); if they fail that bar, the derivatives are wrong, not the tolerance.

**CoolProp is a test-only dependency: never reachable from an import of the package.**
It generates validation data and the offline tables; shipped code never calls it.

**Preserve differentiability, JIT, and vmap.** The code is pure JAX: no Python-side
data-dependent branching that breaks tracing, no NumPy on the traced path. Use `jax.lax`
control flow for anything that depends on a traced value, so `jit`, `vmap`, `grad`, and
`custom_jvp` keep working end-to-end.

**Accuracy is validated against outside references.** The test suite is the acceptance
bar; a change is done when it is green. nist2026 meets the values NIST IR 8608 prints
and CoolProp's Span-Wagner within twice the stated uncertainty
(`tests/test_nist2026_vs_coolprop.py`); Span-Wagner meets CoolProp
(`tests/test_accuracy_vs_coolprop.py`); both saturation tables meet a Maxwell solve on
their surface up to T_c (`tests/test_saturation.py`); the inversions' and the fast path's
gradients meet finite differences and the implicit-function oracle
(`tests/test_inversion_gradients.py`, `tests/test_fastpath_validation.py`); the reference
gases meet the stored CoolProp values of `tests/data/reference_gas_values.csv`, the
transport check values Lemmon and Jacobsen (2004) print and, for helium, its ab initio
table (`tests/test_reference_gases.py`, whose docstring holds the bars and their sources).
