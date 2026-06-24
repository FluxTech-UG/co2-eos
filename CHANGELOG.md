# Changelog

All notable changes to co2-eos are documented here.

## [0.2.0] — 2026-06-24

Performance redesign of the hot path: hand-coded analytic α-derivatives, a fused
`(ρ, u) → {all properties}` entry point, and a table-seeded fixed-iteration
inversion. **Accuracy and gradients are unchanged** — the redesign is validated
to match the previous autodiff EOS and CoolProp to the same tight tolerances.
The public surface gains the fused hot-path functions; the scalar state
functions keep their signatures (see "Migrating" below).

### Why

In an EOS-bound finite-volume simulation (the conserved state is `(ρ, ρu, E)`),
recovering primitives every RHS evaluation was ~95 % of the cost, and inside
that the `T(ρ, u)` inversion alone dominated. Profiling on CPU put
`temperature_from_Du` at **92 %** of the per-step EOS cost. Two things drove it:
every Newton iteration computed Cv by *nested* `jax.grad` of `α`, and the
`while_loop` Newton ran the whole `vmap` batch to the batch-maximum iteration
count (a long tail from a crude initial guess).

### Added

- **`properties_from_rho_u(rho, u, phase_hint=SUPERCRITICAL)`** — the new primary
  hot-path entry point. Batched / array-native (pass 1-D arrays, get a dict of
  1-D arrays). Solves T once from `u(T, ρ) = u` and reuses the α-derivatives to
  return `{temperature, density, pressure, cv, cp, speed_of_sound, enthalpy,
  internal_energy, entropy, gibbs_energy, viscosity, thermal_conductivity}`.
  `density` and `internal_energy` echo the inputs exactly.
- **`temperature_from_rho_u(rho, u, phase_hint=SUPERCRITICAL)`** — batched lean
  inversion returning T only. Correct jvp/vjp via the implicit function theorem.
- **`co2_eos.helmholtz`** — the analytic Helmholtz core: `residual_derivs`,
  `ideal_derivs` (value + all first/second τ,δ derivatives in one fused pass),
  plus the δ-precomputed τ-derivative path used by the Newton inner loop.
- **`co2_eos.core`** — the fused property kernel and the table-seeded inversion.
- **`co2_eos/data/seed_table.npz`** + `scripts/generate_seed_table.py` — a
  precomputed `(ρ, u) → T₀` bilinear seed table. It is a convergence accelerator
  only: the Newton polish sets accuracy and the IFT JVP sets gradients, so the
  table carries no accuracy or differentiability risk.
- `bench/` — profiling and before/after benchmark scripts (CPU and the V100S
  `gpu_bench.sh` harness for flux-compute).

### Changed

- **Derivatives are now analytic, not autodiff.** `α_δ, α_τ, α_δδ, α_ττ, α_δτ`
  are hand-coded and share each term's value, so the transcendentals are
  evaluated once per term instead of being recomputed by nested `jax.grad`.
  Validated against the autodiff derivatives of `span_wagner.alphar`/`alpha0` to
  < 3e-12 relative across the regime and near-critical stress points.
- **The `(ρ, u) → T` inversion** now uses the table seed + a fixed, unrolled
  analytic-Cv Newton (no `while_loop`), with the δ-invariant envelopes
  precomputed once per solve. Branchless and uniform across a `vmap` batch.
- **`transport.thermal_conductivity`** no longer builds its own `jax.grad`
  chain; it takes the analytic reduced derivatives. The fused `(ρ, u)` path feeds
  it the derivatives already computed at `(T, ρ)`, so the critical-enhancement
  term costs only a reference-temperature δ-derivative pair.
- **State functions are analytic-backed.** `properties`, `state_from_PT`,
  `state_from_Ph`, `state_from_Du`, `density_from_PT` keep their signatures but
  derive properties through the analytic kernel.

### Performance (CPU, Apple M-class, float64, N = 4096 batched)

| stage | before (autodiff) | after (analytic) | speedup |
|---|---:|---:|---:|
| `T(ρ, u)` inversion | 19.8 ms | 4.1 ms | **4.8×** |
| property bundle | 3.4 ms | 1.6 ms | 2.1× |
| full hot path (T + P + μ + k) | 21.7 ms | 6.0 ms | **3.6×** |

On the OVH Tesla V100S the redesign is 1.3–2.2× faster at these (latency-bound)
batch sizes and scales to **7.7× (inversion) / 4.9× (full path)** once
compute-bound (N ≳ 64k, measured to N = 2²⁰). Full tables in `bench/PROFILING.md`.

### Accuracy (unchanged)

- Round-trip `T → (ρ, u) → T` ≤ **1.2e-12 K** across the regime and the
  subcritical liquid/vapor branches (requirement: ≤ 1e-8 K).
- Fused properties match `properties(T, ρ)` at the recovered T to < 1e-9.
- CoolProp agreement is identical to v0.1 (same Span-Wagner polynomial); the
  `tests/test_accuracy_vs_coolprop.py` tolerances are unchanged.
- jvp == vjp == central finite differences for every inversion.

### Kept

- Scalar `properties`, `state_from_PT`, `state_from_Ph`, `state_from_Du`,
  `density_from_PT`; the saturation curve API; `viscosity` / `thermal_conductivity`.
- `co2_eos.span_wagner` (autodiff EOS) and `co2_eos.inversions` (original Newton
  solvers) remain importable — they are the validation ground truth and the
  benchmark "before" baseline.

## [0.1.0]

Initial release: pure-JAX Span-Wagner (1996) EOS with autodiff-derived
properties, transport correlations, saturation table, and phase-aware inversions.
