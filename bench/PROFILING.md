# EOS hot-path profiling & benchmark report

Profiling that motivated the v0.2 redesign, and the before/after measurements.
Workload: near-critical CO₂ (T ≈ 300–333 K, ρ ≈ 170–470 kg/m³, supercritical),
float64, batched. Scripts in this directory reproduce every number.

- `profile_baseline.py` — per-building-block timing + Newton iteration histogram
- `compare.py` — before (v0.1 autodiff) vs after (v0.2 analytic) at N = 64–4096
- `seed_study.py`, `fixed_iters_study.py`, `seed_table_study.py` — seed/iteration
  design studies
- `validate_analytic.py`, `validate_core.py` — correctness (also covered by the
  test suite)

## 1. Where the time went (baseline, CPU, N = 4096)

`profile_baseline.py`, µs per batched call:

| block | µs/call | note |
|---|---:|---|
| `internal_energy` | 1204 | 1 autodiff deriv (α_τ) |
| `cv` | 1575 | nested autodiff (α_ττ) |
| `pressure` | 2010 | 1 autodiff deriv (α_δ) |
| `all_properties_shared` | 3060 | 5 autodiff reduced derivs |
| `viscosity` | 249 | no autodiff |
| `thermal_conductivity` | 2153 | autodiff critical enhancement |
| **`temperature_from_Du`** | **20098** | **Newton, the prime suspect** |
| full hot path (T+P+μ+k) | 21813 | — |

**`temperature_from_Du` is 92 % of the full hot path.** Two causes:

1. Every Newton iteration computes Cv by *nested* `jax.grad` of `α`.
2. The `while_loop` Newton runs the whole `vmap` batch to the batch-maximum
   iteration count.

### Newton iteration distribution (baseline seed `u/(3.5R)`, N = 4096)

```
min=3  p50=5  p90=6  p99=13  max=16  mean=5.66
```

The median point converges in 5 iterations but the batch pays the **max of 16** —
a long tail caused by the crude seed (the IIR energy-reference offset throws
`u/(3.5R)` off by ~240 K in the regime).

## 2. The redesign

- **Analytic α-derivatives** (`helmholtz.py`) replace the nested `jax.grad`
  chain; each term's value is shared across all five derivatives. Validated to
  < 3e-12 relative vs autodiff.
- **Table-seeded inversion.** A precomputed `(ρ, u) → T₀` bilinear table seeds
  Newton within **≤ 0.042 K** in the regime, vs ~20 K for the best affine seed and
  ~240 K for the baseline. This collapses the iteration count: from the table seed
  a fixed **N = 5** unrolled Newton reaches float64 round-off (round-trip
  ≤ 1.2e-12 K), where the affine seed needed N ≈ 10.
- **Fixed, unrolled Newton** (no `while_loop`) with the δ-invariant envelopes
  precomputed once per solve — branchless, uniform across the batch, and the
  inner step evaluates only τ-dependent transcendentals.
- **Fusion.** `properties_from_rho_u` solves T once and reuses the α-derivatives
  for every thermodynamic property and for the transport critical-enhancement
  term.

Seed/iteration design data (`seed_table_study.py`, regime round-trip max |ΔT|):

| seed | \|ΔT\| seed | N=3 | N=4 | N=5 |
|---|---:|---:|---:|---:|
| affine (best linear) | 20 K | 4e0 | 2e0 | 6e-1 |
| **(ρ,u) table** | **0.14 K** | 3e-9 | 1e-12 | 1e-12 |

## 3. Before / after (CPU, Apple M-class, float64)

`compare.py`, µs per batched call; speedup = before / after:

| N | invert before | invert after | ×  | full before | full after | ×  |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 881   | 213  | 4.1 | 1010  | 362  | 2.8 |
| 256  | 2683  | 748  | 3.6 | 3121  | 1046 | 3.0 |
| 1024 | 13866 | 2829 | 4.9 | 11952 | 3504 | 3.4 |
| 4096 | 19766 | 4130 | 4.8 | 21723 | 6037 | 3.6 |

Property bundle alone (analytic vs autodiff): 2.1–2.5×. The full-path "before"
already uses the v0.2 analytic transport conductivity, so it *understates* the
real speedup (the original autodiff conductivity was slower still).

## 4. V100S GPU (OVH Tesla V100S-PCIE-32GB, via flux-compute)

Reproduce with:

```bash
flux-compute run --cloud flux-ovh --upload . \
    --script bench/gpu_bench.sh --fetch "bench-out:gpu-results"
```

### Spec batch sizes (N = 64–4096), µs/call

| N | invert before | invert after | × | full before | full after | × |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 866  | 610 | 1.4 | 966  | 721 | 1.3 |
| 256  | 772  | 588 | 1.3 | 878  | 732 | 1.2 |
| 1024 | 1009 | 572 | 1.8 | 1134 | 761 | 1.5 |
| 4096 | 1359 | 616 | 2.2 | 1529 | 774 | 2.0 |

**At the consumer's batch sizes the V100S is latency-bound:** wall time barely
scales with N (a ~600–800 µs floor of kernel-launch / dispatch overhead), so the
GPU is starved and the analytic speedup is a modest 1.3–2.2×. The redesign still
wins — and the fixed-iteration branchless inversion is exactly the right shape
for the launch-bound regime — but the GPU has plenty of unused throughput here.

### Scaling to GPU saturation, µs/call

| N | invert before | invert after | × | full before | full after | × |
|---:|---:|---:|---:|---:|---:|---:|
| 64       | 868    | 632   | 1.4 | 926    | 689   | 1.3 |
| 4096     | 1347   | 625   | 2.2 | 1525   | 829   | 1.8 |
| 16384    | 2922   | 818   | 3.6 | 3101   | 1258  | 2.5 |
| 65536    | 8138   | 1803  | 4.5 | 9141   | 2761  | 3.3 |
| 262144   | 35995  | 5123  | **7.0** | 41155  | 9313  | 4.4 |
| 1048576  | 135962 | 17732 | **7.7** | 164579 | 33780 | **4.9** |

Once compute-bound (N ≳ 64k) the analytic redesign pulls ahead hard: **7.7× on
the inversion and 4.9× on the full hot path** at N = 2²⁰ — larger than the CPU
speedup, because the GPU executes the analytic kernel's arithmetic in parallel
while the autodiff baseline pays for its nested-grad graph and the
`while_loop`-to-batch-max iteration tail.

**Takeaways**

- Below the GPU's saturation point the EOS call is launch-bound, not
  compute-bound; the redesign still helps (1.3–2.2×) but the bigger lever for a
  small-batch RHS is fewer/larger launches (fuse more work per call — which
  `properties_from_rho_u` does — or run a larger grid).
- The analytic + fixed-iteration design is the right one for both regimes: it
  removes the nested-grad graph and the data-dependent iteration count, so it
  wins at small N and scales to ~5–8× at large N.
- Absolute V100S inversion time at N = 4096 is ~0.6 ms vs ~4 ms on CPU.

## 5. Accuracy & gradients (unchanged)

- Round-trip `T → (ρ, u) → T` ≤ 1.2e-12 K across regime / supercritical /
  subcritical liquid / subcritical vapor (requirement ≤ 1e-8 K).
- Fused properties match `properties(T, ρ)` to < 1e-9.
- Analytic derivatives match `jax.grad` to < 3e-12.
- jvp == vjp == central finite differences for every inversion.
- CoolProp agreement unchanged from v0.1 (`tests/test_accuracy_vs_coolprop.py`).
