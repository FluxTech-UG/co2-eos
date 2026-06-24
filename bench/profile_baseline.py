"""Baseline profiling of the co2-eos hot path (pre-redesign).

Measures, on whatever device JAX sees (CPU locally, V100S on flux-compute):

  1. Per-call wall time of the building blocks that dominate the consumer's
     RHS evaluation:
        - internal_energy(T, rho)              [1 autodiff: alpha_tau]
        - cv(T, rho)                           [nested autodiff: alpha_tautau]
        - pressure(T, rho)                     [1 autodiff: alpha_delta]
        - all_properties_shared(T, rho)        [5 reduced derivatives]
        - viscosity(T, rho)                    [no autodiff]
        - thermal_conductivity(T, rho)         [5 autodiff derivatives]
        - temperature_from_Du(rho, u, phase)   [Newton, the prime suspect]
        - FULL consumer hot path (to_primitives): T_from_Du + P + mu + k

  2. Newton iteration-count distribution for temperature_from_Du across the
     near-critical regime (this is what runs to the batch-max under vmap).

Run:  python bench/profile_baseline.py [--json OUT.json]
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import transport as tr
from co2_eos.inversions import temperature_from_Du, SUPERCRITICAL

NS = [64, 256, 1024, 4096]
REPEAT = 50  # timed repetitions per measurement


# ── Regime sampling ─────────────────────────────────────────────────────────
# Near-critical CO2: T in [300, 333] K, rho in [170, 470] kg/m3, supercritical.

def regime_grid(n, seed=0):
    """Return (T, rho, u, phase) arrays of length n sampled in the regime."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(300.0, 333.0, size=n)
    rho = rng.uniform(170.0, 470.0, size=n)
    Tj = jnp.asarray(T, dtype=jnp.float64)
    rhoj = jnp.asarray(rho, dtype=jnp.float64)
    u = np.asarray(sw.internal_energy(Tj, rhoj))
    phase = np.full(n, SUPERCRITICAL, dtype=np.int32)
    return Tj, rhoj, jnp.asarray(u, dtype=jnp.float64), jnp.asarray(phase)


# ── Timing helper ───────────────────────────────────────────────────────────

def time_call(fn, *args, repeat=REPEAT):
    """Compile, warm up, then time `fn(*args)` over `repeat` runs.

    Returns mean seconds per call (block_until_ready each iteration).
    """
    out = fn(*args)
    jax.block_until_ready(out)
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn(*args)
        jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / repeat


# ── Build the consumer hot-path closure (mirrors twe/solver/primitives.py) ──

@jax.jit
def consumer_hot_path(rho, u, phase):
    T = temperature_from_Du(rho, u, phase)
    P = sw.pressure(T, rho)
    mu = tr.viscosity(T, rho)
    k = tr.thermal_conductivity(T, rho)
    return T, P, mu, k


# ── Newton iteration-count distribution (vectorized, jitted) ────────────────
# Mirrors _initial_guess_Du + _newton_body_Du / _newton_cond_Du exactly, but
# counts iterations to convergence per element under vmap+jit (fast).

_du_dT_ref = jax.grad(sw._scalar_internal_energy, argnums=0)


def _count_one(rho, u):
    T0 = jnp.clip(u / (3.5 * sw.R), sw.T_TRIPLE, 800.0)

    def cond(state):
        _, i, conv = state
        return jnp.logical_and(i < 50, jnp.logical_not(conv))

    def body(state):
        T, i, conv = state
        f = sw._scalar_internal_energy(T, rho) - u
        fp = _du_dT_ref(T, rho)
        step = f / jnp.where(jnp.abs(fp) > 1e-30, fp, 1e-30)
        Tn = jnp.clip(T - step, sw.T_TRIPLE, 800.0)
        conv_n = jnp.logical_or(
            jnp.abs(f) < 1e-10,
            jnp.abs(step) < 1e-13 * jnp.maximum(jnp.abs(Tn), 1.0))
        return (Tn, i + 1, conv_n)

    _, i, _ = jax.lax.while_loop(cond, body, (T0, jnp.int32(0), jnp.bool_(False)))
    return i


_count_batch = jax.jit(jax.vmap(_count_one))


def newton_iteration_counts(T, rho, u):
    return np.asarray(_count_batch(rho, u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--repeat", type=int, default=REPEAT)
    args = ap.parse_args()

    dev = jax.devices()[0]
    print(f"JAX device: {dev}  (platform={dev.platform})")
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    print()

    results = {"device": str(dev), "platform": dev.platform, "by_N": {}}

    # JIT-wrapped building blocks
    f_u   = sw.internal_energy
    f_cv  = sw.cv
    f_P   = sw.pressure
    f_aps = sw.all_properties_shared
    f_mu  = tr.viscosity
    f_k   = tr.thermal_conductivity

    @jax.jit
    def f_tdu(rho, u, phase):
        return temperature_from_Du(rho, u, phase)

    header = f"{'N':>6} | {'u':>9} {'cv':>9} {'P':>9} {'aps':>9} {'visc':>9} {'cond':>9} {'T_Du':>9} {'FULL':>9}   (us/call)"
    print(header)
    print("-" * len(header))

    for n in NS:
        T, rho, u, phase = regime_grid(n)
        tu   = time_call(f_u, T, rho, repeat=args.repeat)
        tcv  = time_call(f_cv, T, rho, repeat=args.repeat)
        tP   = time_call(f_P, T, rho, repeat=args.repeat)
        taps = time_call(f_aps, T, rho, repeat=args.repeat)
        tmu  = time_call(f_mu, T, rho, repeat=args.repeat)
        tk   = time_call(f_k, T, rho, repeat=args.repeat)
        ttdu = time_call(f_tdu, rho, u, phase, repeat=args.repeat)
        tfull = time_call(consumer_hot_path, rho, u, phase, repeat=args.repeat)
        us = 1e6
        print(f"{n:>6} | {tu*us:>9.1f} {tcv*us:>9.1f} {tP*us:>9.1f} {taps*us:>9.1f} "
              f"{tmu*us:>9.1f} {tk*us:>9.1f} {ttdu*us:>9.1f} {tfull*us:>9.1f}")
        results["by_N"][n] = {
            "internal_energy": tu, "cv": tcv, "pressure": tP,
            "all_properties_shared": taps, "viscosity": tmu,
            "thermal_conductivity": tk, "temperature_from_Du": ttdu,
            "full_hot_path": tfull,
        }

    # Newton iteration distribution on a fixed regime sample
    print("\nNewton iteration-count distribution (temperature_from_Du), N=4096:")
    T, rho, u, _ = regime_grid(4096)
    counts = newton_iteration_counts(T, rho, u)
    pct = lambda p: int(np.percentile(counts, p))
    print(f"  min={counts.min()}  p50={pct(50)}  p90={pct(90)}  "
          f"p99={pct(99)}  max={counts.max()}  mean={counts.mean():.2f}")
    print(f"  histogram (iters: count): "
          + ", ".join(f"{i}:{int((counts==i).sum())}"
                      for i in range(counts.min(), counts.max() + 1)))
    results["newton_iters"] = {
        "min": int(counts.min()), "max": int(counts.max()),
        "mean": float(counts.mean()), "p50": pct(50),
        "p90": pct(90), "p99": pct(99),
        "histogram": {int(i): int((counts == i).sum())
                      for i in range(counts.min(), counts.max() + 1)},
    }

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
