"""Round-trip T error vs FIXED Newton iteration count, for candidate seeds.

Drives the choice of (seed, N_iters) for the unrolled inversion: we need
max round-trip |ΔT| <= 1e-8 K across the regime with the smallest N.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos.span_wagner import TC, RHOC, R, T_TRIPLE
from co2_eos import core


def make_points(Tlo, Thi, rlo, rhi, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    T = rng.uniform(Tlo, Thi, n)
    rho = rng.uniform(rlo, rhi, n)
    Tj = jnp.asarray(T); rhoj = jnp.asarray(rho)
    u = jax.vmap(lambda t, r: core._u_and_cv(t, r)[0])(Tj, rhoj)
    return Tj, rhoj, u


def solve_fixed(rho, u, T0_fn, niters):
    def one(rho, u):
        T = T0_fn(rho, u)
        def step(_, T):
            fu, cv = core._u_and_cv(T, rho)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            return jnp.clip(T - (fu - u) / cv, T_TRIPLE, 800.0)
        return jax.lax.fori_loop(0, niters, step, T)
    return jax.jit(jax.vmap(one))(rho, u)


SEEDS = {
    "const313": lambda rho, u: jnp.full_like(u, 313.0),
    # affine in u, calibrated to the regime (fit T ~ a + b*u on regime sample)
    "affine":   None,   # filled in main once we have a fit
}


def main():
    # Fit an affine seed T0 = a + b*u on a regime sample.
    Tr, rr, ur = make_points(300, 333, 170, 470, n=3000, seed=7)
    ur_np = np.asarray(ur); Tr_np = np.asarray(Tr)
    b, a = np.polyfit(ur_np, Tr_np, 1)
    print(f"affine seed fit: T0 = {a:.6e} + {b:.6e} * u")
    SEEDS["affine"] = lambda rho, u: jnp.clip(a + b * u, T_TRIPLE, 800.0)

    for label, (Tlo, Thi, rlo, rhi) in {
        "regime (T 300-333, rho 170-470)": (300, 333, 170, 470),
        "wide   (T 250-360, rho 120-900)": (250, 360, 120, 900),
    }.items():
        print(f"\n=== {label} ===")
        T, rho, u = make_points(Tlo, Thi, rlo, rhi)
        for sname, sfn in SEEDS.items():
            row = [f"{sname:>9}"]
            for n in (3, 4, 5, 6, 8):
                Tsol = solve_fixed(rho, u, sfn, n)
                err = np.asarray(jnp.abs(Tsol - T))
                row.append(f"N={n}:{err.max():.1e}")
            print("  " + "  ".join(row))


if __name__ == "__main__":
    main()
