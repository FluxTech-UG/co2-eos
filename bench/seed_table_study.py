"""Does a precomputed T0(rho,u) table seed collapse the near-critical iters?

Compares affine seed vs a coarse bilinear (rho,u) table seed, reporting fixed-
iteration round-trip error in the regime (incl. the near-critical corner).
"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from co2_eos.span_wagner import TC, RHOC, R, T_TRIPLE
from co2_eos import core

def u_of(T, rho):
    return jax.vmap(lambda t, r: core._u_and_cv(t, r)[0])(T, rho)

def make_points(Tlo, Thi, rlo, rhi, n=6000, seed=3):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.uniform(Tlo, Thi, n))
    rho = jnp.asarray(rng.uniform(rlo, rhi, n))
    return T, rho, u_of(T, rho)

# ── Build a coarse T0(rho,u) table over the regime (+margin) by offline solve ─
RHO_LO, RHO_HI = 150.0, 500.0
T_LO, T_HI = 295.0, 340.0
NRHO, NU = 48, 96

def build_table():
    rho_grid = np.linspace(RHO_LO, RHO_HI, NRHO)
    # u-range: span the regime; compute u at (rho, T_LO/T_HI) corners
    Tg = jnp.asarray([T_LO, T_HI])
    u_lo = float(np.min([float(core._u_and_cv(jnp.float64(T_LO), jnp.float64(r))[0]) for r in rho_grid]))
    u_hi = float(np.max([float(core._u_and_cv(jnp.float64(T_HI), jnp.float64(r))[0]) for r in rho_grid]))
    u_grid = np.linspace(u_lo, u_hi, NU)
    # For each (rho, u) node solve T offline with a robust 40-iter Newton.
    T0 = np.zeros((NRHO, NU))
    for i, r in enumerate(rho_grid):
        for j, uu in enumerate(u_grid):
            T = 313.0
            for _ in range(60):
                fu, cv = core._u_and_cv(jnp.float64(T), jnp.float64(r))
                T = float(np.clip(T - (float(fu) - uu) / float(cv), T_LO - 10, T_HI + 10))
            T0[i, j] = T
    return jnp.asarray(rho_grid), jnp.asarray(u_grid), jnp.asarray(T0)

rho_grid, u_grid, T0_tab = build_table()

def table_seed(rho, u):
    # bilinear interp with clamped indices
    ri = jnp.clip(jnp.searchsorted(rho_grid, rho) - 1, 0, NRHO - 2)
    uj = jnp.clip(jnp.searchsorted(u_grid, u) - 1, 0, NU - 2)
    r0, r1 = rho_grid[ri], rho_grid[ri + 1]
    u0, u1 = u_grid[uj], u_grid[uj + 1]
    fr = jnp.clip((rho - r0) / (r1 - r0), 0.0, 1.0)
    fu = jnp.clip((u - u0) / (u1 - u0), 0.0, 1.0)
    c00 = T0_tab[ri, uj]; c01 = T0_tab[ri, uj + 1]
    c10 = T0_tab[ri + 1, uj]; c11 = T0_tab[ri + 1, uj + 1]
    return ((c00 * (1 - fr) + c10 * fr) * (1 - fu)
            + (c01 * (1 - fr) + c11 * fr) * fu)

# affine seed (regime fit)
_Tr, _rr, _ur = make_points(300, 333, 170, 470, n=3000, seed=7)
_b, _a = np.polyfit(np.asarray(_ur), np.asarray(_Tr), 1)
def affine_seed(rho, u):
    return jnp.clip(_a + _b * u, T_TRIPLE, 800.0)

def solve_fixed(rho, u, seed_fn, n):
    def one(rho, u):
        T = seed_fn(rho, u)
        def step(_, T):
            fu, cv = core._u_and_cv(T, rho)
            cv = jnp.where(jnp.abs(cv) > 1e-30, cv, 1e-30)
            return jnp.clip(T - (fu - u) / cv, T_TRIPLE, 800.0)
        return jax.lax.fori_loop(0, n, step, T)
    return jax.jit(jax.vmap(one))(rho, u)

def main():
    print(f"table: {NRHO}x{NU} over rho[{RHO_LO},{RHO_HI}] T[{T_LO},{T_HI}]")
    # seed quality
    T, rho, u = make_points(300, 333, 170, 470)
    for name, sfn in [("affine", affine_seed), ("table", table_seed)]:
        s = sfn(rho, u)
        err = np.asarray(jnp.abs(s - T))
        print(f"  {name} seed |ΔT|: p50={np.percentile(err,50):.3f} "
              f"p99={np.percentile(err,99):.3f} max={err.max():.3f} K")
    print("\nregime (T 300-333, rho 170-470) round-trip max |ΔT| K:")
    for name, sfn in [("affine", affine_seed), ("table", table_seed)]:
        row = [f"{name:>7}"]
        for n in (2, 3, 4, 5, 6, 8, 10, 12):
            err = np.asarray(jnp.abs(solve_fixed(rho, u, sfn, n) - T))
            row.append(f"N={n}:{err.max():.0e}")
        print("  " + " ".join(row))

if __name__ == "__main__":
    main()
