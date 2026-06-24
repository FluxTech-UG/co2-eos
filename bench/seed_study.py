"""Pick the (ρ, u) → T Newton seed: compare seed error and iteration counts.

Compares three seeds across the regime and a wider band:
  crude   : u / (3.5 R)                       (current baseline)
  const   : fixed 313 K (regime midpoint)
  ideal   : invert ideal-gas u_id(T) = u      (ρ-independent, has IIR offset)

For each, reports seed |T0 - T_true| and the analytic-Newton iterations to
reach |ΔT| < 1e-10 K.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos.span_wagner import TC, RHOC, R, T_TRIPLE
from co2_eos import helmholtz as hz
from co2_eos import core


def make_points(Tlo, Thi, rlo, rhi, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(Tlo, Thi, n)
    rho = rng.uniform(rlo, rhi, n)
    Tj = jnp.asarray(T); rhoj = jnp.asarray(rho)
    u = jax.vmap(lambda t, r: core._u_and_cv(t, r)[0])(Tj, rhoj)
    return Tj, rhoj, u


def seed_crude(rho, u):
    return jnp.clip(u / (3.5 * R), T_TRIPLE, 800.0)


def seed_const(rho, u):
    return jnp.full_like(u, 313.0)


def _ideal_u(T):
    a0_t, _ = hz.ideal_tau_only(TC / T)
    return R * TC * a0_t


_dideal = jax.grad(_ideal_u)


def seed_ideal(rho, u):
    """Invert ideal-gas u_id(T)=u with a few fixed Newton steps from 300 K."""
    def step(_, T):
        f = _ideal_u(T) - u
        fp = _dideal(T)
        return jnp.clip(T - f / fp, T_TRIPLE, 800.0)
    T0 = jnp.full_like(u, 300.0)
    return jax.lax.fori_loop(0, 5, step, T0)


seed_ideal_v = jax.jit(jax.vmap(seed_ideal))


def newton_iters_to_conv(rho, u, T0):
    """Count analytic-Newton iterations until |step| < 1e-10 K (cap 40)."""
    def cond(s):
        _, i, conv = s
        return jnp.logical_and(i < 40, jnp.logical_not(conv))
    def body(s):
        T, i, _ = s
        fu, cv = core._u_and_cv(T, rho)
        step = (fu - u) / cv
        Tn = jnp.clip(T - step, T_TRIPLE, 800.0)
        return (Tn, i + 1, jnp.abs(step) < 1e-10)
    _, i, _ = jax.lax.while_loop(cond, body, (T0, jnp.int32(0), jnp.bool_(False)))
    return i


iters_batch = jax.jit(jax.vmap(newton_iters_to_conv))


def report(name, T_true, rho, u, T0):
    err = np.asarray(jnp.abs(T0 - T_true))
    iters = np.asarray(iters_batch(rho, u, T0))
    pct = lambda a, p: np.percentile(a, p)
    print(f"{name:>7} | seed|ΔT| K  p50={pct(err,50):7.2f} p99={pct(err,99):8.2f} "
          f"max={err.max():8.2f} | iters p50={int(pct(iters,50))} "
          f"p99={int(pct(iters,99))} max={int(iters.max())} mean={iters.mean():.2f}")


def main():
    for label, (Tlo, Thi, rlo, rhi) in {
        "regime  (T 300-333, rho 170-470)": (300, 333, 170, 470),
        "wide    (T 240-360, rho 100-1000)": (240, 360, 100, 1000),
    }.items():
        print(f"\n=== {label} ===")
        T, rho, u = make_points(Tlo, Thi, rlo, rhi)
        report("crude", T, rho, u, seed_crude(rho, u))
        report("const", T, rho, u, seed_const(rho, u))
        report("ideal", T, rho, u, seed_ideal_v(rho, u))


if __name__ == "__main__":
    main()
