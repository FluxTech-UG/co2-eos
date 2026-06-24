"""Validate the redesigned core: round-trip, properties, seed/iteration count."""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from co2_eos import core, span_wagner as sw
from co2_eos.span_wagner import TC, RHOC, R, T_TRIPLE

def pts(Tlo, Thi, rlo, rhi, n, seed):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.uniform(Tlo, Thi, n)),
            jnp.asarray(rng.uniform(rlo, rhi, n)))

solveT = jax.jit(jax.vmap(core._temperature_from_Du))
u_of = jax.jit(jax.vmap(lambda T, r: core._u_and_cv(T, r)[0]))

def roundtrip(label, Tlo, Thi, rlo, rhi, n=8000, seed=11):
    T, rho = pts(Tlo, Thi, rlo, rhi, n, seed)
    u = u_of(T, rho)
    phase = jnp.full(n, 2, dtype=jnp.int32)
    Tsol = solveT(rho, u, phase)
    err = np.asarray(jnp.abs(Tsol - T))
    print(f"  {label:42} max|ΔT|={err.max():.2e} K  p99={np.percentile(err,99):.2e}  "
          f"mean={err.mean():.2e}")
    return err.max()

def seed_quality(Tlo, Thi, rlo, rhi, n=8000, seed=5):
    T, rho = pts(Tlo, Thi, rlo, rhi, n, seed)
    u = u_of(T, rho)
    T0 = jax.jit(jax.vmap(core._seed_T))(rho, u)
    err = np.asarray(jnp.abs(T0 - T))
    print(f"  seed |ΔT|: p50={np.percentile(err,50):.3f} p99={np.percentile(err,99):.3f} "
          f"max={err.max():.3f} K")

def properties():
    # Compare _thermo vs the original autodiff bundle (ground truth).
    T, rho = pts(300, 333, 170, 470, 2000, 9)
    new = jax.jit(jax.vmap(core._thermo))(T, rho)[0]   # (P,cv,cp,w,h,u,s,g)
    old = jax.jit(jax.vmap(sw._scalar_all_properties_shared))(T, rho)
    names = ["P", "cv", "cp", "w", "h", "u", "s", "g"]
    print("\nProperty kernel vs original autodiff bundle (regime):")
    worst = 0.0
    for nm, a, b in zip(names, new, old):
        a = np.asarray(a); b = np.asarray(b)
        rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-30)
        worst = max(worst, rel.max())
        print(f"  {nm:>3} max rel err {rel.max():.2e}")
    print(f"  worst {worst:.2e}  -> {'PASS' if worst < 1e-10 else 'CHECK'}")

def main():
    print(f"N_NEWTON = {core._NEWTON_ITERS}, have_seed={core._HAVE_SEED}\n")
    print("Seed quality (regime):")
    seed_quality(300, 333, 170, 470)
    print("\nRound-trip T->u->T:")
    m1 = roundtrip("regime (T 300-333, rho 170-470)", 300, 333, 170, 470)
    roundtrip("near-crit corner (T 300-310, rho 400-470)", 300, 310, 400, 470)
    roundtrip("supercritical (T 305-345, rho 60-700)", 305, 345, 60, 700)
    roundtrip("subcritical liquid (T 260-300, rho 900-1040)", 260, 300, 900, 1040)
    roundtrip("subcritical vapor (T 260-303, rho 60-120)", 260, 303, 60, 120)
    properties()
    print(f"\nRound-trip requirement (<= 1e-8 K): "
          f"{'PASS' if m1 <= 1e-8 else 'FAIL'} (regime max {m1:.2e})")

if __name__ == "__main__":
    main()
