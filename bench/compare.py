"""Before/after benchmark of the EOS hot path: original autodiff vs new analytic.

BEFORE (original, unchanged modules):
  * inversions.temperature_from_Du   — autodiff Cv, while_loop Newton
  * span_wagner.all_properties_shared — 5 autodiff reduced derivatives
  * full hot path = temperature_from_Du + pressure + viscosity + thermal_cond

AFTER (redesigned core):
  * core._temperature_from_Du        — analytic Cv, table seed, fixed Newton
  * core._thermo                     — analytic derivative bundle
  * core._state_from_rho_u           — fused: solve T once, derive everything

Runs at N = 64/256/1024/4096.  Reports µs/call and the after/before speedup.
The full-path BEFORE uses the (already analytic) transport conductivity, so it
*understates* the real speedup — the original autodiff conductivity was slower.
"""
import argparse, json, time
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

from co2_eos import core, span_wagner as sw, transport as tr
from co2_eos import inversions as inv

NS = [64, 256, 1024, 4096]
SUPER = 2


def regime(n, seed=0):
    rng = np.random.default_rng(seed)
    T = jnp.asarray(rng.uniform(300.0, 333.0, n))
    rho = jnp.asarray(rng.uniform(170.0, 470.0, n))
    u = jax.vmap(lambda t, r: core._u_and_cv(t, r)[0])(T, rho)
    phase = jnp.full(n, SUPER, dtype=jnp.int32)
    return T, rho, u, phase


def timed(fn, *args, repeat=50):
    o = fn(*args); jax.block_until_ready(o)
    o = fn(*args); jax.block_until_ready(o)
    t0 = time.perf_counter()
    for _ in range(repeat):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / repeat


# BEFORE
@jax.jit
def before_invert(rho, u, phase):
    return inv.temperature_from_Du(rho, u, phase)

@jax.jit
def before_props(T, rho):
    return sw.all_properties_shared(T, rho)

@jax.jit
def before_full(rho, u, phase):
    T = inv.temperature_from_Du(rho, u, phase)
    P = sw.pressure(T, rho)
    mu = tr.viscosity(T, rho)
    k = tr.thermal_conductivity(T, rho)
    return T, P, mu, k

# AFTER
after_invert = jax.jit(jax.vmap(core._temperature_from_Du))
after_props = jax.jit(jax.vmap(core._thermo))
after_full = jax.jit(jax.vmap(core._state_from_rho_u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--ns", default=None,
                    help="comma-separated batch sizes (overrides default)")
    a = ap.parse_args()
    global NS
    if a.ns:
        NS = [int(x) for x in a.ns.split(",")]
    dev = jax.devices()[0]
    print(f"device={dev} platform={dev.platform} x64={jax.config.jax_enable_x64}\n")
    res = {"device": str(dev), "platform": dev.platform, "by_N": {}}

    print(f"{'N':>6} | {'invert old':>11} {'invert new':>11} {'spd':>5} | "
          f"{'props old':>10} {'props new':>10} {'spd':>5} | "
          f"{'full old':>10} {'full new':>10} {'spd':>5}   (µs/call)")
    print("-" * 110)
    for n in NS:
        T, rho, u, phase = regime(n)
        us = 1e6
        bi = timed(before_invert, rho, u, phase, repeat=a.repeat)
        ai = timed(after_invert, rho, u, phase, repeat=a.repeat)
        bp = timed(before_props, T, rho, repeat=a.repeat)
        ap2 = timed(after_props, T, rho, repeat=a.repeat)
        bf = timed(before_full, rho, u, phase, repeat=a.repeat)
        af = timed(after_full, rho, u, phase, repeat=a.repeat)
        print(f"{n:>6} | {bi*us:>11.1f} {ai*us:>11.1f} {bi/ai:>5.1f} | "
              f"{bp*us:>10.1f} {ap2*us:>10.1f} {bp/ap2:>5.1f} | "
              f"{bf*us:>10.1f} {af*us:>10.1f} {bf/af:>5.1f}")
        res["by_N"][n] = {
            "invert_old": bi, "invert_new": ai, "invert_speedup": bi / ai,
            "props_old": bp, "props_new": ap2, "props_speedup": bp / ap2,
            "full_old": bf, "full_new": af, "full_speedup": bf / af,
        }
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
