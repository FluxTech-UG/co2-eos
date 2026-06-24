"""Validate the hand-coded analytic Helmholtz derivatives against jax.grad.

Checks residual_derivs / ideal_derivs against the autodiff derivatives of
span_wagner.alphar / alpha0 over a dense regime grid plus stress points near
the critical point.  Reports max abs / rel error per derivative.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz

# Autodiff references (same definitions inversions.py / span_wagner.py use)
ar = sw.alphar
ar_d = jax.grad(ar, argnums=1)
ar_t = jax.grad(ar, argnums=0)
ar_dd = jax.grad(ar_d, argnums=1)
ar_tt = jax.grad(ar_t, argnums=0)
ar_dt = jax.grad(ar_d, argnums=0)

a0 = sw.alpha0
a0_t = jax.grad(a0, argnums=0)
a0_tt = jax.grad(a0_t, argnums=0)


def grid():
    """(tau, delta) covering the near-critical regime and beyond."""
    # Regime: T in [300,333], rho in [170,470]; plus wider for safety.
    T = np.linspace(290.0, 360.0, 41)
    rho = np.linspace(120.0, 520.0, 41)
    pts = [(sw.TC / t, r / sw.RHOC) for t in T for r in rho]
    # Stress points very close to critical (delta~1, tau~1) but not exactly.
    for dd in (0.98, 0.995, 0.999, 1.001, 1.005, 1.02):
        for tt in (0.985, 0.998, 1.002, 1.015):
            pts.append((tt, dd))
    return pts


def main():
    pts = grid()
    refs = {"ar": [], "ar_d": [], "ar_t": [], "ar_dd": [], "ar_tt": [], "ar_dt": [],
            "a0": [], "a0_t": [], "a0_tt": []}
    ana = {k: [] for k in refs}
    for tau, delta in pts:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        refs["ar"].append(float(ar(tau, delta)))
        refs["ar_d"].append(float(ar_d(tau, delta)))
        refs["ar_t"].append(float(ar_t(tau, delta)))
        refs["ar_dd"].append(float(ar_dd(tau, delta)))
        refs["ar_tt"].append(float(ar_tt(tau, delta)))
        refs["ar_dt"].append(float(ar_dt(tau, delta)))
        refs["a0"].append(float(a0(tau, delta)))
        refs["a0_t"].append(float(a0_t(tau, delta)))
        refs["a0_tt"].append(float(a0_tt(tau, delta)))

        rar, rar_d, rar_t, rar_dd, rar_tt, rar_dt = hz.residual_derivs(tau, delta)
        ra0, ra0_t, ra0_tt = hz.ideal_derivs(tau, delta)
        ana["ar"].append(float(rar)); ana["ar_d"].append(float(rar_d))
        ana["ar_t"].append(float(rar_t)); ana["ar_dd"].append(float(rar_dd))
        ana["ar_tt"].append(float(rar_tt)); ana["ar_dt"].append(float(rar_dt))
        ana["a0"].append(float(ra0)); ana["a0_t"].append(float(ra0_t))
        ana["a0_tt"].append(float(ra0_tt))

    print(f"{len(pts)} points  (regime grid + near-critical stress points)\n")
    print(f"{'quantity':>8} | {'max abs err':>12} | {'max rel err':>12}")
    print("-" * 40)
    worst_rel = 0.0
    for k in refs:
        r = np.array(refs[k]); a = np.array(ana[k])
        abs_err = np.abs(r - a)
        rel_err = abs_err / np.maximum(np.abs(r), 1e-12)
        worst_rel = max(worst_rel, rel_err.max())
        print(f"{k:>8} | {abs_err.max():12.3e} | {rel_err.max():12.3e}")
    print(f"\nworst relative error across all derivatives: {worst_rel:.3e}")
    print("PASS" if worst_rel < 1e-9 else "FAIL (>1e-9)")


if __name__ == "__main__":
    main()
