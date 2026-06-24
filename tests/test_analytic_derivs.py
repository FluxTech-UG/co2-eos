"""The hand-coded analytic Helmholtz derivatives must match jax.grad.

``helmholtz.residual_derivs`` / ``ideal_derivs`` replace the jax.grad chain in
the hot path; they have to reproduce the autodiff derivatives of
``span_wagner.alphar`` / ``alpha0`` across the regime and the near-critical
stress region to ~1e-10.  The δ-precomputed fast path used by the Newton inner
loop must agree with the full bundle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from co2_eos import span_wagner as sw
from co2_eos import helmholtz as hz

# Autodiff references
_ar_d = jax.grad(sw.alphar, argnums=1)
_ar_t = jax.grad(sw.alphar, argnums=0)
_ar_dd = jax.grad(_ar_d, argnums=1)
_ar_tt = jax.grad(_ar_t, argnums=0)
_ar_dt = jax.grad(_ar_d, argnums=0)
_a0_t = jax.grad(sw.alpha0, argnums=0)
_a0_tt = jax.grad(_a0_t, argnums=0)


def _grid():
    T = np.linspace(290.0, 360.0, 25)
    rho = np.linspace(120.0, 520.0, 25)
    pts = [(sw.TC / t, r / sw.RHOC) for t in T for r in rho]
    # near-critical stress points (δ≈1, τ≈1) but not exactly critical
    for dd in (0.98, 0.995, 1.005, 1.02):
        for tt in (0.985, 0.998, 1.002, 1.015):
            pts.append((tt, dd))
    return pts


PTS = _grid()


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-10)


def test_residual_derivs_match_autodiff():
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt = hz.residual_derivs(tau, delta)
        worst = max(
            worst,
            _rel(ar, sw.alphar(tau, delta)),
            _rel(ar_d, _ar_d(tau, delta)),
            _rel(ar_t, _ar_t(tau, delta)),
            _rel(ar_dd, _ar_dd(tau, delta)),
            _rel(ar_tt, _ar_tt(tau, delta)),
            _rel(ar_dt, _ar_dt(tau, delta)),
        )
    assert worst < 1e-9, f"residual analytic vs autodiff worst rel err {worst:.2e}"


def test_ideal_derivs_match_autodiff():
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        a0, a0_t, a0_tt = hz.ideal_derivs(tau, delta)
        worst = max(
            worst,
            _rel(a0, sw.alpha0(tau, delta)),
            _rel(a0_t, _a0_t(tau, delta)),
            _rel(a0_tt, _a0_tt(tau, delta)),
        )
    assert worst < 1e-10, f"ideal analytic vs autodiff worst rel err {worst:.2e}"


def test_fast_tau_path_matches_full():
    """The δ-precomputed Newton inner-loop path equals the full bundle."""
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        dstate = hz.residual_tau_prep(delta)
        ar_t_fast, ar_tt_fast = hz.residual_tau_fast(tau, dstate)
        _, _, ar_t, _, ar_tt, _ = hz.residual_derivs(tau, delta)
        worst = max(worst, _rel(ar_t_fast, ar_t), _rel(ar_tt_fast, ar_tt))
    assert worst < 1e-12, f"fast τ-path vs full worst rel err {worst:.2e}"
