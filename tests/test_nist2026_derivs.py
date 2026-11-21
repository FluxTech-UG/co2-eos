"""The nist2026 analytic Helmholtz derivatives must match jax.grad.

``nist2026.residual_derivs`` / ``ideal_derivs`` (and ``ideal_tau_only``) have to reproduce the autodiff derivatives of
``nist2026.alphar`` / ``alpha0`` at the repo's bars (1e-9 relative for the
residual part, 1e-10 for the ideal part), over the regime, the near-critical
stress region (both sides of τ = 1, δ within 2 % of 1) and the corners of the
single-phase range (dense liquid near the triple point, dilute hot gas), so
the three Gaussian terms with β up to 10⁶ (terms 22 to 24, centred at
τ ≈ 1.0024 to 1.0131) are exercised on their peaks and their flanks.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from co2_eos import nist2026 as nb

_ar_d = jax.grad(nb.alphar, argnums=1)
_ar_t = jax.grad(nb.alphar, argnums=0)
_ar_dd = jax.grad(_ar_d, argnums=1)
_ar_tt = jax.grad(_ar_t, argnums=0)
_ar_dt = jax.grad(_ar_d, argnums=0)
_a0_t = jax.grad(nb.alpha0, argnums=0)
_a0_tt = jax.grad(_a0_t, argnums=0)

_ref_res = jax.jit(lambda t, d: (nb.alphar(t, d), _ar_d(t, d), _ar_t(t, d),
                                 _ar_dd(t, d), _ar_tt(t, d), _ar_dt(t, d)))
_ref_ideal = jax.jit(lambda t, d: (nb.alpha0(t, d), _a0_t(t, d), _a0_tt(t, d)))
_res = jax.jit(nb.residual_derivs)
_ideal = jax.jit(nb.ideal_derivs)
_ideal_t = jax.jit(nb.ideal_tau_only)


def _grid():
    T = np.linspace(290.0, 360.0, 15)
    rho = np.linspace(120.0, 520.0, 15)
    pts = [(nb.TC / t, r / nb.RHOC) for t in T for r in rho]
    for dd in (0.98, 0.995, 0.999, 1.0, 1.001, 1.005, 1.02):
        for tt in (0.985, 0.998, 0.9998, 1.0, 1.0002, 1.0024, 1.005, 1.0131):
            pts.append((tt, dd))
    for T, r in ((220.0, 1170.0), (250.0, 1080.0), (280.0, 900.0),
                 (400.0, 50.0), (800.0, 5.0), (1500.0, 300.0), (300.0, 0.5)):
        pts.append((nb.TC / T, r / nb.RHOC))
    return pts


PTS = _grid()


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-10)


def test_residual_derivs_match_autodiff():
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        got = _res(tau, delta)
        ref = _ref_res(tau, delta)
        worst = max(worst, *(_rel(g, r) for g, r in zip(got, ref)))
    assert worst < 1e-9, f"residual analytic vs autodiff worst rel err {worst:.2e}"


def test_ideal_derivs_match_autodiff():
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        a0, a0_t, a0_tt = _ideal(tau, delta)
        t_only = _ideal_t(tau)
        r0, r0_t, r0_tt = _ref_ideal(tau, delta)
        worst = max(worst,
                    _rel(a0, r0), _rel(a0_t, r0_t), _rel(a0_tt, r0_tt),
                    _rel(t_only[0], a0_t), _rel(t_only[1], a0_tt))
    assert worst < 1e-10, f"ideal analytic vs autodiff worst rel err {worst:.2e}"
