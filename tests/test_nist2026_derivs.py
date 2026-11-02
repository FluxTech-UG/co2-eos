"""The nist2026 analytic Helmholtz derivatives must match jax.grad.

``nist2026.residual_derivs`` / ``ideal_derivs`` and the δ-precomputed
inner-loop pair have to reproduce the autodiff derivatives of
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
_fast = jax.jit(lambda t, d: nb.residual_tau_fast(t, nb.residual_tau_prep(d)))


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


def test_fast_tau_path_matches_full():
    """The δ-precomputed Newton inner-loop pair equals the full bundle."""
    worst = 0.0
    for tau, delta in PTS:
        tau = jnp.float64(tau); delta = jnp.float64(delta)
        _, _, ar_t, _, ar_tt, _ = _res(tau, delta)
        ft, ftt = _fast(tau, delta)
        worst = max(worst, _rel(ft, ar_t), _rel(ftt, ar_tt))
    assert worst < 1e-12, f"fast τ path vs full bundle worst rel err {worst:.2e}"


def test_kernel_is_jittable_vmappable_and_twice_differentiable():
    taus = jnp.linspace(0.9, 1.1, 7)
    deltas = jnp.linspace(0.5, 1.5, 7)
    out = jax.jit(jax.vmap(nb.residual_derivs))(taus, deltas)
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in out)
    # Third δ-derivative of αʳ through the analytic second derivative.
    d3 = jax.grad(lambda d: nb.residual_derivs(1.01, d)[3])(1.0)
    d3_ref = jax.grad(_ar_dd, argnums=1)(1.01, 1.0)
    assert _rel(d3, d3_ref) < 1e-9


def test_steep_gaussians_underflow_cleanly_at_the_widom_crossing():
    """Terms 22 to 24 (β = 2.2e4, 6e4, 1e6) centre below T_c; at FluxTech's
    Widom crossing (314 K, 472 kg/m³) term 24's exponent is about −1140 and
    its exp underflows to exactly 0.  The bundle, its jit and its gradients
    (a third derivative of α through c_p) stay finite there and at the range's
    far corners."""
    states = [(314.0, 472.0), (2000.0, 50.0), (220.0, 1170.0), (304.2282, 468.0)]
    cp_grad = jax.jit(jax.grad(nb._scalar_cp, argnums=(0, 1)))
    res_grad = jax.jit(jax.jacfwd(nb.residual_derivs, argnums=(0, 1)))
    for T, rho in states:
        tau, delta = nb.TC / T, rho / nb.RHOC
        assert all(np.isfinite(float(x)) for x in _res(tau, delta))
        for row in res_grad(tau, delta):
            assert all(np.isfinite(float(x)) for x in row)
        assert all(np.isfinite(float(x)) for x in cp_grad(T, rho))
