"""The pressure derivatives in the full-state dict, and the batched (ρ, T) call.

``dP_dT_rho`` = (∂P/∂T)_ρ and ``dP_drho_T`` = (∂P/∂ρ)_T come from the analytic
α-derivatives in ``core._thermo``.  They must equal ``jax.grad`` of the
independent autodiff pressure in ``span_wagner``, satisfy the two exact
thermodynamic identities against the bundle's own cp, cv and speed of sound,
and survive jit, vmap and grad.  ``properties_from_rho_T`` must reproduce
``vmap(properties)``.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as co2
from co2_eos import span_wagner as sw

# (P [Pa], T [K], phase hint, label): near-critical, supercritical dense, liquid.
STATES = [
    (7.4e6, 305.0, co2.SUPERCRITICAL, "near_critical"),
    (20.0e6, 330.0, co2.SUPERCRITICAL, "supercritical_dense"),
    (10.0e6, 280.0, co2.LIQUID, "liquid"),
]
IDS = [s[3] for s in STATES]

_dP_dT = jax.grad(sw._scalar_pressure, argnums=0)
_dP_drho = jax.grad(sw._scalar_pressure, argnums=1)


def _rel(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


def _T_rho(P, T, hint):
    rho = co2.density_from_PT(P, T, hint)
    assert np.isfinite(float(rho)), (P, T, hint)
    return jnp.float64(T), rho


@pytest.mark.parametrize("P,T,hint,label", STATES, ids=IDS)
def test_derivatives_match_autodiff_pressure(P, T, hint, label):
    T, rho = _T_rho(P, T, hint)
    s = co2.properties(T, rho)
    assert _rel(s["dP_dT_rho"], _dP_dT(T, rho)) < 1e-9
    assert _rel(s["dP_drho_T"], _dP_drho(T, rho)) < 1e-9


@pytest.mark.parametrize("P,T,hint,label", STATES, ids=IDS)
def test_thermodynamic_identities(P, T, hint, label):
    T, rho = _T_rho(P, T, hint)
    s = co2.properties(T, rho)
    pT, prho = s["dP_dT_rho"], s["dP_drho_T"]
    # c_p − c_v = T (∂P/∂T)_ρ² / (ρ² (∂P/∂ρ)_T)
    assert _rel(T * pT ** 2 / (rho ** 2 * prho), s["cp"] - s["cv"]) < 1e-8
    # a² = (∂P/∂ρ)_T · c_p / c_v
    assert _rel(prho * s["cp"] / s["cv"], s["speed_of_sound"] ** 2) < 1e-8


def _batch():
    T = jnp.array([s[1] for s in STATES])
    rho = jnp.stack([co2.density_from_PT(P, t, h) for P, t, h, _ in STATES])
    return rho, T


def test_rho_T_batched_matches_vmap_scalar():
    rho, T = _batch()
    batched = co2.properties_from_rho_T(rho, T)
    ref = jax.vmap(co2.properties)(T, rho)
    assert set(batched) == set(ref)
    for k in ref:
        np.testing.assert_allclose(np.asarray(batched[k]), np.asarray(ref[k]),
                                   rtol=1e-14, atol=0.0, err_msg=k)


def test_rho_T_batched_rejects_bad_shapes():
    with pytest.raises(ValueError):
        co2.properties_from_rho_T(jnp.array([400.0, 500.0]), jnp.array([305.0]))
    with pytest.raises(ValueError):
        co2.properties_from_rho_T(jnp.float64(400.0), jnp.float64(305.0))


def test_every_full_state_dict_carries_the_derivatives():
    P, T, hint, _ = STATES[1]
    rho = co2.density_from_PT(P, T, hint)
    u = co2.properties(T, rho)["internal_energy"]
    h = co2.properties(T, rho)["enthalpy"]
    dicts = [
        co2.properties(T, rho),
        co2.state_from_PT(P, T, hint),
        co2.state_from_Ph(P, h, hint),
        co2.state_from_Du(rho, u),
        jax.tree_util.tree_map(lambda x: x[0],
                               co2.properties_from_rho_u(rho[None], u[None])),
        jax.tree_util.tree_map(lambda x: x[0],
                               co2.properties_from_rho_T(rho[None], jnp.array([T]))),
    ]
    ref = dicts[0]
    for d in dicts:
        for k in ("dP_dT_rho", "dP_drho_T"):
            assert _rel(d[k], ref[k]) < 1e-9, k


def test_derivatives_survive_jit_vmap_and_grad():
    rho, T = _batch()
    f = jax.jit(jax.vmap(lambda t, r: co2.properties(t, r)["dP_dT_rho"]))
    np.testing.assert_allclose(
        np.asarray(f(T, rho)),
        np.asarray(co2.properties_from_rho_T(rho, T)["dP_dT_rho"]),
        rtol=1e-14, atol=0.0)

    def scalar_obj(rho, T):
        s = co2.properties_from_rho_T(rho, T)
        return jnp.sum(s["dP_dT_rho"] * s["dP_drho_T"])

    g_rho, g_T = jax.grad(scalar_obj, argnums=(0, 1))(rho, T)
    assert np.all(np.isfinite(np.asarray(g_rho)))
    assert np.all(np.isfinite(np.asarray(g_T)))
    # Mixed partial is symmetric: ∂/∂ρ (∂P/∂T) = ∂/∂T (∂P/∂ρ).
    for t, r in zip(T, rho):
        a = jax.grad(lambda r_: co2.properties(t, r_)["dP_dT_rho"])(r)
        b = jax.grad(lambda t_: co2.properties(t_, r)["dP_drho_T"])(t)
        assert _rel(a, b) < 1e-8
