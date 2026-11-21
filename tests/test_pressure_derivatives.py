"""The full-state dict satisfies the two exact thermodynamic identities.

c_p − c_v = T (∂P/∂T)_ρ² / (ρ² (∂P/∂ρ)_T) and a² = (∂P/∂ρ)_T · c_p / c_v, with
every quantity from the bundle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tests.span_wagner_api import co2   # the Span-Wagner path, eos bound

# (P [Pa], T [K], phase hint, label): near-critical, supercritical dense, liquid.
STATES = [
    (7.4e6, 305.0, co2.SUPERCRITICAL, "near_critical"),
    (20.0e6, 330.0, co2.SUPERCRITICAL, "supercritical_dense"),
    (10.0e6, 280.0, co2.LIQUID, "liquid"),
]
IDS = [s[3] for s in STATES]

def _rel(a, b):
    return abs(float(a) - float(b)) / abs(float(b))


def _T_rho(P, T, hint):
    rho = co2.density_from_PT(P, T, hint)
    assert np.isfinite(float(rho)), (P, T, hint)
    return jnp.float64(T), rho


@pytest.mark.parametrize("P,T,hint,label", STATES, ids=IDS)
def test_thermodynamic_identities(P, T, hint, label):
    T, rho = _T_rho(P, T, hint)
    s = co2.properties(T, rho)
    pT, prho = s["dP_dT_rho"], s["dP_drho_T"]
    # c_p − c_v = T (∂P/∂T)_ρ² / (ρ² (∂P/∂ρ)_T)
    assert _rel(T * pT ** 2 / (rho ** 2 * prho), s["cp"] - s["cv"]) < 1e-8
    # a² = (∂P/∂ρ)_T · c_p / c_v
    assert _rel(prho * s["cp"] / s["cv"], s["speed_of_sound"] ** 2) < 1e-8
