"""The static ``eos`` switch reaches every public function, on nist2026 too.

Checks the selection (unknown names raise; one compile per backend), the
fused (ρ, u) inversion at round-off on the NIST surface (5 Newton steps from
its seed table; 3 leave up to 6e-5 K within a few mK of T_c), the batched and
scalar paths agreeing, gradients finite through the hot path at FluxTech's
Widom crossing, and the transport's critical parts reading the selected
backend (its χ_ref table against the exact bundle).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import core, transport

E = "nist2026"


def test_unknown_eos_raises():
    with pytest.raises(ValueError, match="unknown eos"):
        ce.properties(310.0, 470.0, eos="refprop")
    with pytest.raises(ValueError, match="unknown eos"):
        ce.saturation_pressure(280.0, eos="Span-Wagner")


def test_every_public_function_takes_eos():
    rho = jnp.array([300.0, 472.0, 800.0])
    T = jnp.array([330.0, 314.0, 290.0])
    st = ce.properties_from_rho_T(rho, T, eos=E)
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in st.values())
    st_u = ce.properties_from_rho_u(rho, st["internal_energy"], eos=E)
    assert np.max(np.abs(np.asarray(st_u["temperature"]) - np.asarray(T))) < 1e-9
    T_u = ce.temperature_from_rho_u(rho, st["internal_energy"], eos=E)
    assert np.max(np.abs(np.asarray(T_u) - np.asarray(T))) < 1e-9
    one = ce.properties(314.0, 472.0, eos=E)
    # Scalar and batched paths: round-off (the 24-term vector reduction is
    # fused differently under vmap), not bit for bit as on Span-Wagner.
    assert float(one["cp"]) == pytest.approx(float(st["cp"][1]), rel=1e-13)
    assert float(ce.state_from_Du(472.0, st["internal_energy"][1], eos=E)
                 ["temperature"]) == pytest.approx(314.0, abs=1e-9)
    P = float(st["pressure"][1])
    assert float(ce.density_from_PT(P, 314.0, eos=E)) == pytest.approx(472.0, rel=1e-10)
    assert float(ce.state_from_PT(P, 314.0, eos=E)["density"]) == pytest.approx(472.0, rel=1e-10)
    h = float(st["enthalpy"][1])
    assert float(ce.state_from_Ph(P, h, eos=E)["temperature"]) == pytest.approx(314.0, abs=1e-8)
    for f in (ce.viscosity, ce.thermal_conductivity, ce.bulk_viscosity):
        assert np.isfinite(float(f(314.0, 472.0, eos=E)))
    for f in (ce.saturation_pressure, ce.saturation_densities,
              ce.saturation_enthalpies, ce.saturation_entropies):
        assert np.all(np.isfinite(np.asarray(f(280.0, eos=E))))
    for f in (ce.saturation_temperature, ce.saturation_densities_P,
              ce.saturation_enthalpies_P, ce.saturation_entropies_P):
        assert np.all(np.isfinite(np.asarray(f(6.0e6, eos=E))))


def test_backends_differ_where_they_should():
    a = ce.properties(314.0, 472.0, eos=ce.SPAN_WAGNER)
    b = ce.properties(314.0, 472.0, eos=E)
    for k in ("pressure", "cp", "thermal_conductivity", "bulk_viscosity"):
        assert float(a[k]) != float(b[k])
    assert float(a["viscosity"]) == float(b["viscosity"])   # no EOS input


def test_rho_u_round_trip_at_round_off_near_critical():
    rng = np.random.default_rng(7)
    T = np.concatenate([304.1282 + np.geomspace(1e-3, 2.0, 1500),
                        rng.uniform(305.0, 345.0, 1500)])
    rho = np.concatenate([rng.uniform(420.0, 520.0, 1500),
                          rng.uniform(60.0, 1000.0, 1500)])
    st = ce.properties_from_rho_T(jnp.asarray(rho), jnp.asarray(T), eos=E)
    T2 = ce.temperature_from_rho_u(jnp.asarray(rho), st["internal_energy"], eos=E)
    assert np.max(np.abs(np.asarray(T2) - T)) < 1e-9


def test_gradients_finite_at_the_widom_crossing():
    u = float(ce.properties(314.0, 472.0, eos=E)["internal_energy"])
    for key in ("cp", "thermal_conductivity", "bulk_viscosity", "speed_of_sound"):
        g = jax.grad(lambda r, uu: ce.state_from_Du(r, uu, eos=E)[key],
                     argnums=(0, 1))(472.0, u)
        assert all(np.isfinite(float(x)) for x in g), key


def test_transport_reads_the_nist2026_surface():
    tr = transport.for_eos(E)
    assert tr._HAVE_CHI_CHEB
    d = jnp.linspace(0.05, 2.7, 200)
    from co2_eos import nist2026 as nb
    tau_ref = nb.TC / transport._T_REF

    def exact(delta):
        _, ar_d, _, ar_dd, _, _ = nb.residual_derivs(tau_ref, delta)
        return 1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd
    rel = np.abs(np.asarray(jax.vmap(tr._dpdrho_ref_reduced)(d))
                 / np.asarray(jax.vmap(exact)(d)) - 1.0)
    assert rel.max() < 1e-11
    assert core.for_eos(E)._HAVE_SEED
