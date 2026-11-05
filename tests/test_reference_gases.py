"""The reference-gas backends (air, helium, argon) behind the ``eos=`` keyword.

The contract: a consumer's call site is the Span-Wagner call with another name,
every bundle field present; the values meet the stated tolerances against the
stored reference (tests/data/reference_gas_values.csv, generated once by
scripts/generate_reference_gas_values.py, never compared by hash); every
property is a derivative of the one Helmholtz function.
"""

import csv
import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import co2_eos as ce
from co2_eos import backends

GASES = ("air", "helium", "argon")
VALUES = Path(__file__).parent / "data" / "reference_gas_values.csv"

# The stated tolerances of the route: density 0.05 %, c_p 0.2 %, μ and k 1 %.
TOL = {"rho": 5e-4, "cp": 2e-3, "cv": 2e-3, "w": 1e-3, "mu": 1e-2, "k": 1e-2}
# Helium's transport is dilute only: above its 1 % limit (``P_ONE_PERCENT``, Swift's
# top state with its swing) it stays below this bar to P_MAX, measured 1.09 % at 20
# bar and 1.33 % at 25 bar.
TOL_HELIUM_TRANSPORT_ABOVE = 1.5e-2
# Enthalpy and internal energy differ from CoolProp by a reference state; the
# difference across each gas's whole table is what the engine's energy reads.
TOL_ENERGY_SPAN = 2e-3


def _one_percent_limit(gas):
    """Pa, the pressure the 1 % transport bar stands to, past a solve's round-off."""
    data = importlib.import_module("co2_eos.reference_gases_data").GASES[gas]
    limit = data.TRANSPORT.P_ONE_PERCENT if gas == "helium" else data.P_MAX
    return limit * (1.0 + 1e-9)


def _table():
    with VALUES.open() as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    out = {}
    for gas in GASES:
        sel = [r for r in rows if r["gas"] == gas]
        out[gas] = {k: np.array([float(r[k]) for r in sel]) for k in sel[0]
                    if k not in ("gas", "tag")}
        out[gas]["tag"] = np.array([r["tag"] for r in sel])
    return out


TABLE = _table()


def _model(gas):
    t = TABLE[gas]
    rho = np.asarray(jax.vmap(lambda p, T: ce.density_from_PT(p, T, eos=gas))(t["P_Pa"], t["T_K"]))
    return rho, ce.properties_from_rho_T(jnp.asarray(rho), jnp.asarray(t["T_K"]), eos=gas)


def _state_grid(gas):
    surf = backends.get(gas).surface
    pts = []
    for T in (200.0, 290.0, 450.0, 720.0, 1000.0):
        for P in (5e3, 101325.0, 6.9e5, 13.8e5, 20e5):
            pts.append((T, float(ce.density_from_PT(P, T, eos=gas))))
    return surf, pts


# ── The registry and the call site ───────────────────────────────────────────

def test_names_in_the_registry():
    assert set(GASES) <= set(ce.EOS_NAMES)
    assert ce.REFERENCE_GAS_NAMES == GASES
    assert ce.CO2_EOS_NAMES == ("span_wagner", "nist2026")
    assert ce.EOS_NAMES == ce.CO2_EOS_NAMES + ce.REFERENCE_GAS_NAMES
    assert ce.DEFAULT_EOS == "span_wagner"


@pytest.mark.parametrize("gas", GASES)
def test_consumer_call_returns_the_span_wagner_bundle(gas):
    rho = jnp.array([1.2, 1.1])
    T = jnp.array([300.0, 320.0])
    bundle = ce.properties_from_rho_T(rho, T, eos=gas)
    co2 = ce.properties_from_rho_T(jnp.array([300.0]), jnp.array([350.0]))
    assert set(bundle) == set(co2)
    for key, value in bundle.items():
        assert value.shape == rho.shape, key
        assert value.dtype == jnp.float64, key
        assert np.all(np.isfinite(value)), key


@pytest.mark.parametrize("gas", GASES)
def test_saturation_refuses_with_the_reason(gas):
    with pytest.raises(ValueError, match="no saturation"):
        ce.saturation_pressure(150.0, eos=gas)
    with pytest.raises(ValueError, match="no saturation"):
        ce.saturation_densities_P(1e5, eos=gas)


def test_unknown_name_refused():
    with pytest.raises(ValueError, match="unknown eos"):
        ce.properties_from_rho_T(jnp.array([1.0]), jnp.array([300.0]), eos="neon")


# ── Values against the stored reference ──────────────────────────────────────

@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("key,field", [("cp", "cp"), ("cv", "cv"), ("w", "speed_of_sound")])
def test_thermo_against_reference(gas, key, field):
    _, st = _model(gas)
    err = np.abs(np.asarray(st[field]) / TABLE[gas][key] - 1.0)
    i = int(np.argmax(err))
    assert err[i] < TOL[key], (
        f"{gas} {key}: {err[i]:.2e} at {TABLE[gas]['T_K'][i]} K, {TABLE[gas]['P_Pa'][i]} Pa")


@pytest.mark.parametrize("gas", GASES)
def test_density_against_reference(gas):
    rho, _ = _model(gas)
    err = np.abs(rho / TABLE[gas]["rho"] - 1.0)
    i = int(np.argmax(err))
    assert err[i] < TOL["rho"], (
        f"{gas} rho: {err[i]:.2e} at {TABLE[gas]['T_K'][i]} K, {TABLE[gas]['P_Pa'][i]} Pa")


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("key,field", [("mu", "viscosity"), ("k", "thermal_conductivity")])
def test_transport_against_reference(gas, key, field):
    """1 % to P_MAX on air and argon.  Helium, dilute only: 1 % to Swift's top state."""
    _, st = _model(gas)
    err = np.abs(np.asarray(st[field]) / TABLE[gas][key] - 1.0)
    above = TABLE[gas]["P_Pa"] > _one_percent_limit(gas)
    i = int(np.argmax(np.where(above, 0.0, err)))
    assert err[i] < TOL[key], (
        f"{gas} {key}: {err[i]:.2e} at {TABLE[gas]['T_K'][i]} K, {TABLE[gas]['P_Pa'][i]} Pa")
    if gas == "helium":
        assert above.any()
        assert err[above].max() < TOL_HELIUM_TRANSPORT_ABOVE, (
            f"helium {key} above {_one_percent_limit(gas)} Pa: {err[above].max():.2e}")
    else:
        assert not above.any()


def test_helium_transport_limit_covers_swift():
    """The limit is Swift's top state, 13.8 bar, with its swing: every stored Swift row is in."""
    tag = TABLE["helium"]["tag"] == "swift"
    top = TABLE["helium"]["P_Pa"][tag].max()
    limit = _one_percent_limit("helium")
    assert top <= limit
    assert top == pytest.approx(limit, rel=2e-9)


@pytest.mark.parametrize("gas", GASES)
@pytest.mark.parametrize("key,field", [("h", "enthalpy"), ("u", "internal_energy")])
def test_energy_differences_against_reference(gas, key, field):
    _, st = _model(gas)
    model = np.asarray(st[field])
    ref = TABLE[gas][key]
    span = ref.max() - ref.min()
    err = np.abs((model - model[0]) - (ref - ref[0])) / span
    assert err.max() < TOL_ENERGY_SPAN, f"{gas} {key} span error {err.max():.2e}"


def test_rig_states_are_in_the_stored_table():
    assert (TABLE["air"]["tag"] == "hsulai").sum() == 44
    assert (TABLE["helium"]["tag"] == "swift").sum() == 114
    assert (TABLE["argon"]["tag"] == "swift").sum() == 114


# ── Limits where the answer is known ─────────────────────────────────────────

@pytest.mark.parametrize("gas", ("helium", "argon"))
def test_monatomic_limit(gas):
    """c_p → (5/2)R as ρ → 0 at every T; the bulk viscosity is exactly 0."""
    surf = backends.get(gas).surface
    T = jnp.array([200.0, 500.0, 1000.0])
    st = ce.properties_from_rho_T(1e-6 * jnp.ones(3), T, eos=gas)
    assert np.max(np.abs(np.asarray(st["cp"]) / (2.5 * surf.R) - 1.0)) < 1e-6
    assert np.max(np.abs(np.asarray(st["cv"]) / (1.5 * surf.R) - 1.0)) < 1e-6
    w = np.sqrt(5.0 / 3.0 * surf.R * np.asarray(T))
    assert np.max(np.abs(np.asarray(st["speed_of_sound"]) / w - 1.0)) < 1e-6
    assert np.all(np.asarray(st["bulk_viscosity"]) == 0.0)
    assert np.all(np.asarray(st["bulk_viscosity_vibrational"]) == 0.0)


@pytest.mark.parametrize("gas", GASES)
def test_dilute_limit_is_the_ideal_gas_law(gas):
    surf = backends.get(gas).surface
    rho = 1e-7
    for T in (200.0, 500.0, 1000.0):
        st = ce.properties(T, rho, eos=gas)
        assert abs(float(st["pressure"]) / (rho * surf.R * T) - 1.0) < 1e-8


# ── Every property is a derivative of the one function ───────────────────────

_REL_FLOOR = 1e-10


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), _REL_FLOOR)


@pytest.mark.parametrize("gas", GASES)
def test_residual_derivs_match_autodiff(gas):
    surf, pts = _state_grid(gas)
    d = jax.grad(surf.alphar, argnums=1)
    t = jax.grad(surf.alphar, argnums=0)
    worst = 0.0
    for T, rho in pts:
        tau, delta = jnp.float64(surf.TC / T), jnp.float64(rho / surf.RHOC)
        ar, ar_d, ar_t, ar_dd, ar_tt, ar_dt = surf.residual_derivs(tau, delta)
        worst = max(worst,
                    _rel(ar, surf.alphar(tau, delta)),
                    _rel(ar_d, d(tau, delta)),
                    _rel(ar_t, t(tau, delta)),
                    _rel(ar_dd, jax.grad(d, argnums=1)(tau, delta)),
                    _rel(ar_tt, jax.grad(t, argnums=0)(tau, delta)),
                    _rel(ar_dt, jax.grad(d, argnums=0)(tau, delta)))
    assert worst < 1e-9, f"{gas}: residual analytic vs autodiff {worst:.2e}"


@pytest.mark.parametrize("gas", GASES)
def test_ideal_derivs_match_autodiff(gas):
    surf, pts = _state_grid(gas)
    t = jax.grad(surf.alpha0, argnums=0)
    worst = 0.0
    for T, rho in pts:
        tau, delta = jnp.float64(surf.TC / T), jnp.float64(rho / surf.RHOC)
        a0, a0_t, a0_tt = surf.ideal_derivs(tau, delta)
        worst = max(worst,
                    _rel(a0, surf.alpha0(tau, delta)),
                    _rel(a0_t, t(tau, delta)),
                    _rel(a0_tt, jax.grad(t, argnums=0)(tau, delta)))
        at_only, att_only = surf.ideal_tau_only(tau)
        worst = max(worst, _rel(at_only, a0_t), _rel(att_only, a0_tt))
    assert worst < 1e-10, f"{gas}: ideal analytic vs autodiff {worst:.2e}"


@pytest.mark.parametrize("gas", GASES)
def test_fast_tau_path_matches_full(gas):
    surf, pts = _state_grid(gas)
    worst = 0.0
    for T, rho in pts:
        tau, delta = jnp.float64(surf.TC / T), jnp.float64(rho / surf.RHOC)
        ar_t, ar_tt = surf.residual_tau_fast(tau, surf.residual_tau_prep(delta))
        _, _, ar_t_full, _, ar_tt_full, _ = surf.residual_derivs(tau, delta)
        worst = max(worst, _rel(ar_t, ar_t_full), _rel(ar_tt, ar_tt_full))
    assert worst < 1e-12, f"{gas}: fast τ-path vs full {worst:.2e}"


@pytest.mark.parametrize("gas", GASES)
def test_bundle_identities(gas):
    """β, K_T, K_s, w and the heat capacities close against autodiff of a(ρ, T)."""
    surf, pts = _state_grid(gas)

    def helmholtz(rho, T):
        return surf.R * T * surf.alpha(surf.TC / T, rho / surf.RHOC)

    def pressure(rho, T):
        return rho ** 2 * jax.grad(helmholtz, argnums=0)(rho, T)

    def entropy(rho, T):
        return -jax.grad(helmholtz, argnums=1)(rho, T)

    for T, rho in pts:
        st = ce.properties(T, rho, eos=gas)
        rho_j, T_j = jnp.float64(rho), jnp.float64(T)
        P = pressure(rho_j, T_j)
        P_rho = jax.grad(pressure, argnums=0)(rho_j, T_j)
        P_T = jax.grad(pressure, argnums=1)(rho_j, T_j)
        s = entropy(rho_j, T_j)
        cv = T * jax.grad(entropy, argnums=1)(rho_j, T_j)
        a = helmholtz(rho_j, T_j)
        cp = cv + T * P_T ** 2 / (rho ** 2 * P_rho)
        K_T = rho * P_rho
        K_s = K_T * cp / cv
        beta = P_T / K_T
        assert _rel(st["pressure"], P) < 1e-11
        assert _rel(st["dP_drho_T"], P_rho) < 1e-11
        assert _rel(st["dP_dT_rho"], P_T) < 1e-11
        assert _rel(st["cv"], cv) < 1e-10
        assert _rel(st["cp"], cp) < 1e-10
        assert _rel(st["entropy"], s) < 1e-10
        assert _rel(st["internal_energy"], a + T * s) < 1e-10
        assert _rel(st["enthalpy"], a + T * s + P / rho) < 1e-10
        assert _rel(st["gibbs_energy"], a + P / rho) < 1e-10
        assert _rel(float(st["speed_of_sound"]) ** 2 * rho, K_s) < 1e-10
        # β from the bundle's own fields, the form a consumer writes.
        beta_b = float(st["dP_dT_rho"]) / (rho * float(st["dP_drho_T"]))
        assert _rel(beta_b, beta) < 1e-10


# ── Inversions round trip, and their gradients ───────────────────────────────

@pytest.mark.parametrize("gas", GASES)
def test_rho_u_inversion_round_trip(gas):
    _, pts = _state_grid(gas)
    T = jnp.array([p[0] for p in pts])
    rho = jnp.array([p[1] for p in pts])
    u = ce.properties_from_rho_T(rho, T, eos=gas)["internal_energy"]
    back = ce.temperature_from_rho_u(rho, u, eos=gas)
    assert np.max(np.abs(np.asarray(back) - np.asarray(T))) < 1e-9
    st = ce.properties_from_rho_u(rho, u, eos=gas)
    assert np.max(np.abs(np.asarray(st["temperature"]) - np.asarray(T))) < 1e-9


@pytest.mark.parametrize("gas", GASES)
def test_density_from_pt_round_trip(gas):
    for T in (200.0, 450.0, 1000.0):
        for P in (5e3, 101325.0, 13.8e5, 20e5):
            rho = ce.density_from_PT(P, T, eos=gas)
            st = ce.properties(T, rho, eos=gas)
            assert abs(float(st["pressure"]) / P - 1.0) < 1e-12


@pytest.mark.parametrize("gas", GASES)
def test_state_from_ph_round_trip(gas):
    for T in (200.0, 450.0, 1000.0):
        for P in (5e3, 101325.0, 13.8e5, 20e5):
            st = ce.state_from_PT(P, T, eos=gas)
            back = ce.state_from_Ph(P, st["enthalpy"], eos=gas)
            assert abs(float(back["temperature"]) - T) < 1e-8
            assert abs(float(back["density"]) / float(st["density"]) - 1.0) < 1e-11


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("gas", GASES)
def test_jit_vmap_grad_of_the_consumer_call(gas):
    rho = jnp.array([0.2, 0.5, 1.0])    # inside every gas's P_MAX
    T = jnp.array([300.0, 450.0, 700.0])
    plain = ce.properties_from_rho_T(rho, T, eos=gas)
    jitted = jax.jit(lambda r, t: ce.properties_from_rho_T(r, t, eos=gas))(rho, T)
    for key in plain:
        assert np.allclose(plain[key], jitted[key], rtol=1e-14, atol=0.0), key

    def scalar(field, r, t):
        return ce.properties_from_rho_T(jnp.array([r]), jnp.array([t]), eos=gas)[field][0]

    for field in ("pressure", "cp", "speed_of_sound", "viscosity", "thermal_conductivity"):
        d_T = jax.grad(scalar, argnums=2)(field, 1.2, 450.0)
        d_rho = jax.grad(scalar, argnums=1)(field, 1.2, 450.0)
        assert np.isfinite(d_T) and np.isfinite(d_rho), field
        fd_T = _fd(lambda t: scalar(field, 1.2, t), 450.0, 1e-3)
        fd_rho = _fd(lambda r: scalar(field, r, 450.0), 1.2, 1e-5)
        assert abs(float(d_T) - float(fd_T)) <= 1e-6 * max(abs(float(fd_T)), 1e-9), field
        assert abs(float(d_rho) - float(fd_rho)) <= 1e-6 * max(abs(float(fd_rho)), 1e-9), field


@pytest.mark.parametrize("gas", GASES)
def test_inversion_gradients(gas):
    rho0, T0, P0 = 2.0, 450.0, 2e5
    st = ce.properties(T0, rho0, eos=gas)
    u0 = float(st["internal_energy"])

    def T_of_u(u):
        return ce.state_from_Du(rho0, u, eos=gas)["temperature"]

    assert _rel(jax.grad(T_of_u)(u0), _fd(T_of_u, u0, 10.0)) < 1e-6
    assert _rel(jax.grad(T_of_u)(u0), 1.0 / float(st["cv"])) < 1e-12

    def rho_of_P(p):
        return ce.density_from_PT(p, T0, eos=gas)

    assert _rel(jax.grad(rho_of_P)(P0), _fd(rho_of_P, P0, 10.0)) < 1e-6

    def rho_of_T(t):
        return ce.density_from_PT(P0, t, eos=gas)

    assert _rel(jax.grad(rho_of_T)(T0), _fd(rho_of_T, T0, 1e-2)) < 1e-6

    h0 = float(ce.state_from_PT(P0, T0, eos=gas)["enthalpy"])

    def T_of_h(h):
        return ce.state_from_Ph(P0, h, eos=gas)["temperature"]

    def T_of_P(p):
        return ce.state_from_Ph(p, h0, eos=gas)["temperature"]

    assert _rel(jax.grad(T_of_h)(h0), _fd(T_of_h, h0, 10.0)) < 1e-6
    fd = float(_fd(T_of_P, P0, 100.0))
    assert abs(float(jax.grad(T_of_P)(P0)) - fd) <= 1e-5 * max(abs(fd), 1e-9)


# ── The validity domain fails loudly ─────────────────────────────────────────

@pytest.mark.parametrize("gas", GASES)
def test_outside_the_domain_is_nan(gas):
    surf = backends.get(gas).surface
    T_lo, T_hi = surf.T_RANGE
    rho = jnp.array([1.0, 1.0, 1.0])
    T = jnp.array([T_lo - 1.0, 0.5 * (T_lo + T_hi), T_hi + 1.0])
    st = ce.properties_from_rho_T(rho, T, eos=gas)
    assert np.isnan(st["pressure"][0]) and np.isnan(st["pressure"][2])
    assert np.isfinite(st["pressure"][1])
    # Past the pressure limit: the density at twice the limit.
    rho_hi = jnp.array([2.0 * surf.P_MAX / (surf.R * 500.0)])
    assert np.isnan(ce.properties_from_rho_T(rho_hi, jnp.array([500.0]), eos=gas)["pressure"][0])
    # The T-only inversion: an energy far past the hot limit.
    st_hot = ce.properties(T_hi - 1.0, 1.0, eos=gas)
    u_far = float(st_hot["internal_energy"]) + 2000.0 * float(st_hot["cv"])
    assert np.isnan(ce.temperature_from_rho_u(jnp.array([1.0]), jnp.array([u_far]), eos=gas)[0])


# ── The published coefficients are transcribed exactly ───────────────────────

def _published(fluid):
    CP = pytest.importorskip("CoolProp.CoolProp")
    return json.loads(CP.get_fluid_param_string(fluid, "JSON"))[0]


@pytest.mark.parametrize("gas,fluid", [("air", "Air"), ("helium", "Helium"), ("argon", "Argon")])
def test_residual_terms_match_the_published_table(gas, fluid):
    data = importlib.import_module("co2_eos.reference_gases_data")
    eos = _published(fluid)["EOS"][0]
    power = next(t for t in eos["alphar"] if t["type"] == "ResidualHelmholtzPower")
    table = list(zip(power["n"], power["t"], power["d"], power["l"]))
    kept = data.GASES[gas].RESIDUAL_TERMS
    for term in kept:
        assert term in table, (gas, term)
    # Every published power term that feeds δ¹ or δ² is kept.
    for n, t, d, l in table:
        if d == 1 or d == 2:
            assert (n, t, d, l) in kept, (gas, n, t, d, l)
    assert data.GASES[gas].TC == eos["STATES"]["reducing"]["T"]
    assert data.GASES[gas].PC == pytest.approx(eos["STATES"]["reducing"]["p"], rel=2e-5)
    assert data.GASES[gas].RHOC_MOLAR == pytest.approx(eos["STATES"]["reducing"]["rhomolar"],
                                                       rel=1e-9)
    # CoolProp's air M (28.96546) and helium R (8.3144598) differ from the published
    # 28.9586 and 8.314472; the module holds the published values.
    assert data.GASES[gas].M == pytest.approx(eos["molar_mass"], rel=2.5e-4)
    assert data.GASES[gas].R_MOLAR == pytest.approx(eos["gas_constant"], rel=2e-6)


@pytest.mark.parametrize("gas,table", [
    # (T [K], ρ [mol/dm³], η [μPa·s], λ [mW/(m·K)], share): Lemmon and Jacobsen (2004) Table V,
    # p. 28, the rows in the temperature range.  The dense rows lie past P_MAX (air 133 and 124
    # bar, argon 114 and 119 bar), as a fuller check of the same terms.  Their printed λ carries
    # the critical enhancement this backend leaves out; `share` is its fraction of the printed
    # λ (Eqs. 7-10, read once from CoolProp's conductivity contributions), zero on air at 300 K,
    # above T_ref = 265 K (Table I), and at zero density.
    ("air", [(300.0, 0.0, 18.5230, 26.3529, 0.0), (200.0, 10.0, 21.1392, 35.3185, 3.807e-2),
             (300.0, 5.0, 21.3241, 32.6062, 0.0)]),
    ("argon", [(300.0, 0.0, 22.7241, 17.8042, 0.0), (200.0, 10.0, 25.5662, 26.1377, 6.430e-2),
               (300.0, 5.0, 26.3706, 23.2302, 9.891e-4)]),
])
def test_transport_against_the_published_check_values(gas, table):
    surf = backends.get(gas).surface
    for T, rho_molar, eta, lam, share in table:
        rho = max(rho_molar * 1e3, 1e-12) * surf.M
        assert float(ce.viscosity(T, rho, eos=gas)) * 1e6 == pytest.approx(eta, rel=5e-5)
        ours = float(ce.thermal_conductivity(T, rho, eos=gas)) * 1e3
        assert ours / lam == pytest.approx(1.0 - share, abs=5e-5), (gas, T, rho_molar)


def test_helium_transport_reads_its_ab_initio_table():
    data = importlib.import_module("co2_eos.reference_gases_data")
    for T, eta, lam in data.HELIUM_DILUTE_TABLE:
        assert float(ce.viscosity(T, 1.0, eos="helium")) * 1e6 == pytest.approx(eta, rel=5e-6)
        assert float(ce.thermal_conductivity(T, 1.0, eos="helium")) * 1e3 == pytest.approx(
            lam, rel=5e-6)
    # The stored cubics are the refit of the stored table.
    T, eta, lam = np.array(data.HELIUM_DILUTE_TABLE).T
    x = np.log(T / data.HELIUM_FIT_T0)
    for y, c in ((eta, data.GASES["helium"].TRANSPORT.LN_ETA),
                 (lam, data.GASES["helium"].TRANSPORT.LN_LAMBDA)):
        refit = np.polynomial.polynomial.polyfit(x, np.log(y), 3)
        assert np.allclose(refit, c, rtol=1e-9, atol=0.0)


def test_helium_second_virial_against_ab_initio():
    """B(T) of the series against Hurly and Mehl (2007) Table 5, 200 to 1000 K."""
    surf = backends.get("helium").surface
    for T, B_cm3 in ((200.0, 12.1673), (300.0, 11.8212), (500.0, 11.0082), (1000.0, 9.5505)):
        _, ar_d, *_ = surf.residual_derivs(surf.TC / T, 1e-12)
        B = float(ar_d) / surf.RHOC_MOLAR * 1e6
        assert B == pytest.approx(B_cm3, abs=0.1), (T, B)


@pytest.mark.parametrize("gas,fluid", [("air", "Air"), ("argon", "Argon")])
def test_transport_terms_match_the_published_table(gas, fluid):
    data = importlib.import_module("co2_eos.reference_gases_data")
    tr = _published(fluid)["TRANSPORT"]
    d = data.GASES[gas].TRANSPORT
    vis = tr["viscosity"]
    assert d.SIGMA_NM == pytest.approx(vis["sigma_eta"] * 1e9, rel=1e-12)
    assert d.EPS_K == vis["epsilon_over_k"]
    assert d.M_ETA == pytest.approx(vis["dilute"]["molar_mass"] * 1e3, rel=1e-12)
    assert list(d.OMEGA_B) == vis["dilute"]["a"]
    assert np.allclose(d.ETA_RES, [(a * 1e6, t, dd, l) for a, t, dd, l in zip(
        vis["higher_order"]["a"], vis["higher_order"]["t1"],
        vis["higher_order"]["d1"], vis["higher_order"]["l"])], rtol=1e-12, atol=0.0)
    con = tr["conductivity"]
    assert d.LAMBDA_DILUTE_N[0] == pytest.approx(con["dilute"]["A"][0] * 1e3, rel=1e-12)
    assert np.allclose(d.LAMBDA_RES, [(a * 1e3, t, dd, l) for a, t, dd, l in zip(
        con["residual"]["A"], con["residual"]["t"],
        con["residual"]["d"], con["residual"]["l"])], rtol=1e-12, atol=0.0)
