"""The default backend is NIST IR 8608 (2026), ``eos="nist2026"``.

A call with no ``eos`` argument is the same computation as the call that names
``"nist2026"``: the same compiled kernel, so the same bits.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np

import co2_eos as ce


def test_default_eos_is_nist2026():
    assert ce.DEFAULT_EOS == ce.NIST2026 == "nist2026"


def test_every_public_function_defaults_to_nist2026():
    from co2_eos import transport
    functions = [(name, getattr(ce, name)) for name in ce.__all__]
    functions += [(f"transport.{name}", getattr(transport, name))
                  for name in ("viscosity", "thermal_conductivity",
                               "bulk_viscosity")]
    for name, fn in functions:
        if not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        assert "eos" in params, name
        assert params["eos"].default == ce.NIST2026, name


def test_batched_transport_names_its_backend():
    """``transport``'s batched functions take ``eos``; Span-Wagner is named."""
    from co2_eos import transport
    T = jnp.asarray([250.0, 304.2, 400.0])
    rho = jnp.asarray([20.0, 468.0, 150.0])
    sw = transport.for_eos(ce.SPAN_WAGNER)
    np.testing.assert_array_equal(
        transport.thermal_conductivity(T, rho, eos=ce.SPAN_WAGNER),
        jax.jit(jax.vmap(sw._scalar_thermal_conductivity))(T, rho))
    assert not np.array_equal(transport.thermal_conductivity(T, rho),
                              transport.thermal_conductivity(
                                  T, rho, eos=ce.SPAN_WAGNER))


def test_generic_constants_are_the_default_surface():
    """A public constant of a submodule that a backend surface also carries
    is the default surface's value: no generic name holds Span-Wagner's."""
    from co2_eos import backends, core, inversions, saturation, transport
    surface = backends.get(ce.DEFAULT_EOS).surface
    checked = 0
    for mod in (core, inversions, transport, saturation):
        for name in dir(mod):
            if name.isupper() and not name.startswith("_") and hasattr(surface, name):
                checked += 1
                assert getattr(mod, name) == getattr(surface, name), \
                    f"{mod.__name__}.{name}"
    for name in ("TC", "PC", "M", "R_MOLAR", "RHOC_MOLAR"):
        assert not hasattr(transport, name), f"transport.{name}"


def test_critical_and_triple_point_follow_the_backend():
    from co2_eos import nist2026, span_wagner
    assert ce.critical_point() == (nist2026.TC, nist2026.PC, nist2026.RHOC)
    assert ce.critical_point(eos=ce.SPAN_WAGNER) == (
        span_wagner.TC, span_wagner.PC, span_wagner.RHOC)
    assert ce.triple_point() == nist2026.T_TRIPLE
    assert ce.triple_point(eos=ce.SPAN_WAGNER) == span_wagner.T_TRIPLE
    for gas in ce.REFERENCE_GAS_NAMES:
        try:
            ce.triple_point(eos=gas)
        except ValueError as err:
            assert gas in str(err)
        else:
            raise AssertionError(f"triple_point(eos={gas!r}) did not raise")
    try:
        ce.critical_point(eos="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("critical_point accepted an unknown eos")


def test_submodule_names_are_the_nist2026_instance():
    """Every generic module-level name of a submodule is its nist2026 form.

    ``core``, ``inversions`` and ``transport`` bind each kernel of their
    ``for_eos(NIST2026)`` namespace at module level (the same object, so the
    same compiled code); ``saturation``'s lookups take ``eos`` with the
    nist2026 default.  Span-Wagner is reached by its own name only:
    ``span_wagner``, or ``for_eos("span_wagner")``.
    """
    from co2_eos import core, inversions, saturation, transport
    for mod in (core, inversions, transport):
        ns = mod.for_eos(ce.NIST2026)
        for name, value in vars(ns).items():
            if name == "eos":
                continue
            assert getattr(mod, name) is value, f"{mod.__name__}.{name}"
    for name in saturation._LOOKUPS:
        params = inspect.signature(getattr(saturation, name)).parameters
        assert params["eos"].default == ce.NIST2026, f"saturation.{name}"
    assert (inspect.signature(saturation._ensure_loaded).parameters["eos"]
            .default == ce.NIST2026)


def test_default_call_equals_named_nist2026_call():
    T = jnp.asarray([250.0, 304.2, 310.0, 400.0])
    rho = jnp.asarray([20.0, 468.0, 600.0, 150.0])
    default = ce.properties_from_rho_T(rho, T)
    named = ce.properties_from_rho_T(rho, T, eos=ce.NIST2026)
    for key in named:
        np.testing.assert_array_equal(default[key], named[key], err_msg=key)
    P = jnp.asarray([8.0e6, 9.97e6])
    Tp = jnp.asarray([305.0, 320.0])
    np.testing.assert_array_equal(
        jax.vmap(ce.density_from_PT)(P, Tp),
        jax.vmap(lambda p, t: ce.density_from_PT(p, t, eos=ce.NIST2026))(P, Tp))
    TS = jnp.asarray([250.0, 290.0, 303.0])
    np.testing.assert_array_equal(ce.saturation_pressure(TS),
                                  ce.saturation_pressure(TS, eos=ce.NIST2026))
