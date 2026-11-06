"""Evaluate the public API's Span-Wagner path on a fixed grid; write an .npz.

Run by ``test_span_wagner_bit_for_bit`` in a subprocess, once against the
0.5.0 package (a ``git archive`` of its commit; Span-Wagner was its only
backend) as ``PYTHONPATH=<root> python -P span_wagner_probe.py OUT``, and once
against this checkout with the backend named,
``... span_wagner_probe.py OUT span_wagner``.  It calls only names 0.5.0
already had, so the two outputs must agree bit for bit.  Not a test module
(no ``test_`` prefix).
"""
import sys
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import co2_eos as _ce

# With a backend named (argv[2]), every call carries ``eos=<name>``; without
# one, the calls are 0.5.0's, which took no ``eos``.
_EOS = sys.argv[2] if len(sys.argv) > 2 else None
_NAMES = ("properties_from_rho_T", "properties_from_rho_u",
          "temperature_from_rho_u", "properties", "state_from_Du", "viscosity",
          "thermal_conductivity", "bulk_viscosity", "density_from_PT",
          "state_from_PT", "state_from_Ph", "saturation_pressure",
          "saturation_densities", "saturation_enthalpies",
          "saturation_entropies", "saturation_temperature",
          "saturation_densities_P", "saturation_enthalpies_P",
          "saturation_entropies_P")


class _Api:
    AUTO, LIQUID, VAPOR = _ce.AUTO, _ce.LIQUID, _ce.VAPOR
    __file__ = _ce.__file__


ce = _Api()
for _name in _NAMES:
    _fn = getattr(_ce, _name)
    setattr(ce, _name, _fn if _EOS is None else partial(_fn, eos=_EOS))

out = {}


def put(prefix, value):
    if isinstance(value, dict):
        for k, v in value.items():
            out[f"{prefix}/{k}"] = np.asarray(v)
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            out[f"{prefix}/{i}"] = np.asarray(v)
    else:
        out[prefix] = np.asarray(value)


T1 = np.concatenate([np.linspace(220.0, 420.0, 21),
                     [304.1782, 304.2282, 304.3, 305.1282, 314.0]])
R1 = np.concatenate([np.linspace(15.0, 1150.0, 21), [440.0, 468.0, 472.0, 500.0]])
TT, RR = (a.ravel() for a in np.meshgrid(T1, R1, indexing="ij"))
TT = jnp.asarray(TT)
RR = jnp.asarray(RR)

st = ce.properties_from_rho_T(RR, TT)
put("properties_from_rho_T", st)
U = st["internal_energy"]
put("properties_from_rho_u", ce.properties_from_rho_u(RR, U))
put("temperature_from_rho_u", ce.temperature_from_rho_u(RR, U))
put("properties", jax.vmap(ce.properties)(TT, RR))
put("state_from_Du", jax.vmap(ce.state_from_Du)(RR, U))
put("viscosity", jax.vmap(ce.viscosity)(TT, RR))
put("thermal_conductivity", jax.vmap(ce.thermal_conductivity)(TT, RR))
put("bulk_viscosity", jax.vmap(ce.bulk_viscosity)(TT, RR))

P1 = np.array([0.6e6, 2.0e6, 5.0e6, 7.1e6, 7.37e6, 7.5e6, 8.0e6, 10.0e6, 20.0e6])
T2 = np.linspace(225.0, 360.0, 19)
PP, TP = (jnp.asarray(a.ravel()) for a in np.meshgrid(P1, T2, indexing="ij"))
for hint in (ce.AUTO, ce.LIQUID, ce.VAPOR):
    h = jnp.full(PP.shape, hint, dtype=jnp.int32)
    put(f"density_from_PT/{hint}", jax.vmap(ce.density_from_PT)(PP, TP, h))
    put(f"state_from_PT/{hint}", jax.vmap(ce.state_from_PT)(PP, TP, h))

H1 = np.linspace(150e3, 560e3, 23)
PH, HH = (jnp.asarray(a.ravel()) for a in np.meshgrid(P1, H1, indexing="ij"))
for hint in (ce.AUTO, ce.LIQUID, ce.VAPOR):
    h = jnp.full(PH.shape, hint, dtype=jnp.int32)
    put(f"state_from_Ph/{hint}", jax.vmap(ce.state_from_Ph)(PH, HH, h))

TS = jnp.asarray(np.linspace(217.0, 304.1, 40))
PS = jnp.asarray(np.linspace(0.53e6, 7.37e6, 40))
for name in ("saturation_pressure", "saturation_densities",
             "saturation_enthalpies", "saturation_entropies"):
    put(name, getattr(ce, name)(TS))
for name in ("saturation_temperature", "saturation_densities_P",
             "saturation_enthalpies_P", "saturation_entropies_P"):
    put(name, getattr(ce, name)(PS))

# Gradients through the custom_jvp rules (a subset of the grid).
sub = slice(0, None, 7)
put("grad/state_from_Du/cp", jax.vmap(jax.grad(
    lambda r, u: ce.state_from_Du(r, u)["cp"], argnums=(0, 1)))(RR[sub], U[sub]))
put("grad/properties/thermal_conductivity", jax.vmap(jax.grad(
    lambda t, r: ce.properties(t, r)["thermal_conductivity"],
    argnums=(0, 1)))(TT[sub], RR[sub]))
put("grad/properties/bulk_viscosity", jax.vmap(jax.grad(
    lambda t, r: ce.properties(t, r)["bulk_viscosity"],
    argnums=(0, 1)))(TT[sub], RR[sub]))
put("grad/density_from_PT", jax.vmap(jax.grad(
    lambda p, t: ce.density_from_PT(p, t), argnums=(0, 1)))(PP, TP))
put("grad/state_from_Ph/temperature", jax.vmap(jax.grad(
    lambda p, h: ce.state_from_Ph(p, h)["temperature"], argnums=(0, 1)))(PH, HH))
put("grad/saturation_pressure", jax.vmap(jax.grad(ce.saturation_pressure))(TS))

# The submodules' kernels: 0.5.0's module-level names (its only backend), and
# here each submodule's ``for_eos(<name>)`` instance, the same names bound to
# the backend, and ``transport``'s batched functions with ``eos`` named.
from co2_eos import core as _core, inversions as _inv, saturation as _sat, transport as _tr
if _EOS is not None:
    _core, _inv, _sat = (m.for_eos(_EOS) for m in (_core, _inv, _sat))
    _trm = _tr
    _tr = _Api()
    for _name in ("viscosity", "thermal_conductivity", "bulk_viscosity"):
        setattr(_tr, _name, partial(getattr(_trm, _name), eos=_EOS))

for name in ("saturation_pressure", "saturation_densities",
             "saturation_entropies"):
    put(f"saturation.{name}", getattr(_sat, name)(TS))
for name in ("saturation_temperature", "saturation_enthalpies",
             "saturation_densities_P", "saturation_entropies_P"):
    put(f"saturation.{name}", getattr(_sat, name)(PS))
h = jnp.full(PP.shape, _ce.AUTO, dtype=jnp.int32)
put("inversions.density_from_PT", _inv.density_from_PT(TP, PP, h))
put("inversions.state_from_Ph", _inv.state_from_Ph(
    PH, HH, jnp.full(PH.shape, _ce.AUTO, dtype=jnp.int32)))
put("inversions.temperature_from_Du", _inv.temperature_from_Du(
    RR, U, jnp.full(RR.shape, _ce.AUTO, dtype=jnp.int32)))
put("core._state_from_rho_u", jax.vmap(_core._state_from_rho_u)(
    RR, U, jnp.full(RR.shape, _ce.AUTO, dtype=jnp.int32)))
put("transport.viscosity", _tr.viscosity(TT, RR))
put("transport.thermal_conductivity", _tr.thermal_conductivity(TT, RR))
put("transport.bulk_viscosity", _tr.bulk_viscosity(TT, RR))

np.savez(sys.argv[1], **out)
print("co2_eos loaded from", ce.__file__)
