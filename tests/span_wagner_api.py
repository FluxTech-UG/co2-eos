"""co2_eos with ``eos="span_wagner"`` bound: the public API and the submodules.

The tests that validate the Span-Wagner implementation (against CoolProp,
whose CO2 is Span-Wagner, or against the ``span_wagner`` autodiff oracle)
import this in place of ``co2_eos`` and its submodules:
``from tests.span_wagner_api import co2, core, inversions, saturation,
transport``.

* ``co2``: every public function with ``eos="span_wagner"`` bound; constants
  and every name that takes no ``eos`` pass through.
* ``core``, ``inversions``, ``saturation``, ``transport``: the submodule's
  names with its ``for_eos("span_wagner")`` instance over the module-level
  ones, which are the default backend's (nist2026); ``transport``'s batched
  ``viscosity``, ``thermal_conductivity`` and ``bulk_viscosity`` with
  ``eos="span_wagner"`` bound.

Not a test module (no ``test_`` prefix).
"""
import inspect
from functools import partial
from types import SimpleNamespace

import co2_eos as _ce
from co2_eos import core as _core
from co2_eos import inversions as _inversions
from co2_eos import saturation as _saturation
from co2_eos import transport as _transport

SW = _ce.SPAN_WAGNER


def _bound(name):
    value = getattr(_ce, name)
    if callable(value) and "eos" in inspect.signature(value).parameters:
        return partial(value, eos=SW)
    return value


co2 = SimpleNamespace(**{name: _bound(name) for name in _ce.__all__})


def _instance(module, **extra):
    names = {k: v for k, v in vars(module).items() if not k.startswith("__")}
    names.update(vars(module.for_eos(SW)))
    names.update(extra)
    return SimpleNamespace(**names)


core = _instance(_core)
inversions = _instance(_inversions)
saturation = _instance(_saturation)
transport = _instance(_transport, **{
    name: partial(getattr(_transport, name), eos=SW)
    for name in ("viscosity", "thermal_conductivity", "bulk_viscosity")})
