"""co2_eos's public API with ``eos="span_wagner"`` bound on every function.

The tests that validate the Span-Wagner implementation (against CoolProp,
whose CO2 is Span-Wagner, against the ``span_wagner`` autodiff oracle, or
against the ``inversions`` and ``transport`` submodules, which are the
Span-Wagner instance) import this in place of ``co2_eos``:
``from tests.span_wagner_api import co2``.  Constants and every name that
takes no ``eos`` pass through.  Not a test module (no ``test_`` prefix).
"""
import inspect
from functools import partial
from types import SimpleNamespace

import co2_eos as _ce


def _bound(name):
    value = getattr(_ce, name)
    if callable(value) and "eos" in inspect.signature(value).parameters:
        return partial(value, eos=_ce.SPAN_WAGNER)
    return value


co2 = SimpleNamespace(**{name: _bound(name) for name in _ce.__all__})
