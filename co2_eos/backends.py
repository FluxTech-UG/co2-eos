"""The equation-of-state backends and the static switch that selects one.

Every public function of co2-eos takes a keyword ``eos``, a Python string that
is static under ``jax.jit`` (one compiled kernel per backend; no traced
branching), with the default ``DEFAULT_EOS``:

  * ``"span_wagner"``: Span & Wagner (1996), the default.  Its surface is
    ``span_wagner`` (constants, α, the autodiff property functions) and its
    analytic kernel ``helmholtz``; its tables carry no suffix.
  * ``"nist2026"``: Lemmon, Harvey & Hellmann, NIST IR 8608 (2026), 24 terms
    with no nonanalytic terms.  Surface and kernel are both ``nist2026``; its
    tables carry the suffix ``_nist2026``.

A backend's surface module provides ``TC, RHOC, RHOC_MOLAR, R, R_MOLAR, M,
PC, T_TRIPLE`` and ``_scalar_pressure``, ``_scalar_enthalpy``,
``_scalar_internal_energy``, ``_scalar_entropy``; its kernel module provides
``ideal_derivs``, ``ideal_tau_only``, ``residual_derivs``,
``residual_tau_prep``, ``residual_tau_fast``, ``residual_tau_derivs``.  Each
backend carries its own gas constant and molar mass; nothing shared between
backends assumes Span-Wagner's.
"""

from pathlib import Path
from types import SimpleNamespace

from co2_eos import helmholtz as _helmholtz
from co2_eos import nist2026 as _nist2026
from co2_eos import span_wagner as _span_wagner

SPAN_WAGNER = "span_wagner"
NIST2026 = "nist2026"
EOS_NAMES = (SPAN_WAGNER, NIST2026)
DEFAULT_EOS = SPAN_WAGNER

DATA_DIR = Path(__file__).parent / "data"

_BACKENDS = {
    SPAN_WAGNER: SimpleNamespace(name=SPAN_WAGNER, surface=_span_wagner,
                                 kernel=_helmholtz, suffix=""),
    NIST2026: SimpleNamespace(name=NIST2026, surface=_nist2026,
                              kernel=_nist2026, suffix="_nist2026"),
}


def get(eos):
    """The backend record for ``eos``; raises on an unknown name."""
    try:
        return _BACKENDS[eos]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown eos {eos!r}; choose one of {EOS_NAMES}") from None


def table_path(eos, stem):
    """Path of the backend's table ``<stem><suffix>.npz`` in ``co2_eos/data``."""
    return DATA_DIR / f"{stem}{get(eos).suffix}.npz"
