"""The equation-of-state backends and the static switch that selects one.

Every public function of co2-eos takes a keyword ``eos``, a Python string that
is static under ``jax.jit`` (one compiled kernel per backend; no traced
branching), with the default ``DEFAULT_EOS``:

  * ``"nist2026"``: Lemmon, Harvey & Hellmann, NIST IR 8608 (2026), the
    default: 24 terms with no nonanalytic terms.  Surface and kernel are both
    ``nist2026``; its tables carry the suffix ``_nist2026``.
  * ``"span_wagner"``: Span & Wagner (1996), bit for bit 0.5.0.  Its surface
    is ``span_wagner`` (constants, α, the autodiff property functions) and its
    analytic kernel ``span_wagner_kernel``; its tables carry no suffix.

A backend's surface module provides ``TC, RHOC, RHOC_MOLAR, R, R_MOLAR, M,
PC, T_TRIPLE`` and ``_scalar_pressure``, ``_scalar_enthalpy``,
``_scalar_internal_energy``, ``_scalar_entropy``; its kernel module provides
``ideal_derivs``, ``ideal_tau_only``, ``residual_derivs``,
``residual_tau_prep``, ``residual_tau_fast``, ``residual_tau_derivs``.  Each
backend carries its own gas constant and molar mass; nothing shared between
backends assumes Span-Wagner's.

The reference gases ``"air"``, ``"helium"`` and ``"argon"`` are backends of the
same registry: ``reference_gases`` is the surface and the kernel of each, and
the bundle ``properties_from_rho_T`` returns is built by the same code.  They
carry no saturation dome and no near-critical machinery (``has_dome`` False):
``saturation`` refuses them, and ``transport.for_eos`` and ``inversions.for_eos``
hand out the namespaces ``reference_gases`` holds, leaving the CO₂ factories
for the CO₂ backends alone.
"""

from pathlib import Path
from types import SimpleNamespace

from co2_eos import nist2026 as _nist2026
from co2_eos import reference_gases as _reference_gases
from co2_eos import span_wagner as _span_wagner
from co2_eos import span_wagner_kernel as _span_wagner_kernel

SPAN_WAGNER = "span_wagner"
NIST2026 = "nist2026"
AIR = "air"
HELIUM = "helium"
ARGON = "argon"
CO2_EOS_NAMES = (SPAN_WAGNER, NIST2026)
REFERENCE_GAS_NAMES = (AIR, HELIUM, ARGON)
EOS_NAMES = CO2_EOS_NAMES + REFERENCE_GAS_NAMES
DEFAULT_EOS = NIST2026

# The upper temperature bound [K] of the CO₂ inversions' iterates (the fused
# (ρ, u) Newton in ``core``, the guesses and the (ρ, u) Newton in
# ``inversions``), on every CO₂ backend.  A state above it is outside the
# package's range; the reference gases bound their own (``T_CLIP``).
INVERSION_T_MAX = 800.0

DATA_DIR = Path(__file__).parent / "data"

_BACKENDS = {
    SPAN_WAGNER: SimpleNamespace(name=SPAN_WAGNER, surface=_span_wagner,
                                 kernel=_span_wagner_kernel, suffix="", has_dome=True),
    NIST2026: SimpleNamespace(name=NIST2026, surface=_nist2026,
                              kernel=_nist2026, suffix="_nist2026", has_dome=True),
    **{name: SimpleNamespace(name=name, surface=_reference_gases.GASES[name],
                             kernel=_reference_gases.GASES[name], suffix=f"_{name}",
                             has_dome=False)
       for name in REFERENCE_GAS_NAMES},
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
