"""Write the stored reference values the reference-gas tests compare against.

Runs once, offline.  CoolProp is a test-only dependency: this script is the one
place the reference-gas route meets it, and nothing in the package imports it.
CoolProp's air is Lemmon et al. (2000), its argon Tegeler et al. (1999), its
helium Ortiz Vega et al.; its transport for air and argon is Lemmon and
Jacobsen (2004).  Its helium transport is Arp et al. (viscosity) and Hands and
Arp (conductivity), both older than the ab initio fit the helium backend uses,
so the helium transport columns carry that fit's disagreement with them
(tests/test_reference_gases.py states the bars).

    python scripts/generate_reference_gas_values.py

Rows: the wide grid (``grid``, 200 to 1000 K, 0.05 to 20 bar) and the states the
two rigs reach (``hsulai``: air at 101.325 kPa, 290 to 720 K; ``swift``: helium
and argon at 6.9 and 13.8 bar with a 10 % acoustic swing).  ``high`` carries
helium and argon from 20 bar to their 25 bar validity limit.
"""

import csv
from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np

from co2_eos.reference_gases_data import GASES as REFERENCE_GASES

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "reference_gas_values.csv"

FLUID = {"air": "Air", "helium": "Helium", "argon": "Argon"}
BAR = 1e5
GRID_T = np.arange(200.0, 1001.0, 50.0)
GRID_P = np.array([0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]) * BAR
SWIFT_T = np.arange(280.0, 1001.0, 40.0)
SWIFT_P = np.array([6.9 * 0.9, 6.9, 6.9 * 1.1, 13.8 * 0.9, 13.8, 13.8 * 1.1]) * BAR
HSULAI_T = np.arange(290.0, 721.0, 10.0)
HSULAI_P = np.array([101325.0])
HIGH_T = np.arange(200.0, 1001.0, 100.0)
HIGH_P = np.array([25.0]) * BAR

COLUMNS = ("gas", "tag", "T_K", "P_Pa", "rho", "cp", "cv", "w", "mu", "k", "h", "u", "s")
KEYS = ("D", "C", "O", "A", "V", "L", "H", "U", "S")
# CoolProp's mass basis carries M = 28.96546 g/mol for air; Lemmon et al. (2000)
# and Lemmon and Jacobsen (2004) state 28.9586.  The molar surface is the
# published one, so the air columns are moved to the published M: densities by
# M_pub/M_cp, per-mass quantities by M_cp/M_pub, the sound speed by its square root.
M_SCALE = {"air": REFERENCE_GASES["air"].M / CP.PropsSI("M", "Air"), "helium": 1.0, "argon": 1.0}
POWER = {"D": 1.0, "C": -1.0, "O": -1.0, "A": -0.5, "V": 0.0, "L": 0.0, "H": -1.0, "U": -1.0,
         "S": -1.0}


def rows():
    plans = [(gas, "grid", GRID_T, GRID_P) for gas in FLUID]
    plans.append(("air", "hsulai", HSULAI_T, HSULAI_P))
    plans += [(gas, "swift", SWIFT_T, SWIFT_P) for gas in ("helium", "argon")]
    plans += [(gas, "high", HIGH_T, HIGH_P) for gas in ("helium", "argon")]
    for gas, tag, temps, pressures in plans:
        for T in temps:
            for P in pressures:
                values = [CP.PropsSI(k, "T", float(T), "P", float(P), FLUID[gas])
                          * M_SCALE[gas] ** POWER[k] for k in KEYS]
                yield [gas, tag, repr(float(T)), repr(float(P))] + [repr(float(v)) for v in values]


def main():
    OUT.parent.mkdir(exist_ok=True)
    table = list(rows())
    with OUT.open("w", newline="") as f:
        f.write(f"# CoolProp {CoolProp.__version__}, scripts/generate_reference_gas_values.py\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        w.writerows(table)
    print(f"wrote {len(table)} rows to {OUT}")


if __name__ == "__main__":
    main()
