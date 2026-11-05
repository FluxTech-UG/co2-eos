"""Print the fit behind helium's transport: the ab initio dilute part.

Hurly and Mehl (2007) tabulate ab initio η(T) and λ(T) and publish no closed
form.  The backend evaluates ln η and ln λ as cubics fitted once by this script
to the 18 rows of their Table 5 from 200 to 1000 K, held in
``reference_gases_data``; the fit's largest errors against the printed rows are
3.4e-6 (η) and 2.8e-6 (λ), the printed digits' own size.

Helium's transport is dilute only: the backend's 1 % bar and its limit stand in
``reference_gases_data`` (``P_ONE_PERCENT``).

    python scripts/fit_helium_transport.py

tests/test_reference_gases.py holds the stored coefficients to this refit.
"""

import numpy as np
from numpy.polynomial import polynomial as P

from co2_eos.reference_gases_data import HELIUM_DILUTE_TABLE, HELIUM_FIT_T0


def fit_dilute():
    """(c_η, c_λ, largest η error, largest λ error) from the stored table."""
    T, eta, lam = np.array(HELIUM_DILUTE_TABLE).T
    x = np.log(T / HELIUM_FIT_T0)
    c_eta = P.polyfit(x, np.log(eta), 3)
    c_lam = P.polyfit(x, np.log(lam), 3)
    err_eta = np.abs(np.exp(P.polyval(x, c_eta)) / eta - 1.0).max()
    err_lam = np.abs(np.exp(P.polyval(x, c_lam)) / lam - 1.0).max()
    return c_eta, c_lam, err_eta, err_lam


if __name__ == "__main__":
    c_eta, c_lam, err_eta, err_lam = fit_dilute()
    print("LN_ETA", tuple(float(c) for c in c_eta), f"largest error {err_eta:.2e}")
    print("LN_LAMBDA", tuple(float(c) for c in c_lam), f"largest error {err_lam:.2e}")
