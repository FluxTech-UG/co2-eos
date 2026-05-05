"""Generate CO2 saturation table from Span-Wagner EOS.

Solves the Maxwell construction (equal P and equal Gibbs energy) at each
subcritical temperature to find coexisting liquid and vapor densities.
CoolProp provides initial guesses; final values are derived entirely from
the Span-Wagner EOS in co2_eos/span_wagner.py.

Outputs co2_eos/data/saturation_table.npz with cubic spline coefficients
for JIT-compilable interpolation in co2_eos/saturation.py. The .npz lives
inside the package so it ships in the wheel.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from pathlib import Path
import time
import CoolProp.CoolProp as CP

# Ensure JAX float64 before importing span_wagner
import jax
jax.config.update("jax_enable_x64", True)

from co2_eos.span_wagner import (
    TC, RHOC, R, PC,
    _scalar_pressure, _scalar_enthalpy, _scalar_entropy,
)

# ── Constants ────────────────────────────────────────────────────────────
T_TRIPLE = 216.592       # K — CO2 triple point temperature
T_CRIT = TC              # 304.1282 K
T_NEAR_CRIT = T_CRIT - 0.001  # stop table 1 mK below Tc

DATA_DIR = Path(__file__).parent.parent / "co2_eos" / "data"


def gibbs(T, rho):
    """Specific Gibbs energy g = h - T*s [J/kg]."""
    h = float(_scalar_enthalpy(T, rho))
    s = float(_scalar_entropy(T, rho))
    return h - T * s


def maxwell_residual(log_rhos, T):
    """Residual for Maxwell construction in log-density space.

    Variables: [ln(rho_l), ln(rho_v)]
    Equations:
        f1 = (P_l - P_v) / P_c  (equal pressure)
        f2 = (g_l - g_v) / (R*T) (equal Gibbs energy)
    """
    rho_l = np.exp(log_rhos[0])
    rho_v = np.exp(log_rhos[1])

    P_l = float(_scalar_pressure(T, rho_l))
    P_v = float(_scalar_pressure(T, rho_v))
    g_l = gibbs(T, rho_l)
    g_v = gibbs(T, rho_v)

    f1 = (P_l - P_v) / PC
    f2 = (g_l - g_v) / (R * T)
    return [f1, f2]


def build_temperature_grid(n_bulk=300, n_near=300, n_vnear=200):
    """Non-uniform T grid: bulk + near-critical + very-near-critical.

    Returns ascending array of temperatures from T_TRIPLE to T_NEAR_CRIT.
    Three regions with increasing density approaching Tc.
    """
    T_b1 = T_CRIT - 1.0    # bulk → near-critical boundary
    T_b2 = T_CRIT - 0.01   # near-critical → very-near-critical boundary

    T_bulk = np.linspace(T_TRIPLE, T_b1, n_bulk, endpoint=False)
    T_near = np.linspace(T_b1, T_b2, n_near, endpoint=False)
    T_vnear = np.linspace(T_b2, T_NEAR_CRIT, n_vnear)

    return np.concatenate([T_bulk, T_near, T_vnear])


def solve_saturation_curve(T_grid):
    """Solve Maxwell construction at each temperature.

    Returns arrays: P_sat, rho_l, rho_v, h_l, h_v, s_l, s_v
    """
    N = len(T_grid)
    P_sat = np.zeros(N)
    rho_l = np.zeros(N)
    rho_v = np.zeros(N)
    h_l = np.zeros(N)
    h_v = np.zeros(N)
    s_l = np.zeros(N)
    s_v = np.zeros(N)

    for i, T in enumerate(T_grid):
        # CoolProp initial guess
        try:
            rho_l_guess = CP.PropsSI('D', 'T', float(T), 'Q', 0, 'CO2')
            rho_v_guess = CP.PropsSI('D', 'T', float(T), 'Q', 1, 'CO2')
        except Exception:
            # Very near critical — use previous solution with mild extrapolation
            if i > 0:
                rho_l_guess = rho_l[i - 1]
                rho_v_guess = rho_v[i - 1]
            else:
                raise

        x0 = [np.log(rho_l_guess), np.log(rho_v_guess)]

        sol, info, ier, msg = fsolve(
            maxwell_residual, x0, args=(T,), full_output=True,
            xtol=1e-12,
        )

        if ier != 1:
            print(f"  WARNING: fsolve did not converge at T={T:.6f} K: {msg}")
            # Fall back to CoolProp guess (still use SW pressure for consistency)
            sol = x0

        rl = np.exp(sol[0])
        rv = np.exp(sol[1])

        # Ensure rho_l > rho_v
        if rl < rv:
            rl, rv = rv, rl

        rho_l[i] = rl
        rho_v[i] = rv
        P_sat[i] = float(_scalar_pressure(T, rl))
        h_l[i] = float(_scalar_enthalpy(T, rl))
        h_v[i] = float(_scalar_enthalpy(T, rv))
        s_l[i] = float(_scalar_entropy(T, rl))
        s_v[i] = float(_scalar_entropy(T, rv))

        if (i + 1) % 50 == 0 or i == N - 1:
            print(f"  [{i+1:4d}/{N}] T={T:.4f} K  P_sat={P_sat[i]/1e6:.6f} MPa  "
                  f"rho_l={rho_l[i]:.3f}  rho_v={rho_v[i]:.3f} kg/m3")

    return P_sat, rho_l, rho_v, h_l, h_v, s_l, s_v


def fit_and_save(T_grid, P_sat, rho_l, rho_v, h_l, h_v, s_l, s_v):
    """Fit cubic splines and save coefficients to npz."""
    arrays = {}

    # ── T-based splines (primary) ────────────────────────────────────────
    arrays['T_breaks'] = T_grid
    for name, values in [
        ('P_sat', P_sat), ('rho_l', rho_l), ('rho_v', rho_v),
        ('h_l', h_l), ('h_v', h_v), ('s_l', s_l), ('s_v', s_v),
    ]:
        cs = CubicSpline(T_grid, values)
        arrays[f'{name}_c'] = cs.c  # shape (4, N-1)

    # ── P-based splines (for inverse lookups) ────────────────────────────
    # P_sat is monotonically increasing, so we can use it as x-axis
    arrays['P_breaks'] = P_sat
    for name, values in [
        ('T_sat', T_grid), ('rho_l_P', rho_l), ('rho_v_P', rho_v),
        ('h_l_P', h_l), ('h_v_P', h_v), ('s_l_P', s_l), ('s_v_P', s_v),
    ]:
        cs = CubicSpline(P_sat, values)
        arrays[f'{name}_c'] = cs.c

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "saturation_table.npz"
    np.savez(out_path, **arrays)
    print(f"\nSaved saturation table to {out_path}")
    print(f"  T range: [{T_grid[0]:.3f}, {T_grid[-1]:.6f}] K")
    print(f"  P range: [{P_sat[0]/1e6:.6f}, {P_sat[-1]/1e6:.6f}] MPa")
    print(f"  {len(T_grid)} grid points, {len(arrays)} arrays")


def main():
    print("Generating CO2 saturation table from Span-Wagner EOS...")
    t0 = time.time()

    T_grid = build_temperature_grid()
    print(f"Temperature grid: {len(T_grid)} points, "
          f"[{T_grid[0]:.3f}, {T_grid[-1]:.6f}] K")

    P_sat, rho_l, rho_v, h_l, h_v, s_l, s_v = solve_saturation_curve(T_grid)
    fit_and_save(T_grid, P_sat, rho_l, rho_v, h_l, h_v, s_l, s_v)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
