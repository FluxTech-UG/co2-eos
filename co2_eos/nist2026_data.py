"""Constants of the NIST IR 8608 equation of state for CO₂: the single source.

Lemmon, E. W., Harvey, A. H. and Hellmann, R., "Alternative Fundamental
Equation of State for Fluid Carbon Dioxide", NIST IR 8608 (February 2026),
doi:10.6028/NIST.IR.8608.

Every value below is transcribed from the report and verified digit by digit
against the page image.  Page numbers are the report's printed page numbers
(the PDF page is eight more: printed p. 2 is PDF p. 10).  Values the report
marks "This work" are properties computed from the new equation; they are the
report's checkable values (the report prints no separate table of computed
states for implementers), so they are kept here as ``*_TABLE`` constants for
the tests and are never used as inputs, except where the report itself uses
them as reducing parameters (T_c and ρ_c).

Reduced variables: τ = T_c/T, δ = ρ/ρ_c (Eq. 2, p. 4).
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Table 1, p. 2: physical constants and characteristic properties
# ═══════════════════════════════════════════════════════════════════════════
R_MOLAR = 8.314462618          # J/(mol·K), molar gas constant (Mohr et al.)
M = 0.0440095                  # kg/mol (44.0095 g/mol, IUPAC 2007)
TC = 304.1282                  # K, critical temperature (reducing parameter)
RHOC_MOLAR = 10624.9           # mol/m³ (10.6249 mol/L, reducing parameter)
T_TRIPLE = 216.5909            # K, triple-point temperature (Kawamura et al.)

# Computed from the equation ("This work"): the report's check values.
PC_TABLE = 7.37703e6           # Pa, critical pressure p(T_c, ρ_c)
P_TRIPLE_TABLE = 0.51795e6     # Pa, triple-point pressure p_sat(T_t)
ACENTRIC_TABLE = 0.2249        # acentric factor ω

# p. 3, text: "The critical density given in Table 1 is, when converted to
# mass units, 467.597 kg m−3".
RHOC_MASS_TEXT = 467.597       # kg/m³

# p. 5, text: the reference state of n°6 and n°7 (the refrigeration-industry
# convention): saturated liquid at 273.15 K has h = 200 kJ/kg, s = 1 kJ/(kg·K).
REF_T = 273.15                 # K
REF_H_LIQ = 200.0e3            # J/kg
REF_S_LIQ = 1.0e3              # J/(kg·K)

# ═══════════════════════════════════════════════════════════════════════════
# Table 3, p. 5: the ideal-gas part, Eq. (5)
#   α° = ln δ + n°6 + n°7 τ + (n°0 − 1) ln τ
#        + Σ_{i=1..5} n°i ln(1 − exp(−m°i τ / T_c))
# ═══════════════════════════════════════════════════════════════════════════
IDEAL_N0 = 3.50011
IDEAL_N = np.array([1.6429, 0.4, 1.0196, 1.027, 0.202])              # n°1..n°5
IDEAL_M = np.array([946.0, 1063.0, 1985.0, 3502.0, 10884.0])        # m°1..m°5, K
IDEAL_N6 = -6.1251093666397622
IDEAL_N7 = 5.1155660487491881

# ═══════════════════════════════════════════════════════════════════════════
# Table 4, p. 6: the residual part, Eq. (6), p. 5
#   αʳ = Σ_{1..5}   n δ^d τ^t
#      + Σ_{6..12}  n δ^d τ^t exp(−r δ^l)
#      + Σ_{13..24} n δ^d τ^t exp(−η(δ − ε)² − β(τ − γ)²)
# I_Pol = 5, I_Exp = 7, I_GBS = 12 (p. 5).
# ═══════════════════════════════════════════════════════════════════════════
N_POL, N_EXP, N_GBS = 5, 7, 12

RES_N = np.array([
    0.032646485,           # 1
    0.82874959001,         # 2
    -0.718938788655,       # 3
    -0.84276998907051,     # 4
    0.1092257,             # 5
    -0.89202,              # 6
    -0.5989,               # 7
    1.0,                   # 8
    -0.013545,             # 9
    -0.0051172,            # 10
    -0.034668,             # 11
    -0.39452,              # 12
    -0.188,                # 13
    0.22705,               # 14
    0.22518,               # 15
    -0.34052,              # 16
    -0.1,                  # 17
    0.19096,               # 18
    -0.2578465,            # 19
    -0.191,                # 20
    -0.022783,             # 21
    -0.0005,               # 22
    -0.0000044,            # 23
    -0.0000848,            # 24
])

RES_T = np.array([
    1.0, 0.16, 1.1369, 1.1, 0.4217,                          # 1..5
    2.514, 3.666, 1.0, 1.219, 7.25, 8.0, 0.7,                # 6..12
    2.7, 2.4, 2.582, 0.536, 1.46, 1.2, 1.0305,               # 13..19
    1.0, 1.0, 1.0, 1.0, 1.0,                                 # 20..24
])

RES_D = np.array([
    4, 1, 1, 2, 3,
    1, 3, 2, 7, 1, 2, 1,
    2, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
])

# Exponential terms 6..12
EXP_L = np.array([2, 2, 1, 2, 3, 3, 2])
EXP_R = np.array([1.1368, 1.182, 0.81, 0.666, 1.0, 0.7514, 0.4656])

# Gaussian bell-shaped terms 13..24
GBS_ETA = np.array([1.682, 1.7558, 1.8562, 1.337, 1.0, 1.864, 0.9388,
                    10.35, 10.52, 43.0, 200.0, 250.0])
GBS_BETA = np.array([2.655, 0.567, 2.249, 0.74, 2.34, 2.3, 0.0,
                     60.79, 327.0, 22000.0, 60000.0, 1000000.0])
GBS_GAMMA = np.array([1.398, 1.3241, 1.24515, 1.2006, 1.1453, 0.9526, 1.0,
                      1.3085, 1.1085, 1.0131, 1.005, 1.0024])
GBS_EPS = np.array([-0.1762, 0.4031, 0.77185, -0.0849, 0.6133, 0.0379,
                    1.0626, 0.652, 0.8918, 0.972, 0.9, 0.978])

# ═══════════════════════════════════════════════════════════════════════════
# Section 7, pp. 32 to 33: auxiliary equations for the saturation boundary.
# The report states they reproduce the full phase-equilibrium solution of the
# equation within 0.004 % (p_s) and 0.01 % (ρ_L, ρ_V, except very near the
# critical point), Fig. 30, p. 33.  They are check values here, never inputs.
#   Eq. (16)  ln(p_s/p_c) = (T_c/T) Σ a_i (1 − T/T_c)^t_i
#   Eq. (17)  ln(ρ_L/ρ_c) = Σ a_i (1 − T/T_c)^t_i
#   Eq. (18)  ln(ρ_V/ρ_c) = Σ a_i (1 − T/T_c)^t_i
# ═══════════════════════════════════════════════════════════════════════════
AUX_PS_A = np.array([-7.059904, 1.939133, -1.646356, -3.299563])
AUX_PS_T = np.array([1.0, 1.5, 2.0, 4.0])
AUX_PS_TOL = 0.004e-2
AUX_RHOL_A = np.array([1.929233, -0.630916, -0.324714, 0.396134])
AUX_RHOL_T = np.array([0.34, 0.5, 10.0 / 6.0, 11.0 / 6.0])
AUX_RHOV_A = np.array([-1.696815, -0.845878, -4.577673, -10.13093, -29.7956])
AUX_RHOV_T = np.array([0.34, 0.5, 1.0, 7.0 / 3.0, 14.0 / 3.0])
AUX_RHO_TOL = 0.01e-2

# Structural facts the kernel in ``nist2026`` relies on: asserted, not assumed.
assert RES_N.shape == RES_T.shape == RES_D.shape == (N_POL + N_EXP + N_GBS,)
assert EXP_L.shape == EXP_R.shape == (N_EXP,)
assert all(a.shape == (N_GBS,) for a in (GBS_ETA, GBS_BETA, GBS_GAMMA, GBS_EPS))
assert IDEAL_N.shape == IDEAL_M.shape == (5,)
