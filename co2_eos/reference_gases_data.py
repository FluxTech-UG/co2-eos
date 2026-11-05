"""Coefficients of the reference gases: the one home of every number, cited.

Each residual term is the published (n, t, d, l) of n δ^d τ^t exp(−δ^l); the
ones kept are every power term that feeds δ¹ and δ² (``reference_gases`` sums the
series).  What the series leaves out, the terms past δ² and the Gaussian terms of
the argon and helium equations, moves the compressibility factor by at most 4.2e-6
(argon) and 4.7e-6 (helium) over 200 to 1000 K to 20 bar, 8.5e-6 and 7.6e-6 at 25
bar, against the full equations (measured 2026-11-05, 10 K steps).  The Gaussians
alone are 3e-25 on argon and 2.6e-6 on helium.  On air the series meets the density
bar with 3.6e-4 at 200 K, 20 bar.

Sources, each read in the published document:

[L2000]  Lemmon, Jacobsen, Penoncello and Friend, J. Phys. Chem. Ref. Data 29(3),
         331 (2000): Table 13, p. 346 (the residual terms); Table 9, p. 342 (the
         maxcondentherm row, the reducing parameters); R, p. 333; M, the fixed
         points for air, p. 334.
[T1999]  Tegeler, Span and Wagner, J. Phys. Chem. Ref. Data 28(3), 779 (1999):
         Table 30, p. 808 (the residual terms; its footnote holds R = 0.2081333
         kJ/(kg·K), T_c and ρ_c = 535.6 kg/m³).  Its reference range ends at 700 K;
         from there to 1000 K the series follows that equation's own extrapolation
         within the tolerances (tests/test_reference_gases.py), which is no
         comparison against data.
[OV2023] Ortiz Vega, Hall, Holste, Harvey and Lemmon, NIST IR 8474 (2023),
         doi:10.6028/NIST.IR.8474: Table 2, p. 4 (the terms); Table 1, p. 2 (the
         constants).  The report documents the preliminary equation in REFPROP 10.
[LJ2004] Lemmon and Jacobsen, Int. J. Thermophys. 25(1), 21 (2004): Table I, p. 24
         (M, ε/k, σ); Table II, p. 24 (b_i); Table III, p. 25 (viscosity); Table IV,
         p. 26 (conductivity); the forms are Eqs. 1-3, p. 23 and Eqs. 4-6, pp. 24-25.
         The critical enhancement, Eqs. 7-10, p. 27, is left out: it is 0.16 % (air)
         and 0.20 % (argon) of λ at 200 K, 20 bar, and exactly zero where the bracket
         of Eq. 10 goes negative (air at 280 K and above at 20 bar).
[HM2007] Hurly and Mehl, J. Res. NIST 112, 75 (2007): Table 5, pp. 87-88.
[BURCAT] Burcat's thermochemical database, 23 July 2023 version
         (respecth.elte.hu/burcat/THERM.DAT), the low-range NASA-7 coefficients:
         AIR "L 9/95" (N₂ 78.084, O₂ 20.9476, Ar 0.9365, CO₂ 0.0319 %), AR and HE
         "g 5/97".  Ar and He are also McBride, Gordon and Reno, NASA TM-4513
         (1993), Table II, pp. 12 and 33.

Air's gas constant is the R of [L2000].  B(T) alone misses the density bar on air
(0.36 %) and argon (0.30 %), both at 200 K, 20 bar, so both carry C(T); on helium
B(T) alone stands at 0.017 % and C(T) takes it to the 4.7e-6 above.

Transport.  Air and argon carry the dilute and residual parts of [LJ2004]: the
dilute part alone misses the 1 % bar at density (at 200 K: air λ 1.38 % at 5 bar, μ
1.45 % at 10 bar, argon λ 1.89 % at 5 bar, μ 1.74 % at 10 bar; at 280 K, 15.18 bar:
argon μ 1.39 %, λ 3.22 %).  Helium is dilute only, adding no coefficients: its 1 %
bar stands to ``P_ONE_PERCENT``, the top of Swift's helium states.
"""

from types import SimpleNamespace

P_REF = 1.0e5   # Pa, the standard-state pressure of the NASA polynomials

# ── Air ─────────────────────────────────────────────────────────────────────
_AIR = SimpleNamespace(
    NAME="air",
    TC=132.6312,                # K, [L2000] the maxcondentherm, its reducing T
    RHOC_MOLAR=10447.7,         # mol/m³, [L2000]
    PC=3785020.0,               # Pa, [L2000]
    M=0.0289586,                # kg/mol, [L2000], [LJ2004]
    R_MOLAR=8.31451,            # J/(mol·K), [L2000]
    T_RANGE=(200.0, 1000.0),    # K, the low range of the NASA-7 polynomial
    P_MAX=20.0e5,               # Pa, where the density stays within its 0.05 %
    NASA7=(3.56839620, -6.78729429e-04, 1.55371476e-06, -3.29937060e-12,
           -4.66395387e-13, -1.06234659e+03, 3.71582965),   # [BURCAT]
    RESIDUAL_TERMS=(            # [L2000] Table 13, k = 1, 2, 3, 4, 11, 15, 18
        (0.118160747229, 0, 1, 0),
        (0.713116392079, 0.33, 1, 0),
        (-1.61824192067, 1.01, 1, 0),
        (0.0714140178971, 0, 2, 0),
        (-0.101365037912, 1.6, 1, 1),
        (-0.146629609713, 3.6, 1, 2),
        (0.0148287891978, 3.5, 1, 3),
    ),
    TRANSPORT=SimpleNamespace(
        KIND="lemmon_jacobsen",
        M_ETA=28.9586,          # g/mol, [LJ2004] Table I
        SIGMA_NM=0.360,         # nm, [LJ2004] Table I
        EPS_K=103.3,            # K, [LJ2004] Table I
        OMEGA_B=(0.431, -0.4623, 0.08406, 0.005341, -0.00331),  # [LJ2004] Table II
        ETA_RES=(               # (N [μPa·s], t, d, l), [LJ2004] Table III
            (10.72, 0.2, 1, 0),
            (1.122, 0.05, 4, 0),
            (0.002019, 2.4, 9, 0),
            (-8.876, 0.6, 1, 1),
            (-0.02916, 3.6, 8, 1),
        ),
        LAMBDA_DILUTE_N=(1.308, 1.405, -1.036),     # N₁ [mW/(m·K) per μPa·s], N₂, N₃
        LAMBDA_DILUTE_T=(0.0, -1.1, -0.3),          # [LJ2004] Table IV
        LAMBDA_RES=(            # (N [mW/(m·K)], t, d, l), [LJ2004] Table IV
            (8.743, 0.1, 1, 0),
            (14.76, 0.0, 2, 0),
            (-16.62, 0.5, 3, 2),
            (3.793, 2.7, 7, 2),
            (-6.142, 0.3, 7, 2),
            (-0.3778, 1.3, 11, 2),
        ),
    ),
)

# ── Argon ───────────────────────────────────────────────────────────────────
_ARGON = SimpleNamespace(
    NAME="argon",
    TC=150.687,                 # K, [T1999]
    RHOC_MOLAR=535.6 / 0.039948,    # mol/m³, [T1999] ρ_c = 535.6 kg/m³
    PC=4863000.0,               # Pa, [LJ2004] Table I
    M=0.039948,                 # kg/mol, [LJ2004] Table I
    R_MOLAR=8.31451,            # J/(mol·K), [T1999] R·M
    T_RANGE=(200.0, 1000.0),    # K
    P_MAX=25.0e5,               # Pa, the series' measured range (module docstring)
    NASA7=(2.5, 0.0, 0.0, 0.0, 0.0, -7.45375e+02, 4.37967491),  # [BURCAT]
    RESIDUAL_TERMS=(            # [T1999] Table 30, i = 1-9, 13, 14, 22, 23
        (0.088722304990011, 0, 1, 0),
        (0.70514805167298, 0.25, 1, 0),
        (-1.682011565409, 1, 1, 0),
        (-0.14909014431486, 2.75, 1, 0),
        (-0.1202480460094, 4, 1, 0),
        (-0.12164978798599, 0, 2, 0),
        (0.40035933626752, 0.25, 2, 0),
        (-0.27136062699129, 0.75, 2, 0),
        (0.24211924579645, 2.75, 2, 0),
        (-0.32181391750702, 3, 1, 1),
        (0.33230017695794, 3.5, 1, 1),
        (-0.011397840001996, 1, 2, 2),
        (-0.024455169960535, 7, 2, 2),
    ),
    TRANSPORT=SimpleNamespace(
        KIND="lemmon_jacobsen",
        M_ETA=39.948,           # g/mol, [LJ2004] Table I
        SIGMA_NM=0.335,         # nm, [LJ2004] Table I
        EPS_K=143.2,            # K, [LJ2004] Table I
        OMEGA_B=(0.431, -0.4623, 0.08406, 0.005341, -0.00331),  # [LJ2004] Table II
        ETA_RES=(               # (N [μPa·s], t, d, l), [LJ2004] Table III
            (12.19, 0.42, 1, 0),
            (13.99, 0.0, 2, 0),
            (0.005027, 0.95, 10, 0),
            (-18.93, 0.5, 5, 2),
            (-6.698, 0.9, 1, 4),
            (-3.827, 0.8, 2, 4),
        ),
        LAMBDA_DILUTE_N=(0.8158, -0.432),           # N₁ [mW/(m·K) per μPa·s], N₂
        LAMBDA_DILUTE_T=(0.0, -0.77),               # [LJ2004] Table IV
        LAMBDA_RES=(            # (N [mW/(m·K)], t, d, l), [LJ2004] Table IV
            (13.73, 0.0, 1, 0),
            (10.07, 0.0, 2, 0),
            (0.7375, 0.0, 4, 0),
            (-33.96, 0.8, 5, 2),
            (20.47, 1.2, 6, 2),
            (-2.274, 0.8, 9, 2),
            (-3.973, 0.5, 1, 4),
        ),
    ),
)

# ── Helium-4 ────────────────────────────────────────────────────────────────
# (T [K], η [μPa·s], λ [mW/(m·K)]), [HM2007] Table 5, every printed row in the domain.
HELIUM_DILUTE_TABLE = (
    (200.0, 15.1284, 118.308),
    (225.0, 16.3769, 128.062),
    (250.0, 17.5862, 137.510),
    (273.16, 18.6765, 146.027),
    (275.0, 18.7620, 146.695),
    (298.15, 19.8245, 154.994),
    (300.0, 19.9084, 155.649),
    (325.0, 21.0288, 164.400),
    (350.0, 22.1260, 172.970),
    (375.0, 23.2024, 181.375),
    (400.0, 24.2598, 189.633),
    (450.0, 26.3241, 205.753),
    (500.0, 28.3298, 221.413),
    (600.0, 32.1959, 251.596),
    (700.0, 35.9049, 280.550),
    (800.0, 39.4880, 308.517),
    (900.0, 42.9671, 335.670),
    (1000.0, 46.3584, 362.135),
)
HELIUM_FIT_T0 = 300.0   # K, the fit's variable is ln(T/T0)

_HELIUM = SimpleNamespace(
    NAME="helium",
    TC=5.1953,                  # K, [OV2023] Table 1
    RHOC_MOLAR=17383.7,         # mol/m³, [OV2023] Table 1
    PC=228320.0,                # Pa, [OV2023] Table 1
    M=0.004002602,              # kg/mol, [OV2023] Table 1
    R_MOLAR=8.314472,           # J/(mol·K), [OV2023] Table 1
    T_RANGE=(200.0, 1000.0),    # K
    P_MAX=25.0e5,               # Pa, the series' measured range (module docstring)
    NASA7=(2.5, 0.0, 0.0, 0.0, 0.0, -7.45375e+02, 9.28723974e-01),  # [BURCAT]
    RESIDUAL_TERMS=(            # [OV2023] Table 2, i = 2, 3, 4, 5, 7, 8, 10, 11, 12
        (3.0638932, 0.425, 1, 0),
        (-4.2420844, 0.63, 1, 0),
        (0.054418088, 0.69, 2, 0),
        (-0.18971904, 1.83, 2, 0),
        (2.2833566, 0.925, 1, 1),
        (-0.53331595, 1.585, 1, 2),
        (0.99444915, 1.51, 2, 1),
        (-0.30078896, 2.9, 2, 2),
        (-1.6432563, 0.8, 1, 1),
    ),
    TRANSPORT=SimpleNamespace(
        KIND="ab_initio_dilute",
        # ln η [μPa·s] and ln λ [mW/(m·K)], cubics in ln(T/T0), fitted once to
        # HELIUM_DILUTE_TABLE by scripts/fit_helium_transport.py.  Dilute only.
        LN_ETA=(2.9911416450465844, 0.6828630060089349, 0.014526870558021602,
                0.001176271502718677),
        LN_LAMBDA=(5.047604999616816, 0.6822103880501055, 0.014467766624251054,
                   0.0011899549778530115),
        # Pa, where μ and λ hold 1 %: Swift's top state, 13.8 bar, with its 10 %
        # swing.  The worst below it is μ 0.77 % (200 K, 15 bar), λ 0.90 % (360 K,
        # 15.18 bar); above it, μ 1.00 % and λ 1.09 % at 20 bar, μ 1.23 % and λ 1.33 %
        # at 25 bar.  The stored reference is Arp et al. and Hands and Arp, which
        # stand up to 0.44 % from this ab initio fit in the dilute limit.
        P_ONE_PERCENT=1.1 * 13.8e5,
    ),
)

GASES = {"air": _AIR, "helium": _HELIUM, "argon": _ARGON}
