"""CO₂ transport property correlations: pure JAX implementation.

Viscosity:            Laesecke & Muzny, JPCRD 46, 013107 (2017)
Thermal conductivity: Huber, Sykioti, Assael & Perkins, JPCRD 45, 013102 (2016)
Bulk viscosity:       vibrational relaxation (Lin & Trusler, JCED 59, 4099 (2014))
                      plus the critical part (Onuki, PRE 76, 061126 (2007))

Coefficients cross-referenced against CoolProp's CarbonDioxide.json and
TransportRoutines.cpp.  CoolProp is never imported here.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

# ── Shared constants ──────────────────────────────────────────────────────
# Use the same values as the Span-Wagner EOS module.
TC = 304.1282            # K
RHOC_MOLAR = 10624.9063  # mol/m³
M = 0.0440098            # kg/mol
R_MOLAR = 8.31451        # J/(mol·K)
PC = 7377300.0            # Pa
RHOC_MASS = RHOC_MOLAR * M  # kg/m³ ≈ 467.60

# ═══════════════════════════════════════════════════════════════════════════
# Residual Helmholtz derivatives (analytic, shared with the thermo core)
# ═══════════════════════════════════════════════════════════════════════════
# The conductivity critical enhancement needs αʳ_δ, αʳ_δδ, αʳ_ττ, αʳ_δτ and
# α⁰_ττ.  We take them from the analytic ``helmholtz`` kernel rather than
# jax.grad, and let the fused (ρ, u) path pass in the values it already
# computed at (T, ρ) so the enhancement costs no extra Helmholtz evaluation.
from co2_eos import helmholtz as _hz


# ═══════════════════════════════════════════════════════════════════════════
# VISCOSITY: Laesecke & Muzny (2017)
# ═══════════════════════════════════════════════════════════════════════════

# ── Dilute gas η₀(T): Eq. (4) ───────────────────────────────────────────
_ETA0_A = jnp.array([
    1749.354893188350,
    -369.069300007128,
    5423856.34887691,
    -2.21283852168356,
    -269503.247933569,
    73145.021531826,
    5.34368649509278,
])


def _eta_dilute(T):
    """Dilute-gas viscosity η₀(T) [Pa·s].

    One pow: T^(1/6); then T^(1/3) and √T follow by multiplication.
    """
    a = _ETA0_A
    T_sixth = T ** (1.0 / 6.0)
    T_third = T_sixth * T_sixth
    sqrt_T = T_third * T_sixth
    den = (a[0]
           + a[1] * T_sixth
           + a[2] * jnp.exp(a[3] * T_third)
           + (a[4] + a[5] * T_third) * jnp.exp(-T_third)
           + a[6] * sqrt_T)
    return 0.0010055 * sqrt_T / den


# ── Initial density dependence (Rainwater-Friend) ────────────────────────
_EPSILON_OVER_K = 200.76       # K
_SIGMA = 3.78421e-10           # m
_NA = 6.02214129e23            # 1/mol

_RF_B = jnp.array([
    -19.572881, 219.73999, -1015.3226, 2471.0125,
    -3375.1717, 2491.6597, -787.26086, 14.085455, -0.34664158,
])
_RF_T = jnp.array([0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.5, -5.5])


def _eta_initial(T, rho_molar, eta0):
    """Initial-density viscosity contribution [Pa·s].

    η_initial = η₀ · B_η · ρ_molar.  The Rainwater-Friend exponents are all
    multiples of -¼, so every T*^t comes from y = T*^(-¼) (two sqrts and a
    reciprocal) by a multiply ladder, with no pow calls.
    """
    T_star = T / _EPSILON_OVER_K
    y = 1.0 / jnp.sqrt(jnp.sqrt(T_star))     # T*^(-1/4)
    y2 = y * y
    y4 = y2 * y2                              # T*^(-1)
    y6 = y4 * y2
    y10 = y6 * y4
    y22 = y10 * y10 * y2
    b = _RF_B
    # exponents: 0, -.25, -.5, -.75, -1, -1.25, -1.5, -2.5, -5.5
    B_eta_star = (b[0] + b[1] * y + b[2] * y2 + b[3] * y2 * y + b[4] * y4
                  + b[5] * y4 * y + b[6] * y6 + b[7] * y10 + b[8] * y22)
    B_eta = _NA * _SIGMA ** 3 * B_eta_star  # m³/mol
    return eta0 * B_eta * rho_molar


# ── Higher-order (residual): Eqs. (8)-(9) ───────────────────────────────
_TT = 216.592       # K, triple-point temperature
_RHO_TL = 1178.53   # kg/m³, triple-point saturated liquid density
_C1 = 0.360603235428487
_C2 = 0.121550806591497
_GAMMA_VISC = 8.06282737481277

# Reference viscosity scale η_tL: Eq. (9)
_ETA_TL = (_RHO_TL ** (2.0 / 3.0)
           * (R_MOLAR * _TT) ** 0.5
           / (M ** (1.0 / 6.0) * 84446887.43579945))


def _eta_residual(T, rho_mass):
    """Higher-order viscosity contribution [Pa·s]."""
    Tr = T / _TT
    rhor = rho_mass / _RHO_TL
    rhor2 = rhor * rhor
    return _ETA_TL * (_C1 * Tr * rhor2 * rhor
                      + (rhor2 + rhor ** _GAMMA_VISC) / (Tr - _C2))


# ── Scalar viscosity ─────────────────────────────────────────────────────

def _scalar_viscosity(T, rho):
    """Dynamic viscosity μ [Pa·s] at scalar (T, ρ [kg/m³])."""
    rho_molar = rho / M
    eta0 = _eta_dilute(T)
    eta_init = _eta_initial(T, rho_molar, eta0)
    eta_res = _eta_residual(T, rho)
    return eta0 + eta_init + eta_res


# ═══════════════════════════════════════════════════════════════════════════
# THERMAL CONDUCTIVITY: Huber et al. (2016)
# ═══════════════════════════════════════════════════════════════════════════

# ── Dilute gas λ₀(T): Eq. (3) ───────────────────────────────────────────
_LAM0_L = jnp.array([0.0151874307, 0.0280674040, 0.0228564190, -0.00741624210])


def _lambda_dilute(T):
    """Dilute-gas thermal conductivity λ₀(T) [W/(m·K)]."""
    tau = TC / T
    tau2 = tau * tau
    poly = (_LAM0_L[0] + _LAM0_L[1] * tau
            + _LAM0_L[2] * tau2 + _LAM0_L[3] * tau2 * tau)
    return 1.0 / (jnp.sqrt(tau) * poly) / 1000.0  # mW → W


# ── Residual λ_res(T, ρ): polynomial ────────────────────────────────────
_LAM_RES_B = jnp.array([
    0.0100128,  0.0560488, -0.081162,  0.0624337, -0.0206336, 0.00253248,
    0.00430829, -0.0358563, 0.067148, -0.0522855,  0.0174571, -0.00196414,
])
_LAM_RES_D = jnp.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], dtype=jnp.float64)
_LAM_RES_T = jnp.array([0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1], dtype=jnp.float64)

_RHOMASS_REDUCING = 467.6  # kg/m³ (same as RHOC_MASS)


def _lambda_residual(T, rho):
    """Residual thermal conductivity [W/(m·K)].

    Σ b_k · δ^{1..6} · (1 or 1/τ): a δ multiply ladder, no pow calls.
    """
    inv_tau = T / TC
    delta = rho / _RHOMASS_REDUCING
    b = _LAM_RES_B
    out = 0.0
    dk = delta
    for k in range(6):
        out += (b[k] + b[k + 6] * inv_tau) * dk
        dk = dk * delta
    return out


# ── Critical enhancement: simplified Olchowy-Sengers ────────────────────
_KB = 1.3806488e-23    # J/K
_R0 = 1.02             # CO₂-specific amplitude ratio
_GAMMA_CE = 0.052      # CO₂-specific
_GAMMA_EXP = 1.239     # universal critical exponent γ
_NU = 0.63             # universal critical exponent ν
_XI0 = 1.5e-10         # m, correlation-length amplitude ξ₀
_QD = 2.5e9            # 1/m
_T_REF = 456.19        # K

# The enhancement needs dp/dρ at the fixed reference temperature, a smooth
# function of δ alone.  A precomputed degree-100 Chebyshev fit (3e-13 max rel
# error over δ ∈ [0, 2.75]; scripts/generate_chi_ref_table.py) replaces the
# full residual bundle at τ_ref.  Tolerant of a missing file so the generator
# script can import this package to BUILD the table; the exact bundle is the
# fallback, so the fallback changes speed only, never values.
_CHI_PATH = Path(__file__).parent / "data" / "chi_ref_cheb.npz"
_HAVE_CHI_CHEB = False
try:
    _chi = np.load(_CHI_PATH)
    _CHI_COEFS = jnp.asarray(_chi["coefs"])
    _CHI_LO = float(_chi["delta_lo"])
    _CHI_HI = float(_chi["delta_hi"])
    if float(_chi["t_ref"]) != _T_REF:
        raise ValueError(
            f"chi_ref_cheb.npz was generated for T_ref={float(_chi['t_ref'])},"
            f" transport uses {_T_REF}: regenerate the table")
    _HAVE_CHI_CHEB = True
except FileNotFoundError:
    pass


def _dpdrho_ref_reduced(delta):
    """1 + 2δ·αʳ_δ + δ²·αʳ_δδ at T_ref: Chebyshev-Clenshaw (or exact fallback).

    Queries are clamped to the fit box; beyond δ = 2.75 the enhancement is
    zero anyway (χ − χ_ref < 0 there).
    """
    if not _HAVE_CHI_CHEB:
        tau_ref = TC / _T_REF
        _, ar_d, _, ar_dd, _, _ = _hz.residual_derivs(tau_ref, delta)
        return 1.0 + 2.0 * delta * ar_d + delta * delta * ar_dd
    x = jnp.clip((2.0 * delta - (_CHI_LO + _CHI_HI)) / (_CHI_HI - _CHI_LO),
                 -1.0, 1.0)
    b1 = jnp.zeros_like(delta)
    b2 = jnp.zeros_like(delta)
    for c in np.asarray(_CHI_COEFS)[:0:-1]:
        b1, b2 = 2.0 * x * b1 - b2 + float(c), b1
    return x * b1 - b2 + float(np.asarray(_CHI_COEFS)[0])


@jax.custom_jvp
def _xi_of_excess(diff):
    """ξ = ξ₀ (max(Δχ̃, 0)/Γ)^(ν/γ) [m], zero where Δχ̃ ≤ 0.

    The value is the operations the 0.4.0 enhancement used, so every property
    built on it keeps its bits.  The derivative is supplied: the power's own
    derivative at Δχ̃ = 0 is infinite and its product with the clamp's zero
    is NaN, so the tangent is (ν/γ) ξ/Δχ̃ where Δχ̃ > 0 and 0 elsewhere.
    """
    diff_safe = jnp.maximum(diff, 0.0)
    return _XI0 * (diff_safe / _GAMMA_CE) ** (_NU / _GAMMA_EXP)


@_xi_of_excess.defjvp
def _xi_of_excess_jvp(primals, tangents):
    (diff,), (ddiff,) = primals, tangents
    xi = _xi_of_excess(diff)
    positive = diff > 0.0
    slope = jnp.where(positive,
                      (_NU / _GAMMA_EXP) * xi / jnp.where(positive, diff, 1.0),
                      0.0)
    return xi, slope * ddiff


def _correlation_length_and_excess(T, rho, alr_d, alr_dd):
    """Olchowy-Sengers correlation length ξ [m] and the susceptibility excess.

    ξ = ξ₀ (Δχ̃/Γ)^(ν/γ) with Δχ̃ = χ̃(T, ρ) − χ̃(T_ref, ρ) T_ref/T, and ξ = 0
    where Δχ̃ ≤ 0.  Shared by the conductivity enhancement and the critical
    bulk viscosity, so both read one ξ.  Takes αʳ_δ and αʳ_δδ at (T, ρ).

    ξ(Δχ̃) is :func:`_xi_of_excess`, whose derivative is finite where ξ = 0.

    Returns (ξ, Δχ̃).
    """
    rho_molar = rho / M
    delta = rho_molar / RHOC_MOLAR        # == rho / RHOC_MASS

    # ── dp/dρ at (T, ρ) ──
    dp_drho_mol = R_MOLAR * T * (1.0 + 2.0 * delta * alr_d
                                  + delta ** 2 * alr_dd)
    chi = PC / RHOC_MOLAR ** 2 * rho_molar / dp_drho_mol

    # ── dp/dρ at reference temperature T_ref (same δ; precomputed fit) ──
    dp_drho_ref = R_MOLAR * _T_REF * _dpdrho_ref_reduced(delta)
    chi_ref = PC / RHOC_MOLAR ** 2 * rho_molar / dp_drho_ref * _T_REF / T

    diff = chi - chi_ref
    return _xi_of_excess(diff), diff


def _lambda_critical_formula(T, rho, mu, alr_d, alr_dd, alr_tt, alr_dt, al0_tt,
                             guarded):
    """The enhancement's formula; ``guarded`` (static) is its derivative form.

    Unguarded it is the enhancement's value, the 0.4.0 operations bit for bit.
    Guarded, the zeroed branch (ξ = 0) is evaluated at a placeholder ξ, so 1/ξ
    there is finite and the final ``where`` passes a zero tangent rather than
    0·∞ = NaN; where ξ > 0 the two agree.
    """
    rho_molar = rho / M
    tau = TC / T
    delta = rho_molar / RHOC_MOLAR        # == rho / RHOC_MASS

    xi, diff = _correlation_length_and_excess(T, rho, alr_d, alr_dd)
    if guarded:
        xi = jnp.where(diff > 0.0, xi, _XI0)

    # ── Molar heat capacities from the shared EOS derivatives ──
    cv_over_R = -tau ** 2 * (al0_tt + alr_tt)
    num_cp = (1.0 + delta * alr_d - delta * tau * alr_dt) ** 2
    den_cp = 1.0 + 2.0 * delta * alr_d + delta ** 2 * alr_dd
    cp_over_R = cv_over_R + num_cp / den_cp

    cv_molar = R_MOLAR * cv_over_R
    cp_molar = R_MOLAR * cp_over_R

    # ── Ω functions ──
    qd_xi = _QD * xi
    inv_qd_xi = 1.0 / jnp.maximum(qd_xi, 1e-300)

    omega = (2.0 / jnp.pi) * (
        (cp_molar - cv_molar) / cp_molar * jnp.arctan(qd_xi)
        + cv_molar / cp_molar * qd_xi
    )
    omega0 = (2.0 / jnp.pi) * (
        1.0 - jnp.exp(-1.0 / (inv_qd_xi
                                + (qd_xi) ** 2 / (3.0 * delta ** 2)))
    )

    # ── Critical enhancement ──
    xi_safe = jnp.maximum(xi, 1e-300)
    lam_c = (rho_molar * cp_molar * _R0 * _KB * T
             / (6.0 * jnp.pi * mu * xi_safe)
             * (omega - omega0))

    # Zero out when diff <= 0 (no enhancement)
    return jnp.where(diff > 0.0, lam_c, 0.0)


@jax.custom_jvp
def _lambda_critical_shared(T, rho, mu, alr_d, alr_dd, alr_tt, alr_dt, al0_tt):
    """Critical enhancement of thermal conductivity [W/(m·K)].

    Takes the reduced residual derivatives αʳ_δ, αʳ_δδ, αʳ_ττ, αʳ_δτ and the
    ideal α⁰_ττ at (T, ρ); the caller (thermo kernel or fused (ρ, u) path)
    has already computed these, so the enhancement adds only the reference-T
    δ-derivative pair.  μ [Pa·s] is passed in to avoid a circular dependency.

    The reduced derivatives are identical whether δ is formed from molar or
    mass density (δ = ρ/ρc dimensionless), so the shared values apply directly.
    """
    return _lambda_critical_formula(T, rho, mu, alr_d, alr_dd, alr_tt, alr_dt,
                                    al0_tt, guarded=False)


@_lambda_critical_shared.defjvp
def _lambda_critical_shared_jvp(primals, tangents):
    """The value from the plain formula, the tangent from the guarded one."""
    value = _lambda_critical_shared(*primals)
    _, tangent = jax.jvp(
        lambda *a: _lambda_critical_formula(*a, guarded=True), primals, tangents)
    return value, tangent


# ── Scalar thermal conductivity ──────────────────────────────────────────

def _thermal_conductivity_shared(T, rho, mu, alr_d, alr_dd, alr_tt, alr_dt,
                                 al0_tt):
    """λ [W/(m·K)] reusing reduced derivatives already computed at (T, ρ).

    Used by the fused (ρ, u) path so the conductivity costs no extra Helmholtz
    evaluation beyond the reference-T δ-derivatives.
    """
    lam0 = _lambda_dilute(T)
    lam_res = _lambda_residual(T, rho)
    lam_crit = _lambda_critical_shared(T, rho, mu, alr_d, alr_dd, alr_tt,
                                       alr_dt, al0_tt)
    return lam0 + lam_res + lam_crit


def _scalar_thermal_conductivity(T, rho):
    """Thermal conductivity λ [W/(m·K)] at scalar (T, ρ [kg/m³])."""
    mu = _scalar_viscosity(T, rho)
    tau = TC / T
    delta = rho / RHOC_MASS
    _, alr_d, _, alr_dd, alr_tt, alr_dt = _hz.residual_derivs(tau, delta)
    _, _, al0_tt = _hz.ideal_derivs(tau, delta)
    return _thermal_conductivity_shared(T, rho, mu, alr_d, alr_dd, alr_tt,
                                        alr_dt, al0_tt)


# ═══════════════════════════════════════════════════════════════════════════
# BULK VISCOSITY: vibrational relaxation + critical part
# ═══════════════════════════════════════════════════════════════════════════
# ζ = ζ_vib + ζ_crit, both zero-frequency (Newtonian) coefficients, valid while
# ω·τ₁₁ ≪ 1 (single-phase CO₂, about 270 to 500 K, ρ below 1200 kg/m³).  The
# rotational and configurational part, O(μ), is below 0.1 % of ζ_vib and is
# not included.  Pure CO₂: no catalytic shortening of τ₁₁ by water.

# ── Vibrational relaxation time τ₁₁: Lin & Trusler (2014), Eqs. 3 to 7 ──
_P11_A = 9.46e-6        # collision transition probability prefactor
_P11_T = 252.0          # K, its exponential temperature scale
_SIGMA_HS = 0.355e-9    # m, hard-sphere diameter of the collision model
_A_MFP = 1.852          # Madigosky-Litovitz dense mean-free-path coefficient
_RHO_MFP_SWITCH = 896.0  # kg/m³, where the dilute and dense paths are equal
_RHO_MFP_WIDTH = 10.0   # kg/m³, width of the smooth switch between them

# ── Critical part: Onuki (2007), Appendix B, low-frequency limit ──
_R_B = 0.03             # universal amplitude, ζ_crit = R_B ρ c² t_ξ


def _vibrational_relaxation_time(T, rho):
    """Isothermal vibrational relaxation time τ₁₁ [s] at scalar (T, ρ).

    1/τ₁₁ = (v̄/λ)·P₁₁ with the mean speed v̄ = √(8RT/(πM)) and
    P₁₁ = 9.46e-6·exp(T/252 K).  The mean free path λ is the dilute
    hard-sphere 1/(√2πσ²n) below ρ = 896 kg/m³ and the Madigosky-Litovitz
    a(n^(−1/3) − σ) above it.  The two are equal at that density, and the
    switch is written as a logistic blend in ρ of width 10 kg/m³ so τ₁₁ is
    smooth (finite jvp and grad everywhere); it departs from the hard switch
    by under 1e-4 relative below 820 kg/m³.

    Note the two paths also cross near 200 kg/m³, so the switch is not
    max(λ_dil, λ_dense): between the crossings the dense path is the larger.
    """
    n = rho / M * _NA
    v_mean = jnp.sqrt(8.0 * R_MOLAR * T / (jnp.pi * M))
    lam_dilute = 1.0 / (jnp.sqrt(2.0) * jnp.pi * _SIGMA_HS ** 2 * n)
    lam_dense = _A_MFP * (n ** (-1.0 / 3.0) - _SIGMA_HS)
    w = jax.nn.sigmoid((rho - _RHO_MFP_SWITCH) / _RHO_MFP_WIDTH)
    lam = (1.0 - w) * lam_dilute + w * lam_dense
    p11 = _P11_A * jnp.exp(T / _P11_T)
    return lam / (v_mean * p11)


def _bulk_viscosity_parts(T, rho, mu, w, cp, cv, al0_tt, xi):
    """(ζ_vib, ζ_crit, τ₁₁) at scalar (T, ρ) from quantities already at hand.

    ζ_vib = ρc²(γ − 1)Δ(1 − Δ)τ₁₁ with γ = c_p/c_v and Δ = c_vib/c_p: the
    low-frequency limit of the single-relaxation-time absorption (Lin &
    Trusler 2014, Eq. 1).  c_vib = c_v⁰ − (5/2)R/M is the ideal-gas isochoric
    heat capacity less its translational and rotational parts, from the
    Span-Wagner ideal part (c_v⁰/R = −τ²α⁰_ττ).  c, c_p, c_v are the
    equilibrium EOS values.

    ζ_crit = R_B ρc² t_ξ with t_ξ = 6πμξ³/(k_B T) (Onuki 2007, B4), ξ the
    Olchowy-Sengers correlation length and μ the background viscosity
    (Laesecke-Muzny carries no critical enhancement).  ζ_crit = 0 where ξ = 0.
    """
    tau_red = TC / T
    r_mass = R_MOLAR / M
    c_vib = r_mass * (-tau_red ** 2 * al0_tt - 2.5)
    frac = c_vib / cp
    gamma = cp / cv
    rho_c2 = rho * w * w
    tau11 = _vibrational_relaxation_time(T, rho)
    zeta_vib = rho_c2 * (gamma - 1.0) * frac * (1.0 - frac) * tau11
    t_xi = 6.0 * jnp.pi * mu * xi ** 3 / (_KB * T)
    zeta_crit = _R_B * rho_c2 * t_xi
    return zeta_vib, zeta_crit, tau11


def _bulk_viscosity_shared(T, rho, mu, w, cp, cv, al0_tt, xi):
    """ζ [Pa·s] reusing quantities the fused (T, ρ) path already computed."""
    zeta_vib, zeta_crit, _ = _bulk_viscosity_parts(T, rho, mu, w, cp, cv,
                                                   al0_tt, xi)
    return zeta_vib + zeta_crit


def _scalar_bulk_viscosity(T, rho):
    """Bulk viscosity ζ [Pa·s] at scalar (T, ρ [kg/m³]).

    The equilibrium c, c_p, c_v come from the thermo kernel in ``core``
    (imported here, at trace time, since ``core`` imports this module).
    """
    from co2_eos.core import _thermo
    (_, cv, cp, w, *_), aux = _thermo(T, rho)
    ar_d, ar_dd, _, _, a0_tt = aux
    mu = _scalar_viscosity(T, rho)
    xi, _ = _correlation_length_and_excess(T, rho, ar_d, ar_dd)
    return _bulk_viscosity_shared(T, rho, mu, w, cp, cv, a0_tt, xi)


# ═══════════════════════════════════════════════════════════════════════════
# Public API: batched, JIT-compiled
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def viscosity(T, rho):
    """Dynamic viscosity μ [Pa·s]. Batched over leading axis.

    Args:
        T: temperature [K], 1-D array
        rho: mass density [kg/m³], 1-D array

    Returns:
        μ [Pa·s], same shape as inputs
    """
    return jax.vmap(_scalar_viscosity)(T, rho)


@jax.jit
def thermal_conductivity(T, rho):
    """Thermal conductivity λ [W/(m·K)]. Batched over leading axis.

    Args:
        T: temperature [K], 1-D array
        rho: mass density [kg/m³], 1-D array

    Returns:
        λ [W/(m·K)], same shape as inputs
    """
    return jax.vmap(_scalar_thermal_conductivity)(T, rho)


@jax.jit
def bulk_viscosity(T, rho):
    """Bulk viscosity ζ [Pa·s]. Batched over leading axis.

    Args:
        T: temperature [K], 1-D array
        rho: mass density [kg/m³], 1-D array

    Returns:
        ζ [Pa·s], same shape as inputs
    """
    return jax.vmap(_scalar_bulk_viscosity)(T, rho)
