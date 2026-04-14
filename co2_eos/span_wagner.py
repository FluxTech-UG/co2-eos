"""Span-Wagner (1996) equation of state for CO₂ — pure JAX implementation.

Implements the complete reduced Helmholtz energy α(τ,δ) = α⁰(τ,δ) + αʳ(τ,δ)
from: Span & Wagner, J. Phys. Chem. Ref. Data 25, 1509-1596 (1996).

Coefficients sourced from the original paper and cross-referenced against
CoolProp's machine-readable fluids/CarbonDioxide.json.

All thermodynamic properties are derived via jax.grad — no hand-coded
derivative formulas. CoolProp is never imported here.
"""

import jax
import jax.numpy as jnp
from functools import partial

# Enable float64
jax.config.update("jax_enable_x64", True)

# ── Critical constants ──────────────────────────────────────────────────────
TC = 304.1282          # K
RHOC_MOLAR = 10624.9063  # mol/m³
M = 0.0440098          # kg/mol
RHOC = RHOC_MOLAR * M  # kg/m³  ≈ 467.60
R_MOLAR = 8.31451      # J/(mol·K)  (value used in Span-Wagner)
R = R_MOLAR / M        # J/(kg·K)   ≈ 188.9241
PC = 7377300.0          # Pa
T_TRIPLE = 216.592      # K  — CO₂ triple-point temperature


# ═══════════════════════════════════════════════════════════════════════════
# Ideal-gas Helmholtz energy  α⁰(τ, δ)
# ═══════════════════════════════════════════════════════════════════════════
# α⁰ = a1 + a2·τ + (a_logtau - 1)·ln(τ) + ln(δ)
#       + Σ nk · ln(1 - exp(-θk·τ))
#
# The (a_logtau - 1) form comes from CoolProp's convention where the log
# coefficient is the cp0/R ideal-gas heat capacity constant.  For CO₂,
# a_logtau = 2.5  ⟹  coefficient of ln(τ) is 1.5.

# Lead terms (includes IIR enthalpy/entropy offset: a1_off=-14.4979156224319,
# a2_off=8.82013935801453 already folded into a1, a2)
_A0_A1 = 8.37304456 + (-14.4979156224319)
_A0_A2 = -3.70454304 + 8.82013935801453
_A0_LOGTAU = 2.5       # coefficient of ln(τ) — CoolProp IdealGasHelmholtzLogTau

_A0_PE_N = jnp.array([1.99427042, 0.62105248, 0.41195293,
                       1.04028922, 0.08327678])
_A0_PE_T = jnp.array([3.15163, 6.1119, 6.77708, 11.32384, 27.08792])


def alpha0(tau, delta):
    """Ideal-gas reduced Helmholtz energy α⁰(τ, δ).

    Args:
        tau: reduced inverse temperature Tc/T  (scalar)
        delta: reduced density ρ/ρc  (scalar)

    Returns:
        α⁰ value (scalar)
    """
    result = _A0_A1 + _A0_A2 * tau + _A0_LOGTAU * jnp.log(tau) + jnp.log(delta)
    # Planck-Einstein terms
    result += jnp.sum(_A0_PE_N * jnp.log(1.0 - jnp.exp(-_A0_PE_T * tau)))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Residual Helmholtz energy  αʳ(τ, δ)
# ═══════════════════════════════════════════════════════════════════════════

# ── Polynomial + exponential terms (34 terms total) ─────────────────────
# Terms 1-7: l=0 (pure polynomial)
# Terms 8-34: l>0 (exponential: multiply by exp(-δ^l))

_AR_N = jnp.array([
    0.388568232032,    2.93854759427,    -5.5867188535,
   -0.767531995925,    0.317290055804,    0.548033158978,
    0.122794112203,    2.16589615432,     1.58417351097,
   -0.231327054055,    0.0581169164314,  -0.553691372054,
    0.489466159094,   -0.0242757398435,   0.0624947905017,
   -0.121758602252,   -0.370556852701,   -0.0167758797004,
   -0.11960736638,    -0.0456193625088,   0.0356127892703,
   -0.00744277271321, -0.00173957049024, -0.0218101212895,
    0.0243321665592,  -0.0374401334235,   0.143387157569,
   -0.134919690833,   -0.0231512250535,   0.0123631254929,
    0.00210583219729, -0.000339585190264, 0.00559936517716,
   -0.000303351180556,
])

_AR_D = jnp.array([
    1, 1, 1, 1, 2, 2, 3,
    1, 2, 4, 5, 5, 5, 6, 6, 6,
    1, 1, 4, 4, 4, 7, 8,
    2, 3, 3, 5, 5, 6, 7, 8, 10,
    4, 8,
], dtype=jnp.float64)

_AR_T = jnp.array([
    0.0, 0.75, 1.0, 2.0, 0.75, 2.0, 0.75,
    1.5, 1.5, 2.5, 0.0, 1.5, 2.0, 0.0, 1.0, 2.0,
    3.0, 6.0, 3.0, 6.0, 8.0, 6.0, 0.0,
    7.0, 12.0, 16.0, 22.0, 24.0, 16.0, 24.0, 8.0, 2.0,
    28.0, 14.0,
])

_AR_L = jnp.array([
    0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 4, 4, 4, 4, 4, 4,
    5, 6,
], dtype=jnp.float64)

# ── Gaussian bell-shaped terms (5 terms, indices 35-39) ─────────────────

_AR_GAUSS_N = jnp.array([
    -213.654886883, 26641.5691493, -24027.2122046,
    -283.41603424, 212.472844002,
])

_AR_GAUSS_D = jnp.array([2, 2, 2, 3, 3], dtype=jnp.float64)
_AR_GAUSS_T = jnp.array([1.0, 0.0, 1.0, 3.0, 3.0])

_AR_GAUSS_ETA = jnp.array([25.0, 25.0, 25.0, 15.0, 20.0])
_AR_GAUSS_BETA = jnp.array([325.0, 300.0, 300.0, 275.0, 275.0])
_AR_GAUSS_GAMMA = jnp.array([1.16, 1.19, 1.19, 1.25, 1.22])
_AR_GAUSS_EPS = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])

# ── Non-analytic terms (3 terms, indices 40-42) ─────────────────────────

_NA_N = jnp.array([-0.666422765408, 0.726086323499, 0.0550686686128])
_NA_A = jnp.array([3.5, 3.5, 3.0])
_NA_B = jnp.array([0.875, 0.925, 0.875])
_NA_BETA = jnp.array([0.3, 0.3, 0.3])
_NA_BIG_A = jnp.array([0.7, 0.7, 0.7])
_NA_BIG_B = jnp.array([0.3, 0.3, 1.0])
_NA_BIG_C = jnp.array([10.0, 10.0, 12.5])
_NA_BIG_D = jnp.array([275.0, 275.0, 275.0])


def alphar(tau, delta):
    """Residual reduced Helmholtz energy αʳ(τ, δ).

    Args:
        tau: reduced inverse temperature Tc/T  (scalar)
        delta: reduced density ρ/ρc  (scalar)

    Returns:
        αʳ value (scalar)
    """
    # ── Polynomial + exponential terms ──
    # αʳ_poly = Σ nk · δ^dk · τ^tk · exp(-δ^lk)
    # For lk = 0, the exp factor is 1.
    powers = delta ** _AR_D * tau ** _AR_T
    # exp(-δ^l) factor: when l=0, δ^0=1, exp(-1)≠1, so we need a mask
    exp_factor = jnp.where(_AR_L > 0, jnp.exp(-delta ** _AR_L), 1.0)
    poly_exp = jnp.sum(_AR_N * powers * exp_factor)

    # ── Gaussian bell-shaped terms ──
    # αʳ_gauss = Σ nk · δ^dk · τ^tk · exp(-ηk(δ-εk)² - βk(τ-γk)²)
    gauss_powers = delta ** _AR_GAUSS_D * tau ** _AR_GAUSS_T
    gauss_exp = jnp.exp(
        -_AR_GAUSS_ETA * (delta - _AR_GAUSS_EPS) ** 2
        - _AR_GAUSS_BETA * (tau - _AR_GAUSS_GAMMA) ** 2
    )
    gauss = jnp.sum(_AR_GAUSS_N * gauss_powers * gauss_exp)

    # ── Non-analytic terms ──
    # θ = (1 - τ) + A·[(δ-1)²]^(1/(2β))
    # Δ = θ² + B·[(δ-1)²]^a
    # Ψ = exp(-C·(δ-1)² - D·(τ-1)²)
    # αʳ_na = Σ nk · Δ^bk · δ · Ψ
    dm1_sq = (delta - 1.0) ** 2
    theta = (1.0 - tau) + _NA_BIG_A * (dm1_sq ** (1.0 / (2.0 * _NA_BETA)))
    big_delta = theta ** 2 + _NA_BIG_B * (dm1_sq ** _NA_A)
    psi = jnp.exp(-_NA_BIG_C * dm1_sq - _NA_BIG_D * (tau - 1.0) ** 2)

    # Δ^b: use safe power to handle Δ=0 at exact critical point
    big_delta_safe = jnp.maximum(big_delta, 1e-300)
    delta_b = big_delta_safe ** _NA_B

    na = jnp.sum(_NA_N * delta_b * delta * psi)

    return poly_exp + gauss + na


def alpha(tau, delta):
    """Total reduced Helmholtz energy α(τ, δ) = α⁰ + αʳ."""
    return alpha0(tau, delta) + alphar(tau, delta)


# ═══════════════════════════════════════════════════════════════════════════
# Physical property functions  (all via autodiff from α)
# ═══════════════════════════════════════════════════════════════════════════
#
# Working in reduced variables throughout:
#   τ = Tc/T,  δ = ρ/ρc
#
# The Helmholtz free energy per unit mass is:
#   A(T, ρ) = R·T·α(τ, δ)
#
# All properties below take (T, ρ) in physical units and return physical
# units.  Derivatives of α w.r.t. τ and δ are obtained via jax.grad.

# ── Derivative building blocks ──────────────────────────────────────────

def _alpha_of_T_rho(T, rho):
    """α(τ(T), δ(ρ)) as a function of physical T, ρ — for autodiff."""
    tau = TC / T
    delta = rho / RHOC
    return alpha(tau, delta)


def _alphar_of_T_rho(T, rho):
    """αʳ(τ(T), δ(ρ)) as a function of physical T, ρ."""
    tau = TC / T
    delta = rho / RHOC
    return alphar(tau, delta)


# Reduced-variable derivatives of αʳ (used in property formulas)
_dalr_ddelta = jax.grad(alphar, argnums=1)
_dalr_dtau = jax.grad(alphar, argnums=0)
_d2alr_ddelta2 = jax.grad(lambda tau, delta: _dalr_ddelta(tau, delta), argnums=1)
_d2alr_dtau2 = jax.grad(lambda tau, delta: _dalr_dtau(tau, delta), argnums=0)
_d2alr_ddelta_dtau = jax.grad(lambda tau, delta: _dalr_ddelta(tau, delta), argnums=0)

# Reduced-variable derivatives of α⁰
_dal0_ddelta = jax.grad(alpha0, argnums=1)
_dal0_dtau = jax.grad(alpha0, argnums=0)
_d2al0_dtau2 = jax.grad(lambda tau, delta: jax.grad(alpha0, argnums=0)(tau, delta), argnums=0)


# ── Scalar property functions (τ, δ inputs) ────────────────────────────

def _pressure_reduced(tau, delta):
    """P / (ρ·R·T) = 1 + δ·αʳ_δ."""
    return 1.0 + delta * _dalr_ddelta(tau, delta)


def _cv_reduced(tau, delta):
    """Cv / R = -τ²·(α⁰_ττ + αʳ_ττ)."""
    return -tau ** 2 * (_d2al0_dtau2(tau, delta) + _d2alr_dtau2(tau, delta))


def _cp_reduced(tau, delta):
    """Cp / R = Cv/R + (1 + δ·αʳ_δ - δ·τ·αʳ_δτ)² / (1 + 2δ·αʳ_δ + δ²·αʳ_δδ)."""
    cv_r = _cv_reduced(tau, delta)
    alr_d = _dalr_ddelta(tau, delta)
    alr_dd = _d2alr_ddelta2(tau, delta)
    alr_dt = _d2alr_ddelta_dtau(tau, delta)

    num = (1.0 + delta * alr_d - delta * tau * alr_dt) ** 2
    den = 1.0 + 2.0 * delta * alr_d + delta ** 2 * alr_dd
    return cv_r + num / den


def _speed_of_sound_sq_reduced(tau, delta):
    """w² / (R·T) = 1 + 2δ·αʳ_δ + δ²·αʳ_δδ
                     + (1 + δ·αʳ_δ - δ·τ·αʳ_δτ)² / (τ²·(α⁰_ττ + αʳ_ττ))
    Note: the denominator in the last term is -Cv/R (positive for stable fluids).
    """
    alr_d = _dalr_ddelta(tau, delta)
    alr_dd = _d2alr_ddelta2(tau, delta)
    alr_dt = _d2alr_ddelta_dtau(tau, delta)
    al0_tt = _d2al0_dtau2(tau, delta)
    alr_tt = _d2alr_dtau2(tau, delta)

    mechanical = 1.0 + 2.0 * delta * alr_d + delta ** 2 * alr_dd
    num = (1.0 + delta * alr_d - delta * tau * alr_dt) ** 2
    den = tau ** 2 * (al0_tt + alr_tt)  # this is -Cv/R, negative
    return mechanical - num / den  # minus minus = plus


# ── Physical property functions (T, ρ inputs → physical units) ──────────

def _scalar_pressure(T, rho):
    """Pressure [Pa] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    return rho * R * T * _pressure_reduced(tau, delta)


def _scalar_cv(T, rho):
    """Isochoric heat capacity Cv [J/(kg·K)] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    return R * _cv_reduced(tau, delta)


def _scalar_cp(T, rho):
    """Isobaric heat capacity Cp [J/(kg·K)] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    return R * _cp_reduced(tau, delta)


def _scalar_speed_of_sound(T, rho):
    """Speed of sound [m/s] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    w_sq = R * T * _speed_of_sound_sq_reduced(tau, delta)
    return jnp.sqrt(jnp.maximum(w_sq, 0.0))


def _scalar_enthalpy(T, rho):
    """Specific enthalpy h [J/kg] at scalar (T, ρ).

    h = R·T·(τ·(α⁰_τ + αʳ_τ) + δ·αʳ_δ + 1)
    """
    tau = TC / T
    delta = rho / RHOC
    al0_t = _dal0_dtau(tau, delta)
    alr_t = _dalr_dtau(tau, delta)
    alr_d = _dalr_ddelta(tau, delta)
    return R * T * (tau * (al0_t + alr_t) + delta * alr_d + 1.0)


def _scalar_internal_energy(T, rho):
    """Specific internal energy u [J/kg] at scalar (T, ρ).

    u = R·T·τ·(α⁰_τ + αʳ_τ)
    """
    tau = TC / T
    delta = rho / RHOC
    al0_t = _dal0_dtau(tau, delta)
    alr_t = _dalr_dtau(tau, delta)
    return R * T * tau * (al0_t + alr_t)


def _scalar_entropy(T, rho):
    """Specific entropy s [J/(kg·K)] at scalar (T, ρ).

    s = R·(τ·(α⁰_τ + αʳ_τ) - α⁰ - αʳ)
    """
    tau = TC / T
    delta = rho / RHOC
    al0_t = _dal0_dtau(tau, delta)
    alr_t = _dalr_dtau(tau, delta)
    al0_val = alpha0(tau, delta)
    alr_val = alphar(tau, delta)
    return R * (tau * (al0_t + alr_t) - al0_val - alr_val)


# ═══════════════════════════════════════════════════════════════════════════
# Public API — batched, JIT-compiled, vmappable
# ═══════════════════════════════════════════════════════════════════════════

@jax.jit
def pressure(T, rho):
    """Pressure [Pa]. Batched over leading axis."""
    return jax.vmap(_scalar_pressure)(T, rho)


@jax.jit
def cv(T, rho):
    """Isochoric heat capacity Cv [J/(kg·K)]. Batched."""
    return jax.vmap(_scalar_cv)(T, rho)


@jax.jit
def cp(T, rho):
    """Isobaric heat capacity Cp [J/(kg·K)]. Batched."""
    return jax.vmap(_scalar_cp)(T, rho)


@jax.jit
def speed_of_sound(T, rho):
    """Speed of sound [m/s]. Batched."""
    return jax.vmap(_scalar_speed_of_sound)(T, rho)


@jax.jit
def enthalpy(T, rho):
    """Specific enthalpy h [J/kg]. Batched."""
    return jax.vmap(_scalar_enthalpy)(T, rho)


@jax.jit
def internal_energy(T, rho):
    """Specific internal energy u [J/kg]. Batched."""
    return jax.vmap(_scalar_internal_energy)(T, rho)


@jax.jit
def entropy(T, rho):
    """Specific entropy s [J/(kg·K)]. Batched."""
    return jax.vmap(_scalar_entropy)(T, rho)


@jax.jit
def all_properties(T, rho):
    """Compute all properties at once. Returns dict.

    Keys: P [Pa], Cv [J/(kg·K)], Cp [J/(kg·K)], w [m/s],
          h [J/kg], u [J/kg], s [J/(kg·K)]
    """
    return {
        "P": pressure(T, rho),
        "Cv": cv(T, rho),
        "Cp": cp(T, rho),
        "w": speed_of_sound(T, rho),
        "h": enthalpy(T, rho),
        "u": internal_energy(T, rho),
        "s": entropy(T, rho),
    }


# ── Gibbs energy ─────────────────────────────────────────────────────────

def _scalar_gibbs_energy(T, rho):
    """Specific Gibbs energy g = h - T·s [J/kg] at scalar (T, ρ)."""
    tau = TC / T
    delta = rho / RHOC
    al0_t = _dal0_dtau(tau, delta)
    alr_t = _dalr_dtau(tau, delta)
    alr_d = _dalr_ddelta(tau, delta)
    al0_val = alpha0(tau, delta)
    alr_val = alphar(tau, delta)
    # g = h - T·s = R·T·(1 + δ·αʳ_δ + α⁰ + αʳ)
    return R * T * (1.0 + delta * alr_d + al0_val + alr_val)


@jax.jit
def gibbs_energy(T, rho):
    """Specific Gibbs energy g [J/kg]. Batched."""
    return jax.vmap(_scalar_gibbs_energy)(T, rho)


# ═══════════════════════════════════════════════════════════════════════════
# Shared-derivative property bundle (simulation hot path)
# ═══════════════════════════════════════════════════════════════════════════

def _scalar_all_properties_shared(T, rho):
    """Compute all 7 properties + Gibbs energy from 5 shared reduced derivatives.

    Computes αʳ_δ, αʳ_ττ, αʳ_δδ, αʳ_δτ, α⁰_ττ once, then derives everything.
    """
    tau = TC / T
    delta = rho / RHOC

    # ── Evaluate the 5 reduced derivatives (+ values needed for h, s, g) ──
    alr_d = _dalr_ddelta(tau, delta)
    alr_tt = _d2alr_dtau2(tau, delta)
    alr_dd = _d2alr_ddelta2(tau, delta)
    alr_dt = _d2alr_ddelta_dtau(tau, delta)
    al0_tt = _d2al0_dtau2(tau, delta)

    # For enthalpy, entropy, Gibbs: also need τ-first-derivatives and values
    al0_t = _dal0_dtau(tau, delta)
    alr_t = _dalr_dtau(tau, delta)
    al0_val = alpha0(tau, delta)
    alr_val = alphar(tau, delta)

    # ── Derive all properties from shared derivatives ──

    # P = ρ·R·T·(1 + δ·αʳ_δ)
    P = rho * R * T * (1.0 + delta * alr_d)

    # Cv = -R·τ²·(α⁰_ττ + αʳ_ττ)
    cv_val = -R * tau ** 2 * (al0_tt + alr_tt)

    # Cp numerator/denominator terms (reused for speed of sound)
    num = (1.0 + delta * alr_d - delta * tau * alr_dt) ** 2
    den = 1.0 + 2.0 * delta * alr_d + delta ** 2 * alr_dd

    # Cp = Cv + R·num/den
    cp_val = cv_val + R * num / den

    # w² = R·T·(den - num / (τ²·(α⁰_ττ + αʳ_ττ)))
    w_sq = R * T * (den - num / (tau ** 2 * (al0_tt + alr_tt)))
    w_val = jnp.sqrt(jnp.maximum(w_sq, 0.0))

    # h = R·T·(τ·(α⁰_τ + αʳ_τ) + δ·αʳ_δ + 1)
    h_val = R * T * (tau * (al0_t + alr_t) + delta * alr_d + 1.0)

    # u = R·T·τ·(α⁰_τ + αʳ_τ)
    u_val = R * T * tau * (al0_t + alr_t)

    # s = R·(τ·(α⁰_τ + αʳ_τ) - α⁰ - αʳ)
    s_val = R * (tau * (al0_t + alr_t) - al0_val - alr_val)

    # g = h - T·s = R·T·(1 + δ·αʳ_δ + α⁰ + αʳ)
    g_val = R * T * (1.0 + delta * alr_d + al0_val + alr_val)

    return P, cv_val, cp_val, w_val, h_val, u_val, s_val, g_val


@jax.jit
def all_properties_shared(T, rho):
    """Compute all properties from shared reduced derivatives. Returns dict.

    Equivalent to all_properties() but computes the 5 key reduced derivatives
    once and reuses them, avoiding redundant autodiff evaluations.

    Keys: P [Pa], Cv [J/(kg·K)], Cp [J/(kg·K)], w [m/s],
          h [J/kg], u [J/kg], s [J/(kg·K)], g [J/kg]
    """
    P, Cv, Cp, w, h, u, s, g = jax.vmap(_scalar_all_properties_shared)(T, rho)
    return {"P": P, "Cv": Cv, "Cp": Cp, "w": w, "h": h, "u": u, "s": s, "g": g}
