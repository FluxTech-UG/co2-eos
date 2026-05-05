# CO2-EOS

Differentiable CO₂ thermodynamic properties in JAX.

CO2-EOS is a pure-JAX implementation of the Span-Wagner equation of state for carbon dioxide. Every property evaluation is JIT-compiled, vectorisable with `vmap`, and fully differentiable with `jax.grad` — enabling gradient-based optimisation of any system that depends on CO₂ thermodynamics.

```python
import co2_eos as co2
import jax

# Properties from (T, ρ)
state = co2.properties(T=310.0, rho=350.0)
print(state['pressure'], state['cp'], state['speed_of_sound'])

# State from (P, T) — phase-aware, robust near the critical point
state = co2.state_from_PT(P=8e6, T=310.0)

# Differentiate anything — e.g. ∂ρ/∂T at constant P
# (peaks near the Widom line at these conditions)
drho_dT = jax.grad(lambda T: co2.state_from_PT(P=8e6, T=T)['density'])(310.0)

# Vectorise across conditions
import jax.numpy as jnp
T_array = jnp.linspace(280, 340, 100)
states = jax.vmap(lambda T: co2.state_from_PT(P=8e6, T=T))(T_array)
```

## Why this exists

If you're building CO₂ system models in Python and need thermodynamic properties, CoolProp is the standard choice. It's excellent — accurate, well-tested, and covers 110+ fluids.

But CoolProp is a C++ library with Python bindings. You can't `jax.grad` through it. You can't JIT-compile a simulation loop that calls it. And the per-call overhead from the Python↔C++ boundary adds up fast in tight integration loops.

CO2-EOS solves this for CO₂ by implementing the same reference EOS (Span-Wagner 1996) directly in JAX:

- **Differentiable.** `jax.grad` through any property, any inversion, any combination. No finite differences. Exact gradients via autodiff, including second and higher derivatives for free.
- **Fast.** JIT-compiled evaluation of all thermodynamic properties in ~1.8 μs/point — roughly 185× faster than CoolProp's Python interface.
- **Composable.** `jit`, `vmap`, `grad`, `custom_vjp` — the full JAX transformation stack works. Embed property evaluations inside your own JIT-compiled simulation and differentiate end-to-end.
- **Phase-aware.** Robust inversions near the critical point using Halley's method with step damping and bisection fallback. Two-phase dome detection that avoids convergence to thermodynamically unstable spinodal states.

## What's included

**Equation of state** — Full Span-Wagner (1996) formulation for CO₂. Helmholtz free energy as A(T, ρ), with all thermodynamic properties derived by autodiff:

- Pressure, internal energy, enthalpy, entropy
- Isochoric and isobaric heat capacities (Cv, Cp)
- Speed of sound
- Gibbs energy

**Transport properties:**

- Viscosity — Laesecke & Muzny (2017), including dilute-gas, initial-density, and residual contributions, plus critical enhancement
- Thermal conductivity — Huber, Sykioti, Assael & Perkins (2016), with critical enhancement via the simplified crossover model

**Saturation curve:**

- Saturation pressure, densities, enthalpies, and entropies as functions of T or P
- Cubic spline interpolation of a precomputed Maxwell-construction table, JIT-compilable and differentiable

**State inversions:**

- `state_from_PT(P, T)` — Halley's method with pressure-aware critical-region initial guess, step damping, and bisection safety net
- `state_from_Ph(P, h, phase_hint=)` — Phase-aware: detects the two-phase dome at subcritical pressures and returns saturation properties directly, bypassing single-phase Newton iteration inside the dome
- `state_from_Du(rho, u)` — For simulation codes carrying (ρ, u) as conserved variables. Uses `custom_vjp` for clean gradient propagation through the implicit solve

## Valid range

CO2-EOS covers the full fluid region of the Span-Wagner EOS:

- **Temperature:** 216.6 K (triple point) to 1100 K
- **Pressure:** up to 800 MPa
- **Saturation curve:** triple point to within 1 mK of the critical point (304.1282 K, 7.3773 MPa)

Transport properties follow the valid ranges of their respective correlations (broadly: gas and liquid up to 100 MPa, with reduced accuracy in the immediate critical region for thermal conductivity).

## Installation

```bash
pip install co2-eos
```

Requires Python ≥ 3.10 and JAX ≥ 0.4. No other dependencies.

For development:

```bash
git clone https://github.com/John-FluxTech/co2-eos.git
cd co2-eos
pip install -e ".[dev]"
```

The `[dev]` extra adds CoolProp (for validation tests only) and pytest.

## Validation

CO2-EOS is validated against CoolProp across the full valid range. The test suite covers:

- Single-phase properties on a dense (T, ρ) grid spanning gas, liquid, and supercritical regions
- Saturation curve properties from triple point to near-critical
- Transport properties (viscosity and thermal conductivity) across all phases
- Inversions (P,T → ρ), (P,h → state), (ρ,u → state) including near-critical conditions
- Comparison of all Helmholtz derivatives against CoolProp's analytical values

Maximum relative errors against CoolProp are below 10⁻¹⁰ for thermodynamic properties (limited by floating-point evaluation order, not by formulation differences — both implement the same Span-Wagner polynomial).

Run the validation suite:

```bash
pytest tests/ -v
```

## Scope and philosophy

CO2-EOS is a CO₂-first library. Rather than expanding to cover more fluids, we go deeper on CO₂ — improving accuracy, coverage, and robustness in the regions that matter for real engineering: near-critical, transcritical, and two-phase.

There are excellent general-purpose thermodynamic libraries: CoolProp for broad fluid coverage, teqp for autodiff-native multi-model EOS evaluation, FeOs for SAFT models and density functional theory, and Clapeyron.jl in the Julia ecosystem. CO2-EOS complements these by providing a JAX-native path for the single fluid where differentiability, speed, and near-critical robustness matter most.

Contributions that extend CO₂ capability are welcome:

- Better transport property models as new correlations are published
- Improved near-critical formulations (crossover EOS, scaled equations)
- Mixture models where CO₂ is the primary component (CO₂ + impurities for CCS, CO₂ + lubricant for heat pumps)
- Fitting to experimental data for conditions where Span-Wagner accuracy is insufficient

## Physical references

The implementations follow these published formulations:

- **EOS:** Span, R. and Wagner, W. (1996). "A New Equation of State for Carbon Dioxide." *J. Phys. Chem. Ref. Data*, 25(6), 1509–1596.
- **Viscosity:** Laesecke, A. and Muzny, C. D. (2017). "Reference Correlation for the Viscosity of Carbon Dioxide." *J. Phys. Chem. Ref. Data*, 46, 013107.
- **Thermal conductivity:** Huber, M. L., Sykioti, E. A., Assael, M. J., and Perkins, R. A. (2016). "Reference Correlation of the Thermal Conductivity of Carbon Dioxide from the Triple Point to 1100 K and up to 200 MPa." *J. Phys. Chem. Ref. Data*, 45, 013102.

Solver design is informed by:

- Bell, I. H. et al. (2014). CoolProp solver architecture — phase-aware initial guesses and fallback chains.
- Bell, I. H., Deiters, U. K., and Leal, A. M. M. (2022). "Implementing an Equation of State without Derivatives: teqp." *Ind. Eng. Chem. Res.*, 61(17), 6010–6027.
- Bell, I. H. and Alpert, B. K. (2018). "Exceptionally Reliable Density-Solving Algorithms." *Fluid Phase Equilibria*, 477, 87–97.

## Acknowledgements

CO2-EOS was developed by Claude and John Patrick Therrien during research on transcritical CO₂ thermoacoustic engines at FluxTech UG in Berlin. The Helmholtz-first autodiff design was validated against CoolProp and informed by the solver strategies documented in teqp and the CoolProp codebase.

## License

Apache-2.0
