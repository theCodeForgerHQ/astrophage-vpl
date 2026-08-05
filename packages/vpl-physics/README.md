# vpl-physics

The forward physics operator `F₁` of [doc 03](../../docs/03-physics-models.md): everything
that turns a set of control parameters into a `PlasmaState`.

One interface, four fidelity levels (doc 03 §1). The inverse solver never learns which one
it is talking to ([doc 00](../../docs/00-charter.md) E1).

| Module | Level | Physics | Status |
|---|---|---|---|
| `vpl.physics.analytic` | **L0** | Bohm criterion, matrix sheath, collisionless Child–Langmuir, collisional (mobility-limited) sheath | implemented |
| `vpl.physics.fluid` | L1 | drift-diffusion ions + Boltzmann electrons + Poisson, FEniCSx | not yet |
| `vpl.physics.kinetic` | L2 | 1D3V electrostatic PIC with Monte-Carlo collisions | not yet |
| `vpl.physics.surrogate` | L3 | GP / neural-operator emulator trained on L2 | not yet |

## L0 — analytic sheath models

`vpl.physics.analytic.sheath` implements doc 03 §2 verbatim. It is the framework's
verification anchor: **verification gate V-03** (doc 07) requires L1 and L2 to reproduce
its `s` and `J_i` to within 5 % in the collisionless high-bias limit, and doc 03 §2.3's
`Γ_E ≈ 6.6 kW·m⁻²` at the reference operating point RP-1 is "the number every other model
must reproduce in this limit".

```python
from vpl.core.state import PlasmaParams, Species
from vpl.core.units import Q_
from vpl.physics.analytic import AnalyticSheathSolver, ion_energy_flux

params = PlasmaParams(
    species=Species(name="Ar+", mass=Q_(39.948, "u"), charge_number=1),
    n_0=Q_(1e17, "m**-3"),
    T_e=Q_(3.0, "eV"),
    T_i=Q_(0.05, "eV"),
    T_g=Q_(300.0, "K"),
    pressure=Q_(5.0, "mTorr"),
    bias=Q_(-250.0, "V"),
    gamma_se=0.10,
    kappa=1.0,
)

ion_energy_flux(params).to("kW/m**2")  # 6.58 kW/m^2
AnalyticSheathSolver().solve(params)  # PlasmaState(L0, 101 points, steady, 5 fields, no IVDF)
```

### Two discrepancies in the Baseline specification, surfaced not resolved

Doc 00 C4 forbids hidden assumptions, so both of these are exposed as explicit arguments
with documented defaults rather than picked silently. See the module docstring of
`vpl.physics.analytic.sheath` for the full statement.

1. **`γ_i` in the Bohm speed.** Doc 03 §2.1 writes `c_s = √(e(T_e + γ_i T_i)/m_i)` with
   `γ_i = 3`; doc 01 §2.2 quotes `c_s = √(e T_e/m_i) = 2.69 km/s`, which is `γ_i = 0`. The
   default is `γ_i = 0`, reproducing the tabulated number. At RP-1 the two differ by 2.5 %.
2. **Which density sets `λ_D` in the Child–Langmuir thickness.** Doc 01 §2.2 evaluates
   `λ_D` at `n_0` and gets `s ≈ 0.89 mm`. The Bohm-flux matching that *derives* that
   formula requires `λ_D` at the sheath-edge density `n_s = h_l n_0`, which gives
   `1.14 mm`. The default reproduces doc 01; passing `h_l` gives the self-consistent value.

## Testing

```bash
uv run pytest packages/vpl-physics -m physics     # verification tests
uv run pytest packages/vpl-physics -m "not physics"
```
