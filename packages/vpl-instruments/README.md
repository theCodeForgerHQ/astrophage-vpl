# vpl-instruments

The measurement operator `F₂` of [doc 04](../../docs/04-measurement-models.md): plasma
state in, observable out.

```
x (PlasmaState)
  └─► F₂  emission response      ← this package
        └─► F₃  optical transport      (vpl-optics)
              └─► F₄  detection        (vpl-detectors)
                    └─► y (Measurement)
```

## What is here

`vpl.instruments.oes` — doc 04 §2 and §5.

| Module | Doc section | What it does |
|---|---|---|
| `levels` | §2.2 | The level system: levels, electron-impact channels, radiative channels |
| `cr` | §2.2 | The quasi-static collisional-radiative solve for `n_u` |
| `escape` | §2.3 | Holstein–Biberman escape factors `Λ_ul` |
| `lineshape` | §2.4 | Doppler, natural and Voigt profiles |
| `emissivity` | §2.1 | `ε_ul = n_u A_ul h ν_ul / 4π`, and the line-of-sight radiance |
| `spectrograph` | §5, §6.1 | Grating dispersion, finite slit, instrument function |
| `instrument` | §9 | `OesInstrument`, implementing the doc 08 §4 `Instrument` protocol |

## What is deliberately not here

Named up front rather than discovered later (doc 00 C4). Each module docstring repeats its
own omissions with a bound; this is the index.

- **Ray tracing.** doc 04 §6.3 buys Raysect for `F₃`; this package integrates the
  emissivity along a straight chord of the doc 02 §3.1 chamber and states the error that
  costs. Vignetting, aberration and the depth-of-field weighting of doc 04 §6.2 are
  `vpl-optics`.
- **The detector chain.** doc 04 §7's eighteen noise sources are `vpl-detectors`.
  `OesInstrument.observe` applies photon shot noise (N1) and the imperfect radiometric
  calibration of doc 04 §7.3, and nothing else.
- **Ar II lines.** The CR model is written for any level system; the four Ar II lines of
  doc 02 §6.3 need an ion-stage CR model with a Saha-like coupling to Ar I, which is not
  implemented.
- **Stark and Zeeman broadening.** doc 01 §4.2 makes Stark negligible here; Zeeman is
  modelled in the LIF channel (doc 04 §3.3), not in OES.

## Verification

`tests/` is where the claims are. The externally-anchored ones:

| Test | Anchor |
|---|---|
| `test_escape.py::test_monochromatic_escape_matches_exponential_integral` | A brute-force position/angle double quadrature written in the test, against the closed form `(1 − 2E₃(τ))/2τ` the module uses |
| `test_escape.py::test_lorentz_escape_reaches_its_analytic_thick_limit` | `Λ√τ → 4/(3√π)`, derived in `escape.py`. Reproduced to 8 digits |
| `test_escape.py::test_doppler_escape_converges_to_its_analytic_thick_limit` | `Λ → √(ln τ)/(τ√π)`, same derivation |
| `test_escape.py::test_escape_factor_tends_to_one_as_the_gas_thins` | doc 04 §8 **V-26** |
| `test_lineshape.py::test_voigt_fwhm_matches_olivero_longbothum` | Olivero & Longbothum (1977), stated accurate to 0.02 %, measured off the profile |
| `test_lineshape.py::test_doppler_width_matches_the_textbook_coefficient` | The `7.1623e-7 √(T/M)` prefactor |
| `test_cr.py::test_de_excitation_obeys_detailed_balance_for_a_maxwellian` | Klein–Rosseland, analytic, exact in the discrete sense |
| `test_cr.py::test_high_density_limit_is_boltzmann` | doc 04 §8 **V-25**, LTE limit |
| `test_cr.py::test_low_density_limit_is_corona` | doc 04 §8 **V-25**, corona limit |
| `test_spectrograph.py::test_reciprocal_dispersion_matches_the_oes_s3_specification` | doc 02 OES-S3, 0.62 nm/mm |
| `test_spectrograph.py::test_angular_dispersion_matches_a_numerical_derivative` | Finite difference of the grating equation. **This one found a real 2.1× bug** |
| `test_instrument.py::test_observe_and_forward_agree_exactly_...` | doc 04 §9's one-code-path requirement |
| `test_instrument.py::test_the_instrument_layer_does_not_import_the_inverse_layer` | doc 08 §1 principle 5, against the import graph |

**Not verified: doc 04 §8 V-24, "CR model vs published Ar line ratios".** No number in
this package has been compared against a real argon measurement or against a published
argon CR calculation. Every anchor above is an analytic limit, an independent numerical
route, or a row of doc 02 §6.2 — all of which check that the equations are solved
correctly, and none of which check that they are the right equations for argon (doc 07 §1).
The blocker is that doc 09 §5 keeps bulk atomic data out of the repository, so there is no
Ar level system here to run a benchmark on; closing V-24 needs an LXCat/NIST fetch plus a
transcribed table from a named paper, in the style of
`vpl-validation/src/vpl/validation/data/swarm-benchmarks.yaml`.
