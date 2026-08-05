# ADR-011 — The two-term solver omitted the inelastic contribution to momentum transfer

**Status:** Accepted · **Date:** 2026-08-05 · **Amends:** [ADR-009](ADR-009-eedf-solver.md) ·
**Closes:** doc 11 WBS 1.8's acceptance criterion

## Context

Doc 11 WBS 1.8's acceptance criterion is that rate coefficients **reproduce published
values**. Gate G-1 recorded WBS 1.8 as done on the strength of tests that were all
*internal*: the Maxwellian and Druyvesteyn limits, the Einstein relation, a closed-form
mobility, and second-order convergence of the solver's own discretisation.

Those tests pass against a solver that is wrong, provided it is wrong consistently. This
ADR records what happened when the published half of the criterion was finally supplied.

## What the benchmark found

The standard model gases for verifying an electron Boltzmann code are the **Reid ramp**
(Reid 1979) and **Lucas-Saelee** (Lucas & Saelee 1975). Run against Reid's own tabulated
two-term column:

| `E/N` | `<eps>` | `W` | `N D_T` |
|---|---|---|---|
| 1 Td | +0.05 % | **+1.5 %** | +0.5 % |
| 12 Td | **+9.3 %** | **+28.4 %** | **+24.7 %** |
| 24 Td | **+21.6 %** | **+52.0 %** | **+54.4 %** |

A refinement study ruled out discretisation first: the answers were stable to 0.01 %
across grids from 1 000 to 6 000 cells and domains from 2 to 12 eV. The disagreement was
physics.

## Cause

Reid's §2 derives the momentum-conservation equation with elastic *and* inelastic
collisions present and shows it retains the elastic form only if `sigma_m,e` is replaced
by an equivalent cross section. For isotropic inelastic scattering that reduces to his
eq. (4):

```
sigma_m(eps) = sigma_m,e(eps) + sum_k sigma_k(eps)
```

An inelastic collision randomises the electron's direction just as an elastic one does; it
impedes the drift whatever it does to the energy. Reid then says, in the paper's own
italics of emphasis:

> if equation (5b) is used to determine `f(eps)`, the momentum transfer cross section that
> appears in the equation is the **total** cross section for momentum transfer (as given by
> equation 4) and **not** the cross section for momentum transfer in elastic collisions.

Our solver used the elastic cross section alone, in three places: the field-heating term
of the flux operator, the mobility integral (Reid eq. 6) and the diffusion integral
(Reid eq. 7). Electrons were therefore too mobile, and — through the field-heating term —
too hot.

**Why it survived every existing test.** The error scales with `sum_k sigma_k / sigma_m,e`.
Every synthetic gas in the suite was argon-like, with an inelastic cross section around
1.5 % of the elastic one, so the defect moved nothing measurably. The Reid gas is the
opposite by construction: at 1 eV its ramp is `8 x 10^-20 m^2` against an elastic
`6 x 10^-20 m^2`. Reid's title is *"…gases with large inelastic cross sections"*, and this
is exactly the regime his paper exists to probe.

## Decision

**Implement Reid eq. (4) as an explicit effective cross section, and use it where the
derivation puts it.**

`ElectronKinetics` gains `effective_momentum_transfer_m2` and its boundary counterpart.
`InelasticChannel` gains `sigma_edge_m2`, because the flux terms and the mobility sum are
evaluated at cell boundaries and an effective cross section that existed only at cell
centres would fix the energy balance and leave the transport coefficients wrong.

Which cross section goes where follows Reid's **eq. (5a)**, the exact form:

| Term | Cross section |
|---|---|
| Field heating, mobility, diffusion | **Effective** — `sigma_m,e + sum_k sigma_k` |
| Elastic energy loss (`2 m_e/M` per collision) | **Elastic** — an inelastic collision removes its threshold instead, already handled by the inelastic operator |

Reid's eq. (5b) substitutes the effective cross section into the loss terms as well and
calls the error negligible. It is: the two forms differ by 0.03 % in mean energy at 24 Td.
The exact form is implemented anyway, since it costs one array. Notably eq. (5b) matches
MultiBolt's `MB(2)` *very slightly* better, which is evidence that the residual difference
is this documented approximation rather than a defect in either code.

## Result

Against Flynn et al's `MB(2)`, with the gate for comparison — the pass criterion is that
our disagreement be smaller than what the two-term approximation itself costs:

| `E/N` | `<eps>` | `W` | `N D_T` | Two-term approximation costs |
|---|---|---|---|---|
| 1 Td | +0.054 % | +0.035 % | +0.002 % | 0.00 % / 0.31 % |
| 12 Td | +0.038 % | −0.011 % | +0.017 % | 1.71 % / 2.79 % |
| 24 Td | +0.034 % | −0.025 % | +0.005 % | 2.03 % / 2.91 % |

Against Reid's own 1979 table, all eight tabulated fields from 1 to 40 Td: worst
disagreement **0.34 %** in `W` and **0.10 %** in `<eps>`, against the 0.2 % uncertainty
Reid quotes for his own Boltzmann code.

Lucas-Saelee at 30 Td, the only published check on a **rate coefficient** and the reason
that gas is carried at all:

| Quantity | Agreement with `MB(2)` |
|---|---|
| `k_iz` | **+1.83 %** |
| `<eps>` | −1.48 % |
| `W` | +1.07 % |

## Consequences

- **doc 11 WBS 1.8's acceptance criterion is now met**, and was not before. Gate G-1
  recorded it as satisfied on internal evidence alone; [G-1](../gates/G-1-foundation.md)
  is amended accordingly.
- **Every rate coefficient this solver has produced was wrong for argon too**, though by
  far less: argon's inelastic cross sections are ~1.5 % of its elastic one near threshold
  and rise above it, so the mobility error is of that order rather than 50 %. Nothing
  downstream had consumed a rate table yet — the CR model is P2 work — so no result is
  invalidated.
- **The residual Lucas-Saelee disagreement (1–2 %) is larger than the Reid one (0.05 %)
  and is not yet explained.** Two candidates, both already named in ADR-009 as carried
  simplifications: the omitted growth-model correction, which matters precisely for a
  non-conservative swarm, and the ionisation energy-sharing rule, which neither published
  source states. A test asserts the published value lies between the one-takes-all and
  equal-sharing brackets, so the ambiguity is bounded rather than assumed away. It should
  be closed when the CR model consumes these rates.
- **The negative control had to be replaced, and the reason is worth recording.** The
  first one perturbed the mass ratio, and doubling it moved the mean energy by under
  0.01 % — because at 24 Td the Reid gas loses essentially all its energy inelastically.
  A control that cannot fail proves nothing, so the gate is now guarded by two that can:
  reconstructing the elastic-only momentum transfer, and steepening the ramp by 20 %.
- **This is the argument for external benchmarks, in one data point.** 1 488 internal
  tests, `mypy --strict`, 98 % coverage and a green gate report did not find a 52 % error.
  One published table did, in its first run.
