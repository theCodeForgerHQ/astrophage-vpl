# ADR-009 — EEDF / Boltzmann solver: neither BOLSIG+ nor `bolos`

**Status:** Accepted · **Date:** 2026-08-05 · **Blocks:** P1 (WBS 1.8), and everything
downstream of a rate coefficient — L1 (doc 03 §3.2), the CR model (doc 04 §2.2), the
Thomson spectrum (doc 04 §4.2)

## Context

Doc 08 §2's build/buy table is binding, and it says:

| Capability | Decision | Choice |
|---|---|---|
| EEDF / Boltzmann solver | **Buy** | BOLSIG+ or `bolos` |

Doc 03 §9 repeats it with the reasoning: *"The standard tool; reimplementation is pure
risk."* That reasoning is sound in the abstract. This ADR exists because both named
options were tried on the actual toolchain and both failed for reasons that have nothing
to do with the quality of the physics in them.

The requirement itself is not in doubt. Doc 03 §8 lists assumption **A6, "Maxwellian EEDF
for rate coefficients"**, and its Handling column reads, in full: *"**Replaced** by
two-term Boltzmann solver output (§3.2)."* Doc 03 §3.2 then specifies the deliverable —
`k_iz(E/N)`, `k_ex,j(E/N)`, `µ_e(E/N)`, `D_e(E/N)`, `⟨ε⟩(E/N)` — and doc 04 §4.2 requires
the EEDF itself to survive, not only its moments, because the Thomson spectrum is computed
from it. Something has to produce these numbers.

## Evidence

### `bolos` — installs, imports, and cannot run

Attempted on the project toolchain (Python 3.12.13, macOS/arm64):

```
$ uv pip install bolos
 + bolos==0.2
 + numpy==2.5.1
 + scipy==1.18.0
$ python -c "from bolos import solver, parser, grid; print('ok imports')"
ok imports
```

So far so good. Running an actual solve is where it stops. Three findings, in order of
severity:

**1. It calls a SciPy API that no longer exists.** `bolos` 0.2 uses
`scipy.integrate.simps` in three places — `solver.py:601`, `:766`, `:802` — which are
respectively the **normalisation of `f0`**, the **mobility integral** and the **diffusion
integral**. That alias was deprecated in SciPy 1.10 and **removed in SciPy 1.14**.
`packages/vpl-physics/pyproject.toml` requires `scipy>=1.14`; the installed version is
1.18.0:

```
$ python -c "import scipy.integrate as si; print(hasattr(si,'simps'), hasattr(si,'simpson'))"
False True

AttributeError: module 'scipy.integrate' has no attribute 'simps'. Did you mean: 'simpson'?
```

This is not fixable by pinning around it: the three broken call sites are exactly the three
outputs doc 03 §3.2 asks for.

**2. It divides by zero on any energy grid that starts at zero.** With
`grid.LinearGrid(0, 60., 200)` — the usage its own documentation shows — the first
iteration raises:

```
solver.py:587: RuntimeWarning: divide by zero encountered in divide
  sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy) / GAMMA
solver.py:620: RuntimeWarning: invalid value encountered in divide
solver.py:497: MatrixRankWarning: Matrix is exactly singular
```

The project's pytest configuration is `filterwarnings = ["error"]` (root
`pyproject.toml`), so under the test harness this is a hard failure rather than noise. It
is also a real numerical defect, not a cosmetic one — the singular-matrix warning says the
first row of the linear system is garbage.

**3. It is unmaintained, and its licence is not clearly stated.** The last commit to
`github.com/aluque/bolos` touching code is **2018-03-27**; the repository has been pushed
to since but not developed. It contains **no LICENSE file at all** — the only licence
statement anywhere is `License: LGPLv2` in the PyPI metadata, alongside a
`Programming Language :: Python :: 2.7` classifier. Doc 09's whole premise is that every
input must be traceable to a citable, licensed source; a dependency whose licence exists
only as a one-line PyPI field, in a project whose own licence (ADR-001) is still open and
leaning Apache-2.0, is a liability we would be adding deliberately.

Upstream `master` has the same `simps` calls, so there is no unreleased fix to pull.

### BOLSIG+ — cannot be run here, and cannot be shipped anywhere

From the author's own distribution and copyright pages (LAPLACE, Université de Toulouse,
retrieved 2026-08-05):

| Fact | Consequence for this project |
|---|---|
| *"made available 'as is' and in **binary executable form only**. No source code, assistance or support are provided."* | No Python API. Coupling means writing files, shelling out, and parsing text output — and no way to verify the numerics we would be depending on. |
| Current version (07/2024) ships for **Windows and Linux only**. macOS was dropped after 12/2019. | The development machine is macOS/arm64 (doc 10 §1 names WSL2/Ubuntu as the *reference* machine, not the only one). A dependency that cannot run on a developer's laptop is a dependency the tests will be written around. |
| *"The BOLSIG+ software and associated documentation and input files are **not to be distributed by third parties or used for commercial purposes**."* | Directly incompatible with doc 12's open-core commercialisation model, and with doc 13's reproducibility requirement that a reader be able to re-run an archived manifest. |
| *"Permission to use BOLSIG+ is hereby granted, provided that proper reference is made in any publications making use of data generated by this software."* | Fine, and honoured either way — the physics is cited regardless of who implements it. |

The non-commercial clause alone is decisive. Doc 12 §7 argues that adoption is the moat and
that the framework must be auditable and re-runnable; a core numerical component that
cannot be redistributed and cannot be used commercially makes both impossible.

### What was left

Nothing on PyPI. `LoKI-B` (Tejero-del-Caz et al., *Plasma Sources Sci. Technol.* **28**
(2019) 043001) is MATLAB. `METHES` is MATLAB. The Monte-Carlo alternatives solve a
different and much more expensive problem, and the framework already has a Monte-Carlo
kinetic level in L2 — using one here would collapse the L1/L2 distinction doc 03 §1 exists
to maintain.

## Decision

**Implement the two-term expansion directly, in `vpl.physics.eedf`, from Hagelaar &
Pitchford (2005).** This is a deviation from doc 08 §2 and doc 03 §9, and it is the reason
this ADR exists.

The justification is doc 00 C2. C2 forbids **unsolved** problems, not textbook ones. The
two-term expansion is:

> G. J. M. Hagelaar and L. C. Pitchford, "Solving the Boltzmann equation to obtain electron
> transport coefficients and rate coefficients for fluid models", *Plasma Sources Sci.
> Technol.* **14** (2005) 722–733.

which is the paper BOLSIG+ itself implements. Every equation in
`vpl/physics/eedf/solver.py` is quoted in its module docstring, together with the two
closed-form limits that fall out of it. The solver module is **208 statements** as measured
by coverage — smaller than the 286-statement LXCat parser that feeds it, and far smaller
than the ~1500-line PIC kernel ADR-003 is already leaning towards building.

The distinction that makes this different from "reimplementation is pure risk": **the
two-term equation has exact solutions in three limits, and they are not the limits the
solver was written for.** A reimplementation that could only be checked against another
code would indeed be pure risk. This one is checked against closed forms.

## Verification, and the numbers it produced

Measured, not asserted. All fixtures are synthetic gases written in the test modules, per
doc 09 §5.

| Check | Closed form | Measured | Relative error |
|---|---|---|---|
| Maxwellian limit: `⟨ε⟩ → 1.5 kT_g` at `E/N → 0` | 0.03877800 eV | 0.03877834 eV | **8.8 × 10⁻⁶** |
| Maxwellian limit: shape of `f0` | `exp(-ε/kT_g)` | — | **1.8 × 10⁻⁵** peak-relative |
| Einstein relation `D_e/µ_e = kT_e/e` | 0.02585200 eV | 0.02585534 eV | **1.3 × 10⁻⁴** |
| Druyvesteyn `⟨ε⟩ = ε₀ Γ(5/4)/Γ(3/4)`, `ε₀ = (E/N)/(σ_m √(3m/M))` | 17.369946 eV | 17.370051 eV | **6.0 × 10⁻⁶** |
| Druyvesteyn shape | `exp(-(ε/ε₀)²)` | — | **9.1 × 10⁻⁶** peak-relative |
| Reduced mobility at constant collision frequency, `µ_e N = γ/(2σ₀√ε_ref)` = `e/(m_e ν)` | 1.4827423952 × 10²⁴ m⁻¹V⁻¹s⁻¹ | 1.4827423952 × 10²⁴ | **2 × 10⁻¹⁶** (machine precision), at `E/N` = 0.5, 1.5 and 3.0 Td |
| Normalisation `∫ f₀ √ε dε` | 1 | 1.000000000000000 | exact |
| Growth eigenvalue vs `k_iz` | identical by construction | — | **< 10⁻¹³** |
| Order of accuracy in cell width | 2 (design) | **1.997** (R² = 0.9999997; pairwise 1.995, 1.998, 1.999) | — |

The mobility check deserves emphasis. With `σ_m = σ₀√(ε_ref/ε)` the collision frequency is
energy-independent and the two-term mobility integral collapses, by an exact discrete Abel
summation, to the normalisation of `f0` — so `µ_e N` becomes **independent of the EEDF**
and equal to the textbook `µ = e/(m_e ν)` (Lieberman & Lichtenberg §5.3). It came out to
machine precision at three different fields. That isolates the transport integral from
every other part of the solve, and it is not a check any comparison against another
Boltzmann code could give.

### What could not be verified in the repository, and why

Doc 11 WBS 1.8's done-when is *"rate coefficients reproduce published Ar values"*, and
**this ADR does not claim that gate is met.** It cannot be met by a unit test: reproducing
published argon transport requires the LXCat argon tables, and doc 09 §5 forbids them from
entering the repository. The two requirements are in direct conflict and the conflict is
recorded here rather than resolved by quietly committing a table.

What has been built instead is the machinery to close it outside the repository:
`tabulate_each_electron_set(store, ...)` runs the doc 03 §3.2 tabulation under all three
electron databases of doc 09 §2.1 straight off a verified `AtomicDataStore`, and
`DatabaseSpread.relative_spread` returns the disagreement between them as an array — the
doc 06 §4 term-2 contribution, as a function of `E/N` rather than as a remembered
percentage. **The argon comparison belongs in the benchmark suite (doc 07), against the
cached data, and it is an open item against G-1.** The synthetic-gas closed forms above are
a stronger *verification* than an argon comparison would be, and a weaker *validation*;
both are needed and only one is done.

## Consequences

**Doc 08 §2's row should read "Build" when doc 08 is next revised.** Baseline documents are
superseded, not patched (per ADR-007's note), so neither doc 08 nor doc 03 §9 is edited
here.

**No new dependency.** The implementation uses NumPy and SciPy, both already required by
`vpl-physics`. No optional-dependency extra was added. This is a real saving against the
"buy" option, which would have needed either an LGPL dependency of uncertain provenance or
a binary that half the developers cannot run.

**Three simplifications are carried, and all three are stated in the module docstring
rather than discovered later (doc 00 C4):**

| Omission | Bound |
|---|---|
| Hagelaar & Pitchford's **growth-model correction** to `σ_m` (the `ν̄/(γ√ε)` term in the `f1` relation) | Reported per solve as `EedfSolution.growth_correction`, so it is bounded rather than assumed small. On the argon-like synthetic gas it measures 4.9 × 10⁻¹¹ at 5 Td, 3.2 × 10⁻⁵ at 20 Td, 1.5 × 10⁻⁴ at 40 Td and **5.6 × 10⁻⁴ at 80 Td** — i.e. it is growing, and a run far up the `E/N` axis should read the number rather than trust this row. |
| **Superelastic collisions** | Need the excited-state populations, which are doc 04 §2.2's CR model — a *consumer* of this module. Valid while the metastable fraction is small; doc 05 §2.2 makes that fraction an inferred nuisance parameter, so the assumption is at least visible in the posterior. |
| **Electron-electron collisions** | Negligible at the ionisation degree of doc 01 §2.1 (`n_e/n_g ~ 10⁻⁶`). |

**The ionisation energy-sharing rule is exposed, not fixed.** `IonisationSharing` offers
one-takes-all and equal-sharing, which bracket any real sharing distribution. They are
measurably different — equal sharing puts more electrons in the lowest cell, because the
ionising population sits just above threshold and half of a small excess is nearly zero —
so it is a `StrEnum` manifest field that doc 07's ablation can sweep rather than a
constant somebody picked.

**Extrapolation above the tabulated cross sections stays at `RAISE`.** The atomic layer's
default is kept rather than softened: doc 03 §4.5 calls the tail the most consequential
atomic-data choice in the project, and a grid that runs past its data is a configuration
error worth a stack trace, not a silent power law. Both policies remain keyword arguments
so a manifest records what was used.

**ADR-003 is unaffected but rhymes.** The same argument — a textbook algorithm, small,
verifiable against analytic limits, and easier to couple than an adapted general-purpose
code — is the one ADR-003 is making for the 1D3V PIC kernel. Two ADRs reaching the same
conclusion from independent evidence is worth noticing; it suggests doc 08 §2's default
towards "buy" is calibrated for larger components than these.
