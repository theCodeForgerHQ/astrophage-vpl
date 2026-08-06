# ADR-012: The Laplace posterior ships UNVALIDATED, and coverage does not change that

**Status:** Accepted
**Date:** 2026-08-06
**Supersedes:** none
**Related:** ADR-011 (self-consistent is not correct), doc 05 §5, doc 05 §6, doc 11 §9 item 5

## Context

doc 11 §9 item 5 — the coverage test — is the only item on the compressed critical path
marked **Do not cut**. Its selling point is stated plainly: "uncertainty that has been
*checked* — the strongest differentiator."

To check an interval, the project first has to produce one. doc 05 §5 lists five posterior
engines. Only MAP existed. The cheapest way to get from a point estimate to an interval is
the Laplace approximation, which doc 05 §5 costs at "+ one Hessian" — and then constrains
in the same table row:

> Valid only when the posterior is near-Gaussian; **its validity is tested, not assumed**.

and in the following paragraph:

> The framework runs Laplace and NUTS on a subset of cases and reports the divergence
> between them; Laplace is only permitted where that divergence is below a threshold.

**NUTS is not built, and will not be built before the deadline.** It requires a
differentiable forward model, which requires the L3 surrogate, which requires the DS-TRAIN
ensemble — 5 000 L2 runs, blocked on the GPU. So the permission doc 05 §5 describes cannot
be granted for any case. Two decisions follow, and both are about what may be *claimed*
rather than about what to compute.

## Decision 1 — the validity label lives on the object, not in a document

Every `LaplacePosterior` carries `validity = LaplaceValidity.UNVALIDATED`, and its `repr`
carries it too. There is no code path that produces a `VALIDATED` posterior, because
nothing in this project has actually run the comparison that would justify one.

The alternative — computing Laplace posteriors and recording the caveat in a report — was
rejected for the reason `vpl.validation.sealed` already exists: a caveat that lives only in
a document is a caveat that gets dropped. A number is lifted into a slide, the sentence
qualifying it is not, and by the time anyone asks, the provenance is gone. This is the same
failure mode as reporting T1 as T2, and it gets the same structural treatment.

## Decision 2 — a singular Hessian is refused, not regularised

At a flat direction the Hessian is singular and `numpy.linalg.inv` returns an enormous
finite matrix rather than an error. Reporting that as a covariance silently converts

> this combination of parameters is not identified by this measurement (doc 05 §6)

into

> this combination is measured, with a large error bar.

The first is a result about the experiment and is genuinely interesting. The second is a
fabrication. doc 05 §6.2 predicts a specific instance of exactly this — `Gamma_i ~ n_0
sqrt(T_e)` makes `n_0` and `T_e` correlated — so the singular case is expected here, not
hypothetical.

Adding a ridge to make the inverse exist was rejected: the flat direction of the resulting
covariance would be governed entirely by the size of the ridge, a number chosen for
numerical convenience and then reported as physics. `laplace_posterior` raises instead, and
names the offending eigenvector, because *which* combination is unconstrained is the part a
reader can act on.

Negative curvature is refused on the same principle. A MAP that stopped at a saddle reports
`converged=True` exactly as readily as one that found the mode, and the Hessian is the
cheapest place to catch it.

## Decision 3 — coverage is measured with the PIT, and reported with its own interval

The naive coverage test counts hits at one credible level and prints a fraction. Two
problems, both of which would have shipped:

**The fraction is itself an estimate.** At doc 11 §9's 400 cases, the standard error on a
coverage estimate near 0.9 is `sqrt(0.9 x 0.1 / 400) ~ 1.5 %`, so a 95 % interval around it
spans roughly `+/- 3 %`. A measured 0.88 is therefore fully consistent with a perfectly
calibrated posterior, and 0.91 is not evidence of success. **A project whose central claim
is calibrated uncertainty cannot make an uncalibrated statement about its own
calibration** — so `CoverageReport` offers no way to obtain a coverage number without its
interval. The interval is Wilson's, not the normal approximation, which degenerates to
`[1, 1]` at a perfect hit rate and can place bounds outside `[0, 1]`.

**One level discards most of the information.** If the truth is drawn from the posterior
then `F_posterior(truth)` is Uniform(0, 1) exactly, so one array of PIT values tests every
credible level at once and a KS test is a single decision rule. It also separates three
failure modes that a coverage number conflates — overconfident (PIT at the edges, the one
that misleads a reader), underconfident (PIT at the centre, wasteful but honest) and biased
(PIT skewed, which can look almost right at one level). The report names which, because
"not calibrated" does not tell anyone whether the published intervals are too generous or
too mean.

## Decision 4 — passing coverage does **not** upgrade the validity label

This is the decision most likely to be argued with later, so it is recorded explicitly.

It is tempting to treat a passing coverage test as the empirical validation doc 05 §5 asks
for: if the 90 % intervals contain the truth 90 % of the time, what more is there to check?

The answer is that calibration is a statement about an **ensemble average**, and the
Gaussian-shape assumption is a statement about **each posterior**. A set of intervals can be
well-calibrated on average while being over-wide in one region of parameter space and
over-narrow in another, the two cancelling in the aggregate. Coverage cannot see skew that
averages out; NUTS-vs-Laplace divergence can. Passing coverage is therefore **necessary and
not sufficient**, and `LaplaceValidity` stays `UNVALIDATED`.

Treating coverage as sufficient would be precisely the ADR-011 failure: a self-consistent
check mistaken for an external one.

## Consequences

- Any figure quoting a credible interval from this project must carry `UNVALIDATED`
  alongside its tier label. Two labels, not one.
- The coverage result is a genuine and defensible headline — "our 90 % intervals contain
  the truth in X % of 400+ cases, +/- 3 %" — and it must be stated at that precision, not
  as a bare percentage.
- When the GPU is resolved and the L3 surrogate lands, NUTS becomes buildable and
  `LaplaceValidity.VALIDATED` becomes reachable. Nothing should set it before then.
- A refused singular Hessian is a **result**, not a crash: it identifies a null direction
  doc 05 §6 predicts. The ensemble driver must record such cases and continue, not abort.
