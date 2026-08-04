# 12 — Commercialisation

Version 1.0 · Status: **Baseline** · Owner: Team Astrophage

---

## 1. What is actually being sold

**Not a simulator.** The simulation exists to validate the reconstruction; nobody buys a
validation harness.

**The product is the measurement.** A customer has a plasma system and does not know how much
energy is hitting their surfaces. The product tells them, with a defensible error bar, without
putting anything into their plasma.

```
Customer's optical data  ──►  ┌──────────────────────┐  ──►  Ion energy flux, resolved
                              │  Physics-constrained │       Credible interval
Customer's operating point ──►│      inversion       │  ──►  IEDF at the surface
                              └──────────────────────┘       Erosion / heating prediction
                                                             Data-quality flags
```

The simulation framework becomes the **evidence package** that the measurement can be trusted
— and, separately, a product in its own right for customers who want to design a diagnostic
before buying one.

---

## 2. Product ladder

| Tier | Product | Buyer | Delivered |
|---|---|---|---|
| **T1** | **Diagnostic design service** | Anyone planning a plasma diagnostic | "Here is what your proposed optical setup can and cannot measure, and to what precision" — the optimal-experiment-design capability of doc 05 §9 |
| **T2** | **Inversion software licence** | Labs that already have optical diagnostics | The inverse engine, applied to their data |
| **T3** | **Qualification-as-a-service** | Thruster and EP developers | Erosion/lifetime qualification reports from measurement campaigns |
| **T4** | **Integrated diagnostic package** | Fabs, thruster manufacturers | Optical hardware + inversion software as an instrument |

**T1 is the wedge, and it is available immediately — with no hardware at all.** It is the only
tier the current constraint set (no lab, no data) permits us to deliver today, and it is
genuinely valuable: the framework can answer "would E-FISH be worth the fs laser?" or "will a
HeNe interferometer see anything at my density?" before a purchase order is written. Doc 01
§5.4 and doc 02 §7.1 are worked examples of exactly this service, produced as a by-product of
our own design process.

That last point is worth stating in a pitch: **we did not invent the T1 product; we discovered
we had built it while specifying our own instrument.**

---

## 3. Markets

From the proposal deck, with the honest qualifier attached to each.

| Market | Size (2030, third-party forecasts) | Our applicability today |
|---|---|---|
| Electric & space propulsion | ~$20 B, ~12 % CAGR | **Strong.** Xe/Kr plumes are electropositive; erosion is the binding lifetime constraint |
| Semiconductor plasma etch | $37 B+ etch equipment | **Qualified.** Real etch plasmas are electronegative; doc 03 assumption A8 excludes negative ions in v1. Extension is straightforward but not done |
| Fusion plasma-facing components | Emerging research market | **Partial.** Divertor conditions are far outside the specified envelope; the method transfers, the parameterisation does not |

**The etch qualifier is stated deliberately.** Claiming a $37 B market while the model does not
handle the chemistry that market runs on is the kind of overclaim that a technical judge or a
technical investor will catch, and it discredits the parts that are true. The correct pitch is
"electropositive today, electronegative is a specified extension" — which is still a large
market and is a claim that survives scrutiny.

---

## 4. Why this is defensible

| Moat | Strength |
|---|---|
| **Validation evidence** | The closed-loop benchmark suite, coverage validation and error budget are years of work to replicate. Anyone can write an inversion; almost nobody can show theirs is calibrated |
| **The forward operator** | A verified, layered, instrument-realistic forward chain is the hard asset. The inversion is comparatively easy once it exists |
| **Failure-boundary knowledge** | Knowing precisely where the method breaks is what makes it deployable — and it is only obtainable by having done the sweeps |
| **Public benchmark** | Defining the standard benchmark for this problem class (doc 09 §4.3) makes competitors measure themselves against our axes |
| **Not a moat** | The physics, the algorithms and the atomic data are all public. **This is deliberate** (doc 00 C2) and it is honest to say so |

---

## 5. TRL assessment

Assessed conservatively. Inflated TRL claims are the most common credibility failure in
deep-tech pitching.

| Component | TRL now | TRL after P7 | Blocker to the next level |
|---|---|---|---|
| Inverse methodology | 2 | **4** (validated in a laboratory-analogous environment — computational) | Real data |
| Forward model | 2 | **4** | Experimental validation |
| Instrument design | 2 | 3 | Fabrication |
| Integrated diagnostic | 1 | 2 | Everything above |

**Honest headline: the project reaches TRL 4 on the computational core and no higher, because
TRL 5+ requires hardware we have chosen not to build.** Saying this is a strength: it shows we
know what the ladder means. A team claiming TRL 6 from a simulation has told the evaluator
they cannot be trusted on anything else.

---

## 6. Path to real data

The single blocking dependency for every tier above T1.

| Route | What it needs | Attractiveness |
|---|---|---|
| **RRCAT / DAE facility access** | Incubation relationship (already the stated pathway in the proposal deck) | **Highest** — laser and vacuum infrastructure already exists |
| **University partnership** | A group with an ICP/CCP source and OES + a probe | High — low cost, fast, and even probe-only data validates the density/temperature chain |
| **Published-data validation** | No access at all — validate against datasets in the literature | **Available immediately and currently unexploited.** Several published RF-sheath IEDF and LIF datasets exist; running the inversion against them would be genuine, if partial, experimental validation |
| **Customer pilot** | An EP developer with existing optical access | Highest commercial value, hardest to obtain first |

**The published-data route deserves attention now.** It requires nothing we do not have, it
converts "closed-loop validated" into "partially experimentally validated" — the single
largest credibility jump available to the project — and it costs literature search plus
digitisation rather than capital. It should be promoted into the roadmap as a P5-parallel task.

---

## 7. IP posture

| Asset | Posture | Rationale |
|---|---|---|
| Core framework | **Open source** (ADR-001, leaning Apache-2.0) | Adoption is the moat. A closed inverse framework nobody can audit is worth less than an open one everybody validates against |
| Benchmark suite & datasets | Open | Standard-setting |
| Trained surrogates for specific hardware | Proprietary | The customer-specific asset |
| Qualification reports & methodology | Commercial service | Where revenue lives |
| Patents | **Probably not defensible** | The physics and algorithms are published. A patent on an integration would be weak and expensive. Trade-secret the customer-specific calibrations instead |

**The recommendation to *not* pursue patents is deliberate and should be defended, not
hidden.** Doc 00 C2 forbids novel methods; a project that uses only published methods by
design cannot simultaneously claim patentable novelty in those methods. Consistency here is
more valuable than an aspirational IP slide.

---

## 8. Revenue model

| Stream | Structure | Timing |
|---|---|---|
| Diagnostic design studies (T1) | Fixed-fee engagements | **Available now** |
| Software licence (T2) | Annual, per-seat or per-site | After P7 |
| Qualification service (T3) | Per-campaign | After facility access |
| Integrated systems (T4) | Capital sale + support | 3+ years |

Open-core: the framework is free; the trained hardware-specific surrogates, the qualification
reports, the support and the traceable calibration chain are paid.

---

## 9. What would make this fail commercially

| Risk | Honest assessment |
|---|---|
| Customers do not believe non-intrusive inference | The comparative benchmark (doc 07 §5.3) is the direct counter, but it is simulated. Real side-by-side against an RFEA is what actually closes this |
| The uncertainty is too large to be decision-useful | 17 % (doc 06 §4) is useful for erosion trending, marginal for absolute lifetime qualification. **This is a real limitation, not a marketing problem** |
| Optical access does not exist on customer hardware | Many production tools have no viewports. This constrains the addressable market more than market-size figures suggest |
| The market prefers a cheap probe that is "good enough" | Probes are cheap and understood. The value case rests on environments where probes cannot survive or cannot be tolerated — which narrows the beachhead to high-flux EP and research applications |

Listing these is not pessimism. An evaluator who raises one of these and finds it already
analysed concludes the team is credible; one who raises it and finds a blank concludes the
opposite.

---

## 10. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Four-tier product ladder with T1 identified as immediately deliverable; conservative TRL; recommendation against patenting; published-data validation route promoted. |
