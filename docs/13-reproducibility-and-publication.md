# 13 — Reproducibility and Publication

Version 1.0 · Status: **Baseline** · Owner: Danushika N

---

## 1. The standard

Every number, figure and claim the project produces must be regenerable by a third party from
the repository alone. Concretely:

```
(manifest, git commit, environment lock, seed)  ──►  identical result
```

No "the plot was made in a notebook that is no longer around". No hand-edited figures. No
numbers typed into slides from a terminal.

---

## 2. What is captured per run

```yaml
run:
  id: 20260804-b02-4a7f2e91
  manifest_sha256: 4a7f2e91...
  git_commit: 9c1d8b3...
  git_dirty: false                 # a dirty tree fails the run in release mode
  environment_lock_sha256: e81c...  # micromamba lock file
  container_digest: sha256:...
  seeds:
    plasma: 20260804
    collisions: 20260805
    photons: 20260806
    detector: 20260807
    sampler: 20260808
  hardware: {cpu: "…", gpu: "RTX A4000", cores: 16, ram_gb: 64}
  solver_versions: {dolfinx: "0.8.0", petsc: "3.20", numpyro: "0.15"}
  data_versions: {lxcat_phelps: "sha256:…", nist_asd: "v5.11"}
  tier: T2
  started_utc: …
  duration_s: …
  status: completed
  quarantined_cases: 0
```

**Per-stream seeds, not one global seed** (doc 10 §5). This is what allows the noise model to
change without perturbing the plasma solve, which the ablation matrix requires.

**`git_dirty` fails the run in release mode.** A result produced from uncommitted code is not
reproducible, and permitting it means the archive quietly fills with results nobody can
regenerate.

---

## 3. The figure pipeline

### 3.1 Rule

**No figure is produced by hand.** Every figure is a declarative specification consumed by the
publication engine.

```yaml
figure:
  id: fig-03-flux-reconstruction
  source_run: 20260804-b02-4a7f2e91
  kind: profile_with_credible_band
  x: {var: z, units: mm, label: "Distance from wall"}
  y: {var: gamma_E, units: kW/m^2, label: "Ion energy flux"}
  series:
    - {data: truth, style: truth, label: "Ground truth (sealed)"}
    - {data: posterior_mean, style: primary, label: "Reconstruction"}
    - {data: posterior_ci95, style: band, label: "95 % credible interval"}
    - {data: rfea_simulated, style: comparison, label: "Simulated RFEA"}
  annotations:
    tier: true                   # T0/T1/T2 label — mandatory
    provenance: true             # commit + manifest hash in the corner
  outputs: [pdf, svg, png, pgf]
```

### 3.2 Style

| Aspect | Standard |
|---|---|
| Backend | Matplotlib + SciencePlots |
| Formats | Vector PDF and SVG primary; PGF for LaTeX; PNG for the web |
| Fonts | Journal-matched, embedded |
| Units | SI, via `pint` — the axis label is generated from the quantity, never typed |
| Uncertainty | Always shown. A point estimate without an interval is rejected by the engine |
| Colour | Colourblind-safe; also distinguishable in greyscale |
| Tier label | **Mandatory on every accuracy figure** (doc 07 G-V7) |
| Provenance | Commit hash and manifest hash rendered in the figure margin |

**The provenance-in-the-margin rule is a small thing with outsized effect.** A figure in a
slide deck that carries its own commit hash can be traced back months later. One that does not
becomes an orphan the moment it is copied.

### 3.3 Standard figure set

Generated for every benchmark, without being asked:

| Figure | Content |
|---|---|
| Flux reconstruction | Truth, reconstruction, CI, comparators |
| IEDF comparison | True vs recovered distribution at the wall |
| Residuals | Per channel, standardised, with whiteness test |
| Reliability diagram | Nominal vs empirical coverage (doc 06 §7.3) |
| SBC rank histogram | Uniformity test |
| Posterior corner plot | Parameter correlations, degeneracies visible |
| FIM spectrum | Eigenvalues; the near-null space |
| Information contribution | Per-channel entropy reduction |
| Sobol indices | First-order and total effect |
| Error budget | Stacked contribution bar |
| Convergence | Mesh, timestep, `N_ppc` |
| MMS order verification | Log–log with fitted slope |
| Identifiability map | Envelope coloured by condition number |
| Ablation summary | CI inflation per removed channel |

---

## 4. Automated reports

`vpl report <run-id>` produces a PDF structured like a laboratory report:

1. Configuration — the full manifest, rendered
2. Provenance — commit, environment, seeds, data versions
3. Forward simulation — plasma state, fields, IEDF
4. Synthetic measurements — per instrument, including raw detector frames
5. Reconstruction — posterior, credible intervals, diagnostics
6. Validation — comparison to sealed truth, metrics
7. Uncertainty — error budget, coverage, calibration
8. Identifiability — FIM, profiles, information content
9. Limitations — **auto-populated from the assumptions register** (doc 03 §8) for the
   assumptions this run actually relied on
10. Bibliography — only the sources this run touched (doc 09 §6)

**Section 9 is the one that matters most and the one that would never survive being written by
hand.** Because it is generated from the assumptions actually in play, it cannot be quietly
omitted when the results are good.

---

## 5. Archival and retention

| Class | Retention | Location |
|---|---|---|
| Manifests, code, provenance | Forever | Git |
| Reduced artifacts (46 GB, doc 10 §7) | Forever | Git-LFS / DVC remote |
| Raw fields (860 GB) | 90 days, then regenerate on demand | Local NVMe (treated as cache) |
| Published figures | Forever | Git |
| Release datasets | Forever | Zenodo, with DOI |

**Raw data is a cache, not an asset**, because it is deterministically regenerable. This is
what keeps the archive at tens of gigabytes instead of a terabyte.

---

## 6. Reproducibility testing

Reproducibility is tested, not assumed:

| Test | Frequency |
|---|---|
| Re-run a random archived manifest; compare bit-for-bit | Nightly |
| Rebuild the environment from lock; re-run | Weekly |
| Rebuild the container from scratch; re-run | On release |
| Regenerate every published figure from its manifest | On release |
| Cold-start test: fresh clone, fresh machine, documented setup only | On release |

**The cold-start test is the only one that catches undocumented local state**, and it is the
one most projects skip. It is scheduled on every release for exactly that reason.

---

## 7. Publication plan

| Output | Venue class | Content |
|---|---|---|
| Framework paper | Computer Physics Communications / SoftwareX | The architecture, verification, and the open framework |
| Methods paper | Plasma Sources Science & Technology / Rev. Sci. Instrum. | The inverse method, identifiability analysis, error budget |
| Benchmark paper | J. Open Source Software / data journal | The public benchmark suite and its rationale |
| Negative/boundary result | Same | **Where the method fails** — the failure-boundary map. Genuinely publishable and rarely written |

**The fourth is the one worth planning for now.** "We map the operating conditions under which
ion energy flux is *not* identifiable from optical diagnostics" is a more distinctive
contribution than another successful reconstruction, and it comes free from the sweeps already
scheduled in doc 07.

---

## 8. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-04 | Baseline. Per-stream seeding, declarative figures with embedded provenance, auto-populated limitations section, cache-not-asset retention policy. |
