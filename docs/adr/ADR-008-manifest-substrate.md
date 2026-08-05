# ADR-008 — The manifest substrate: OmegaConf now, Hydra at the sweep layer

**Status:** Accepted · **Date:** 2026-08-05 · **Blocks:** P1 (WBS 1.3, gate G-1.3), and
every experiment thereafter

## Context

Doc 08 §2's build/buy table is binding and reads:

| Capability | Decision | Choice |
|---|---|---|
| Configuration | Buy | **Hydra + OmegaConf** |

and doc 08 §2 lists the manifest language itself as **Build — "thin layer over Hydra"**.
Doc 08 §6 then specifies what that layer must do, and its own usage is:

```bash
vpl run experiments/b02-reference-operating-point.yaml
vpl reproduce <run-id>
vpl compare <run-id-a> <run-id-b>
```

Implementing WBS 1.3 against that specification exposed a mismatch between what the table
names and what §6 asks for.

## Evidence

### 1. Doc 08 §6 asks for a file path; Hydra's unit of work is a composed config group

Hydra's entry point is `@hydra.main(version_base=…, config_path=…, config_name=…)`, and its
value is *composition*: `defaults:` lists, config groups, `--multirun`, launcher plugins.
Doc 08 §6's manifest is a single self-contained document — "**One file, one experiment**" —
addressed by an ordinary filesystem path. Running an arbitrary path under Hydra means either
`--config-dir`/`--config-name` gymnastics per invocation or `compose()` from the
Compose API, at which point Hydra is being used as an OmegaConf wrapper.

### 2. Hydra owns a working directory and a run tree; doc 13 owns those already

| Concern | Hydra's behaviour | This project's specification |
|---|---|---|
| Working directory | `hydra.job.chdir` changes it (opt-out, but on by default historically) | doc 13 §2 records `git_commit` from the checkout; `vpl.core.provenance._default_repo_root` walks up from `__file__` *specifically so a run launched from `experiments/` records the same commit* |
| Output directory | Creates `outputs/YYYY-MM-DD/HH-MM-SS/` and writes `.hydra/` into it | doc 13 §2 fixes the run identity as `<date>-<name>-<manifest-digest>`, and doc 13 §5 fixes the retention policy for that tree |
| Logging | Installs its own logging config | Not asked for |

Two run-directory managers in one framework is a conflict, and the one that must win is
doc 13's, because the identity is the manifest digest and doc 00 E3 is stated in terms of it.

### 3. Hydra injects into the config, and the config is the run identity

Hydra adds `hydra:` and `job:` nodes to the composed configuration. `manifest_sha256` is
taken over the manifest document (doc 08 §7 stores it in every artifact), so anything Hydra
adds becomes part of the run's identity — meaning a Hydra upgrade could change the digest of
an unchanged experiment. Doc 13 §5 keeps manifests and digests *forever*.

### 4. OmegaConf alone is not a downgrade — it is load-bearing

Measured on the installed stack (`omegaconf 2.3.1`, `PyYAML 6.0.3`, Python 3.12.13):

```
>>> yaml.safe_load("a: 1.0e17")      →  {'a': '1.0e17'}   ← str
>>> OmegaConf.to_container(OmegaConf.create("a: 1.0e17"))
                                     →  {'a': 1e+17}      ← float
```

PyYAML implements YAML 1.1, whose float resolver requires a **signed** exponent. Doc 08 §6's
own plasma block writes `n0: {value: 1.0e17, units: m^-3}`. **A manifest engine built on
PyYAML could not read the example manifest in its own specification** — it would see the
reference density as a string and reject it, or worse, coerce it. This is not a hypothetical:
it was found by loading `docs/08-software-architecture.md` §6 verbatim as
`packages/vpl-experiment/examples/b02-reference-operating-point.yaml`.

OmegaConf also supplies the two things the layer above will need: interpolation, and
`from_dotlist` overrides — which is exactly the mechanism `vpl sweep` (doc 10 §6) requires,
and which this implementation already exposes as `vpl run <manifest> --set key=value`.

### 5. Nothing about the deferral is irreversible

Doc 10 §6 specifies sweep execution as "`vpl sweep <manifest>` → local process pool; SLURM
adapter for Tier 2". That is where Hydra's multirun, config groups and launcher plugins earn
their keep, and adopting it there does not disturb anything decided here: a sweep composes
overrides and hands each resulting document to the same `load_manifest`.

## Decision

**OmegaConf is the manifest substrate. Hydra is deferred to the doc 10 §6 sweep layer and is
not a dependency of `vpl-experiment`.**

| Concern | Doc 08 §2 | Adopted | Deviation? |
|---|---|---|---|
| YAML loading, interpolation, resolution | Hydra + OmegaConf | **OmegaConf** | Partial — half the named choice, and the half §6 uses |
| Config-group composition, multirun, launchers | Hydra | **Deferred to `vpl sweep`** (doc 10 §6) | **Yes — timing, not substitution** |
| Schema validation and unknown-key rejection | (unspecified) | **Typed dataclasses in `vpl.experiment.manifest.schema`** | New |
| Run directory, run identity, run index | (Hydra, by implication) | **`vpl.experiment.run.store`, per doc 13 §2** | **Yes** |

Schema validation is hand-written rather than delegated to OmegaConf structured configs
because the manifest's blocks are frozen slotted dataclasses carrying `pint` quantities and
`StrEnum` members — types OmegaConf's structured-config layer does not model — and because
the error messages are the product here. `vpl.core.params.catalogue` already sets the house
pattern for a strict, suggesting loader, and the manifest reader follows it.

## Consequences

- **Positive.** No `@hydra.main`, so `vpl run` works on any path from any working directory,
  and doc 08 §11's interactive backend can call `execute()` directly — which doc 08 §11
  requires ("the backend exposes exactly the same operations as the CLI").
- **Positive.** The run identity contains nothing but the manifest, so `manifest_sha256` is
  stable against a configuration-library upgrade. Doc 13 §5's "forever" is affordable.
- **Positive.** One dependency (`omegaconf`, plus its `antlr4-python3-runtime`) instead of
  two, in a package doc 08 §3 wants installable without the scientific stack.
- **Negative.** Config-group composition is unavailable, so a manifest cannot yet say
  `defaults: [instruments: full_set]`. Nothing in doc 08 §6 asks for it, and `--set`
  overrides cover the sweep case that motivated it.
- **Negative.** Two places will resolve configuration once `vpl sweep` lands. The boundary is
  stated once, here: **Hydra composes documents; OmegaConf loads and resolves one.**
- **Neutral.** Nothing else in doc 08 §2 changes. No scientific dependency is added, removed
  or substituted.

## Related finding — doc 08 §4's `ForwardSolver.solve` cannot express a steady solve

Surfaced by the same work and recorded here rather than in its own ADR, because it is a
one-line correction to a signature rather than a decision between alternatives.

Doc 08 §4 declares:

```python
def solve(self, params: PlasmaParams, t: TimeGrid) -> PlasmaState: ...
```

`TimeGrid` is not optional. But the data model `vpl-core` already ships says it should be:

| Type | What it already says |
|---|---|
| `PlasmaState.time` | `TimeGrid \| None` — steady is a first-class state |
| `ScalarField` | `(n_z,)` when `time is None`, `(n_t, n_z)` otherwise |
| `IonEnergyFlux.is_steady` | a published view, defined as `time is None` |
| `AnalyticSheathSolver.solve` | `time` defaults to `None`, and its docstring says a time grid "would imply a time dependence this level does not model" |

L0 is analytic and steady; L1 has steady and transient modes. Under doc 08 §4 as written, a
steady solve is inexpressible: passing a `TimeGrid` obliges the solver to return `(n_t, n_z)`
fields it never computed.

`vpl.experiment.solvers.ManifestSolver` is therefore doc 08 §4's contract with `t` widened to
`TimeGrid | None`, and without `cost_estimate` (which belongs to the doc 10 §6 work queue and
would have obliged the L0 adapter to invent a wall-clock figure — a doc 00 C1 defect).

**The consequence a reviewer should weigh:** parameter types are contravariant, so a solver
implementing doc 08 §4 *exactly* is not assignable to `ManifestSolver`. Until doc 08 §4 is
revised, a solver written for the manifest engine must widen `t`. The alternative — requiring
a `TimeGrid` and having steady solvers ignore it — would mean a manifest could state a time
grid that silently did nothing, which is the failure mode doc 08 §6's unknown-key rule exists
to prevent, arriving through the contract instead of the schema.

**Recommendation for doc 08 §4 when it is next revised:** `solve(self, params: PlasmaParams,
t: TimeGrid | None) -> PlasmaState`.

## Related finding — "bit-for-bit" needs one exclusion, and exactly one

Doc 00 E3 and gate G-1.3 say "bit-for-bit". Doc 08 §7 requires every artifact to embed
`created_utc`. The two cannot both hold literally: two honest runs of one manifest *must*
differ in their bytes.

`vpl.experiment.digest` resolves this by hashing the artifact's stored **values and
attributes**, excluding `created_utc` and nothing else — the commit, the seed, the
environment lock hash, the tier and the manifest digest are all inside the digest, so a
reproduction that changed any of them fails the gate. Walking the values rather than the file
also makes the comparison survive an h5py or Parquet-codec upgrade, which doc 13 §5's
"forever" retention makes a certainty.

The evidence that this is the right amount of exclusion is a test:
`test_only_the_creation_timestamp_differs_between_the_two_artifact_files` asserts that the
two files' raw bytes differ *and* that their content digests agree.
