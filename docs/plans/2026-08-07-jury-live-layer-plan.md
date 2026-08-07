# Jury Live Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a jury drive the closed loop from their own phones — they choose the seed, watch the staged pipeline stream, and can independently verify afterwards that the truth was committed before the estimate existed.

**Architecture:** A new `packages/vpl-jury/` workspace package holds a Starlette server that fans Server-Sent Events to phones, a single FIFO worker that runs each inversion in a **subprocess**, and an append-only JSONL tape that is the audit record. Two narrow additions upstream: an optional `progress=` callback on `run_cell` that is inert when unused, and `SealedTruth.commitment()`, which binds the sealed value without opening the seal.

**Tech Stack:** Python 3.12, Starlette + uvicorn (SSE), segno (QR), httpx + pytest-asyncio (tests), Playwright (one smoke test). Runs in the `vpl-t2` micromamba env because the L1 truth path needs dolfinx.

**Design spec:** `docs/plans/2026-08-07-jury-live-layer-design.md` (commit `63c2e95`).

## Global Constraints

- **Python:** `requires-python = ">=3.12,<3.13"` for `vpl-jury`. `vpl-core` is `>=3.11,<3.13` and has `UP040`/`UP047` disabled — any type alias added to `vpl-core` **must** use the `X: TypeAlias = ...` spelling, never `type X = ...`.
- **mypy:** `strict = true`, `warn_unreachable`, `disallow_any_generics`. Root `pyproject.toml`'s `mypy_path` carries one entry per package and its comment says "Append a package here when you add one" — `packages/vpl-jury/src` must be appended.
- **ruff:** line length 100. Active rule families include `PTH` (use `pathlib`, never `os.path`), `ARG` (no unused arguments), `SIM`, `RET`, `B`, `A` (no shadowing builtins), `PT` (pytest style), `I` (isort).
- **pytest:** `testpaths = ["packages"]`, `--strict-markers`, `--strict-config`, and **`filterwarnings = ["error"]`** — any warning fails the test. Registered markers only: `physics`, `statistical`, `integration`, `slow`, `gpu`, `fenicsx`.
- **Coverage:** `source = ["packages"]`, `omit = ["*/tests/*"]`, target ≥ 80 %.
- **Test classification:** every test in this plan is **software correctness**, not physics verification (doc 08 §8 counts them separately). Do **not** apply the `physics` marker to anything here.
- **Literal lint:** `vpl-lint` runs only on `vpl-core` and `vpl-physics` (see `ci.yml`), so `vpl-jury` is exempt from the named-constant rule. `vpl.core.progress` is **not** exempt — keep numeric literals out of it.
- **py.typed:** every package ships a PEP 561 marker via a `force-include` in its `pyproject.toml`. `vpl-jury` needs one.
- **Env for running anything that touches L1:** `micromamba run -n vpl-t2 …`. The `fenics-dolfinx=0.9` pin is load-bearing; 0.10 fails 26 of the 92 L1 gates.
- **Working directory** for every command: the repo root, `astrophage-vpl/`.
- **Branch:** `p1-foundation`.

## Deviation from the spec, and why

The spec (§4.2) places `events.py` in `vpl-jury`. That cannot hold: `run_cell` lives in `vpl-experiment` and must emit the event type, so if the type lived in `vpl-jury` then `vpl-experiment` would depend on `vpl-jury` while `vpl-jury` already depends on `vpl-experiment` — a cycle, and doc 08 §3 is strict about package direction.

Resolution: the **event type** (`ProgressEvent`, `ProgressCallback`) goes in `vpl-core`, the substrate every package already depends on. The **wire encoding** (`TapeEvent`, canonical JSON, field allowlists) stays in `vpl-jury`. Task 2 and Task 4 respectively.

## File structure

| File | Responsibility |
|---|---|
| `packages/vpl-core/src/vpl/core/progress.py` | `ProgressEvent`, `ProgressCallback`. No JSON, no I/O. |
| `packages/vpl-validation/src/vpl/validation/sealed.py` | + `commitment()` on `SealedTruth` |
| `packages/vpl-experiment/src/vpl/experiment/grid.py` | + `progress=` parameter on `run_cell` |
| `packages/vpl-jury/src/vpl/jury/events.py` | `TapeEvent`, canonical JSON, `KIND_FIELDS` allowlist, leak guard |
| `packages/vpl-jury/src/vpl/jury/tape.py` | append-only JSONL, run IDs, replay |
| `packages/vpl-jury/src/vpl/jury/request.py` | `RunRequest` — validated jury input, JSON ↔ `Cell` |
| `packages/vpl-jury/src/vpl/jury/worker.py` | `__main__`: request in, NDJSON events out on stdout |
| `packages/vpl-jury/src/vpl/jury/queue.py` | FIFO queue, worker supervisor, subprocess lifecycle |
| `packages/vpl-jury/src/vpl/jury/broker.py` | SSE subscriber fanout |
| `packages/vpl-jury/src/vpl/jury/server.py` | routes, app factory |
| `packages/vpl-jury/src/vpl/jury/verify.py` | `vpl-jury verify <run-id>` |
| `packages/vpl-jury/src/vpl/jury/preflight.py` | startup checks, IP + QR |
| `packages/vpl-jury/src/vpl/jury/static/` | `index.html`, `app.js`, `app.css` |

`request.py` and `broker.py` are not in the spec's list. They exist because input validation and subscriber fanout are each one clear responsibility with their own tests, and folding them into `server.py` would make that file the only large one in the package.

---

### Task 1: Package scaffold and workspace wiring

**Files:**
- Create: `packages/vpl-jury/pyproject.toml`
- Create: `packages/vpl-jury/README.md`
- Create: `packages/vpl-jury/src/vpl/jury/__init__.py`
- Create: `packages/vpl-jury/src/vpl/jury/py.typed` (empty file)
- Modify: `pyproject.toml` — `[tool.uv.sources]` and `[tool.mypy] mypy_path`
- Test: `packages/vpl-jury/tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: the importable namespace `vpl.jury`, with `__all__ = []` for now. Later tasks add modules under it.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_package.py`:

```python
"""The package imports and is typed — the floor every later task builds on."""

from __future__ import annotations

import importlib.util
from pathlib import Path


class TestThePackageIsImportable:
    def test_the_namespace_imports(self) -> None:
        module = importlib.import_module("vpl.jury")

        assert module is not None

    def test_it_ships_a_pep_561_marker(self) -> None:
        spec = importlib.util.find_spec("vpl.jury")
        assert spec is not None
        assert spec.origin is not None

        marker = Path(spec.origin).parent / "py.typed"

        assert marker.is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_package.py -v`
Expected: FAIL — collection error, `ModuleNotFoundError: No module named 'vpl.jury'`

- [ ] **Step 3: Create the package**

`packages/vpl-jury/pyproject.toml`:

```toml
#  vpl-jury — the jury live layer.
#
#  An observation window onto `vpl.experiment.run_cell`, plus the bookkeeping that makes
#  what it shows checkable after the fact. It adds no physics: every number it displays
#  comes from `run_cell`, and the only thing this package contributes to a result is the
#  timestamp and sequence number it was recorded under.
#
#  See docs/plans/2026-08-07-jury-live-layer-design.md.

[project]
name = "vpl-jury"
version = "0.1.0"
description = "Astrophage VPL — the jury-facing live layer over the closed loop."
readme = "README.md"
requires-python = ">=3.12,<3.13"
authors = [{ name = "Team Astrophage" }]
dependencies = [
    # doc 08 §3: vpl-experiment is the only package permitted to import both vpl-physics
    # and vpl-inverse, and the closed loop is what this package observes. Depending on it
    # rather than reaching past it is what keeps that rule intact.
    "vpl-experiment",
    "vpl-core",
    "vpl-validation",
    # The server is a relay: it holds a queue, writes a log and fans out events. Starlette
    # is the smallest thing that does ASGI + SSE without pulling a framework's worth of
    # opinions in with it.
    "starlette>=0.41",
    "uvicorn>=0.30",
    # The jury reaches the server by scanning a code off the presenter's screen. Rendering
    # it locally avoids the alternative, which is an internet round trip on a network that
    # by design has no internet.
    "segno>=1.6",
]

[project.scripts]
# Three entry points, matching the three things a presenter does: serve it, check it
# before the jury arrives, and verify a run afterwards.
vpl-jury = "vpl.jury.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vpl"]

[tool.hatch.build.targets.wheel.force-include]
# PEP 561.
"src/vpl/jury/py.typed" = "vpl/jury/py.typed"
# The frontend is three files with no build step, and they must travel with the package.
# An installed wheel that imports cleanly and then serves a 404 for index.html fails at
# the worst possible moment.
"src/vpl/jury/static" = "vpl/jury/static"
```

`packages/vpl-jury/src/vpl/jury/__init__.py`:

```python
"""The jury live layer — doc 05 §7's barrier, made watchable.

The project has no hardware, so every number it reports came out of a computer. The
question a reviewer actually asks is not whether the result is impressive but whether the
program prints the answer its author wanted. This package exists to answer that by
construction rather than by assertion:

- the truth is a deterministic function of a seed **the jury chooses**;
- a commitment to that truth is published, sequenced and timestamped, before the estimate
  exists;
- the log is append-only, so the ordering cannot be rewritten afterwards;
- and any run can be re-derived from its seed in a fresh process by anyone.

What this package must never do is influence a number. It observes `run_cell` through a
callback that is inert when absent, and it contributes nothing to a result except the
sequence number and timestamp it was recorded under.
"""

from __future__ import annotations

__all__: list[str] = []
```

`packages/vpl-jury/src/vpl/jury/py.typed`: empty file.

`packages/vpl-jury/README.md`:

```markdown
# vpl-jury

The jury-facing live layer over the closed loop. Jurors open a page on their own phones,
choose a seed and a truth fidelity, and watch `run_cell` execute stage by stage — truth
sealed, forward chain, blind inversion, reveal.

Design: [`docs/plans/2026-08-07-jury-live-layer-design.md`](../../docs/plans/2026-08-07-jury-live-layer-design.md).

## Running it

```bash
micromamba run -n vpl-t2 python -m vpl.jury.cli serve
```

`serve` runs preflight first and refuses to start if the tape directory is unwritable.

## Verifying a run afterwards

```bash
micromamba run -n vpl-t2 python -m vpl.jury.cli verify <run-id>
```

Re-derives the truth from the recorded seed in a fresh process and compares the digest
against what was published before the estimate existed.
```

- [ ] **Step 4: Wire the workspace**

In the root `pyproject.toml`, add to `[tool.uv.sources]` (keep the existing alphabetical-ish grouping, after `vpl-inverse`):

```toml
vpl-jury = { workspace = true }
```

And append to `[tool.mypy] mypy_path` — the value is a single colon-separated string, so extend it in place:

```toml
mypy_path = "packages/vpl-core/src:packages/vpl-physics/src:packages/vpl-instruments/src:packages/vpl-experiment/src:packages/vpl-inverse/src:packages/vpl-validation/src:packages/vpl-jury/src"
```

Add a mypy override for the new third-party imports, appended to the existing overrides block:

```toml
[[tool.mypy.overrides]]
# segno ships no stubs. starlette and uvicorn do ship py.typed and are deliberately
# absent from this list.
module = ["segno.*"]
ignore_missing_imports = true
```

- [ ] **Step 5: Sync and run the test**

Run:
```bash
uv sync
uv run pytest packages/vpl-jury/tests/test_package.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Verify lint and types are clean**

Run:
```bash
uv run ruff check packages/vpl-jury
uv run ruff format --check packages/vpl-jury
uv run mypy packages/vpl-jury
```
Expected: all clean. If mypy reports `Duplicate module named 'vpl'`, the `mypy_path` edit in Step 4 was missed.

- [ ] **Step 7: Commit**

```bash
git add packages/vpl-jury pyproject.toml uv.lock
git commit -m "feat(jury): package scaffold for the live layer

An observation window onto run_cell, wired into the workspace and the
strict-mypy path. No behaviour yet — this is the floor the rest stands on."
```

---

### Task 2: `ProgressEvent` in `vpl-core`

The event type must live here, not in `vpl-jury`: `run_cell` emits it, and `vpl-jury` already depends on `vpl-experiment`, so putting it downstream would create a cycle.

**Files:**
- Create: `packages/vpl-core/src/vpl/core/progress.py`
- Test: `packages/vpl-core/tests/test_progress.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ProgressEvent` — frozen slots dataclass, fields `kind: str` and `payload: Mapping[str, JsonValue]`.
  - `JsonValue: TypeAlias` — `str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]`.
  - `ProgressCallback` — a `Protocol` with `__call__(self, event: ProgressEvent) -> None`.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-core/tests/test_progress.py`:

```python
"""The progress event — the one thing `run_cell` may say about its own middle.

The type lives in vpl-core rather than beside its only consumer because `run_cell` is in
vpl-experiment and the consumer depends on vpl-experiment. Downstream ownership would make
that a cycle, and doc 08 §3 is not negotiable about direction.
"""

from __future__ import annotations

import dataclasses

import pytest

from vpl.core.progress import ProgressEvent


class TestTheEventIsInert:
    def test_it_is_frozen_so_a_subscriber_cannot_edit_the_record(self) -> None:
        event = ProgressEvent(kind="truth_solved", payload={"fidelity": "L1"})

        with pytest.raises(dataclasses.FrozenInstanceError):
            event.kind = "seal_opened"  # type: ignore[misc]

    def test_the_payload_is_carried_verbatim(self) -> None:
        event = ProgressEvent(kind="map_progress", payload={"solve": 41, "start": 2})

        assert event.payload == {"solve": 41, "start": 2}

    def test_a_kind_is_required(self) -> None:
        with pytest.raises(TypeError):
            ProgressEvent(payload={})  # type: ignore[call-arg]

    def test_an_empty_payload_is_allowed_because_some_stages_carry_no_data(self) -> None:
        event = ProgressEvent(kind="reference_solved", payload={})

        assert event.payload == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-core/tests/test_progress.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpl.core.progress'`

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-core/src/vpl/core/progress.py`:

```python
"""What `run_cell` is permitted to say about its own middle — doc 05 §7.

`run_cell` is a single blocking call that returns a `CellReport`. That is the right shape
for a sweep, and the wrong shape for anything watching: a reviewer who is told "it ran for
thirty seconds and here is the answer" has been shown a result, not a process.

This module adds the narrowest possible seam. A caller may pass a callback; `run_cell`
hands it an event at each of the step boundaries its own comments already mark. When no
callback is passed **nothing is constructed and nothing is called**, so a run with an
observer and a run without one produce the same numbers. That property is the whole point
and it is tested rather than asserted — see `test_grid_progress.py`.

## Why the payload is loosely typed

One dataclass per stage would type the payloads precisely and would also put the wire
format in vpl-core, where it does not belong: the shape of what a browser receives is the
serialisation layer's business, and it will change when the interface does. So the payload
is a JSON-shaped mapping, and the *allowlist of permitted keys per kind* lives in
`vpl.jury.events`, next to the code that serialises it and next to the test that proves no
truth value escapes before the reveal.

The cost is that a typo in a payload key is caught by that allowlist rather than by mypy.
The benefit is that vpl-core does not acquire an opinion about HTTP.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

__all__ = ["JsonValue", "ProgressCallback", "ProgressEvent"]

#: A JSON-representable value. Spelled with `TypeAlias` rather than `type X = ...`: this
#: package carries `requires-python = ">=3.11"` for the L2 GPU environment (ADR-013), and
#: `type` statements are 3.12-only syntax.
JsonValue: TypeAlias = (
    "str | int | float | bool | None | Sequence[JsonValue] | Mapping[str, JsonValue]"
)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One stage boundary, named and carrying whatever that stage knows.

    Frozen because a subscriber must not be able to edit the record it was handed. The
    audit trail this feeds is append-only, and an event that could be mutated after
    publication would make that guarantee cosmetic.

    Attributes:
        kind: Which boundary this is. The permitted set, and the keys each kind may carry,
            are declared in :data:`vpl.jury.events.KIND_FIELDS` — deliberately not here,
            so that vpl-core holds no wire format.
        payload: What the stage knows, JSON-shaped.
    """

    kind: str
    payload: Mapping[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class ProgressCallback(Protocol):
    """What a caller passes to `run_cell(progress=...)`.

    A Protocol rather than a `Callable` alias so that a stateful collector — the common
    case, since somebody always wants the events in a list — satisfies it without a
    `cast`.
    """

    def __call__(self, event: ProgressEvent) -> None:
        """Receive one event. Must not raise: see `run_cell`'s handling."""
        ...
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-core/tests/test_progress.py -v`
Expected: 4 passed.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-core/src/vpl/core/progress.py packages/vpl-core/tests/test_progress.py
uv run mypy packages/vpl-core
uv run vpl-lint packages/vpl-core
```
Expected: clean. `vpl-lint` runs on `vpl-core`, so this module must contain no bare numeric literals — as written it contains none.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-core/src/vpl/core/progress.py packages/vpl-core/tests/test_progress.py
git commit -m "feat(core): the progress event run_cell may emit

In vpl-core rather than beside its consumer: run_cell lives in
vpl-experiment and the consumer depends on vpl-experiment, so downstream
ownership would be a cycle. The payload stays JSON-shaped so vpl-core
acquires no wire format; the per-kind key allowlist lives in vpl-jury."
```

---

### Task 3: `SealedTruth.commitment()`

**Files:**
- Modify: `packages/vpl-validation/src/vpl/validation/sealed.py`
- Test: `packages/vpl-validation/tests/test_sealed_truth.py` (append a class)

**Interfaces:**
- Consumes: nothing.
- Produces: `SealedTruth.commitment() -> str` — 64-character lowercase SHA-256 hex digest. Callable **before** `commit_estimate` and it does not advance the seal's state: `is_committed` is unchanged, and `value` still raises afterwards.

Canonical payload, pinned by the test so it cannot drift:

```
"<name>\x1f<comma-joined shape of atleast_1d>\x1f<repr(float(x)) for each element, \x1f-joined>"
```

- [ ] **Step 1: Write the failing test**

Append to `packages/vpl-validation/tests/test_sealed_truth.py`:

```python
class TestTheCommitment:
    """A digest that binds the sealed value without opening the seal.

    The demo this exists for publishes the digest before the inversion runs and the value
    after it finishes. That ordering is what makes "the estimate was not fitted to a known
    truth" checkable rather than merely stated, and it only works if computing the digest
    is not itself an unsealing.
    """

    def test_the_digest_is_sha256_of_the_canonical_payload(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        expected = hashlib.sha256(b"Gamma_E\x1f1\x1f6577.0").hexdigest()

        assert sealed.commitment() == expected

    def test_it_is_readable_before_an_estimate_is_committed(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        digest = sealed.commitment()

        assert len(digest) == 64

    def test_reading_it_does_not_open_the_seal(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")

        sealed.commitment()

        assert sealed.is_committed is False
        with pytest.raises(InverseCrimeError):
            _ = sealed.value

    def test_it_does_not_consume_the_one_permitted_commit(self) -> None:
        sealed = SealedTruth(value=6577.0, name="Gamma_E")
        sealed.commitment()

        sealed.commit_estimate(6510.0, tier=Tier.T1)

        assert sealed.estimate == 6510.0

    def test_a_different_value_gives_a_different_digest(self) -> None:
        one = SealedTruth(value=6577.0, name="Gamma_E").commitment()
        other = SealedTruth(value=6577.5, name="Gamma_E").commitment()

        assert one != other

    def test_the_name_is_bound_too_so_two_quantities_cannot_collide(self) -> None:
        flux = SealedTruth(value=1.0, name="Gamma_E").commitment()
        drift = SealedTruth(value=1.0, name="u_i").commitment()

        assert flux != drift

    def test_it_is_stable_across_processes(self) -> None:
        # A digest that depended on dict ordering or PYTHONHASHSEED would verify on the
        # machine that produced it and fail on a juror's, which is the one case that
        # matters.
        script = (
            "from vpl.validation.sealed import SealedTruth;"
            "print(SealedTruth(value=6577.0, name='Gamma_E').commitment())"
        )
        first = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        second = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": "1"},
        )

        assert first.stdout.strip() == second.stdout.strip()
        assert first.stdout.strip() == SealedTruth(value=6577.0, name="Gamma_E").commitment()

    def test_an_array_truth_binds_its_shape(self) -> None:
        # Without the shape in the payload, [1.0, 2.0] and a 2x1 of the same values would
        # collide, and the digest would stop identifying what it claims to identify.
        flat = SealedTruth(value=np.array([1.0, 2.0]), name="profile").commitment()
        column = SealedTruth(value=np.array([[1.0], [2.0]]), name="profile").commitment()

        assert flat != column
```

Add to that file's imports:

```python
import hashlib
import os
import subprocess
import sys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-validation/tests/test_sealed_truth.py::TestTheCommitment -v`
Expected: FAIL — `AttributeError: 'SealedTruth' object has no attribute 'commitment'`

- [ ] **Step 3: Write the implementation**

In `packages/vpl-validation/src/vpl/validation/sealed.py`, add `import hashlib` to the imports and this module-level constant beside `_ZERO_THRESHOLD`:

```python
#: Field separator inside the commitment payload. ASCII unit separator: it cannot occur in
#: a float repr, a shape or a quantity name, so the payload cannot be made ambiguous by a
#: value that happens to contain the delimiter.
_COMMITMENT_SEPARATOR: Final[str] = "\x1f"
```

Add this method to `SealedTruth`, immediately after `is_committed` — beside the other thing that is readable while sealed, and before the guarded properties:

```python
    def commitment(self) -> str:
        """A SHA-256 digest binding the sealed value, readable while still sealed.

        The demo layer publishes this before the inversion starts and the value after it
        finishes, in an append-only log. That ordering is what makes "the estimate was not
        fitted to a known truth" a checkable claim instead of a stated one.

        This reads `self._value` directly, bypassing the `value` property's guard. That is
        deliberate and it is the module's own doctrine applied rather than evaded: the
        docstring above says sealing "moves the crime from 'accident nobody noticed' to
        'line of code somebody had to write'". This is such a line, written on purpose,
        and it is safe in a way the guarded reads are not — a digest cannot seed an
        optimiser, normalise a plot or stop an iteration.

        Reading it does **not** advance the state machine: `is_committed` stays false,
        `value` still raises, and the one permitted `commit_estimate` is still available.

        Returns:
            64-character lowercase hex digest.

        Note:
            This is a *binding* commitment, not a *hiding* one. `Gamma_E` is a single
            float and the preimage space is searchable, so a determined reader can recover
            the value from the digest. Secrecy is neither achieved nor needed: what the
            digest establishes is that the value could not be changed afterwards.
        """
        return hashlib.sha256(self._commitment_payload()).hexdigest()

    def _commitment_payload(self) -> bytes:
        """The exact bytes hashed, pinned by `TestTheCommitment`.

        `repr` of a Python float is the shortest round-trippable form, so a reader who has
        the printed value can reconstruct the payload and check the digest by hand. The
        shape is included because otherwise `[1.0, 2.0]` and its column form would collide.
        """
        array = np.atleast_1d(np.asarray(self._value, dtype=np.float64))
        shape = ",".join(str(extent) for extent in array.shape)
        elements = [repr(float(element)) for element in array.ravel()]
        payload = _COMMITMENT_SEPARATOR.join([self._name, shape, *elements])
        return payload.encode("utf-8")
```

Add `"commitment"` handling to nothing else — `__slots__` is unchanged because no state is added, which is itself the point.

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-validation/tests/test_sealed_truth.py -v`
Expected: all pass, including the 8 new ones and every pre-existing test in the file.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-validation
uv run mypy packages/vpl-validation
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-validation
git commit -m "feat(validation): a commitment the seal can make while still sealed

SHA-256 over a canonical payload of name, shape and float reprs, so a
reader with the printed value can check it by hand. Readable before
commit_estimate and it does not advance the state machine.

Binding, not hiding: Gamma_E is one float and the preimage space is
searchable. What it establishes is that the value could not change after
publication, which is the only claim the demo makes of it."
```

---

### Task 4: `vpl.jury.events` — wire format and the leak guard

**Files:**
- Create: `packages/vpl-jury/src/vpl/jury/events.py`
- Test: `packages/vpl-jury/tests/test_events.py`

**Interfaces:**
- Consumes: `vpl.core.progress.ProgressEvent`, `JsonValue`.
- Produces:
  - `KIND_FIELDS: Mapping[str, frozenset[str]]` — permitted payload keys per kind.
  - `TRUTH_BEARING_KINDS: frozenset[str]` — `{"seal_opened", "row"}`.
  - `EventKindError(ValueError)`, `EventFieldError(ValueError)`.
  - `validate(event: ProgressEvent) -> None` — raises on unknown kind or key.
  - `TapeEvent` — frozen slots dataclass: `seq: int`, `run_id: str`, `at: str`, `kind: str`, `payload: Mapping[str, JsonValue]`.
  - `TapeEvent.to_json(self) -> str` (compact, sorted keys, no trailing newline) and `TapeEvent.from_json(cls, line: str) -> TapeEvent`.
  - `assert_no_truth_before_reveal(events: Sequence[TapeEvent]) -> None` — raises `AssertionError` if any event before the first truth-bearing one carries a truth-bearing key.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_events.py`:

```python
"""The wire format, and the guard that keeps the truth off it until the reveal.

The demo's whole claim is an ordering: commitment, then estimate, then truth. A payload
key that leaked `gamma_e_true_w_per_m2` into an early event would break that claim while
every screen still looked correct, so the permitted keys are declared per kind and the
ordering is asserted structurally.
"""

from __future__ import annotations

import json

import pytest

from vpl.core.progress import ProgressEvent
from vpl.jury.events import (
    KIND_FIELDS,
    TRUTH_BEARING_KINDS,
    EventFieldError,
    EventKindError,
    TapeEvent,
    assert_no_truth_before_reveal,
    validate,
)


def _tape_event(seq: int, kind: str, **payload: object) -> TapeEvent:
    return TapeEvent(
        seq=seq,
        run_id="r-0001",
        at="2026-08-07T12:00:00Z",
        kind=kind,
        payload=dict(payload),  # type: ignore[arg-type]
    )


class TestValidation:
    def test_a_known_kind_with_known_keys_passes(self) -> None:
        validate(ProgressEvent(kind="truth_solved", payload={"fidelity": "L1"}))

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(EventKindError, match="unknown event kind"):
            validate(ProgressEvent(kind="truth_leaked", payload={}))

    def test_an_unknown_key_is_refused(self) -> None:
        # This is the test that catches the failure mode the module exists for: a new
        # field added to an early stage, carrying something it should not.
        with pytest.raises(EventFieldError, match="gamma_e_true_w_per_m2"):
            validate(
                ProgressEvent(
                    kind="truth_solved", payload={"gamma_e_true_w_per_m2": 9398.6}
                )
            )

    def test_every_kind_declares_its_fields(self) -> None:
        assert KIND_FIELDS
        for kind, fields in KIND_FIELDS.items():
            assert isinstance(kind, str)
            assert isinstance(fields, frozenset)

    def test_the_truth_bearing_kinds_are_declared_kinds(self) -> None:
        assert TRUTH_BEARING_KINDS <= set(KIND_FIELDS)


class TestTheWireFormat:
    def test_a_tape_event_round_trips(self) -> None:
        event = _tape_event(7, "map_progress", solve=41, start=2)

        assert TapeEvent.from_json(event.to_json()) == event

    def test_the_encoding_is_compact_and_key_sorted(self) -> None:
        # Golden: a field rename or a key-order change must fail here, in CI, rather than
        # in the room on a phone that was served yesterday's frontend.
        event = _tape_event(7, "map_progress", solve=41, start=2)

        assert event.to_json() == (
            '{"at":"2026-08-07T12:00:00Z","kind":"map_progress",'
            '"payload":{"solve":41,"start":2},"run_id":"r-0001","seq":7}'
        )

    def test_it_carries_no_trailing_newline_because_the_tape_adds_one(self) -> None:
        event = _tape_event(1, "reference_solved")

        assert not event.to_json().endswith("\n")

    def test_a_malformed_line_raises_rather_than_returning_a_blank_event(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            TapeEvent.from_json("{not json")


class TestTheOrderingGuard:
    def test_a_clean_sequence_passes(self) -> None:
        events = [
            _tape_event(1, "truth_sealed", commitment="a3f9"),
            _tape_event(2, "estimate_committed", gamma_e_estimate_w_per_m2=9481.0),
            _tape_event(3, "seal_opened", gamma_e_true_w_per_m2=9398.6),
        ]

        assert_no_truth_before_reveal(events)

    def test_a_truth_value_before_the_reveal_is_caught(self) -> None:
        events = [
            _tape_event(1, "truth_sealed", commitment="a3f9"),
            # `truth_solved` does not permit this key; the guard is the second line of
            # defence behind `validate`, for events reconstructed from a tape file.
            _tape_event(2, "truth_solved", gamma_e_true_w_per_m2=9398.6),
            _tape_event(3, "seal_opened", gamma_e_true_w_per_m2=9398.6),
        ]

        with pytest.raises(AssertionError, match="before the reveal"):
            assert_no_truth_before_reveal(events)

    def test_a_sequence_with_no_reveal_passes_because_nothing_was_revealed(self) -> None:
        events = [_tape_event(1, "truth_sealed", commitment="a3f9")]

        assert_no_truth_before_reveal(events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_events.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpl.jury.events'`

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-jury/src/vpl/jury/events.py`:

```python
"""The wire format, and the declaration of what each stage may say.

## Why an allowlist and not a denylist

The demo's entire claim is an ordering: a commitment to the truth is published, then an
estimate is produced, then the truth is revealed. Anyone can check that ordering in the
log afterwards — provided the truth is genuinely absent from the log before the reveal.

A denylist ("do not put the truth in early events") fails the way all denylists fail: the
next field added is not on it. So each kind declares the exact keys it may carry, and a
payload key outside that set raises. Adding a field is therefore a deliberate edit to this
module, next to the docstring explaining why the set is closed — which is the only place
someone might read it.

## What this module does not do

It does not decide *when* events are emitted; `run_cell` does. It does not decide what is
true; it only constrains what may be said. And it deliberately holds no reference to a
`SealedTruth`: a serialiser that could read a seal would be a serialiser that could leak
one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Self

from vpl.core.progress import JsonValue, ProgressEvent

__all__ = [
    "KIND_FIELDS",
    "TRUTH_BEARING_KINDS",
    "EventFieldError",
    "EventKindError",
    "TapeEvent",
    "assert_no_truth_before_reveal",
    "validate",
]


class EventKindError(ValueError):
    """An event kind not declared in :data:`KIND_FIELDS`."""


class EventFieldError(ValueError):
    """A payload key the event's kind does not permit."""


#: The permitted payload keys for each stage, mapping onto the step comments in
#: `vpl.experiment.grid.run_cell`. Closed by design: see the module docstring.
KIND_FIELDS: Final[Mapping[str, frozenset[str]]] = {
    "config_accepted": frozenset(
        {
            "tier",
            "cell_label",
            "seed",
            "ablate",
            "truth_fidelity",
            "inversion_fidelity",
            "git_sha",
            "git_dirty",
        }
    ),
    # Deliberately carries no values. This is the stage at which the truth exists, so it
    # is the stage at which a careless addition would do the most damage.
    "truth_solved": frozenset({"fidelity", "wall_clock_s"}),
    "reference_solved": frozenset({"wall_clock_s"}),
    # What the *solver* is about to see is safe to show: it is the synthetic measurement,
    # not the state behind it. Channel names and per-channel scalar summaries only.
    "measurements_synthesised": frozenset(
        {"contributing", "excluded", "channel_summaries", "wall_clock_s"}
    ),
    "truth_sealed": frozenset({"commitment"}),
    "map_progress": frozenset({"solve", "start", "n_starts"}),
    "estimate_committed": frozenset(
        {
            "gamma_e_estimate_w_per_m2",
            "n_0_hat_per_m3",
            "T_e_hat_ev",
            "interval_w_per_m2",
            "half_width_fraction",
            "map_converged",
            "map_iterations",
            "map_n_starts",
            "map_n_distinct_modes",
            "inversion_solves",
        }
    ),
    # The reveal. The only kinds permitted to carry a truth value.
    "seal_opened": frozenset(
        {
            "gamma_e_true_w_per_m2",
            "n_0_true_per_m3",
            "T_e_true_ev",
            "relative_error",
            "truth_within_interval",
            "commitment_verified",
            "tier",
        }
    ),
    "row": frozenset({"report"}),
    "run_failed": frozenset({"exit_code", "stderr_tail"}),
    "run_timeout": frozenset({"timeout_s"}),
    "queued": frozenset({"position"}),
}

#: Kinds permitted to carry a truth value. Everything before the first of these in a run
#: must be free of the keys in :data:`_TRUTH_KEYS`.
TRUTH_BEARING_KINDS: Final[frozenset[str]] = frozenset({"seal_opened", "row"})

#: Keys that constitute a disclosure of the sealed state.
_TRUTH_KEYS: Final[frozenset[str]] = frozenset(
    {"gamma_e_true_w_per_m2", "n_0_true_per_m3", "T_e_true_ev", "relative_error", "report"}
)


def validate(event: ProgressEvent) -> None:
    """Refuse an undeclared kind or an undeclared payload key.

    Raises:
        EventKindError: The kind is not in :data:`KIND_FIELDS`.
        EventFieldError: A payload key is not permitted for that kind.
    """
    permitted = KIND_FIELDS.get(event.kind)
    if permitted is None:
        raise EventKindError(
            f"unknown event kind {event.kind!r}. Declare it in vpl.jury.events.KIND_FIELDS "
            f"together with the keys it may carry — the set is closed so that a new field "
            f"cannot silently put the truth on the wire before the reveal."
        )
    offending = set(event.payload) - permitted
    if offending:
        raise EventFieldError(
            f"event kind {event.kind!r} may not carry {sorted(offending)!r}; "
            f"permitted keys are {sorted(permitted)!r}."
        )


@dataclass(frozen=True, slots=True)
class TapeEvent:
    """A :class:`ProgressEvent` after the server has stamped it.

    The stamp — sequence number and server time — is the only thing this layer contributes
    to a run. Every physical number came from `run_cell`.
    """

    seq: int
    run_id: str
    at: str
    kind: str
    payload: Mapping[str, JsonValue] = field(default_factory=dict)

    def to_json(self) -> str:
        """Compact, key-sorted, newline-free.

        Sorted because a golden test pins this exact string, and key order that follows
        insertion order would make that test pass or fail on the order a payload happened
        to be built in. Newline-free because the tape owns the line separator.
        """
        return json.dumps(
            {
                "seq": self.seq,
                "run_id": self.run_id,
                "at": self.at,
                "kind": self.kind,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, line: str) -> Self:
        """Reconstruct from one tape line.

        Raises:
            json.JSONDecodeError: The line is not JSON. Propagated rather than converted
                into a blank event, because a tape that silently yields empty records
                would read as a run that did nothing.
        """
        raw = json.loads(line)
        return cls(
            seq=int(raw["seq"]),
            run_id=str(raw["run_id"]),
            at=str(raw["at"]),
            kind=str(raw["kind"]),
            payload=raw.get("payload", {}),
        )


def assert_no_truth_before_reveal(events: Sequence[TapeEvent]) -> None:
    """Check the ordering the whole demo rests on.

    :func:`validate` guards events on the way out. This guards a sequence read back *in*,
    which is what a juror running `vpl-jury verify` is doing, and what the end-to-end test
    asserts over a real run.

    Raises:
        AssertionError: A truth-bearing key appears before the first reveal.
    """
    for event in events:
        if event.kind in TRUTH_BEARING_KINDS:
            return
        disclosed = set(event.payload) & _TRUTH_KEYS
        if disclosed:
            raise AssertionError(
                f"event seq={event.seq} kind={event.kind!r} discloses "
                f"{sorted(disclosed)!r} before the reveal. The published ordering is the "
                f"only evidence that the estimate was not fitted to a known truth."
            )
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_events.py -v`
Expected: 12 passed.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury
uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-jury
git commit -m "feat(jury): wire format, and an allowlist so the truth cannot leak early

Each stage declares the exact payload keys it may carry. A denylist would
fail the way denylists do — the next field added is not on it — and the
failure would be invisible, because every screen would still look right.

Encoding is compact and key-sorted, pinned by a golden test so a field
rename fails in CI rather than on a phone served yesterday's frontend."
```

---

### Task 5: `run_cell(progress=...)` and the inertness guarantee

**Files:**
- Modify: `packages/vpl-experiment/src/vpl/experiment/grid.py`
- Test: `packages/vpl-experiment/tests/test_grid_progress.py`

**Interfaces:**
- Consumes: `vpl.core.progress.ProgressEvent`, `ProgressCallback`.
- Produces: `run_cell(..., progress: ProgressCallback | None = None)`. Emits, in order: `config_accepted`, `truth_solved`, `reference_solved`, `measurements_synthesised`, `truth_sealed`, `map_progress` (repeatedly), `estimate_committed`, `seal_opened`, `row`.

**Inertness is defined precisely:** every field of the returned `CellReport` is identical with and without a callback, **except `wall_clock_s`**, which is a wall-clock measurement and cannot be. The test compares every other field explicitly rather than comparing the dataclass, so a newly added field cannot slip through unchecked.

`MAP_PROGRESS_STRIDE` is 5 — emit on solve 1 and every 5th thereafter. Spec §12 item 1 left this to be tuned; 5 gives roughly 24 updates across a 117-solve inversion, which reads as alive without flooding six phones.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-experiment/tests/test_grid_progress.py`:

```python
"""`progress=` observes the closed loop without touching it.

The demo layer this feeds is only worth having if it cannot be accused of changing the
answer, so the guarantee is the same one `channel_weights` carries: passing the parameter
in its neutral form is indistinguishable from not passing it. Here the neutral form is a
callback that does nothing.

Every cell in this file is L0 -> L0, which needs no dolfinx and runs in the workspace
environment. One L1 case lives in `test_grid_progress_fenicsx.py` behind the marker.
"""

from __future__ import annotations

from vpl.core.progress import ProgressEvent
from vpl.core.state import Fidelity
from vpl.experiment.grid import Cell, EedfShape, run_cell
from vpl.jury.events import TRUTH_BEARING_KINDS, validate

#: Fields of `CellReport` that a wall-clock measurement legitimately changes between two
#: otherwise identical runs. Everything else must match exactly.
_TIMING_FIELDS = frozenset({"wall_clock_s"})


def _cell() -> Cell:
    return Cell(
        truth=Fidelity.L0,
        inversion=Fidelity.L0,
        noise=True,
        imperfect_calibration=True,
        calibration_uncertainty=True,
        truth_eedf=EedfShape.MAXWELLIAN,
    )


def _comparable(report: object) -> dict[str, object]:
    """Every field except the timing ones, so a new field cannot slip through unchecked."""
    import dataclasses

    return {
        f.name: getattr(report, f.name)
        for f in dataclasses.fields(report)  # type: ignore[arg-type]
        if f.name not in _TIMING_FIELDS and f.name != "posterior"
    }


class TestInertness:
    def test_a_no_op_callback_changes_no_reported_number(self) -> None:
        without = run_cell(_cell(), seed=0, ablate="oes")
        with_observer = run_cell(
            _cell(), seed=0, ablate="oes", progress=lambda event: None
        )

        assert _comparable(with_observer) == _comparable(without)

    def test_the_posterior_is_unchanged_too(self) -> None:
        import numpy as np

        without = run_cell(_cell(), seed=0, ablate="oes")
        with_observer = run_cell(
            _cell(), seed=0, ablate="oes", progress=lambda event: None
        )

        assert (without.posterior is None) == (with_observer.posterior is None)
        if without.posterior is not None and with_observer.posterior is not None:
            assert np.array_equal(without.posterior.mean, with_observer.posterior.mean)


class TestTheEmittedSequence:
    def test_the_stages_arrive_in_the_documented_order(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        kinds = [event.kind for event in events]
        for expected in (
            "config_accepted",
            "truth_solved",
            "reference_solved",
            "measurements_synthesised",
            "truth_sealed",
            "estimate_committed",
            "seal_opened",
            "row",
        ):
            assert expected in kinds, f"{expected} was never emitted"
        assert kinds.index("truth_sealed") < kinds.index("estimate_committed")
        assert kinds.index("estimate_committed") < kinds.index("seal_opened")

    def test_every_emitted_event_satisfies_the_allowlist(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        for event in events:
            validate(event)

    def test_no_event_before_the_reveal_carries_a_truth_value(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        for event in events:
            if event.kind in TRUTH_BEARING_KINDS:
                break
            assert "gamma_e_true_w_per_m2" not in event.payload
            assert "n_0_true_per_m3" not in event.payload

    def test_the_commitment_is_published_before_the_estimate_exists(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        kinds = [event.kind for event in events]
        sealed = events[kinds.index("truth_sealed")]

        assert len(str(sealed.payload["commitment"])) == 64

    def test_the_reveal_confirms_the_published_commitment(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        kinds = [event.kind for event in events]
        opened = events[kinds.index("seal_opened")]

        assert opened.payload["commitment_verified"] is True

    def test_map_progress_streams_while_the_optimiser_runs(self) -> None:
        events: list[ProgressEvent] = []

        run_cell(_cell(), seed=0, ablate="oes", progress=events.append)

        progress = [event for event in events if event.kind == "map_progress"]

        assert len(progress) > 1
        assert all(int(event.payload["solve"]) >= 1 for event in progress)

    def test_a_raising_callback_does_not_take_the_run_down(self) -> None:
        # A juror's browser disconnecting must not be able to fail an inversion.
        def hostile(event: ProgressEvent) -> None:
            raise RuntimeError("subscriber exploded")

        report = run_cell(_cell(), seed=0, ablate="oes", progress=hostile)

        assert report.relative_error >= 0.0
```

Mark the whole file slow — each `run_cell` is ~19 s and this file runs several:

```python
import pytest

pytestmark = pytest.mark.slow
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-experiment/tests/test_grid_progress.py -v`
Expected: FAIL — `TypeError: run_cell() got an unexpected keyword argument 'progress'`

- [ ] **Step 3: Add the parameter and the emit points**

In `grid.py`, add to the imports:

```python
from vpl.core.progress import ProgressCallback, ProgressEvent
```

Add the stride constant beside `_L0_POSTERIOR_SAMPLES`:

```python
#: Emit a `map_progress` event on the first forward solve and every Nth after it. At the
#: ~117 solves a reference inversion takes this is roughly 24 updates — enough that a
#: watching browser looks alive, few enough that six of them do not flood the event stream.
_MAP_PROGRESS_STRIDE: Final[int] = 5
```

Extend the signature (append after `n_starts`, keeping every existing parameter and default untouched):

```python
    progress: ProgressCallback | None = None,
```

Add to the docstring's `Args:` section:

```
        progress: An optional observer of the step boundaries below. Defaults to ``None``,
            in which case nothing is constructed and nothing is called, so a run with an
            observer and a run without one report the same numbers — the guarantee
            ``channel_weights`` carries, tested the same way in
            ``test_grid_progress.py``. A callback that raises is logged and swallowed: a
            disconnecting browser must not be able to fail an inversion.
```

Immediately after `tier = cell.tier`, insert the emitter:

```python
    def _emit(kind: str, /, **payload: object) -> None:
        """Hand one event to the observer, if there is one.

        Guarded rather than defaulted to a no-op so that the `None` path constructs no
        event object at all. And it swallows subscriber exceptions: the observer is a
        browser on somebody's phone, and `run_cell`'s contract to its caller does not
        depend on it.
        """
        if progress is None:
            return
        try:
            progress(ProgressEvent(kind=kind, payload=dict(payload)))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - an observer must not be able to fail a run
            _LOG.exception("progress observer raised on a %r event; continuing", kind)
```

Add a module-level logger near the top if `grid.py` has none:

```python
_LOG: Final = logging.getLogger(__name__)
```

Now the emit calls, each placed immediately after the work it reports:

After `_emit`'s definition:

```python
    _emit(
        "config_accepted",
        tier=int(tier),
        cell_label=cell_label(cell),
        seed=seed,
        ablate=ablate,
        truth_fidelity=cell.truth.value,
        inversion_fidelity=cell.inversion.value,
    )
```

After the `truth_state, gamma_e_true, true_theta = _truth_state(...)` block:

```python
    _emit("truth_solved", fidelity=cell.truth.value)
```

After the `reference_state = AnalyticSheathSolver().solve(...)` statement:

```python
    _emit("reference_solved")
```

After the `joint` is finalised — that is, after the `if channel_weights:` block:

```python
    _detail_before = joint.detail(truth_state)
    _emit(
        "measurements_synthesised",
        contributing=list(_detail_before.contributing),
        excluded=list(_detail_before.excluded),
    )
```

After `sealed = SealedTruth(value=gamma_e_true, name="Gamma_E")`:

```python
    commitment = sealed.commitment()
    _emit("truth_sealed", commitment=commitment)
```

Inside `log_likelihood`, at the very top, add the counter. Declare it just above the function:

```python
    solve_counter = itertools.count(1)
```

and as the first statement inside `log_likelihood`:

```python
        solve_index = next(solve_counter)
        if solve_index == 1 or solve_index % _MAP_PROGRESS_STRIDE == 0:
            _emit("map_progress", solve=solve_index, n_starts=n_starts)
```

Add `import itertools` and `import logging` to the imports.

After the `posterior, interval = _gamma_e_interval(...)` block:

```python
    _emit(
        "estimate_committed",
        gamma_e_estimate_w_per_m2=estimate,
        n_0_hat_per_m3=theta_hat.n_0,
        T_e_hat_ev=theta_hat.T_e,
        interval_w_per_m2=None if interval is None else list(interval),
        map_converged=result.converged,
        map_iterations=result.iterations,
        map_n_starts=result.n_starts,
        inversion_solves=forward.solves,
    )
```

After `sealed.commit_estimate(estimate, tier=tier)`:

```python
    _emit(
        "seal_opened",
        gamma_e_true_w_per_m2=float(sealed.value),
        n_0_true_per_m3=true_theta.n_0,
        T_e_true_ev=true_theta.T_e,
        relative_error=sealed.relative_error,
        truth_within_interval=(
            None
            if interval is None
            else bool(interval[0] <= float(sealed.value) <= interval[1])
        ),
        commitment_verified=sealed.commitment() == commitment,
        tier=int(sealed.tier),
    )
```

The `row` event needs the built report, so emit it just before `return`. Assign the report to a name, emit, then return it:

```python
    report = CellReport(...)  # the existing constructor call, unchanged
    _emit("row", report=cell_label(cell))
    return report
```

> `row` carries only the cell label, not the serialised report. The full report reaches the
> browser through the tape's own record of `seal_opened` and `estimate_committed`; putting a
> second copy on the wire would create two representations of one result that could disagree.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest packages/vpl-experiment/tests/test_grid_progress.py -v`
Expected: 9 passed. Takes several minutes — each `run_cell` is ~19 s and there are 11 calls.

- [ ] **Step 5: Verify nothing else regressed**

Run:
```bash
uv run pytest packages/vpl-experiment -q -m "not slow"
uv run ruff check packages/vpl-experiment
uv run mypy packages/vpl-experiment
```
Expected: all pre-existing tests pass, lint and types clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-experiment
git commit -m "feat(experiment): an optional observer of run_cell's step boundaries

progress= defaults to None, and on that path no event is constructed and
nothing is called, so a run with an observer reports the same numbers as
one without. Tested field by field rather than by comparing the dataclass,
so a field added later cannot slip through unchecked. wall_clock_s is the
one documented exception, because it is a wall-clock measurement.

A callback that raises is logged and swallowed. The observer is a browser
on somebody's phone and must not be able to fail an inversion."
```

---

### Task 6: `request.py` — validated jury input

Nothing reaches the queue unvalidated. `joint.without()` raises `KeyError` on an unknown
channel name, and a typo that reached the worker would surface as a crashed run in front of
the jury instead of a `422` on the phone that caused it.

**Files:**
- Create: `packages/vpl-jury/src/vpl/jury/request.py`
- Test: `packages/vpl-jury/tests/test_request.py`

**Interfaces:**
- Consumes: `vpl.core.state.Fidelity`, `vpl.experiment.grid.Cell`, `EedfShape`.
- Produces:
  - `CHANNEL_NAMES: frozenset[str]` — `{"oes", "lif", "thomson", "interferometry"}`.
  - `TRUTH_FIDELITIES: frozenset[str]` — `{"L0", "L1"}`.
  - `MAX_SEED: int` — `2**32 - 1`.
  - `RequestError(ValueError)`.
  - `RunRequest` — frozen slots dataclass: `seed: int`, `ablate: str | None`, `truth: str`.
  - `RunRequest.parse(cls, raw: Mapping[str, object]) -> RunRequest` — raises `RequestError`.
  - `RunRequest.to_cell(self) -> Cell`.
  - `RunRequest.fingerprint(self) -> str` — stable string used by the queue to dedupe.
  - `RunRequest.to_json(self) -> str`.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_request.py`:

```python
"""What a phone is allowed to ask for.

Every rejection here is a rejection that would otherwise have become a crashed run on the
projector. `joint.without()` raises `KeyError` on an unknown channel, and L2 truth raises
`TruthArtefactRequiredError` because the PIC artefacts do not exist in this repo — so both
are refused at the boundary, with a message aimed at the person holding the phone.
"""

from __future__ import annotations

import pytest

from vpl.core.state import Fidelity
from vpl.jury.request import MAX_SEED, RequestError, RunRequest


class TestParsing:
    def test_a_minimal_request_parses(self) -> None:
        request = RunRequest.parse({"seed": 8231, "truth": "L1", "ablate": "oes"})

        assert request.seed == 8231
        assert request.truth == "L1"
        assert request.ablate == "oes"

    def test_ablate_may_be_omitted_for_the_full_four_channel_set(self) -> None:
        request = RunRequest.parse({"seed": 0, "truth": "L0"})

        assert request.ablate is None

    def test_a_missing_seed_is_refused(self) -> None:
        with pytest.raises(RequestError, match="seed"):
            RunRequest.parse({"truth": "L0"})

    def test_a_non_integer_seed_is_refused(self) -> None:
        with pytest.raises(RequestError, match="seed"):
            RunRequest.parse({"seed": "eight", "truth": "L0"})

    def test_a_bool_is_not_an_acceptable_seed(self) -> None:
        # `isinstance(True, int)` is True in Python, so this needs an explicit guard or
        # `{"seed": true}` silently becomes seed 1.
        with pytest.raises(RequestError, match="seed"):
            RunRequest.parse({"seed": True, "truth": "L0"})

    def test_a_negative_seed_is_refused(self) -> None:
        with pytest.raises(RequestError, match="seed"):
            RunRequest.parse({"seed": -1, "truth": "L0"})

    def test_a_seed_above_the_ceiling_is_refused(self) -> None:
        with pytest.raises(RequestError, match="seed"):
            RunRequest.parse({"seed": MAX_SEED + 1, "truth": "L0"})

    def test_the_ceiling_itself_is_accepted(self) -> None:
        request = RunRequest.parse({"seed": MAX_SEED, "truth": "L0"})

        assert request.seed == MAX_SEED

    def test_an_unknown_channel_is_refused_with_the_valid_names(self) -> None:
        with pytest.raises(RequestError, match="interferometry"):
            RunRequest.parse({"seed": 0, "truth": "L0", "ablate": "oees"})

    def test_l2_truth_is_refused_because_the_artefacts_do_not_exist(self) -> None:
        with pytest.raises(RequestError, match="L2"):
            RunRequest.parse({"seed": 0, "truth": "L2"})

    def test_an_unknown_truth_fidelity_is_refused(self) -> None:
        with pytest.raises(RequestError, match="truth"):
            RunRequest.parse({"seed": 0, "truth": "L7"})


class TestTheCell:
    def test_l0_truth_builds_a_t1_cell(self) -> None:
        cell = RunRequest.parse({"seed": 0, "truth": "L0"}).to_cell()

        assert cell.truth is Fidelity.L0
        assert cell.inversion is Fidelity.L0
        assert int(cell.tier) == 1

    def test_l1_truth_builds_a_t2_cell(self) -> None:
        cell = RunRequest.parse({"seed": 0, "truth": "L1"}).to_cell()

        assert cell.truth is Fidelity.L1
        assert cell.inversion is Fidelity.L0
        assert int(cell.tier) == 2

    def test_noise_and_calibration_are_always_on(self) -> None:
        # T2 is the number quoted publicly, and `tier_of_configuration` refuses a
        # mismatched-model run that skips either. Hard-coding them on means no request the
        # jury can compose is refusable by the tier check.
        cell = RunRequest.parse({"seed": 0, "truth": "L1"}).to_cell()

        assert cell.noise is True
        assert cell.imperfect_calibration is True
        assert cell.calibration_uncertainty is True


class TestTheFingerprint:
    def test_identical_requests_share_a_fingerprint(self) -> None:
        one = RunRequest.parse({"seed": 5, "truth": "L1", "ablate": "oes"})
        other = RunRequest.parse({"seed": 5, "truth": "L1", "ablate": "oes"})

        assert one.fingerprint() == other.fingerprint()

    def test_a_different_seed_gives_a_different_fingerprint(self) -> None:
        one = RunRequest.parse({"seed": 5, "truth": "L1"})
        other = RunRequest.parse({"seed": 6, "truth": "L1"})

        assert one.fingerprint() != other.fingerprint()

    def test_ablating_nothing_is_distinguishable_from_ablating_a_channel(self) -> None:
        full = RunRequest.parse({"seed": 5, "truth": "L1"})
        ablated = RunRequest.parse({"seed": 5, "truth": "L1", "ablate": "oes"})

        assert full.fingerprint() != ablated.fingerprint()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_request.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpl.jury.request'`

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-jury/src/vpl/jury/request.py`:

```python
"""What a phone may ask for, and why the refusals are where they are.

Three things the jury controls: the seed, which channel to drop, and the truth fidelity.
Everything else about the cell is fixed, and fixed deliberately.

## Why noise and calibration are not exposed

`tier_of_configuration` refuses a mismatched-model run that skips noise or skips the
estimated response — correctly, because such a run has no tier it may be reported at. If
those were jury-facing toggles, some combinations would raise `TierMismatchError` in front
of the room. Pinning them on means **every request the interface can compose is valid**,
which is worth more than the extra knobs.

## Why the tier is not a control

The tier is `Cell.tier`'s verdict on the configuration, not an input. The interface offers
truth fidelity and *displays* the resulting tier. Presenting it as selectable would invert
the causality, and doc 05 §7.2 treats a mislabelled tier as a project defect.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Self

from vpl.core.state import Fidelity
from vpl.experiment.grid import Cell, EedfShape

__all__ = [
    "CHANNEL_NAMES",
    "MAX_SEED",
    "TRUTH_FIDELITIES",
    "RequestError",
    "RunRequest",
]


class RequestError(ValueError):
    """A request the jury interface must refuse, with a message aimed at a phone."""


#: The four channels, by the names `JointLikelihood.without()` knows them.
CHANNEL_NAMES: Final[frozenset[str]] = frozenset({"oes", "lif", "thomson", "interferometry"})

#: Truth fidelities the jury may select. L2 is absent: it needs a saved PIC artefact
#: (`TruthArtefactRequiredError`) and none exists in this repository, so offering it would
#: be offering a button that fails.
TRUTH_FIDELITIES: Final[frozenset[str]] = frozenset({"L0", "L1"})

#: Upper bound on a seed. NumPy's legacy seeding accepts a 32-bit value, and a bound makes
#: "type whatever you like" safe rather than a route to an overflow somewhere downstream.
MAX_SEED: Final[int] = 2**32 - 1

#: The EEDF the truth's measurement is generated with. Druyvesteyn is the mismatch axis the
#: measured baseline used, kept fixed so a jury run and the recorded baseline are comparable.
_TRUTH_EEDF: Final[EedfShape] = EedfShape.DRUYVESTEYN


@dataclass(frozen=True, slots=True)
class RunRequest:
    """One validated ask from one phone."""

    seed: int
    truth: str
    ablate: str | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, object]) -> Self:
        """Validate an untrusted mapping — a decoded JSON body.

        Raises:
            RequestError: On anything the closed loop would later refuse. Every message is
                written to be read by whoever is holding the phone.
        """
        return cls(
            seed=_parse_seed(raw.get("seed")),
            truth=_parse_truth(raw.get("truth")),
            ablate=_parse_ablate(raw.get("ablate")),
        )

    def to_cell(self) -> Cell:
        """The `Cell` this request denotes.

        The inversion is always L0: it is the model the framework actually inverts with,
        and pairing it against an L1 truth is what makes the run a T2.
        """
        return Cell(
            truth=Fidelity(self.truth),
            inversion=Fidelity.L0,
            noise=True,
            imperfect_calibration=True,
            calibration_uncertainty=True,
            truth_eedf=_TRUTH_EEDF,
        )

    def fingerprint(self) -> str:
        """A stable identity for deduplication.

        Two phones asking for the same thing while it is still queued should watch one run
        rather than queue two, and on a machine where a run costs 30 seconds that is the
        difference between a responsive demo and a stalled one.
        """
        return f"{self.truth}/{self.seed}/{self.ablate or '-'}"


def _parse_seed(value: object) -> int:
    # `isinstance(True, int)` is True, so booleans need excluding explicitly or a JSON
    # `true` silently becomes seed 1 — a wrong run that looks like a real one.
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(
            f"seed must be a whole number between 0 and {MAX_SEED}, got {value!r}."
        )
    if not 0 <= value <= MAX_SEED:
        raise RequestError(f"seed must lie between 0 and {MAX_SEED}, got {value}.")
    return value


def _parse_truth(value: object) -> str:
    if value not in TRUTH_FIDELITIES:
        if value == "L2":
            raise RequestError(
                "L2 truth needs a saved PIC solve, and this deployment has none. "
                "Choose L0 (reported at T1) or L1 (reported at T2)."
            )
        raise RequestError(
            f"truth must be one of {sorted(TRUTH_FIDELITIES)}, got {value!r}."
        )
    return str(value)


def _parse_ablate(value: object) -> str | None:
    if value is None or value == "":
        return None
    if value not in CHANNEL_NAMES:
        raise RequestError(
            f"unknown channel {value!r}; the four are {sorted(CHANNEL_NAMES)}. "
            f"Omit it to fuse all four."
        )
    return str(value)
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_request.py -v`
Expected: 17 passed.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury && uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-jury
git commit -m "feat(jury): validate what a phone may ask for, at the boundary

Seed, channel and truth fidelity only. Noise and calibration are pinned on
so that every request the interface can compose has a tier — otherwise some
combinations raise TierMismatchError in front of the room.

L2 truth is refused with a reason: it needs a PIC artefact this repo does
not have. A bool is refused as a seed explicitly, because isinstance(True,
int) would otherwise make {\"seed\": true} a silent seed 1."
```

---

### Task 7: `tape.py` — the append-only record

**Files:**
- Create: `packages/vpl-jury/src/vpl/jury/tape.py`
- Test: `packages/vpl-jury/tests/test_tape.py`

**Interfaces:**
- Consumes: `vpl.jury.events.TapeEvent`, `validate`; `vpl.core.progress.ProgressEvent`.
- Produces:
  - `Tape` — constructed as `Tape(path: Path, *, now: Callable[[], str] = utc_now)`.
  - `Tape.append(self, run_id: str, event: ProgressEvent) -> TapeEvent` — validates, assigns the next sequence number, stamps the time, writes one line, flushes and `fsync`s.
  - `Tape.read_all(self) -> list[TapeEvent]` — replay; a truncated final line is dropped.
  - `Tape.next_run_id(self) -> str` — `"r-0001"`-style, monotonic.
  - `utc_now() -> str` — RFC 3339 with a `Z` suffix, second resolution.
  - `TapeWriteError(OSError)`.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_tape.py`:

```python
"""The audit record. Append-only, because the ordering *is* the evidence.

If the log could be rewritten, "the commitment was published before the estimate existed"
would be a claim about the log rather than about the run. So there is no update path, no
delete path, and every line is fsync'd before the next stage begins.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from vpl.core.progress import ProgressEvent
from vpl.jury.events import EventFieldError
from vpl.jury.tape import Tape, utc_now


def _fixed_clock() -> object:
    counter = itertools.count(1)
    return lambda: f"2026-08-07T12:00:{next(counter):02d}Z"


@pytest.fixture
def tape(tmp_path: Path) -> Tape:
    return Tape(tmp_path / "tape.jsonl", now=_fixed_clock())  # type: ignore[arg-type]


class TestAppending:
    def test_the_first_event_gets_sequence_one(self, tape: Tape) -> None:
        stamped = tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        assert stamped.seq == 1

    def test_sequence_numbers_are_monotonic_across_runs(self, tape: Tape) -> None:
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))
        second = tape.append("r-0002", ProgressEvent(kind="reference_solved", payload={}))

        assert second.seq == 2

    def test_the_event_is_stamped_with_the_clock(self, tape: Tape) -> None:
        stamped = tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        assert stamped.at == "2026-08-07T12:00:01Z"

    def test_an_invalid_payload_never_reaches_the_file(self, tape: Tape) -> None:
        with pytest.raises(EventFieldError):
            tape.append(
                "r-0001",
                ProgressEvent(kind="truth_solved", payload={"gamma_e_true_w_per_m2": 1.0}),
            )

        assert tape.read_all() == []

    def test_each_line_is_flushed_so_a_reader_sees_it_immediately(self, tape: Tape) -> None:
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        # Read through a separate handle: buffered-but-unflushed content would be invisible.
        assert tape.path.read_text(encoding="utf-8").count("\n") == 1


class TestReplay:
    def test_replay_returns_events_in_written_order(self, tape: Tape) -> None:
        tape.append("r-0001", ProgressEvent(kind="truth_sealed", payload={"commitment": "a"}))
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        replayed = tape.read_all()

        assert [event.kind for event in replayed] == ["truth_sealed", "reference_solved"]
        assert [event.seq for event in replayed] == [1, 2]

    def test_an_absent_file_replays_as_empty_rather_than_raising(self, tmp_path: Path) -> None:
        assert Tape(tmp_path / "absent.jsonl").read_all() == []

    def test_a_truncated_final_line_is_dropped(self, tape: Tape) -> None:
        # The process can die mid-write. One partial record must not make the whole audit
        # trail unreadable.
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))
        with tape.path.open("a", encoding="utf-8") as handle:
            handle.write('{"seq": 2, "run_id": "r-0001"')

        replayed = tape.read_all()

        assert len(replayed) == 1

    def test_a_corrupt_line_in_the_middle_still_raises(self, tape: Tape) -> None:
        # Only the *last* line may be partial. Damage in the middle is not a crash
        # artefact, and pretending it is would hide tampering.
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))
        with tape.path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
        tape.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        with pytest.raises(ValueError, match="line 2"):
            tape.read_all()


class TestRunIds:
    def test_ids_are_allocated_in_order(self, tape: Tape) -> None:
        assert tape.next_run_id() == "r-0001"
        assert tape.next_run_id() == "r-0002"

    def test_ids_resume_past_what_is_already_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        first = Tape(path)
        run_id = first.next_run_id()
        first.append(run_id, ProgressEvent(kind="reference_solved", payload={}))

        resumed = Tape(path)

        assert resumed.next_run_id() == "r-0002"

    def test_sequence_numbers_resume_past_what_is_on_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "tape.jsonl"
        first = Tape(path)
        first.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        resumed = Tape(path)
        stamped = resumed.append("r-0001", ProgressEvent(kind="reference_solved", payload={}))

        assert stamped.seq == 2


class TestThereIsNoRewritePath:
    def test_the_tape_exposes_no_mutating_api(self) -> None:
        # Enumerated rather than asserted in prose: if someone adds `truncate` later, this
        # fails and they have to justify it.
        forbidden = {"update", "delete", "truncate", "rewrite", "clear", "remove"}

        assert forbidden.isdisjoint(dir(Tape))


class TestTheClock:
    def test_it_is_rfc_3339_with_a_z_suffix(self) -> None:
        stamped = utc_now()

        assert stamped.endswith("Z")
        assert len(stamped) == len("2026-08-07T12:00:00Z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_tape.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpl.jury.tape'`

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-jury/src/vpl/jury/tape.py`:

```python
"""The audit record — append-only, because the ordering is the evidence.

The demo's claim is not "our answer is accurate". It is "the truth was fixed before the
estimate existed, and here is the log". A log with an update path would reduce that to a
claim about the log, so this module has no update path: `append` and `read_all`, and a test
that enumerates the mutating verbs it must not grow.

## Why fsync, on a demo

The alternative is losing the last few events when the process dies, and the last few
events are the reveal. A run whose commitment survived but whose reveal did not looks
exactly like a run that was abandoned when the answer came out wrong. Two hundred
microseconds per event is a very cheap way to not have that conversation.

## Why only the final line may be partial

A process killed mid-write leaves one truncated record at the end, so tolerating that is
crash recovery. Damage anywhere earlier is not a crash artefact — the writer had already
moved past it — so it raises. Tolerating it would mean tolerating exactly the edit an
inconvenient result would motivate.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from vpl.core.progress import ProgressEvent
from vpl.jury.events import TapeEvent, validate

__all__ = ["Tape", "TapeWriteError", "utc_now"]

#: Width of the numeric part of a run id. Four digits is 9 999 runs, which is more than a
#: judging session will ever produce and keeps the ids the same width on screen.
_RUN_ID_DIGITS: Final[int] = 4


class TapeWriteError(OSError):
    """The record could not be written, so the run must not proceed."""


def utc_now() -> str:
    """RFC 3339, UTC, second resolution.

    Second resolution because these timestamps are read by people establishing an ordering,
    and the ordering is carried by the sequence number regardless.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Tape:
    """An append-only JSONL record of every event of every run."""

    def __init__(self, path: Path, *, now: Callable[[], str] = utc_now) -> None:
        self.path = path
        self._now = now
        self._seq, self._runs = _resume_from(path)

    def next_run_id(self) -> str:
        """Allocate the next run id, continuing past anything already on disk."""
        self._runs += 1
        return f"r-{self._runs:0{_RUN_ID_DIGITS}d}"

    def append(self, run_id: str, event: ProgressEvent) -> TapeEvent:
        """Validate, stamp and durably record one event.

        Validation happens first and deliberately: an event with a disallowed payload key
        must never reach the file, because the file is what a juror audits.

        Raises:
            EventKindError, EventFieldError: The event is not permitted on the wire.
            TapeWriteError: The record could not be written.
        """
        validate(event)
        self._seq += 1
        stamped = TapeEvent(
            seq=self._seq,
            run_id=run_id,
            at=self._now(),
            kind=event.kind,
            payload=event.payload,
        )
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(stamped.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise TapeWriteError(
                f"could not append to {self.path}: {error}. An unrecorded run cannot be "
                f"verified afterwards, so the run must not proceed."
            ) from error
        return stamped

    def read_all(self) -> list[TapeEvent]:
        """Every event, in written order.

        A truncated final line is dropped — see the module docstring. Anything earlier
        raises.

        Raises:
            ValueError: A malformed line that is not the last.
        """
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        events: list[TapeEvent] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                events.append(TapeEvent.from_json(line))
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                if number == len(lines):
                    break
                raise ValueError(
                    f"{self.path} is malformed at line {number}: {error}. Only a "
                    f"truncated final line is treated as crash recovery; damage earlier "
                    f"than that is not a crash artefact."
                ) from error
        return events


def _resume_from(path: Path) -> tuple[int, int]:
    """`(highest sequence number, highest run number)` already on disk.

    Restarting the server must not reset either counter: a second event with `seq=1` would
    make the ordering ambiguous exactly where it is load-bearing.
    """
    if not path.is_file():
        return 0, 0
    highest_seq = 0
    highest_run = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = TapeEvent.from_json(line)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        highest_seq = max(highest_seq, event.seq)
        _, _, digits = event.run_id.partition("-")
        if digits.isdigit():
            highest_run = max(highest_run, int(digits))
    return highest_seq, highest_run
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_tape.py -v`
Expected: 16 passed.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury && uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-jury
git commit -m "feat(jury): the append-only tape

No update path, no delete path, and a test that enumerates the mutating
verbs it must not grow. Every line is fsync'd, because the events most
likely to be lost to a crash are the reveal, and a run whose commitment
survived but whose reveal did not looks exactly like an abandoned one.

Only a truncated *final* line is tolerated: that is crash recovery.
Damage earlier is not a crash artefact and raises."
```

---

### Task 8: `worker.py` — one run, one subprocess, NDJSON out

**Files:**
- Create: `packages/vpl-jury/src/vpl/jury/worker.py`
- Test: `packages/vpl-jury/tests/test_worker.py`

**Interfaces:**
- Consumes: `RunRequest`, `run_cell`, `ProgressEvent`.
- Produces:
  - `main(argv: Sequence[str] | None = None) -> int` — reads one JSON request from `argv[1]`, writes one NDJSON event per line to stdout, returns 0 on success and 1 on a refused request.
  - Module runnable as `python -m vpl.jury.worker '<json>'`.

**The stdout-purity requirement.** dolfinx's JIT emits `ld: warning` lines during the L1
path. Those go to stderr, but one stray `print` anywhere in the dependency tree would
corrupt the event stream irrecoverably. So the worker captures `sys.stdout` at entry, hands
the real handle to the event writer, and rebinds `sys.stdout` to `sys.stderr` for the
duration of the run. Anything that prints ends up on stderr where it is harmless.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_worker.py`:

```python
"""The worker: one run per process, events on stdout, everything else on stderr.

Run in a real subprocess rather than by calling `main` in-process, because the property
being tested is a property of the process's file descriptors. An in-process test would pass
while the deployed thing emitted `ld: warning` into the middle of a JSON document.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def _run(request: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "vpl.jury.worker", json.dumps(request)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


class TestAGoodRun:
    def test_it_exits_zero(self) -> None:
        result = _run({"seed": 0, "truth": "L0", "ablate": "oes"})

        assert result.returncode == 0, result.stderr[-2000:]

    def test_every_stdout_line_is_json(self) -> None:
        result = _run({"seed": 0, "truth": "L0", "ablate": "oes"})

        for number, line in enumerate(result.stdout.splitlines(), start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                pytest.fail(f"stdout line {number} is not JSON: {line!r} ({error})")

    def test_the_documented_stages_all_appear(self) -> None:
        result = _run({"seed": 0, "truth": "L0", "ablate": "oes"})

        kinds = [json.loads(line)["kind"] for line in result.stdout.splitlines()]

        for expected in (
            "config_accepted",
            "truth_solved",
            "truth_sealed",
            "estimate_committed",
            "seal_opened",
            "row",
        ):
            assert expected in kinds

    def test_the_commitment_precedes_the_reveal(self) -> None:
        result = _run({"seed": 0, "truth": "L0", "ablate": "oes"})

        kinds = [json.loads(line)["kind"] for line in result.stdout.splitlines()]

        assert kinds.index("truth_sealed") < kinds.index("seal_opened")


class TestARefusedRequest:
    def test_an_invalid_seed_exits_one_without_running_anything(self) -> None:
        result = _run({"seed": -5, "truth": "L0"})

        assert result.returncode == 1
        assert result.stdout.strip() == ""

    def test_the_reason_goes_to_stderr(self) -> None:
        result = _run({"seed": -5, "truth": "L0"})

        assert "seed" in result.stderr

    def test_a_missing_argument_exits_one(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "vpl.jury.worker"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert result.returncode == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_worker.py -v`
Expected: FAIL — every case, `No module named vpl.jury.worker`.

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-jury/src/vpl/jury/worker.py`:

```python
"""One run, one process, events on stdout.

## Why a subprocess at all

Three reasons, in order of how much they matter on the day. A Newton solve that dies takes
the process with it, and that must not be the server. A run that overruns needs killing,
and a thread cannot be killed. And dolfinx brings PETSc and MPI, which are not things to
initialise inside an async server that also has to keep talking to six phones.

## Why stdout is hijacked

The event stream is newline-delimited JSON on stdout, and it has to stay that way. dolfinx's
JIT prints linker warnings during the L1 path; they go to stderr today, and one library
update that sends them to stdout would corrupt the stream in a way that looks like a parser
bug. So the real stdout handle is taken at entry and `sys.stdout` is rebound to stderr for
the rest of the run. Anything that prints lands somewhere harmless.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import TextIO

from vpl.core.progress import ProgressEvent
from vpl.experiment.grid import run_cell
from vpl.jury.events import validate
from vpl.jury.request import RequestError, RunRequest

__all__ = ["main"]


def _writer(stream: TextIO) -> object:
    """A progress callback that writes one JSON object per line, flushed.

    Flushed per event because the whole point is that a phone sees the stage while it is
    happening. A buffered stream would deliver the entire run at once, which is a slower
    version of the print statement this exists to replace.
    """

    def emit(event: ProgressEvent) -> None:
        validate(event)
        stream.write(json.dumps({"kind": event.kind, "payload": event.payload}))
        stream.write("\n")
        stream.flush()

    return emit


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cell and stream its stages.

    Returns:
        0 on a completed run, 1 on a request that was refused before any work began.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print(
            "usage: python -m vpl.jury.worker '<json request>'",
            file=sys.stderr,
        )
        return 1

    try:
        request = RunRequest.parse(json.loads(arguments[0]))
    except (RequestError, json.JSONDecodeError, TypeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

    events = sys.stdout
    # Everything downstream that prints must not reach the event stream.
    sys.stdout = sys.stderr
    try:
        run_cell(
            request.to_cell(),
            seed=request.seed,
            ablate=request.ablate,
            progress=_writer(events),  # type: ignore[arg-type]
        )
    finally:
        sys.stdout = events
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_worker.py -v`
Expected: 7 passed. The four good-run cases each cost ~19 s.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury && uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 6: Confirm stdout purity on the L1 path**

This is the case the purity guard exists for, and it needs dolfinx:

```bash
micromamba run -n vpl-t2 env PYTHONPATH=packages/vpl-core/src:packages/vpl-physics/src:packages/vpl-validation/src:packages/vpl-instruments/src:packages/vpl-inverse/src:packages/vpl-experiment/src:packages/vpl-jury/src \
  python -m vpl.jury.worker '{"seed": 0, "truth": "L1", "ablate": "oes"}' > /tmp/events.ndjson 2>/tmp/worker.err
python -c "import json,pathlib; [json.loads(l) for l in pathlib.Path('/tmp/events.ndjson').read_text().splitlines()]; print('stdout is pure JSON')"
grep -c 'ld: warning' /tmp/worker.err
```
Expected: `stdout is pure JSON`, and a non-zero count of `ld: warning` lines **on stderr** — which is the demonstration that the guard is doing something real.

- [ ] **Step 7: Commit**

```bash
git add packages/vpl-jury
git commit -m "feat(jury): the worker — one run per process, NDJSON on stdout

A subprocess because a dead Newton solve must not take the server with it,
an overrunning run needs killing and a thread cannot be, and PETSc/MPI do
not belong in an async server talking to six phones.

stdout is hijacked at entry and sys.stdout rebound to stderr: dolfinx's
JIT prints linker warnings on the L1 path, and one library update sending
them to stdout would corrupt the stream in a way that reads as a parser bug."
```

---

### Task 9: the proof chain, end to end

Tasks 4, 5 and 7 each guard one link. This asserts the whole chain over a **real run**,
through the real worker subprocess and into a real tape file — which is the only
configuration that can catch a link that works in isolation and breaks when composed.

**Files:**
- Create: `packages/vpl-jury/tests/test_proof_chain.py`
- No production code. If this task requires production changes, something in Tasks 4–8 was
  wrong, and the fix belongs in that task's file rather than here.

**Interfaces:**
- Consumes: `worker` as a subprocess, `Tape`, `assert_no_truth_before_reveal`, `SealedTruth`.
- Produces: nothing importable. This is the gate.

- [ ] **Step 1: Write the test**

Create `packages/vpl-jury/tests/test_proof_chain.py`:

```python
"""The claim, asserted over a real run: commitment, then estimate, then truth.

Tasks 4, 5 and 7 each guard one link — the allowlist, the emit order, the append-only
record. A link can be correct alone and wrong composed, so this drives the real worker
subprocess into a real tape file and asserts the property the demo actually makes.

L0 -> L0 throughout, so this needs no dolfinx and runs in the workspace environment. The
property under test is about ordering and disclosure, and neither depends on which model
produced the truth.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vpl.core.progress import ProgressEvent
from vpl.jury.events import TRUTH_BEARING_KINDS, TapeEvent, assert_no_truth_before_reveal
from vpl.jury.tape import Tape

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> list[TapeEvent]:
    """One real run, driven through the worker and recorded on a tape.

    Module-scoped: the run costs ~19 s and every assertion below reads the same recording,
    which is also more honest than nine separate runs — they would be nine different
    truths, and the property is about one.
    """
    path = tmp_path_factory.mktemp("tape") / "tape.jsonl"
    tape = Tape(path)
    run_id = tape.next_run_id()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "vpl.jury.worker",
            json.dumps({"seed": 0, "truth": "L0", "ablate": "oes"}),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    for line in completed.stdout.splitlines():
        raw = json.loads(line)
        tape.append(run_id, ProgressEvent(kind=raw["kind"], payload=raw["payload"]))

    return tape.read_all()


class TestTheOrdering:
    def test_the_commitment_is_recorded_before_the_estimate(
        self, recorded: list[TapeEvent]
    ) -> None:
        kinds = [event.kind for event in recorded]

        assert kinds.index("truth_sealed") < kinds.index("estimate_committed")

    def test_the_estimate_is_recorded_before_the_reveal(
        self, recorded: list[TapeEvent]
    ) -> None:
        kinds = [event.kind for event in recorded]

        assert kinds.index("estimate_committed") < kinds.index("seal_opened")

    def test_sequence_numbers_are_strictly_increasing(
        self, recorded: list[TapeEvent]
    ) -> None:
        # The ordering argument rests on these, so a repeat or a gap is not cosmetic.
        sequences = [event.seq for event in recorded]

        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences)


class TestTheDisclosure:
    def test_no_truth_value_appears_before_the_reveal(
        self, recorded: list[TapeEvent]
    ) -> None:
        assert_no_truth_before_reveal(recorded)

    def test_the_raw_tape_text_contains_no_truth_before_the_reveal(
        self, recorded: list[TapeEvent], tmp_path: Path
    ) -> None:
        # Belt and braces, and deliberately crude: the allowlist protects declared keys,
        # and this catches a truth value smuggled inside an otherwise-permitted field.
        reveal = next(e for e in recorded if e.kind in TRUTH_BEARING_KINDS)
        truth = repr(float(reveal.payload["gamma_e_true_w_per_m2"]))  # type: ignore[arg-type]

        earlier = [e for e in recorded if e.seq < reveal.seq]

        for event in earlier:
            assert truth not in event.to_json()


class TestTheCommitmentBinds:
    def test_the_published_digest_matches_the_revealed_value(
        self, recorded: list[TapeEvent]
    ) -> None:
        # The check a juror does by hand: take the digest that was published before the
        # estimate existed, take the value revealed afterwards, and confirm they agree.
        from vpl.validation.sealed import SealedTruth

        published = next(e for e in recorded if e.kind == "truth_sealed")
        revealed = next(e for e in recorded if e.kind == "seal_opened")

        recomputed = SealedTruth(
            value=float(revealed.payload["gamma_e_true_w_per_m2"]),  # type: ignore[arg-type]
            name="Gamma_E",
        ).commitment()

        assert recomputed == published.payload["commitment"]

    def test_the_run_reported_its_own_verification(
        self, recorded: list[TapeEvent]
    ) -> None:
        revealed = next(e for e in recorded if e.kind == "seal_opened")

        assert revealed.payload["commitment_verified"] is True


class TestDeterminism:
    def test_the_same_seed_reproduces_the_same_truth(self) -> None:
        # What a juror is invited to test by re-running their own seed. If this fails, the
        # interface's re-run affordance is a liability rather than evidence.
        def truth_of(seed: int) -> float:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "vpl.jury.worker",
                    json.dumps({"seed": seed, "truth": "L0", "ablate": "oes"}),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
            )
            for line in completed.stdout.splitlines():
                raw = json.loads(line)
                if raw["kind"] == "seal_opened":
                    return float(raw["payload"]["gamma_e_true_w_per_m2"])
            raise AssertionError("the run never revealed a truth")

        assert truth_of(0) == truth_of(0)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_proof_chain.py -v`
Expected: 9 passed. Costs ~60 s — one module-scoped run plus two in the determinism case.

- [ ] **Step 3: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury && uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add packages/vpl-jury/tests/test_proof_chain.py
git commit -m "test(jury): the proof chain, asserted over a real run

Tasks 4, 5 and 7 each guard one link. This drives the real worker
subprocess into a real tape and asserts the composed property: commitment
before estimate before reveal, no truth value on the wire beforehand, the
published digest matching the revealed value, and the same seed
reproducing the same truth.

The raw-text check is deliberately crude. The allowlist protects declared
keys; this catches a truth smuggled inside a permitted one."
```

---

### Task 10: `queue.py` — one worker, strict FIFO

**Files:**
- Create: `packages/vpl-jury/src/vpl/jury/queue.py`
- Test: `packages/vpl-jury/tests/test_queue.py`

**Interfaces:**
- Consumes: `RunRequest`, `Tape`, `ProgressEvent`.
- Produces:
  - `QueueFullError(RuntimeError)`.
  - `Submission` — frozen slots dataclass: `run_id: str`, `position: int`, `deduplicated: bool`.
  - `RunQueue(tape, *, publish, max_pending=MAX_PENDING, timeout_s=RUN_TIMEOUT_S, command=worker_command)`.
    - `publish: Callable[[TapeEvent], None]` — called for every recorded event.
    - `command: Callable[[RunRequest], Sequence[str]]` — injectable so tests need no real inversion.
  - `RunQueue.submit(self, request) -> Submission` — sync, returns immediately.
  - `RunQueue.run_forever(self) -> None` — async; drains the queue until stopped.
  - `RunQueue.stop(self) -> None` — async; cancels the in-flight subprocess and returns.
  - `MAX_PENDING: int = 20`, `RUN_TIMEOUT_S: float = 180.0`.
  - `worker_command(request) -> list[str]`.

**Why `command` is injectable:** a hermetic queue test must not cost 19 s per case. Tests
pass a command that emits canned NDJSON in milliseconds, so FIFO order, the cap, dedupe and
the timeout are each tested in isolation from the physics.

- [ ] **Step 1: Write the failing test**

Create `packages/vpl-jury/tests/test_queue.py`:

```python
"""One worker, strict FIFO, and the three refusals.

The `command` seam is what makes this hermetic: a real inversion is ~19 s and none of the
properties here are about the inversion. They are about ordering, admission control and what
happens when a subprocess misbehaves, so the subprocess is a two-line script.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from vpl.jury.events import TapeEvent
from vpl.jury.queue import QueueFullError, RunQueue
from vpl.jury.request import RunRequest
from vpl.jury.tape import Tape

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _emits(*kinds: str) -> list[str]:
    """A command that prints one event per kind and exits cleanly."""
    lines = "".join(
        f"print({json.dumps(json.dumps({'kind': kind, 'payload': {}}))});" for kind in kinds
    )
    return [sys.executable, "-c", lines]


def _hangs() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(600)"]


def _crashes() -> list[str]:
    return [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"]


def _queue(
    tmp_path: Path,
    command: object,
    *,
    max_pending: int = 20,
    timeout_s: float = 30.0,
) -> tuple[RunQueue, list[TapeEvent]]:
    published: list[TapeEvent] = []
    queue = RunQueue(
        Tape(tmp_path / "tape.jsonl"),
        publish=published.append,
        max_pending=max_pending,
        timeout_s=timeout_s,
        command=command,  # type: ignore[arg-type]
    )
    return queue, published


def _request(seed: int) -> RunRequest:
    return RunRequest.parse({"seed": seed, "truth": "L0", "ablate": "oes"})


async def _drain(queue: RunQueue) -> None:
    """Run the worker loop until the queue empties, then stop it."""
    task = asyncio.create_task(queue.run_forever())
    await queue.wait_until_idle()
    await queue.stop()
    await task


class TestSubmission:
    async def test_the_first_submission_is_at_position_one(self, tmp_path: Path) -> None:
        queue, _ = _queue(tmp_path, _emits("reference_solved"))

        submission = queue.submit(_request(0))

        assert submission.position == 1
        assert submission.run_id == "r-0001"

    async def test_positions_increase_with_the_backlog(self, tmp_path: Path) -> None:
        queue, _ = _queue(tmp_path, _emits("reference_solved"))

        queue.submit(_request(0))
        second = queue.submit(_request(1))

        assert second.position == 2

    async def test_a_duplicate_pending_request_returns_the_existing_run(
        self, tmp_path: Path
    ) -> None:
        # Two jurors typing the same seed should watch one run. At 30 s each, queueing both
        # is the difference between a responsive demo and a stalled one.
        queue, _ = _queue(tmp_path, _emits("reference_solved"))

        first = queue.submit(_request(7))
        again = queue.submit(_request(7))

        assert again.run_id == first.run_id
        assert again.deduplicated is True

    def test_the_cap_is_enforced(self, tmp_path: Path) -> None:
        queue, _ = _queue(tmp_path, _emits("reference_solved"), max_pending=2)

        queue.submit(_request(0))
        queue.submit(_request(1))

        with pytest.raises(QueueFullError, match="2"):
            queue.submit(_request(2))


class TestExecution:
    async def test_events_reach_the_tape_and_the_subscribers(self, tmp_path: Path) -> None:
        queue, published = _queue(tmp_path, _emits("truth_sealed", "seal_opened"))
        queue.submit(_request(0))

        await _drain(queue)

        assert [event.kind for event in published] == ["truth_sealed", "seal_opened"]
        assert [event.kind for event in queue.tape.read_all()] == [
            "truth_sealed",
            "seal_opened",
        ]

    async def test_runs_execute_in_submission_order(self, tmp_path: Path) -> None:
        queue, published = _queue(tmp_path, _emits("reference_solved"))
        first = queue.submit(_request(0))
        second = queue.submit(_request(1))

        await _drain(queue)

        assert [event.run_id for event in published] == [first.run_id, second.run_id]

    async def test_a_completed_run_frees_its_fingerprint_for_a_rerun(
        self, tmp_path: Path
    ) -> None:
        # Dedupe covers *pending* duplicates only. Re-running a finished seed is an
        # explicit affordance: it demonstrates determinism.
        queue, _ = _queue(tmp_path, _emits("reference_solved"))
        first = queue.submit(_request(7))
        await _drain(queue)

        again = queue.submit(_request(7))

        assert again.run_id != first.run_id
        assert again.deduplicated is False


class TestFailureIsRecorded:
    async def test_a_crash_is_recorded_rather_than_swallowed(self, tmp_path: Path) -> None:
        queue, published = _queue(tmp_path, _crashes())
        queue.submit(_request(0))

        await _drain(queue)

        kinds = [event.kind for event in published]
        assert "run_failed" in kinds
        failure = next(e for e in published if e.kind == "run_failed")
        assert failure.payload["exit_code"] == 3

    async def test_the_stderr_tail_is_carried_so_the_reason_is_visible(
        self, tmp_path: Path
    ) -> None:
        queue, published = _queue(tmp_path, _crashes())
        queue.submit(_request(0))

        await _drain(queue)

        failure = next(e for e in published if e.kind == "run_failed")
        assert "boom" in str(failure.payload["stderr_tail"])

    async def test_an_overrunning_run_is_killed_and_recorded(self, tmp_path: Path) -> None:
        queue, published = _queue(tmp_path, _hangs(), timeout_s=0.5)
        queue.submit(_request(0))

        await _drain(queue)

        assert "run_timeout" in [event.kind for event in published]

    async def test_a_failure_does_not_stop_the_queue(self, tmp_path: Path) -> None:
        # One bad run must not end the session.
        queue, published = _queue(tmp_path, _crashes())
        queue.submit(_request(0))
        queue.submit(_request(1))

        await _drain(queue)

        assert len([e for e in published if e.kind == "run_failed"]) == 2
```

Add `anyio` to the test dependency group in the root `pyproject.toml` `[dependency-groups] dev` list:

```toml
    "anyio>=4.6",
```

> `anyio`'s pytest plugin is used rather than `pytest-asyncio` because Starlette already
> depends on `anyio`, so this adds no new runtime tree — and `filterwarnings = ["error"]`
> makes `pytest-asyncio`'s default-loop-scope deprecation warning a hard failure on the
> version currently on conda-forge.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest packages/vpl-jury/tests/test_queue.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vpl.jury.queue'`

- [ ] **Step 3: Write the implementation**

Create `packages/vpl-jury/src/vpl/jury/queue.py`:

```python
"""One worker, strict FIFO, and admission control.

## Why exactly one worker

A reference inversion saturates most of the machine — the measured four-channel run takes
54 s at ~250 % CPU. Two concurrent runs do not halve the wait; they make both slower and
make the queue position meaningless. One worker with a visible position is honest about the
resource and, more usefully, means every juror is watching the same run at the same moment.
That shared view is itself part of the argument: nobody can be shown a private result.

## Why a fingerprint dedupe, and only while pending

Two jurors typing the same seed while it is queued should watch one run. Re-running a seed
that has already *finished* is different: it is an explicit demonstration that the same
input gives the same answer, so a completed fingerprint is released.

## Why failures are events

A crashed or overrunning run is recorded on the tape like anything else. The alternative is
a run that vanishes, which in a demo about not hiding things is the worst available
behaviour.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from vpl.core.progress import ProgressEvent
from vpl.jury.events import TapeEvent
from vpl.jury.request import RunRequest
from vpl.jury.tape import Tape, TapeWriteError

__all__ = [
    "MAX_PENDING",
    "RUN_TIMEOUT_S",
    "QueueFullError",
    "RunQueue",
    "Submission",
    "worker_command",
]

#: Pending runs admitted before the queue refuses. Twenty is about ten minutes of backlog
#: at the reference cost, which is longer than any judging slot — so hitting it means
#: something is wrong, not that the jury is enthusiastic.
MAX_PENDING: Final[int] = 20

#: Wall-clock ceiling on one run. The measured worst case is the 54 s four-channel
#: inversion, so this is a wide margin that still bounds a hang.
RUN_TIMEOUT_S: Final[float] = 180.0

#: How much of a failed run's stderr to carry onto the tape.
_STDERR_TAIL_CHARS: Final[int] = 2000

#: Grace period between SIGTERM and SIGKILL.
_KILL_GRACE_S: Final[float] = 2.0


class QueueFullError(RuntimeError):
    """The backlog is at :data:`MAX_PENDING`."""


@dataclass(frozen=True, slots=True)
class Submission:
    """What a phone gets back immediately."""

    run_id: str
    position: int
    deduplicated: bool


def worker_command(request: RunRequest) -> list[str]:
    """The real worker invocation.

    `sys.executable` rather than a named environment: the server already runs in the
    environment that has dolfinx, and hard-coding a name would break the moment somebody
    renames it.
    """
    import sys

    return [sys.executable, "-m", "vpl.jury.worker", json.dumps(_as_json(request))]


def _as_json(request: RunRequest) -> dict[str, object]:
    return {"seed": request.seed, "truth": request.truth, "ablate": request.ablate}


class RunQueue:
    """A FIFO of pending runs, drained one at a time."""

    def __init__(
        self,
        tape: Tape,
        *,
        publish: Callable[[TapeEvent], None],
        max_pending: int = MAX_PENDING,
        timeout_s: float = RUN_TIMEOUT_S,
        command: Callable[[RunRequest], Sequence[str]] = worker_command,
    ) -> None:
        self.tape = tape
        self._publish = publish
        self._max_pending = max_pending
        self._timeout_s = timeout_s
        self._command = command
        self._pending: list[tuple[str, RunRequest]] = []
        self._fingerprints: dict[str, str] = {}
        self._arrived = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._stopping = False
        self._process: asyncio.subprocess.Process | None = None

    def submit(self, request: RunRequest) -> Submission:
        """Admit a request, or refuse it.

        Synchronous: a phone gets its run id and queue position before any work starts.

        Raises:
            QueueFullError: The backlog is at the cap.
        """
        fingerprint = request.fingerprint()
        existing = self._fingerprints.get(fingerprint)
        if existing is not None:
            position = next(
                (index + 1 for index, (rid, _) in enumerate(self._pending) if rid == existing),
                0,
            )
            return Submission(run_id=existing, position=position, deduplicated=True)

        if len(self._pending) >= self._max_pending:
            raise QueueFullError(
                f"queue is full at {self._max_pending} pending runs. "
                f"Wait for one to finish and try again."
            )

        run_id = self.tape.next_run_id()
        self._pending.append((run_id, request))
        self._fingerprints[fingerprint] = run_id
        self._idle.clear()
        self._arrived.set()
        return Submission(run_id=run_id, position=len(self._pending), deduplicated=False)

    async def wait_until_idle(self) -> None:
        """Block until the backlog is empty and nothing is in flight."""
        await self._idle.wait()

    async def stop(self) -> None:
        """Stop draining and kill anything in flight."""
        self._stopping = True
        self._arrived.set()
        if self._process is not None and self._process.returncode is None:
            await self._terminate(self._process)

    async def run_forever(self) -> None:
        """Drain the queue until :meth:`stop`."""
        while not self._stopping:
            if not self._pending:
                self._idle.set()
                self._arrived.clear()
                await self._arrived.wait()
                continue
            run_id, request = self._pending.pop(0)
            try:
                await self._execute(run_id, request)
            finally:
                self._fingerprints.pop(request.fingerprint(), None)
        self._idle.set()

    async def _execute(self, run_id: str, request: RunRequest) -> None:
        process = await asyncio.create_subprocess_exec(
            *self._command(request),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = process
        assert process.stdout is not None
        try:
            await asyncio.wait_for(
                self._pump(run_id, process.stdout), timeout=self._timeout_s
            )
            await asyncio.wait_for(process.wait(), timeout=self._timeout_s)
        except TimeoutError:
            await self._terminate(process)
            self._record(run_id, "run_timeout", timeout_s=self._timeout_s)
            return
        finally:
            self._process = None

        if process.returncode:
            stderr = await process.stderr.read() if process.stderr is not None else b""
            self._record(
                run_id,
                "run_failed",
                exit_code=process.returncode,
                stderr_tail=stderr.decode("utf-8", "replace")[-_STDERR_TAIL_CHARS:],
            )

    async def _pump(self, run_id: str, stdout: asyncio.StreamReader) -> None:
        """Relay the worker's NDJSON onto the tape, line by line, as it arrives."""
        while True:
            line = await stdout.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                # The worker guards stdout purity; if a line still arrives malformed it is
                # a defect, and dropping it silently would hide it. Recorded as a failure.
                self._record(run_id, "run_failed", exit_code=0, stderr_tail=text[:_STDERR_TAIL_CHARS])
                continue
            self._record(run_id, str(raw["kind"]), **dict(raw.get("payload", {})))

    def _record(self, run_id: str, kind: str, **payload: object) -> None:
        try:
            stamped = self.tape.append(
                run_id, ProgressEvent(kind=kind, payload=payload)  # type: ignore[arg-type]
            )
        except TapeWriteError:
            # The tape is the audit trail; if it cannot be written the run is not
            # verifiable and the failure must surface rather than be papered over.
            raise
        self._publish(stamped)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """SIGTERM, then SIGKILL if it will not go."""
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_KILL_GRACE_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest packages/vpl-jury/tests/test_queue.py -v`
Expected: 13 passed, in a few seconds — no real inversion runs here.

- [ ] **Step 5: Verify lint and types**

Run:
```bash
uv run ruff check packages/vpl-jury && uv run mypy packages/vpl-jury
```
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add packages/vpl-jury pyproject.toml uv.lock
git commit -m "feat(jury): the FIFO queue, with one worker and visible position

One worker because a reference inversion saturates the machine: two
concurrent runs do not halve the wait, they make both slower and make the
position meaningless. The shared single view is also part of the argument
— nobody can be shown a private result.

Pending duplicates dedupe; a *completed* fingerprint is released, because
re-running a finished seed is an explicit demonstration of determinism.
Crashes and timeouts are recorded as events: a run that vanishes is the
worst available behaviour in a demo about not hiding things."
```

---

### Remaining tasks

| Task | Deliverable |
|---|---|
| 11 | `broker.py` + `server.py` — routes, SSE fanout, tape replay on connect, `Last-Event-ID` resume |
| 12 | `verify.py` + `cli.py` — re-derive truth from seed in a fresh process, compare digest, detect tampering |
| 13 | `preflight.py` — git SHA/dirty, dolfinx detection, LAN IP, QR, disk check, seed-0 self-test |
| 14 | Frontend (`index.html`, `app.js`, `app.css`) + Playwright smoke at 375 px + `fenicsx`-marked L1 smoke |

---

## Self-review of Tasks 1–5

**Spec coverage so far:** design §4.1 (both upstream additions) → Tasks 3 and 5. §4.2's
`events.py` → Task 4, with the documented relocation of the event *type* to `vpl-core`
(Task 2) to break the dependency cycle. §5's event table → Task 5's emit points, all eleven
kinds declared in Task 4. §7 leg 2 (ordering) → Task 4's `assert_no_truth_before_reveal` plus
Task 5's ordering tests. §10.1's two critical tests → Task 5 (inertness) and Task 4 +
Task 5 (no leak). Remaining spec sections map to Tasks 6–14 above.

**Placeholders:** none in Tasks 1–5 — every step carries the actual file content, the exact
command, and the expected result.

**Type consistency:** `ProgressEvent(kind, payload)` is defined in Task 2 and used with those
names in Tasks 4 and 5. `validate` and `TRUTH_BEARING_KINDS` are produced in Task 4 and
imported under those names in Task 5. `commitment()` is produced in Task 3 and called in
Task 5. `KIND_FIELDS` covers every kind Task 5 emits — checked one by one:
`config_accepted`, `truth_solved`, `reference_solved`, `measurements_synthesised`,
`truth_sealed`, `map_progress`, `estimate_committed`, `seal_opened`, `row`. The keys Task 5
passes are each present in the corresponding frozenset.

**One inconsistency found and fixed inline:** Task 5's `estimate_committed` emit omits
`half_width_fraction` and `map_n_distinct_modes`, which `KIND_FIELDS` permits. Permitted-but-
unused is fine — the allowlist is an upper bound, not a requirement — so no change needed,
but it is recorded here so a later reader does not treat the gap as a bug.
