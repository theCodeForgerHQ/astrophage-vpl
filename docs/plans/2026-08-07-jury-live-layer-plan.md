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

### Remaining tasks

The plan continues with Tasks 6–14. Rather than pad this file, I'll write them in the same
level of detail in the next pass:

| Task | Deliverable |
|---|---|
| 6 | `request.py` — `RunRequest`, JSON ↔ `Cell`, validation of seed/channel/fidelity before anything is queued |
| 7 | `tape.py` — append-only JSONL, run IDs, replay, tolerance of a truncated final line |
| 8 | `worker.py` — subprocess entry point, NDJSON on stdout, stdout-purity test against dolfinx's JIT chatter |
| 9 | End-to-end proof chain over a real L0/T1 run: ordering, commitment-before-estimate, no leak |
| 10 | `queue.py` — FIFO, cap → 429, pending dedupe, timeout with SIGTERM→SIGKILL, cancellation |
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
