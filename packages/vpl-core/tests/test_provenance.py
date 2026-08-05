"""Artifact provenance — doc 08 §7.

The contract: every artifact the pipeline emits can name the manifest, the commit, the
environment, the seed and the tier that produced it. Doc 00 E3 promises bit-for-bit
reproducibility from exactly that tuple, so the parts of it that are cheap to get
subtly wrong are the parts these tests pin — the canonical hash, and what the record
says when the environment cannot be interrogated.

None of these tests may depend on the state of the machine's own checkout. The two that
touch the real repository assert only what the repository guarantees about itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from vpl import core
from vpl.core import provenance
from vpl.core.provenance import (
    UNKNOWN_COMMIT,
    EnvironmentLockSource,
    Provenance,
    Tier,
    environment_lock_hash,
    git_state,
    manifest_sha256,
)

# Stand-in digests. Real ones come from `manifest_sha256`; these only need the shape.
_MANIFEST_DIGEST = "4a7f2e91" + "0" * 56
_LOCK_DIGEST = "e81c" + "f" * 60

_REPO_ROOT = Path(__file__).resolve().parents[3]

_NEEDS_GIT = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _record(**overrides: Any) -> Provenance:
    """A complete record, so that each test can vary the one field it is about."""
    fields: dict[str, Any] = {
        "manifest_sha256": _MANIFEST_DIGEST,
        "git_commit": "9c1d8b3" + "0" * 33,
        "git_dirty": False,
        "seed": 20260804,
        "environment_lock_hash": _LOCK_DIGEST,
        "environment_lock_source": EnvironmentLockSource.UV_LOCK,
        "created_utc": datetime(2026, 8, 4, 12, 0, 0, tzinfo=UTC),
        "vpl_version": "0.1.0",
        "solver_versions": {"dolfinx": "0.8.0", "petsc": "3.20"},
        "tier": Tier.T2,
    }
    return Provenance(**(fields | overrides))


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Run git in a repository isolated from the developer's own git configuration.

    A global `commit.gpgsign` or `core.excludesfile` on the machine running the suite
    would otherwise change what these tests observe, which is the dependency the suite
    is not allowed to have.
    """
    env = os.environ | {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    return subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)


@pytest.fixture
def committed_repo(tmp_path: Path) -> Path:
    """A repository with one commit and a clean working tree."""
    _git(tmp_path, "init", "--quiet")
    (tmp_path / "tracked.txt").write_text("first\n")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


class TestTier:
    def test_the_labels_are_the_ones_reports_and_figures_carry(self) -> None:
        # doc 05 §7.2 and doc 13 §3.2: the tier label is rendered onto every accuracy
        # figure, so the enum's values are the label text, not an internal code.
        assert (Tier.T0.value, Tier.T1.value, Tier.T2.value) == ("T0", "T1", "T2")

    def test_a_tier_is_a_string_so_it_can_be_stored_as_an_hdf5_attribute(self) -> None:
        assert isinstance(Tier.T2, str)
        assert str(Tier.T2) == "T2"

    def test_an_unrecognised_label_is_rejected(self) -> None:
        # doc 05 §7.2 calls reporting T1 as if it were T2 a project defect. A tier that
        # parses loosely is the first step towards exactly that.
        with pytest.raises(ValueError, match="T3"):
            Tier("T3")


class TestManifestSha256:
    def test_key_order_does_not_change_the_hash(self) -> None:
        # Two manifests that differ only in how a YAML loader happened to order the keys
        # are the same experiment, and must be recognised as a rerun rather than a
        # new run.
        assert manifest_sha256({"seed": 1, "tier": "T2"}) == manifest_sha256(
            {"tier": "T2", "seed": 1}
        )

    def test_nested_key_order_does_not_change_the_hash(self) -> None:
        assert manifest_sha256({"plasma": {"Te": 3.0, "gas": "argon"}}) == manifest_sha256(
            {"plasma": {"gas": "argon", "Te": 3.0}}
        )

    def test_changing_a_value_changes_the_hash(self) -> None:
        assert manifest_sha256({"seed": 1}) != manifest_sha256({"seed": 2})

    def test_a_one_ulp_float_difference_changes_the_hash(self) -> None:
        # doc 00 E3 promises bit-for-bit reproducibility. A hash that rounded before
        # digesting would let two runs that genuinely diverged claim one identity.
        nudged = math.nextafter(1.0, math.inf)

        assert manifest_sha256({"Te": 1.0}) != manifest_sha256({"Te": nudged})

    def test_list_order_changes_the_hash(self) -> None:
        # Unlike mapping keys, sequence order is meaningful: doc 08 §6 lists instruments
        # in the order they are applied.
        assert manifest_sha256({"channels": ["oes", "lif"]}) != manifest_sha256(
            {"channels": ["lif", "oes"]}
        )

    def test_a_boolean_does_not_hash_like_its_integer_value(self) -> None:
        # `seal_truth: true` and `seal_truth: 1` are not the same manifest, even though
        # Python considers True == 1.
        assert manifest_sha256({"seal_truth": True}) != manifest_sha256({"seal_truth": 1})

    def test_the_digest_is_sha256_over_the_canonical_json_encoding(self) -> None:
        # Pins the canonical form itself, not merely self-consistency: sorted keys, no
        # whitespace, UTF-8. Without this, a later change of separators would silently
        # renumber every archived run.
        payload = {"b": 1, "a": {"d": [1, 2], "c": True}}
        canonical = b'{"a":{"c":true,"d":[1,2]},"b":1}'

        assert manifest_sha256(payload) == hashlib.sha256(canonical).hexdigest()

    def test_returns_a_full_length_lowercase_hex_digest(self) -> None:
        digest = manifest_sha256({"seed": 1})

        assert len(digest) == 64
        assert digest == digest.lower()

    def test_a_non_finite_float_is_rejected(self) -> None:
        # JSON has no spelling for nan, and the non-standard one Python emits by default
        # cannot be re-read by another tool — so the archive would hold a hash nobody
        # else could reproduce.
        with pytest.raises(ValueError, match=r"nan|inf|finite"):
            manifest_sha256({"Te": float("nan")})

    def test_a_value_json_cannot_represent_is_rejected_by_name(self) -> None:
        with pytest.raises(TypeError, match="datetime"):
            manifest_sha256({"created": datetime(2026, 8, 4, tzinfo=UTC)})


class TestGitState:
    @_NEEDS_GIT
    def test_reports_the_head_commit_of_a_clean_tree(self, committed_repo: Path) -> None:
        commit, dirty = git_state(committed_repo)

        assert len(commit) == 40
        assert int(commit, 16) >= 0  # a full hexadecimal object name
        assert dirty is False

    @_NEEDS_GIT
    def test_reports_dirty_when_a_tracked_file_is_modified(self, committed_repo: Path) -> None:
        (committed_repo / "tracked.txt").write_text("second\n")

        _, dirty = git_state(committed_repo)

        assert dirty is True

    @_NEEDS_GIT
    def test_an_untracked_file_also_counts_as_dirty(self, committed_repo: Path) -> None:
        # An untracked module can be imported by the run and will not exist for anyone
        # else, so it is exactly as unreproducible as an uncommitted edit.
        (committed_repo / "stray_module.py").write_text("x = 1\n")

        _, dirty = git_state(committed_repo)

        assert dirty is True

    @_NEEDS_GIT
    def test_a_directory_that_is_not_a_repository_is_unknown_and_assumed_dirty(
        self, tmp_path: Path
    ) -> None:
        commit, dirty = git_state(tmp_path)

        assert commit == UNKNOWN_COMMIT
        assert dirty is True

    @_NEEDS_GIT
    def test_a_repository_with_no_commit_yet_is_unknown_and_assumed_dirty(
        self, tmp_path: Path
    ) -> None:
        _git(tmp_path, "init", "--quiet")

        commit, dirty = git_state(tmp_path)

        assert commit == UNKNOWN_COMMIT
        assert dirty is True

    def test_an_absent_git_binary_is_unknown_and_assumed_dirty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Claiming a clean tree that could not be verified is the failure mode that
        # matters: doc 13 §2 makes `git_dirty: false` the thing that lets a run into the
        # release archive.
        empty = tmp_path / "no-tools"
        empty.mkdir()
        monkeypatch.setenv("PATH", str(empty))

        assert git_state(tmp_path) == (UNKNOWN_COMMIT, True)

    def test_a_commit_whose_cleanliness_cannot_be_established_is_assumed_dirty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The half-failure: HEAD resolves but the status call does not. Reached only by
        # stubbing, and worth pinning anyway — it is the branch that decides whether an
        # unverified tree can reach the doc 13 §2 release gate.
        known = "9c1d8b3" + "0" * 33
        answers = iter([known, None])
        monkeypatch.setattr(provenance, "_git_output", lambda *_args: next(answers))

        assert git_state(Path()) == (known, True)

    @_NEEDS_GIT
    def test_discovers_the_surrounding_checkout_when_given_no_root(self) -> None:
        # The production path: `Provenance.capture()` is normally called without a root,
        # from a working directory that may be anywhere under the repository.
        if not (_REPO_ROOT / ".git").exists():
            pytest.skip("not running from a git checkout")

        assert git_state() == git_state(_REPO_ROOT)

    @_NEEDS_GIT
    def test_the_real_repository_reports_a_full_commit_sha(self) -> None:
        # The one test that exercises the real path. It asserts what this checkout
        # guarantees about itself and nothing about whether it happens to be clean.
        if not (_REPO_ROOT / ".git").exists():
            pytest.skip("not running from a git checkout")

        commit, dirty = git_state(_REPO_ROOT)

        assert len(commit) == 40
        assert isinstance(dirty, bool)


class TestEnvironmentLockHash:
    def test_prefers_the_lock_file_and_names_it_as_the_source(self, tmp_path: Path) -> None:
        lock = tmp_path / "uv.lock"
        lock.write_bytes(b"version = 1\n")

        digest, source = environment_lock_hash(tmp_path)

        assert source is EnvironmentLockSource.UV_LOCK
        assert digest == hashlib.sha256(lock.read_bytes()).hexdigest()

    def test_the_digest_tracks_the_lock_contents(self, tmp_path: Path) -> None:
        lock = tmp_path / "uv.lock"
        lock.write_bytes(b"version = 1\n")
        before, _ = environment_lock_hash(tmp_path)

        lock.write_bytes(b"version = 2\n")
        after, _ = environment_lock_hash(tmp_path)

        assert before != after

    def test_falls_back_to_the_installed_distributions_and_says_so(self, tmp_path: Path) -> None:
        # A weaker guarantee than a resolved lock — it pins versions but not the
        # resolution that produced them — so the record must let a reader tell which
        # one they are holding.
        digest, source = environment_lock_hash(tmp_path)

        assert source is EnvironmentLockSource.INSTALLED_DISTRIBUTIONS
        assert len(digest) == 64

    def test_the_fallback_is_stable_within_one_environment(self, tmp_path: Path) -> None:
        # Distribution scan order is filesystem-dependent; unsorted, the same
        # environment would hash differently on consecutive runs.
        assert environment_lock_hash(tmp_path) == environment_lock_hash(tmp_path)

    def test_discovers_the_surrounding_checkout_when_given_no_root(self) -> None:
        if not (_REPO_ROOT / "uv.lock").exists():
            pytest.skip("no uv.lock in this checkout")

        assert environment_lock_hash() == environment_lock_hash(_REPO_ROOT)

    def test_the_real_repository_is_pinned_by_its_lock_file(self) -> None:
        lock = _REPO_ROOT / "uv.lock"
        if not lock.exists():
            pytest.skip("no uv.lock in this checkout")

        digest, source = environment_lock_hash(_REPO_ROOT)

        assert source is EnvironmentLockSource.UV_LOCK
        assert digest == hashlib.sha256(lock.read_bytes()).hexdigest()


class TestCapture:
    def test_fills_the_environment_derived_fields(self, committed_repo: Path) -> None:
        before = datetime.now(UTC)

        record = Provenance.capture(
            manifest_sha256=_MANIFEST_DIGEST,
            seed=20260804,
            tier=Tier.T1,
            solver_versions={"petsc": "3.20"},
            repo_root=committed_repo,
        )

        assert record.git_commit == git_state(committed_repo)[0]
        assert record.git_dirty is False
        assert record.vpl_version == core.__version__
        assert before <= record.created_utc <= datetime.now(UTC)

    def test_created_utc_is_timezone_aware_and_in_utc(self, tmp_path: Path) -> None:
        # A naive timestamp in an archive is unorderable against one written in another
        # timezone, which is the whole point of recording it.
        record = Provenance.capture(
            manifest_sha256=_MANIFEST_DIGEST,
            seed=1,
            tier=Tier.T0,
            solver_versions={},
            repo_root=tmp_path,
        )

        assert record.created_utc.tzinfo is not None
        assert record.created_utc.utcoffset() == timedelta(0)

    def test_passes_the_callers_fields_through_unchanged(self, tmp_path: Path) -> None:
        record = Provenance.capture(
            manifest_sha256=_MANIFEST_DIGEST,
            seed=20260804,
            tier=Tier.T1,
            solver_versions={"numpyro": "0.15"},
            repo_root=tmp_path,
        )

        assert record.manifest_sha256 == _MANIFEST_DIGEST
        assert record.seed == 20260804
        assert record.tier is Tier.T1
        assert dict(record.solver_versions) == {"numpyro": "0.15"}

    def test_later_mutation_of_the_callers_dict_cannot_rewrite_history(
        self, tmp_path: Path
    ) -> None:
        versions = {"petsc": "3.20"}

        record = Provenance.capture(
            manifest_sha256=_MANIFEST_DIGEST,
            seed=1,
            tier=Tier.T2,
            solver_versions=versions,
            repo_root=tmp_path,
        )
        versions["petsc"] = "3.21"

        assert record.solver_versions["petsc"] == "3.20"


class TestRecordIsImmutable:
    def test_a_field_cannot_be_reassigned(self) -> None:
        record = _record()

        with pytest.raises(AttributeError):
            record.tier = Tier.T2  # type: ignore[misc]

    def test_carries_no_instance_dict(self) -> None:
        # `slots=True`: a typo'd attribute must fail rather than attach itself to a
        # record that is about to be written into an archive.
        assert not hasattr(_record(), "__dict__")

    def test_solver_versions_cannot_be_mutated_through_the_record(self) -> None:
        record = _record()

        with pytest.raises(TypeError, match="item assignment"):
            record.solver_versions["petsc"] = "3.21"  # type: ignore[index]

    def test_records_with_the_same_contents_are_equal(self) -> None:
        assert _record() == _record()

    def test_records_differing_in_solver_versions_are_not_equal(self) -> None:
        assert _record() != _record(solver_versions={"petsc": "3.21"})

    def test_is_hashable_despite_carrying_a_mapping(self) -> None:
        # Provenance records are used as dictionary keys when grouping artifacts by run.
        assert len({_record(), _record()}) == 1

    def test_equal_records_hash_equally(self) -> None:
        assert hash(_record()) == hash(_record())


class TestRecordValidation:
    def test_rejects_a_manifest_hash_that_is_not_a_sha256_digest(self) -> None:
        # Catches the plausible mistake of storing the manifest's path or a truncated
        # display hash, which would be indistinguishable from a real one in an archive.
        with pytest.raises(ValueError, match="manifest_sha256"):
            _record(manifest_sha256="experiments/b02.yaml")

    def test_rejects_an_environment_lock_hash_that_is_not_a_sha256_digest(self) -> None:
        with pytest.raises(ValueError, match="environment_lock_hash"):
            _record(environment_lock_hash="uv.lock")

    def test_rejects_a_naive_timestamp(self) -> None:
        with pytest.raises(ValueError, match="created_utc"):
            _record(created_utc=datetime(2026, 8, 4, 12, 0, 0))

    def test_normalises_a_timestamp_given_in_another_zone(self) -> None:
        offset = timezone(timedelta(hours=5, minutes=30))

        record = _record(created_utc=datetime(2026, 8, 4, 17, 30, tzinfo=offset))

        assert record.created_utc == datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
        assert record.created_utc.utcoffset() == timedelta(0)

    def test_rejects_a_negative_seed(self) -> None:
        # A seed is an index into a generator's state, not a signed quantity; numpy
        # rejects it far downstream, where the failure costs a whole run.
        with pytest.raises(ValueError, match="seed"):
            _record(seed=-1)


class TestRoundTrip:
    def test_to_dict_then_from_dict_returns_an_equal_record(self) -> None:
        record = _record()

        assert Provenance.from_dict(record.to_dict()) == record

    def test_to_dict_carries_exactly_the_documented_fields(self) -> None:
        documented = {
            "manifest_sha256",
            "git_commit",
            "git_dirty",
            "seed",
            "environment_lock_hash",
            "created_utc",
            "vpl_version",
            "solver_versions",
            "tier",
        }

        keys = set(_record().to_dict())

        assert documented <= keys
        # The single deliberate addition to doc 08 §7: how strong the lock hash is.
        assert keys - documented == {"environment_lock_source"}

    def test_to_dict_yields_only_types_a_yaml_sidecar_can_hold(self) -> None:
        # These land in HDF5 attributes and YAML sidecars (doc 08 §7), neither of which
        # can store an enum member or a datetime object.
        for key, value in _record().to_dict().items():
            assert isinstance(value, str | bool | int | dict), key
            assert not isinstance(value, Tier | EnvironmentLockSource), key

    def test_to_dict_is_json_serialisable(self) -> None:
        assert json.loads(json.dumps(_record().to_dict()))["tier"] == "T2"

    def test_the_serialised_timestamp_is_iso_8601_with_an_explicit_offset(self) -> None:
        created = _record().to_dict()["created_utc"]

        assert isinstance(created, str)
        assert datetime.fromisoformat(created) == datetime(2026, 8, 4, 12, 0, tzinfo=UTC)

    def test_from_dict_rejects_a_missing_field(self) -> None:
        incomplete = _record().to_dict()
        del incomplete["tier"]

        with pytest.raises(ValueError, match="tier"):
            Provenance.from_dict(incomplete)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("seed", "20260804"),
            ("git_commit", 12345),
            ("git_dirty", "false"),
            ("solver_versions", ["petsc", "3.20"]),
        ],
    )
    def test_from_dict_rejects_a_field_of_the_wrong_type(self, key: str, value: object) -> None:
        # A record that has been through YAML may have been hand-edited on the way, and
        # `git_dirty: "false"` is truthy — the one silent misreading that would let an
        # unreproducible run past the doc 13 §2 gate.
        wrong = _record().to_dict() | {key: value}

        with pytest.raises(TypeError, match=key):
            Provenance.from_dict(wrong)

    def test_from_dict_rejects_an_unparseable_timestamp(self) -> None:
        wrong = _record().to_dict() | {"created_utc": "last Tuesday"}

        with pytest.raises(ValueError, match="created_utc"):
            Provenance.from_dict(wrong)

    def test_from_dict_rejects_an_unrecognised_tier(self) -> None:
        wrong = _record().to_dict() | {"tier": "T2-ish"}

        with pytest.raises(ValueError, match="tier"):
            Provenance.from_dict(wrong)

    def test_from_dict_rejects_an_unrecognised_environment_lock_source(self) -> None:
        wrong = _record().to_dict() | {"environment_lock_source": "vibes"}

        with pytest.raises(ValueError, match="environment_lock_source"):
            Provenance.from_dict(wrong)
