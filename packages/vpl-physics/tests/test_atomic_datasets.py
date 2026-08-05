"""The dataset registry, version lock and store — doc 09 §1, §2, §5, §6.

Doc 09 §5 is the binding constraint on everything in this file:

    The repository stores *references and loaders*, not bulk third-party data. A setup
    script fetches from the primary source at install time, records the version hash, and
    fails loudly if the upstream data has changed.

So every test here builds its own cache in ``tmp_path`` out of synthetic files. Nothing
is downloaded, nothing is committed, and the fetch path is exercised with an injected
downloader that never touches a socket — which is also how the setup script is expected
to be tested, since a loader that can only be verified with a network connection cannot
be verified in CI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vpl.core.params import ProvenanceClass
from vpl.physics.atomic import dataset as dataset_module
from vpl.physics.atomic.dataset import (
    DATASETS,
    LOCK_FILENAME,
    BundledDataError,
    DatasetId,
    DatasetLock,
    DatasetNotRecordedError,
    DatasetSpec,
    DatasetVersion,
    ElectronDatabase,
    Licence,
    LockIntegrityError,
    UpstreamDataChangedError,
    bibliography,
    fetch_dataset,
)
from vpl.physics.atomic.lxcat import ProcessType
from vpl.physics.atomic.store import AtomicDataStore

ACCESSED = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)

_LXCAT_ELECTRON = """\
DATABASE:         {database}
HOW TO REFERENCE: {database} database, www.lxcat.net, retrieved on August 5, 2026.

EFFECTIVE
Ar
 1.360000e-5
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar, Effective
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 0.000000e+0	1.000000e-20
 1.000000e+2	8.000000e-21
-----------------------------

EXCITATION
Ar -> Ar*(11.5eV)
 1.150000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + Ar*(11.5eV), Excitation
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.150000e+1	0.000000e+0
 1.000000e+2	5.000000e-22
-----------------------------

IONIZATION
Ar -> Ar^+
 1.580000e+1
SPECIES: e / Ar
PROCESS: E + Ar -> E + E + Ar+, Ionization
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.580000e+1	0.000000e+0
 1.000000e+2	3.000000e-20
-----------------------------
"""

_LXCAT_ION = """\
DATABASE:         Phelps
HOW TO REFERENCE: Phelps database, www.lxcat.net, retrieved on August 5, 2026.

ELASTIC
Ar+ -> Ar+
SPECIES: Ar+ / Ar
PROCESS: Ar+ + Ar -> Ar+ + Ar, Elastic
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.000000e-2	1.000000e-18
 1.000000e+3	1.000000e-19
-----------------------------

CHARGE EXCHANGE
Ar+ -> Ar
SPECIES: Ar+ / Ar
PROCESS: Ar+ + Ar -> Ar + Ar+, Charge exchange
COLUMNS: Energy (eV) | Cross section (m2)
-----------------------------
 1.000000e-2	8.000000e-19
 1.000000e+3	3.000000e-19
-----------------------------
"""

_NIST_ASD = (
    "element\tsp_num\tobs_wl_air(nm)\tAki(s^-1)\tAcc\tEi(eV)\tEk(eV)\t"
    "conf_k\tterm_k\tJ_k\tg_i\tg_k\n"
    "Ar\t1\t750.3869\t4.45e+07\tAA\t11.828071\t13.479770\t3s23p5\t2[1/2]*\t1\t1\t3\n"
    "Ar\t1\t811.5311\t3.31e+07\tAA\t11.548357\t13.076142\t3s23p5\t2[3/2]\t3\t5\t7\n"
)


def _payload(dataset_id: DatasetId) -> str:
    if dataset_id is DatasetId.NIST_ASD_ARGON:
        return _NIST_ASD
    if dataset_id is DatasetId.LXCAT_PHELPS_ION:
        return _LXCAT_ION
    return _LXCAT_ELECTRON.format(database=dataset_id.value)


def _populate(root: Path, *, only: tuple[DatasetId, ...] | None = None) -> DatasetLock:
    """Write a synthetic cache and the lock that describes it."""
    lock = DatasetLock(versions={})
    for dataset_id in only if only is not None else tuple(DATASETS):
        spec = DATASETS[dataset_id]
        payload = _payload(dataset_id).encode("utf-8")
        (root / spec.filename).write_bytes(payload)
        lock = lock.with_version(
            DatasetVersion.of(
                spec, payload, upstream_version="synthetic-fixture", accessed_utc=ACCESSED
            )
        )
    lock.write(root / LOCK_FILENAME)
    return lock


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    _populate(tmp_path)
    return tmp_path


# ── the catalogue ───────────────────────────────────────────────────────────────


class TestTheCatalogue:
    """doc 09 §2. What the framework knows how to consume, and on what terms."""

    def test_every_identifier_has_a_specification(self) -> None:
        assert set(DATASETS) == set(DatasetId)

    def test_three_independent_electron_sets_are_registered(self) -> None:
        # doc 09 §2.1: "Three independent electron sets are retained deliberately."
        # The spread between them is a budgeted error term (doc 06 §4, term 2), which it
        # cannot be if the framework only knows how to load one.
        assert len(ElectronDatabase) == 3
        assert {db.dataset_id for db in ElectronDatabase} <= set(DATASETS)

    def test_the_electron_sets_are_phelps_biagi_and_ist_lisbon(self) -> None:
        assert {db.value for db in ElectronDatabase} == {"phelps", "biagi", "ist-lisbon"}

    def test_the_ion_set_is_separate_from_the_electron_sets(self) -> None:
        # doc 03 §4.5 needs Ar+/Ar charge exchange, which no electron database carries.
        assert DatasetId.LXCAT_PHELPS_ION not in {db.dataset_id for db in ElectronDatabase}

    def test_every_dataset_is_a_standard_evaluated_database(self) -> None:
        # doc 09 §1: MEASURED is "from a standard evaluated database, with version and
        # access date". Anything here that were not would be an ASSUMED-class defect.
        assert all(spec.provenance_class is ProvenanceClass.MEASURED for spec in DATASETS.values())

    def test_raw_lxcat_tables_may_not_be_redistributed(self) -> None:
        # doc 09 §5, and the reason this package ships a loader rather than a data file.
        for database in ElectronDatabase:
            assert not DATASETS[database.dataset_id].licence.redistribute_raw

    def test_derived_quantities_from_lxcat_may_be_redistributed_with_attribution(self) -> None:
        assert DATASETS[DatasetId.LXCAT_PHELPS].licence.redistribute_derived

    def test_nist_asd_is_public_domain_and_fully_redistributable(self) -> None:
        licence = DATASETS[DatasetId.NIST_ASD_ARGON].licence

        assert licence.redistribute_raw
        assert licence.redistribute_derived

    def test_openadas_raw_files_may_not_be_redistributed(self) -> None:
        assert not DATASETS[DatasetId.OPENADAS_ADF11_ARGON].licence.redistribute_raw
        assert DATASETS[DatasetId.OPENADAS_ADF11_ARGON].licence.redistribute_derived

    def test_citation_keys_are_unique(self) -> None:
        keys = [spec.citation_key for spec in DATASETS.values()]

        assert len(set(keys)) == len(keys)

    def test_filenames_are_unique(self) -> None:
        names = [spec.filename for spec in DATASETS.values()]

        assert len(set(names)) == len(names)

    def test_every_specification_names_a_primary_source(self) -> None:
        assert all(spec.source_url.startswith("https://") for spec in DATASETS.values())

    def test_a_specification_with_no_citation_key_is_rejected(self) -> None:
        spec = DATASETS[DatasetId.LXCAT_PHELPS]

        with pytest.raises(ValueError, match="citation"):
            DatasetSpec(
                dataset_id=spec.dataset_id,
                database_name=spec.database_name,
                title=spec.title,
                source_url=spec.source_url,
                filename=spec.filename,
                provenance_class=spec.provenance_class,
                licence=spec.licence,
                citation_key="",
                citation_author=spec.citation_author,
                citation_note=spec.citation_note,
            )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [("filename", "", "filename"), ("source_url", "ftp://old.invalid", "https")],
    )
    def test_a_specification_that_cannot_be_fetched_or_cached_is_rejected(
        self, field: str, value: str, message: str
    ) -> None:
        spec = DATASETS[DatasetId.LXCAT_PHELPS]
        fields = {
            "dataset_id": spec.dataset_id,
            "database_name": spec.database_name,
            "title": spec.title,
            "source_url": spec.source_url,
            "filename": spec.filename,
            "provenance_class": spec.provenance_class,
            "licence": spec.licence,
            "citation_key": spec.citation_key,
            "citation_author": spec.citation_author,
            "citation_note": spec.citation_note,
        }

        with pytest.raises(ValueError, match=message):
            DatasetSpec(**{**fields, field: value})  # type: ignore[arg-type]

    def test_the_catalogue_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            DATASETS[DatasetId.LXCAT_PHELPS] = DATASETS[DatasetId.LXCAT_BIAGI]  # type: ignore[index]


# ── the recorded version ────────────────────────────────────────────────────────


class TestDatasetVersion:
    def test_the_digest_is_the_sha256_of_the_bytes_as_fetched(self) -> None:
        payload = b"whatever the upstream sent"
        version = DatasetVersion.of(
            DATASETS[DatasetId.LXCAT_PHELPS], payload, upstream_version="v1", accessed_utc=ACCESSED
        )

        assert version.sha256 == hashlib.sha256(payload).hexdigest()
        assert version.n_bytes == len(payload)

    def test_it_round_trips_through_the_plain_types_json_can_hold(self) -> None:
        version = DatasetVersion.of(
            DATASETS[DatasetId.LXCAT_PHELPS], b"x", upstream_version="v1", accessed_utc=ACCESSED
        )

        assert DatasetVersion.from_dict(version.to_dict()) == version

    def test_a_naive_access_time_is_rejected(self) -> None:
        # doc 09 §1 requires an access date. One with no offset cannot be ordered against
        # a record written on another machine, which is the only reason to keep it.
        with pytest.raises(ValueError, match="timezone"):
            DatasetVersion.of(
                DATASETS[DatasetId.LXCAT_PHELPS],
                b"x",
                upstream_version="v1",
                accessed_utc=datetime(2026, 8, 5, 12, 0),
            )

    def test_a_digest_that_is_not_a_sha256_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sha256"):
            DatasetVersion(
                dataset_id=DatasetId.LXCAT_PHELPS,
                sha256="deadbeef",
                source_url="https://example.invalid",
                upstream_version="v1",
                accessed_utc=ACCESSED,
                n_bytes=1,
            )

    def test_an_empty_upstream_version_is_rejected(self) -> None:
        # doc 09 §1: MEASURED means "with version and access date". A blank version label
        # makes the class a claim the record cannot support.
        with pytest.raises(ValueError, match="upstream_version"):
            DatasetVersion.of(
                DATASETS[DatasetId.LXCAT_PHELPS], b"x", upstream_version="", accessed_utc=ACCESSED
            )

    def test_a_zero_length_payload_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_bytes"):
            DatasetVersion(
                dataset_id=DatasetId.LXCAT_PHELPS,
                sha256="0" * 64,
                source_url="https://example.invalid",
                upstream_version="v1",
                accessed_utc=ACCESSED,
                n_bytes=0,
            )

    def test_a_malformed_record_is_rejected_on_the_way_back_in(self) -> None:
        with pytest.raises(ValueError, match="dataset_id"):
            DatasetVersion.from_dict({"dataset_id": "not.a.dataset"})

    def test_a_record_with_the_wrong_field_types_is_rejected(self) -> None:
        record = DatasetVersion.of(
            DATASETS[DatasetId.LXCAT_PHELPS], b"x", upstream_version="v1", accessed_utc=ACCESSED
        ).to_dict()

        with pytest.raises(ValueError, match="must be a string"):
            DatasetVersion.from_dict({**record, "sha256": 1})
        with pytest.raises(ValueError, match="must be an integer"):
            DatasetVersion.from_dict({**record, "n_bytes": "many"})


# ── the lock, and failing loudly ────────────────────────────────────────────────


class TestTheVersionLock:
    def test_it_round_trips_through_a_file(self, tmp_path: Path) -> None:
        lock = _populate(tmp_path)

        assert DatasetLock.read(tmp_path / LOCK_FILENAME) == lock

    def test_reading_a_lock_that_does_not_exist_says_how_to_make_one(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="fetch"):
            DatasetLock.read(tmp_path / LOCK_FILENAME)

    def test_verification_passes_on_the_bytes_that_were_recorded(self, tmp_path: Path) -> None:
        lock = _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))
        payload = _payload(DatasetId.LXCAT_PHELPS).encode("utf-8")

        assert lock.verify(DatasetId.LXCAT_PHELPS, payload).n_bytes == len(payload)

    def test_changed_upstream_data_fails_loudly(self, tmp_path: Path) -> None:
        # doc 09 §5, verbatim: "fails loudly if the upstream data has changed". Silence
        # here would mean a rerun of an archived manifest quietly used a different
        # cross-section table from the one the archived result was computed with.
        lock = _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))

        with pytest.raises(UpstreamDataChangedError) as excinfo:
            lock.verify(DatasetId.LXCAT_PHELPS, b"a revised table")

        message = str(excinfo.value)
        assert "lxcat.phelps" in message
        assert "2026-08-05" in message

    def test_the_failure_shows_both_digests(self, tmp_path: Path) -> None:
        lock = _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))

        with pytest.raises(UpstreamDataChangedError) as excinfo:
            lock.verify(DatasetId.LXCAT_PHELPS, b"a revised table")

        assert hashlib.sha256(b"a revised table").hexdigest() in str(excinfo.value)

    def test_an_unrecorded_dataset_is_not_silently_accepted(self, tmp_path: Path) -> None:
        lock = _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))

        with pytest.raises(DatasetNotRecordedError, match=r"lxcat\.biagi"):
            lock.verify(DatasetId.LXCAT_BIAGI, b"x")

    def test_recording_a_version_leaves_the_original_lock_alone(self, tmp_path: Path) -> None:
        lock = _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))

        extended = lock.with_version(
            DatasetVersion.of(
                DATASETS[DatasetId.LXCAT_BIAGI], b"x", upstream_version="v1", accessed_utc=ACCESSED
            )
        )

        assert len(lock) == 1
        assert len(extended) == 2

    def test_the_lock_file_carries_a_digest_of_itself(self, tmp_path: Path) -> None:
        # Integrated with vpl.core.provenance.manifest_sha256, which is the same
        # canonical hash the run identity of doc 08 §7 is built from. A lock that can be
        # edited undetectably is not a lock.
        _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))
        document = json.loads((tmp_path / LOCK_FILENAME).read_text(encoding="utf-8"))

        assert len(document["lock_sha256"]) == 64

    def test_an_edited_lock_file_is_detected(self, tmp_path: Path) -> None:
        _populate(tmp_path, only=(DatasetId.LXCAT_PHELPS,))
        path = tmp_path / LOCK_FILENAME
        document = json.loads(path.read_text(encoding="utf-8"))
        document["datasets"]["lxcat.phelps"]["sha256"] = "0" * 64
        path.write_text(json.dumps(document), encoding="utf-8")

        with pytest.raises(LockIntegrityError, match="edited"):
            DatasetLock.read(path)

    def test_a_lock_file_that_is_not_a_lock_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / LOCK_FILENAME
        path.write_text('{"datasets": {}}', encoding="utf-8")

        with pytest.raises(LockIntegrityError, match="lock_sha256"):
            DatasetLock.read(path)

    def test_a_lock_whose_datasets_field_is_not_a_mapping_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / LOCK_FILENAME
        path.write_text(json.dumps({"lock_sha256": "0" * 64, "datasets": []}), encoding="utf-8")

        with pytest.raises(LockIntegrityError, match="mapping"):
            DatasetLock.read(path)

    def test_the_recorded_identifiers_are_reported_in_a_stable_order(self, cache: Path) -> None:
        lock = DatasetLock.read(cache / LOCK_FILENAME)

        assert lock.ids() == tuple(sorted(lock.ids()))
        assert DatasetId.LXCAT_PHELPS in lock

    def test_the_repr_names_what_is_locked(self, cache: Path) -> None:
        assert "lxcat.phelps" in repr(DatasetLock.read(cache / LOCK_FILENAME))

    def test_a_lock_is_not_equal_to_something_that_is_not_a_lock(self, cache: Path) -> None:
        assert DatasetLock.read(cache / LOCK_FILENAME) != "not a lock"


# ── fetching, which is the only step that would touch a network ─────────────────


class TestFetching:
    def test_the_downloader_is_injected_and_is_the_only_thing_that_leaves_the_process(
        self, tmp_path: Path
    ) -> None:
        # There is no default downloader. A loader with a hard-wired urlopen cannot be
        # tested offline, and doc 09 §5 puts the fetch in a setup script rather than in
        # the import path precisely so that everything else stays testable without it.
        requested: list[str] = []

        def download(url: str) -> bytes:
            requested.append(url)
            return b"payload"

        spec = DATASETS[DatasetId.LXCAT_PHELPS]
        version = fetch_dataset(
            spec, download=download, destination=tmp_path, upstream_version="2026-08-05"
        )

        assert requested == [spec.source_url]
        assert (tmp_path / spec.filename).read_bytes() == b"payload"
        assert version.sha256 == hashlib.sha256(b"payload").hexdigest()

    def test_the_access_time_is_recorded_at_fetch(self, tmp_path: Path) -> None:
        version = fetch_dataset(
            DATASETS[DatasetId.LXCAT_PHELPS],
            download=lambda _: b"payload",
            destination=tmp_path,
            upstream_version="v1",
            now=lambda: ACCESSED,
        )

        assert version.accessed_utc == ACCESSED

    def test_the_default_clock_is_timezone_aware(self, tmp_path: Path) -> None:
        version = fetch_dataset(
            DATASETS[DatasetId.LXCAT_PHELPS],
            download=lambda _: b"payload",
            destination=tmp_path,
            upstream_version="v1",
        )

        assert version.accessed_utc.tzinfo is not None

    def test_an_empty_download_is_rejected_rather_than_recorded(self, tmp_path: Path) -> None:
        # A proxy error page or a truncated transfer hashes just as well as real data.
        with pytest.raises(ValueError, match="empty"):
            fetch_dataset(
                DATASETS[DatasetId.LXCAT_PHELPS],
                download=lambda _: b"",
                destination=tmp_path,
                upstream_version="v1",
            )

    def test_it_refuses_to_write_third_party_data_into_the_installed_package(self) -> None:
        # This is doc 09 §5 as an executable rule. Caching inside the package directory
        # is how bulk third-party data ends up inside a wheel, which for LXCat and
        # OpenADAS is a licence breach and for all of them is against the architecture.
        inside = Path(dataset_module.__file__).parent

        with pytest.raises(BundledDataError, match="doc 09"):
            fetch_dataset(
                DATASETS[DatasetId.LXCAT_PHELPS],
                download=lambda _: b"payload",
                destination=inside,
                upstream_version="v1",
            )


# ── the store ───────────────────────────────────────────────────────────────────


class TestTheStore:
    def test_it_opens_from_a_cache_directory(self, cache: Path) -> None:
        assert len(AtomicDataStore.open(cache).lock) == len(DATASETS)

    def test_it_refuses_a_root_inside_the_installed_package(self) -> None:
        with pytest.raises(BundledDataError, match="doc 09"):
            AtomicDataStore.open(Path(dataset_module.__file__).parent)

    def test_a_missing_file_names_the_dataset_and_the_setup_step(self, tmp_path: Path) -> None:
        _populate(tmp_path)
        (tmp_path / DATASETS[DatasetId.LXCAT_BIAGI].filename).unlink()
        store = AtomicDataStore.open(tmp_path)

        with pytest.raises(FileNotFoundError, match=r"lxcat\.biagi"):
            store.electron_cross_sections(ElectronDatabase.BIAGI)

    def test_a_modified_cache_file_fails_loudly_on_read(self, cache: Path) -> None:
        (cache / DATASETS[DatasetId.LXCAT_PHELPS].filename).write_text("edited", encoding="utf-8")
        store = AtomicDataStore.open(cache)

        with pytest.raises(UpstreamDataChangedError):
            store.electron_cross_sections(ElectronDatabase.PHELPS)

    def test_an_electron_set_parses_into_cross_sections(self, cache: Path) -> None:
        sections = AtomicDataStore.open(cache).electron_cross_sections(ElectronDatabase.PHELPS)

        assert sections.process_types() >= {ProcessType.EXCITATION, ProcessType.IONIZATION}

    def test_the_ion_set_carries_charge_exchange(self, cache: Path) -> None:
        assert AtomicDataStore.open(cache).ion_cross_sections().charge_exchange().target == "Ar"

    def test_the_line_list_parses(self, cache: Path) -> None:
        lines = AtomicDataStore.open(cache).argon_lines()

        assert len(lines) == 2

    def test_the_line_list_is_returned_from_cache_on_the_second_call(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        assert store.argon_lines() is store.argon_lines()

    def test_it_reports_the_cache_it_was_opened_on(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        assert store.root == cache.resolve()
        assert str(cache.resolve()) in repr(store)

    def test_all_three_electron_sets_are_reachable_in_one_loop(self, cache: Path) -> None:
        # doc 09 §2.1 wants the inference run under all three and the spread reported.
        # If that needs three hand-written call sites it will be done once and then not
        # again, which is how a budgeted error term becomes an unstated risk.
        store = AtomicDataStore.open(cache)

        databases = [database for database, _ in store.each_electron_set()]

        assert databases == list(ElectronDatabase)

    def test_each_electron_set_yields_a_parsed_set(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        assert all(len(sections) == 3 for _, sections in store.each_electron_set())

    def test_a_parsed_dataset_is_returned_from_cache_on_the_second_call(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        first = store.electron_cross_sections(ElectronDatabase.PHELPS)

        assert store.electron_cross_sections(ElectronDatabase.PHELPS) is first

    def test_reading_raw_bytes_verifies_them_too(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        assert store.read_bytes(DatasetId.OPENADAS_ADF11_ARGON)

    def test_the_path_of_a_dataset_is_reportable_without_reading_it(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)

        assert store.path(DatasetId.LXCAT_PHELPS).name == DATASETS[DatasetId.LXCAT_PHELPS].filename
        assert store.touched == ()


# ── the citation ledger ─────────────────────────────────────────────────────────


class TestTheCitationLedger:
    """doc 09 §6: cite what the run touched, not a static list."""

    def test_nothing_is_touched_before_anything_is_read(self, cache: Path) -> None:
        assert AtomicDataStore.open(cache).touched == ()

    def test_reading_a_dataset_records_it(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.electron_cross_sections(ElectronDatabase.PHELPS)

        assert store.touched == (DatasetId.LXCAT_PHELPS,)

    def test_a_dataset_is_recorded_once_however_often_it_is_read(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.electron_cross_sections(ElectronDatabase.PHELPS)
        store.electron_cross_sections(ElectronDatabase.PHELPS)

        assert store.touched == (DatasetId.LXCAT_PHELPS,)

    def test_a_run_that_used_only_phelps_does_not_cite_biagi(self, cache: Path) -> None:
        # doc 09 §6, verbatim: "A run that used only the Phelps set does not cite Biagi."
        store = AtomicDataStore.open(cache)
        store.electron_cross_sections(ElectronDatabase.PHELPS)

        entries = store.bibliography()

        assert DATASETS[DatasetId.LXCAT_PHELPS].citation_key in entries
        assert DATASETS[DatasetId.LXCAT_BIAGI].citation_key not in entries

    def test_the_entry_records_the_retrieval_date_and_the_digest(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.argon_lines()
        version = store.lock.versions[DatasetId.NIST_ASD_ARGON]

        entries = store.bibliography()

        assert "2026-08-05" in entries
        assert version.sha256[:12] in entries

    def test_the_lxcat_required_wording_survives_into_the_entry(self, cache: Path) -> None:
        # doc 09 §2.1: LXCat is free but "citation of the specific database is required",
        # and the database states the wording it wants.
        store = AtomicDataStore.open(cache)
        store.electron_cross_sections(ElectronDatabase.BIAGI)

        assert "www.lxcat.net" in store.bibliography()

    def test_the_ledger_is_emitted_in_a_stable_order(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.electron_cross_sections(ElectronDatabase.PHELPS)
        store.argon_lines()

        first = store.bibliography()
        store.electron_cross_sections(ElectronDatabase.PHELPS)

        assert store.bibliography() == first

    def test_the_whole_catalogue_can_be_written_as_the_accumulated_ledger(self) -> None:
        # doc 09 §6: refs/CITATIONS.bib accumulates every source; the report emits the
        # subset. Both come from the same renderer so they cannot disagree.
        entries = bibliography(DATASETS.values())

        assert entries.count("@misc{") == len(DATASETS)

    def test_an_entry_is_syntactically_a_bibtex_record(self) -> None:
        entry = DATASETS[DatasetId.NIST_ASD_ARGON].bibtex()

        assert entry.startswith("@misc{")
        assert entry.rstrip().endswith("}")
        assert "title" in entry

    def test_an_entry_without_a_version_omits_the_digest_rather_than_inventing_one(self) -> None:
        entry = DATASETS[DatasetId.NIST_ASD_ARGON].bibtex()

        assert "SHA-256" not in entry

    def test_the_touched_specifications_are_available_for_a_report(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.ion_cross_sections()

        specs: tuple[DatasetSpec, ...] = store.touched_specs()

        assert [spec.dataset_id for spec in specs] == [DatasetId.LXCAT_PHELPS_ION]

    def test_the_touched_set_can_be_reset_between_runs(self, cache: Path) -> None:
        store = AtomicDataStore.open(cache)
        store.ion_cross_sections()
        store.reset_touched()

        assert store.touched == ()


class TestTheLicenceModel:
    def test_a_licence_that_permits_raw_but_not_derived_redistribution_is_rejected(self) -> None:
        # Every licence in doc 09 §5 that permits the raw tables also permits anything
        # derived from them. The reverse combination is not a licence anyone grants, and
        # accepting it would let a typo produce a rule the code would then enforce.
        with pytest.raises(ValueError, match="derived"):
            Licence(
                name="nonsense",
                terms="raw yes, derived no",
                redistribute_raw=True,
                redistribute_derived=False,
            )

    def test_the_licence_terms_are_carried_verbatim_from_doc_09(self) -> None:
        licences: Mapping[DatasetId, str] = {
            spec.dataset_id: spec.licence.terms for spec in DATASETS.values()
        }

        assert "citation" in licences[DatasetId.LXCAT_PHELPS].lower()
        assert "public domain" in licences[DatasetId.NIST_ASD_ARGON].lower()
