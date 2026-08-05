"""Posterior artifacts — doc 08 §7 ("Zarr, chunked") and ADR-005.

ADR-005's leaning decision is "thinned samples (retaining ESS >= 400 per parameter) +
full summary statistics + the manifest", and its stated consequence is the thing these
tests exist to hold: **thinning must be ESS-aware rather than fixed-stride, or coverage
statistics will be biased.** Coverage is gate G-V4 and the evidence for doc 00 S4, so a
storage layer that picked a convenient stride would corrupt the one number the project
exists to defend.

The second half of the decision matters just as much and is easier to get wrong quietly:
the summaries must be computed from the **full** chains, before thinning. Summaries
recomputed from the thinned draws would be a lossy copy of a lossy copy, and would agree
with the archive well enough that nothing would ever flag it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from vpl.core.provenance import EnvironmentLockSource, Provenance, Tier
from vpl.core.state import (
    ParameterLevel,
    Posterior,
    SamplerDiagnostics,
)
from vpl.core.storage import (
    MissingProvenanceError,
    read_posterior,
    write_posterior,
)

_N_CHAINS = 4
_N_DRAWS = 1000
_NAMES = ("V_w", "T_e")


@pytest.fixture
def record() -> Provenance:
    return Provenance(
        manifest_sha256="4a7f2e91" + "0" * 56,
        git_commit="9c1d8b3" + "0" * 33,
        git_dirty=False,
        seed=20260804,
        environment_lock_hash="e81c" + "f" * 60,
        environment_lock_source=EnvironmentLockSource.UV_LOCK,
        created_utc=datetime(2026, 8, 4, 12, 30, 15, 123456, tzinfo=UTC),
        vpl_version="0.1.0",
        solver_versions={"numpyro": "0.15", "jax": "0.4.35"},
        tier=Tier.T2,
    )


def _samples() -> NDArray[np.float64]:
    rng = np.random.default_rng(20260804)
    return rng.standard_normal((_N_CHAINS, _N_DRAWS, len(_NAMES))) * np.pi


def _diagnostics(ess: float) -> SamplerDiagnostics:
    return SamplerDiagnostics(
        r_hat=dict.fromkeys(_NAMES, 1.001),
        ess=dict.fromkeys(_NAMES, ess),
        divergences=0,
        e_bfmi=(0.9, 0.91, 0.92, 0.93),
    )


def _posterior(
    *,
    ess: float = 3200.0,
    samples: NDArray[np.float64] | None = None,
    derived: dict[str, NDArray[np.float64]] | None = None,
) -> Posterior:
    return Posterior(
        samples=_samples() if samples is None else samples,
        names=_NAMES,
        levels={"V_w": ParameterLevel.CONTROL, "T_e": ParameterLevel.NUISANCE},
        tier=Tier.T2,
        diagnostics=_diagnostics(ess),
        derived={} if derived is None else derived,
    )


class TestArchivalIsEssAware:
    def test_the_stride_is_the_one_the_posterior_itself_computes(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # ADR-005: the stride must come from the ESS floor, not from a constant chosen
        # for file size. `Posterior.max_safe_stride` already implements that rule, and
        # the writer must not grow a second, differently-biased copy of it.
        posterior = _posterior()

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.stride == posterior.max_safe_stride()
        assert archived.stride > 1

    def test_the_archived_draws_are_exactly_the_thinned_ones(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        posterior = _posterior()
        expected = posterior.thin(posterior.max_safe_stride())

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.posterior.samples.tobytes() == expected.samples.tobytes()

    def test_the_archived_posterior_still_clears_the_ess_floor(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # Gate G-V3 is checked on whatever is in the archive, so what is in the archive
        # has to be able to pass it.
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.posterior.diagnostics.is_clean() is True

    def test_a_posterior_already_below_the_floor_is_refused_rather_than_thinned(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # No stride restores an ESS that was never there. Writing it anyway would put an
        # unusable posterior in an archive that is retained forever (doc 13 §5).
        with pytest.raises(ValueError, match="ESS"):
            write_posterior(tmp_path / "post.zarr", _posterior(ess=100.0), provenance=record)

    def test_a_short_chain_is_archived_whole(self, tmp_path: Path, record: Provenance) -> None:
        # When no thinning is affordable the stride is 1 and nothing is dropped; ADR-005
        # is a budget, not an obligation to discard.
        rng = np.random.default_rng(1)
        short = _posterior(ess=420.0, samples=rng.standard_normal((_N_CHAINS, 120, 2)) * np.pi)

        write_posterior(tmp_path / "post.zarr", short, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.stride == 1
        assert archived.posterior.n_draws == 120


class TestSummariesComeFromTheFullChains:
    def test_the_mean_is_the_full_chain_mean_not_the_thinned_one(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # ADR-005 keeps "full summary statistics" precisely because thinning loses tail
        # resolution. Summaries recomputed from the retained draws would reproduce the
        # loss they exist to compensate for, and would agree with the archive well enough
        # that nothing would flag it.
        stride = _posterior().max_safe_stride()
        spiked = np.zeros((_N_CHAINS, _N_DRAWS, len(_NAMES)))
        spiked[:, ::stride, :] = 1.0
        posterior = _posterior(samples=spiked)

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.summary.mean["V_w"] == posterior.mean("V_w")
        assert archived.posterior.mean("V_w") == 1.0
        assert archived.summary.mean["V_w"] != archived.posterior.mean("V_w")

    def test_the_credible_interval_is_the_full_chain_interval(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        posterior = _posterior()

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        expected = posterior.credible_interval("T_e")
        assert archived.summary.credible["T_e"].lower.tobytes() == expected.lower.tobytes()
        assert archived.summary.credible["T_e"].upper.tobytes() == expected.upper.tobytes()
        assert archived.summary.credible["T_e"].level == expected.level

    def test_the_full_run_diagnostics_are_preserved_beside_the_thinned_ones(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # The thinned posterior reports the ESS it actually has, which is the honest
        # number to gate on. The full-run ESS is what says whether the chains were long
        # enough in the first place, and only the summary can still answer that.
        posterior = _posterior()

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.summary.diagnostics == posterior.diagnostics
        assert archived.summary.n_draws == _N_DRAWS
        assert archived.posterior.diagnostics.worst_ess < posterior.diagnostics.worst_ess

    def test_derived_quantities_are_summarised_and_thinned_alongside(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        # doc 05 §10 requires Gamma_E in every emitted posterior, and it is a profile,
        # not a scalar — so the summary has to hold arrays, not just numbers.
        rng = np.random.default_rng(3)
        profile = rng.standard_normal((_N_CHAINS, _N_DRAWS, 5)) * np.pi
        posterior = _posterior(derived={"Gamma_E": profile})

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)
        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.posterior.derived_names == ("Gamma_E",)
        assert archived.summary.mean["Gamma_E"].shape == (5,)
        assert archived.summary.mean["Gamma_E"].tobytes() == posterior.mean("Gamma_E").tobytes()


class TestPosteriorRoundTrip:
    def test_a_written_posterior_reads_back_equal_to_the_thinned_one(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        posterior = _posterior()
        expected = posterior.thin(posterior.max_safe_stride())

        write_posterior(tmp_path / "post.zarr", posterior, provenance=record)

        assert read_posterior(tmp_path / "post.zarr").posterior == expected

    def test_the_parameter_levels_survive(self, tmp_path: Path, record: Provenance) -> None:
        # doc 05 §10 requires the theta_c / theta_n / theta_f split in the emitted
        # artifact: it separates the answer from the systematics that had to be inferred.
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        archived = read_posterior(tmp_path / "post.zarr")

        assert archived.posterior.level_of("V_w") is ParameterLevel.CONTROL
        assert archived.posterior.level_of("T_e") is ParameterLevel.NUISANCE

    def test_the_tier_survives_as_the_enum(self, tmp_path: Path, record: Provenance) -> None:
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        assert read_posterior(tmp_path / "post.zarr").posterior.tier is Tier.T2

    def test_the_provenance_survives(self, tmp_path: Path, record: Provenance) -> None:
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        assert read_posterior(tmp_path / "post.zarr").provenance == record


class TestPosteriorOnDisk:
    def test_the_sample_array_is_chunked(self, tmp_path: Path, record: Provenance) -> None:
        # doc 08 §7 chose Zarr for chains that are "large, appended incrementally".
        # Chunking along the draw axis, one chain per chunk, is what makes that possible.
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        group = zarr.open_group(store=tmp_path / "post.zarr", mode="r")
        samples = group["samples"]
        # `zarr.Group.__getitem__` may return a sub-group; narrowing here rather than
        # ignoring keeps the assertion honest about what it is measuring.
        assert isinstance(samples, zarr.Array)

        assert samples.chunks[0] == 1
        assert samples.chunks != samples.shape

    def test_the_writer_returns_the_path_it_wrote(self, tmp_path: Path, record: Provenance) -> None:
        written = write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)

        assert written == tmp_path / "post.zarr"
        assert written.is_dir()


class TestPosteriorRefusesToLoseProvenance:
    def test_writing_without_provenance_raises_rather_than_warning(self, tmp_path: Path) -> None:
        with pytest.raises(MissingProvenanceError):
            write_posterior(
                tmp_path / "post.zarr",
                _posterior(),
                provenance=None,  # type: ignore[arg-type]
            )

    def test_reading_a_store_whose_provenance_was_stripped_raises(
        self, tmp_path: Path, record: Provenance
    ) -> None:
        write_posterior(tmp_path / "post.zarr", _posterior(), provenance=record)
        group = zarr.open_group(store=tmp_path / "post.zarr", mode="a")
        del group.attrs["seed"]

        with pytest.raises(MissingProvenanceError, match="seed"):
            read_posterior(tmp_path / "post.zarr")
