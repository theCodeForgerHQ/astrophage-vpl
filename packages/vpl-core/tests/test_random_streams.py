"""Per-stream seeding — doc 10 §5.

    Seeds are per-stream, not global. A change to the noise model must not perturb the
    plasma solve's random sequence.

    Per-stream seeding is a small decision with large consequences. With a single global
    RNG, adding one noise source shifts every subsequent random draw, and two runs that
    should be comparable are not.

The ablation matrix of doc 07 §5.2 is the thing that stops working without this: it
switches noise sources on and off one at a time and compares the results. If switching
one off shifted the plasma solve's draws, the comparison would be measuring the RNG.
"""

from __future__ import annotations

import numpy as np
import pytest

from vpl.core.random import Stream, generator, stream_seed


class TestStreamsAreIndependent:
    def test_the_same_root_and_stream_give_the_same_sequence(self) -> None:
        a = generator(20260804, Stream.COLLISIONS).normal(size=8)
        b = generator(20260804, Stream.COLLISIONS).normal(size=8)

        np.testing.assert_array_equal(a, b)

    def test_different_streams_give_different_sequences(self) -> None:
        collisions = generator(20260804, Stream.COLLISIONS).normal(size=8)
        detector = generator(20260804, Stream.DETECTOR_NOISE).normal(size=8)

        assert not np.array_equal(collisions, detector)

    def test_different_roots_give_different_sequences(self) -> None:
        a = generator(20260804, Stream.COLLISIONS).normal(size=8)
        b = generator(20260805, Stream.COLLISIONS).normal(size=8)

        assert not np.array_equal(a, b)

    def test_drawing_from_one_stream_does_not_advance_another(self) -> None:
        # The property the whole module exists for. A generator is independent state, so
        # exhausting the detector-noise stream must leave the plasma solve untouched.
        untouched = generator(20260804, Stream.PLASMA_INIT).normal(size=4)

        noise = generator(20260804, Stream.DETECTOR_NOISE)
        noise.normal(size=10_000)
        after = generator(20260804, Stream.PLASMA_INIT).normal(size=4)

        np.testing.assert_array_equal(untouched, after)


class TestSeedsDeriveFromTheStreamName:
    def test_a_stream_seed_is_derived_from_the_name_not_a_position(self) -> None:
        # This is what makes the guarantee survive the future. If streams were numbered
        # by declaration order, inserting one would renumber every stream after it and
        # silently change every run that used them — the exact failure doc 10 §5 forbids,
        # reintroduced by the mechanism meant to prevent it.
        assert stream_seed(20260804, Stream.COLLISIONS) == stream_seed(20260804, Stream.COLLISIONS)
        assert stream_seed(20260804, Stream.COLLISIONS) != stream_seed(20260804, Stream.PHOTONS)

    def test_an_unregistered_stream_name_is_rejected(self) -> None:
        # A typo'd stream name would silently create an independent stream that nothing
        # else ever uses, and the run would still finish. Streams are enumerated.
        with pytest.raises(ValueError, match="collisons"):
            stream_seed(20260804, "collisons")  # type: ignore[arg-type]

    def test_a_negative_root_seed_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            stream_seed(-1, Stream.COLLISIONS)


class TestReproducibilityAcrossProcesses:
    def test_stream_seeds_are_stable_against_a_recorded_value(self) -> None:
        # Python's built-in hash() is salted per process, so a name-derived seed built on
        # it would differ between runs on the same machine. doc 00 E3 promises bit-for-bit
        # reproducibility, which makes that disqualifying. These recorded values fail if
        # the derivation ever changes — which it must not, silently.
        assert stream_seed(20260804, Stream.PLASMA_INIT) == 17279061067737022531
        assert stream_seed(20260804, Stream.COLLISIONS) == 15856258057404517983

    def test_first_draws_are_stable_against_a_recorded_value(self) -> None:
        draws = generator(20260804, Stream.COLLISIONS).random(3)

        np.testing.assert_allclose(
            draws, [0.043400299878447846, 0.30281820611131494, 0.9196305563811578]
        )


class TestTheStreamCatalogue:
    def test_every_stream_doc_10_5_names_exists(self) -> None:
        # doc 10 §5 lists them: "plasma initialisation, collisions, photon statistics,
        # detector noise and sampler proposals".
        names = {s.value for s in Stream}

        assert {
            "plasma_init",
            "collisions",
            "photons",
            "detector_noise",
            "sampler",
        } <= names

    def test_stream_values_are_unique(self) -> None:
        values = [s.value for s in Stream]

        assert len(values) == len(set(values))
