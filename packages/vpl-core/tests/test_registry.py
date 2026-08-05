"""Plugin discovery — doc 08 §10.

The contract: a capability arrives as an installed distribution declaring an entry point,
and the core neither knows nor changes (doc 08 §1 principle 3, doc 00 E1). These tests
never rely on what happens to be installed in the working environment — every one of them
stubs the metadata scan, so the suite states the contract rather than the machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib.metadata import EntryPoint

import pytest

from vpl.core import registry
from vpl.core.registry import (
    PluginConflictError,
    PluginGroup,
    PluginLoadError,
    PluginNotFoundError,
    available,
    clear_registrations,
    discover,
    load,
    register,
)

# Entry-point targets used as stand-in plugins. They point at the standard library so the
# tests exercise the real import machinery without shipping a fixture package.
_REAL = "json:JSONDecoder"
_ALSO_REAL = "json:JSONEncoder"
_MISSING_MODULE = "vpl_plugin_nobody_installed:Solver"
_MISSING_ATTRIBUTE = "json:NoSuchSolver"


@pytest.fixture(autouse=True)
def _isolated_registry() -> Iterator[None]:
    """Leave no programmatic registration or memoised discovery behind.

    The registry is process-global by design — a manifest resolves plugins once per run —
    so without this the order of the test file would change its result.
    """
    clear_registrations()
    yield
    clear_registrations()


def _ep(name: str, value: str, group: PluginGroup) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=str(group))


def _stub_entry_points(monkeypatch: pytest.MonkeyPatch, *entries: EntryPoint) -> list[str]:
    """Replace the metadata scan with a fixed set. Returns the log of groups queried."""
    queried: list[str] = []

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        queried.append(group)
        return [entry for entry in entries if entry.group == group]

    monkeypatch.setattr(registry, "entry_points", fake_entry_points)
    return queried


class TestPluginGroups:
    def test_names_the_two_groups_doc_08_writes_out(self) -> None:
        # doc 08 §10 shows these two literal table names in a third-party pyproject.toml.
        # They are part of the published extension contract and cannot be renamed.
        assert PluginGroup.SOLVERS.value == "vpl.solvers"
        assert PluginGroup.INSTRUMENTS.value == "vpl.instruments"

    def test_names_the_two_groups_the_protocols_imply(self) -> None:
        # doc 08 §4 gives InverseEngine and NoiseModel the same protocol treatment as
        # ForwardSolver and Instrument, and doc 00 E1 requires all four to be plugins.
        assert PluginGroup.ENGINES.value == "vpl.engines"
        assert PluginGroup.NOISE.value == "vpl.noise"

    def test_accepts_a_plain_string_for_a_known_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A manifest is data (doc 08 §1 principle 4); it yields strings, not enum members.
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _REAL, PluginGroup.SOLVERS))

        assert available("vpl.solvers") == ("pic1d3v",)

    def test_rejects_an_unknown_group_and_lists_the_real_ones(self) -> None:
        # A typo'd group would otherwise report "nothing installed" for every plugin in
        # it, which reads as a packaging problem and sends the user to the wrong place.
        with pytest.raises(ValueError, match=r"vpl\.solvers.*vpl\.instruments"):
            available("vpl.solver")


class TestAvailable:
    def test_lists_installed_names_in_sorted_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(
            monkeypatch,
            _ep("pic1d3v", _REAL, PluginGroup.SOLVERS),
            _ep("analytic", _REAL, PluginGroup.SOLVERS),
        )

        assert available(PluginGroup.SOLVERS) == ("analytic", "pic1d3v")

    def test_lists_only_the_group_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(
            monkeypatch,
            _ep("pic1d3v", _REAL, PluginGroup.SOLVERS),
            _ep("efish", _REAL, PluginGroup.INSTRUMENTS),
        )

        assert available(PluginGroup.SOLVERS) == ("pic1d3v",)
        assert available(PluginGroup.INSTRUMENTS) == ("efish",)

    def test_is_empty_when_nothing_declares_the_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch)

        assert available(PluginGroup.NOISE) == ()

    def test_does_not_import_the_plugins_it_lists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # doc 08 §10's promise is that installing a package makes it visible. If listing
        # imported every plugin, one broken package would hide all the working ones — and
        # the user would be told their solver does not exist when it does.
        _stub_entry_points(
            monkeypatch,
            _ep("broken", _MISSING_MODULE, PluginGroup.SOLVERS),
            _ep("working", _REAL, PluginGroup.SOLVERS),
        )

        assert available(PluginGroup.SOLVERS) == ("broken", "working")

    def test_includes_programmatic_registrations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _REAL, PluginGroup.SOLVERS))
        register(PluginGroup.SOLVERS, "in_process", object())

        assert available(PluginGroup.SOLVERS) == ("in_process", "pic1d3v")


class TestLoad:
    def test_returns_the_object_the_entry_point_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        _stub_entry_points(monkeypatch, _ep("pic1d3v", _REAL, PluginGroup.SOLVERS))

        assert load(PluginGroup.SOLVERS, "pic1d3v") is json.JSONDecoder

    def test_unknown_name_reports_what_is_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The single most likely user-facing failure is a manifest naming a solver that is
        # not installed. "KeyError: 'pic1d3v'" tells the user nothing they did not know.
        _stub_entry_points(
            monkeypatch,
            _ep("analytic", _REAL, PluginGroup.SOLVERS),
            _ep("fluid", _REAL, PluginGroup.SOLVERS),
        )

        with pytest.raises(PluginNotFoundError, match="analytic, fluid") as caught:
            load(PluginGroup.SOLVERS, "pic1d3v")

        message = str(caught.value)
        assert "pic1d3v" in message
        assert "vpl.solvers" in message

    def test_unknown_name_says_so_plainly_when_nothing_is_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch)

        with pytest.raises(PluginNotFoundError, match="none"):
            load(PluginGroup.SOLVERS, "pic1d3v")

    def test_a_broken_plugin_raises_naming_the_plugin_and_the_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Skipping it would let a manifest run with a *different* solver than it names,
        # which doc 00 E3 forbids: the run would no longer be defined by the manifest.
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _MISSING_MODULE, PluginGroup.SOLVERS))

        with pytest.raises(PluginLoadError) as caught:
            load(PluginGroup.SOLVERS, "pic1d3v")

        message = str(caught.value)
        assert "pic1d3v" in message
        assert "vpl.solvers" in message
        assert _MISSING_MODULE in message

    def test_a_broken_plugin_chains_the_original_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _MISSING_MODULE, PluginGroup.SOLVERS))

        with pytest.raises(PluginLoadError) as caught:
            load(PluginGroup.SOLVERS, "pic1d3v")

        assert isinstance(caught.value.__cause__, ModuleNotFoundError)

    def test_an_importable_module_missing_the_named_object_also_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other half of "broken": the distribution installed, but its entry point
        # points at something the module does not define — a stale declaration.
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _MISSING_ATTRIBUTE, PluginGroup.SOLVERS))

        with pytest.raises(PluginLoadError, match="pic1d3v"):
            load(PluginGroup.SOLVERS, "pic1d3v")

    def test_a_programmatic_registration_shadows_an_entry_point(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch, _ep("pic1d3v", _REAL, PluginGroup.SOLVERS))
        stand_in = object()
        register(PluginGroup.SOLVERS, "pic1d3v", stand_in)

        assert load(PluginGroup.SOLVERS, "pic1d3v") is stand_in


class TestDiscover:
    def test_maps_every_installed_name_to_its_loaded_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        _stub_entry_points(
            monkeypatch,
            _ep("decoder", _REAL, PluginGroup.SOLVERS),
            _ep("encoder", _ALSO_REAL, PluginGroup.SOLVERS),
        )

        assert dict(discover(PluginGroup.SOLVERS)) == {
            "decoder": json.JSONDecoder,
            "encoder": json.JSONEncoder,
        }

    def test_scans_the_metadata_once_per_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Metadata scanning walks every installed distribution. A manifest resolves the
        # same group repeatedly, so the scan is memoised.
        queried = _stub_entry_points(monkeypatch, _ep("decoder", _REAL, PluginGroup.SOLVERS))

        first = discover(PluginGroup.SOLVERS)
        second = discover(PluginGroup.SOLVERS)

        assert first is second
        assert queried == ["vpl.solvers"]

    def test_returns_a_mapping_the_caller_cannot_mutate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The memo is shared. A caller who could write to it would be editing what every
        # later manifest resolves, from anywhere in the process.
        _stub_entry_points(monkeypatch, _ep("decoder", _REAL, PluginGroup.SOLVERS))

        found = discover(PluginGroup.SOLVERS)

        with pytest.raises(TypeError):
            found["decoder"] = object()  # type: ignore[index]

    def test_includes_programmatic_registrations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(monkeypatch)
        stand_in = object()
        register(PluginGroup.SOLVERS, "in_process", stand_in)

        assert dict(discover(PluginGroup.SOLVERS)) == {"in_process": stand_in}

    def test_does_not_silently_skip_a_broken_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(
            monkeypatch,
            _ep("broken", _MISSING_MODULE, PluginGroup.SOLVERS),
            _ep("working", _REAL, PluginGroup.SOLVERS),
        )

        with pytest.raises(PluginLoadError, match="broken"):
            discover(PluginGroup.SOLVERS)


class TestProgrammaticRegistration:
    def test_a_registered_plugin_loads_with_no_entry_point_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch)
        stand_in = object()

        register(PluginGroup.ENGINES, "fake_nuts", stand_in)

        assert load(PluginGroup.ENGINES, "fake_nuts") is stand_in

    def test_clearing_removes_the_registration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_entry_points(monkeypatch)
        register(PluginGroup.ENGINES, "fake_nuts", object())

        clear_registrations()

        assert available(PluginGroup.ENGINES) == ()

    def test_clearing_one_group_leaves_the_others_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch)
        register(PluginGroup.ENGINES, "fake_nuts", object())
        register(PluginGroup.NOISE, "fake_shot", object())

        clear_registrations(PluginGroup.ENGINES)

        assert available(PluginGroup.ENGINES) == ()
        assert available(PluginGroup.NOISE) == ("fake_shot",)

    def test_registering_after_a_discovery_invalidates_the_memo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Otherwise a test that registers a stand-in would silently get whatever an
        # earlier call had already resolved.
        _stub_entry_points(monkeypatch)
        assert dict(discover(PluginGroup.SOLVERS)) == {}
        stand_in = object()

        register(PluginGroup.SOLVERS, "in_process", stand_in)

        assert dict(discover(PluginGroup.SOLVERS)) == {"in_process": stand_in}

    def test_clearing_after_a_discovery_invalidates_the_memo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_entry_points(monkeypatch)
        register(PluginGroup.SOLVERS, "in_process", object())
        assert "in_process" in discover(PluginGroup.SOLVERS)

        clear_registrations(PluginGroup.SOLVERS)

        assert dict(discover(PluginGroup.SOLVERS)) == {}


class TestConflictingDeclarations:
    def test_two_distributions_claiming_one_name_is_an_error_not_a_coin_flip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Whichever distribution the metadata scan happened to reach first would decide
        # what the manifest ran, and the answer could change on reinstall. doc 00 E3
        # requires the manifest plus the code version to determine the run.
        _stub_entry_points(
            monkeypatch,
            _ep("pic1d3v", _REAL, PluginGroup.SOLVERS),
            _ep("pic1d3v", _ALSO_REAL, PluginGroup.SOLVERS),
        )

        with pytest.raises(PluginConflictError, match="pic1d3v") as caught:
            available(PluginGroup.SOLVERS)

        message = str(caught.value)
        assert _REAL in message
        assert _ALSO_REAL in message

    def test_the_same_declaration_twice_is_harmless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two metadata directories for one distribution resolve to the same object, so
        # there is nothing ambiguous to reject.
        _stub_entry_points(
            monkeypatch,
            _ep("pic1d3v", _REAL, PluginGroup.SOLVERS),
            _ep("pic1d3v", _REAL, PluginGroup.SOLVERS),
        )

        assert available(PluginGroup.SOLVERS) == ("pic1d3v",)


class TestErrorTaxonomy:
    def test_every_registry_failure_is_catchable_as_one_kind(self) -> None:
        # A manifest runner wants a single `except` around plugin resolution that
        # distinguishes "your manifest is wrong" from "your install is broken".
        assert issubclass(PluginNotFoundError, registry.PluginError)
        assert issubclass(PluginLoadError, registry.PluginError)
        assert issubclass(PluginConflictError, registry.PluginError)

    def test_a_missing_plugin_is_a_lookup_failure(self) -> None:
        assert issubclass(PluginNotFoundError, LookupError)

    def test_a_broken_plugin_is_an_import_failure(self) -> None:
        assert issubclass(PluginLoadError, ImportError)
