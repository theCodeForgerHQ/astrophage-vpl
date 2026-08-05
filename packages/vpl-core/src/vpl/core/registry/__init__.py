"""Runtime plugin discovery — doc 08 §10.

Capability arrives as an installed distribution, never as an edit to this package
(doc 08 §1 principle 3). A third party declares an entry point::

    [project.entry-points."vpl.solvers"]
    smilei = "vpl_plugin_smilei:SmileiSolver"

and ``pip install vpl-plugin-smilei`` makes ``forward.solver: smilei`` resolvable in every
manifest with no core change — which is doc 00 E1 stated as a mechanism.

The mechanism is ``importlib.metadata`` and nothing else. Doc 00 §6 lists the
reimplementation trap; a bespoke plugin system would be a small, avoidable instance of it,
and standard entry points already give packaging, uninstall and version pinning for free.

**Groups.** Doc 08 §10 writes out ``vpl.instruments`` and ``vpl.solvers``. Two more are
defined here: doc 08 §4 gives ``InverseEngine`` and ``NoiseModel`` exactly the same
protocol treatment as ``ForwardSolver`` and ``Instrument``, and doc 00 E1 requires *every*
solver, instrument, noise model and inference engine to be discovered at runtime. Two
groups with the protocol treatment and two without would leave half of E1 unsatisfiable.
(E1 also names regularisers; that group is deferred until doc 05's prior layer fixes what
a regulariser's contract is, since inventing the group before the protocol would be
guessing.)

**Failure is loud.** A plugin that fails to import raises. Dropping it would let a manifest
naming ``pic1d3v`` run on whatever else was installed, and the run would no longer be
determined by the manifest — the property doc 00 E3 exists to guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from importlib.metadata import EntryPoint, entry_points
from types import MappingProxyType
from typing import Final

__all__ = [
    "GroupLike",
    "PluginConflictError",
    "PluginError",
    "PluginGroup",
    "PluginLoadError",
    "PluginNotFoundError",
    "available",
    "clear_registrations",
    "discover",
    "load",
    "register",
]


class PluginGroup(StrEnum):
    """The entry-point groups the framework defines.

    A ``StrEnum`` because manifests are data (doc 08 §1 principle 4) and yield strings:
    ``PluginGroup("vpl.solvers")`` round-trips a manifest field into a checked member,
    and ``str(group)`` writes it back out for provenance.
    """

    SOLVERS = "vpl.solvers"
    INSTRUMENTS = "vpl.instruments"
    ENGINES = "vpl.engines"
    NOISE = "vpl.noise"


#: A group named either as a member or as the raw string a manifest carries.
type GroupLike = PluginGroup | str


class PluginError(Exception):
    """Base for every failure to resolve a plugin.

    One base so that a manifest runner can wrap plugin resolution in a single ``except``
    and still tell the subclasses apart: :class:`PluginNotFoundError` means the manifest
    names something that is not installed, :class:`PluginLoadError` means the installation
    itself is broken. Those two have different fixes and different owners.
    """


class PluginNotFoundError(PluginError, LookupError):
    """No plugin of that name is installed or registered in that group."""


class PluginLoadError(PluginError, ImportError):
    """A plugin is declared but its target could not be imported.

    The originating exception is always chained; a plugin's import can fail for any reason
    its author's code can fail for, and discarding that traceback would leave the user
    with the symptom and none of the cause.
    """


class PluginConflictError(PluginError):
    """Two installed distributions declare the same name in the same group."""


# ── process-global state ────────────────────────────────────────────────────────
#
# Mutable module state, deliberately. Discovery is a property of the *process* — of what
# is installed and what a host application has registered — and threading a registry
# object through every call site would put the plugin mechanism into the signature of
# every function that resolves one, which is the coupling doc 08 §1 principle 3 forbids.

#: Plugins registered in-process. Shadows entry points; see :func:`register`.
_REGISTERED: Final[dict[PluginGroup, dict[str, object]]] = {}

#: Memoised results of :func:`discover`, keyed by group.
_MEMO: Final[dict[PluginGroup, Mapping[str, object]]] = {}


def _coerce_group(group: GroupLike) -> PluginGroup:
    """Resolve a group name, rejecting anything the framework does not define.

    A mistyped group would otherwise report an empty group, which reads to the user as a
    packaging problem and sends them to reinstall a package that is already there.
    """
    try:
        return PluginGroup(group)
    except ValueError as exc:
        known = ", ".join(member.value for member in PluginGroup)
        raise ValueError(f"unknown plugin group {str(group)!r}; the groups are: {known}") from exc


def _declared(group: PluginGroup) -> dict[str, EntryPoint]:
    """Entry points declared for ``group``, by name. Reads metadata; imports nothing."""
    declared: dict[str, EntryPoint] = {}

    for entry in entry_points(group=str(group)):
        existing = declared.get(entry.name)
        # Two metadata directories for one distribution name the same target, which is
        # harmless. Two *different* targets are not: whichever the scan reached first
        # would decide what the manifest ran, and that can change on reinstall.
        if existing is not None and existing.value != entry.value:
            raise PluginConflictError(
                f"two installed distributions declare {entry.name!r} in group "
                f"{group.value!r}: {existing.value!r} and {entry.value!r}. "
                "Uninstall one of them; while both are present, which one a manifest "
                "resolves to is not determined by the manifest (doc 00 E3)."
            )
        declared[entry.name] = entry

    return declared


def _resolve(entry: EntryPoint, group: PluginGroup) -> object:
    """Import an entry point's target, reporting failure with enough to act on."""
    try:
        plugin: object = entry.load()
    except Exception as exc:
        raise PluginLoadError(
            f"plugin {entry.name!r} in group {group.value!r} is installed but failed to "
            f"import: it points at {entry.value!r}, which raised "
            f"{type(exc).__name__}: {exc}. The declaration and the package that provides "
            "it disagree; this is a fault in the plugin, not in the manifest."
        ) from exc

    return plugin


def _not_found(group: PluginGroup, name: str, names: tuple[str, ...]) -> PluginNotFoundError:
    """Build the error a mistyped or uninstalled manifest entry deserves."""
    installed = ", ".join(names) if names else "none"
    return PluginNotFoundError(
        f"no plugin named {name!r} in group {group.value!r}. Installed: {installed}. "
        f"A plugin becomes available by installing a distribution that declares it under "
        f'[project.entry-points."{group.value}"] (doc 08 §10) — either the name in the '
        "manifest is misspelled, or the package that provides it is not installed."
    )


def available(group: GroupLike) -> tuple[str, ...]:
    """Names declared for ``group``, sorted, without importing any of them.

    Metadata only. Doc 08 §10's promise is that installing a package makes it visible, and
    visibility must not depend on every *other* installed plugin importing cleanly — one
    broken package would otherwise hide every working one, and the user would be told
    their solver does not exist when in fact it does.

    Args:
        group: A :class:`PluginGroup` member or its string value.

    Returns:
        The plugin names, sorted.

    Raises:
        ValueError: If ``group`` is not a group the framework defines.
        PluginConflictError: If two distributions claim one name in ``group``.
    """
    resolved = _coerce_group(group)
    names = set(_declared(resolved)) | set(_REGISTERED.get(resolved, {}))
    return tuple(sorted(names))


def load(group: GroupLike, name: str) -> object:
    """Import and return one plugin.

    Returns ``object`` rather than a protocol type: the entry point's target is whatever
    the plugin author wrote, so the check that it satisfies :mod:`vpl.core.protocols`
    belongs to the caller that knows which protocol it wanted.

    Args:
        group: A :class:`PluginGroup` member or its string value.
        name: The plugin name as it appears in the manifest.

    Returns:
        The object the entry point names — normally the plugin class, uninstantiated.

    Raises:
        ValueError: If ``group`` is not a group the framework defines.
        PluginNotFoundError: If nothing declares ``name``. The message lists what does.
        PluginLoadError: If ``name`` is declared but its target will not import.
    """
    resolved = _coerce_group(group)

    registered = _REGISTERED.get(resolved, {})
    if name in registered:
        return registered[name]

    declared = _declared(resolved)
    entry = declared.get(name)
    if entry is None:
        raise _not_found(resolved, name, tuple(sorted(set(declared) | set(registered))))

    return _resolve(entry, resolved)


def discover(group: GroupLike) -> Mapping[str, object]:
    """Import every plugin in ``group`` and return them by name, memoised.

    Memoised because scanning distribution metadata walks the whole environment and a
    single manifest resolves the same group repeatedly. The returned mapping is read-only:
    it is shared with every other caller in the process, so a caller able to write to it
    would be editing what later manifests resolve.

    Use :func:`load` unless all of them are genuinely needed — ``discover`` imports every
    installed plugin in the group, so one broken plugin fails the whole call. That is
    deliberate (see the module docstring) but it is why :func:`available` exists.

    Args:
        group: A :class:`PluginGroup` member or its string value.

    Returns:
        A read-only mapping of name to loaded object.

    Raises:
        ValueError: If ``group`` is not a group the framework defines.
        PluginConflictError: If two distributions claim one name in ``group``.
        PluginLoadError: If any plugin in ``group`` will not import.
    """
    resolved = _coerce_group(group)

    memo = _MEMO.get(resolved)
    if memo is not None:
        return memo

    loaded = {name: _resolve(entry, resolved) for name, entry in _declared(resolved).items()}
    # Registrations last: :func:`load` prefers them, and the two must agree.
    loaded.update(_REGISTERED.get(resolved, {}))

    found: Mapping[str, object] = MappingProxyType(loaded)
    _MEMO[resolved] = found
    return found


def register(group: GroupLike, name: str, plugin: object) -> None:
    """Add a plugin to this process only, shadowing any entry point of the same name.

    **This is not the extension path.** The supported way to add a solver, instrument,
    engine or noise model is to install a distribution that declares an entry point
    (doc 08 §10) — that is what makes it reproducible from a manifest, what records it in
    the environment capture, and what doc 00 E1 requires. Registration here exists for two
    narrower jobs: tests that need a stand-in without building a wheel, and applications
    that embed the framework and construct their plugins in-process.

    Nothing registered this way appears in an environment capture, so a manifest that
    depends on one is not reproducible by anyone else.

    Args:
        group: A :class:`PluginGroup` member or its string value.
        name: The name the manifest would use.
        plugin: The object to return from :func:`load` and :func:`discover`.

    Raises:
        ValueError: If ``group`` is not a group the framework defines.
    """
    resolved = _coerce_group(group)
    _REGISTERED.setdefault(resolved, {})[name] = plugin
    # A memo built before this call does not contain it, and would be silently stale.
    _MEMO.pop(resolved, None)


def clear_registrations(group: GroupLike | None = None) -> None:
    """Drop programmatic registrations — all of them, or one group's.

    Entry-point plugins are untouched; only what :func:`register` added is removed. Any
    memoised discovery for the affected groups is dropped too, since it may have been
    built while a registration was live.

    Args:
        group: The group to clear, or ``None`` for every group.

    Raises:
        ValueError: If ``group`` is given and is not a group the framework defines.
    """
    if group is None:
        _REGISTERED.clear()
        _MEMO.clear()
        return

    resolved = _coerce_group(group)
    _REGISTERED.pop(resolved, None)
    _MEMO.pop(resolved, None)
