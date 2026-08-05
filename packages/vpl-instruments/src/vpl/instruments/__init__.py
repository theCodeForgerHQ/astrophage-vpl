"""The measurement operator ``F2`` — doc 04 §2-§5.

Doc 04 §1 lays the forward chain out as three strictly separated stages, and this package
is the first of them: a plasma state in, spectral radiance out. The two below it —
``F3`` optical transport (``vpl-optics``) and ``F4`` detection (``vpl-detectors``) — are
separate packages because doc 04 §1's rule that "``F4`` never sees the plasma" is only
enforceable if the boundary is real.

Only the OES channel exists so far. It is deliberately first: doc 02 §6.1 calls it "the
fast backbone channel", and doc 04 §2.3 makes it the one whose easy version is wrong in a
way that is invisible unless you know to look, which is the standard the rest are held to.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
