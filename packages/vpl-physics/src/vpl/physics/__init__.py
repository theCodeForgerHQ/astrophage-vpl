"""The forward physics operator ``F1`` — doc 03.

Four fidelity levels behind one interface (doc 03 §1): ``analytic`` (L0), ``fluid`` (L1),
``kinetic`` (L2) and ``surrogate`` (L3). The inverse solver never learns which one it is
talking to, which is doc 00 E1 stated as a package layout.

Only L0 exists so far. It is deliberately first: doc 03 §2.3 makes it the framework's
"primary analytic verification target", and verification gate V-03 (doc 07) checks the
levels above it against numbers this package produces.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
