"""The parameter registry — doc 02 §12, doc 08 §5, doc 09 §1.

    The registry is the single source of truth. No number appears in code as a literal.

The registry holds **numbers**. Categorical choices — which gas, which electrode
material, which collision set — are manifest fields, not registry entries: the rule this
package enforces is about unattributed magnitudes, and "argon" has no uncertainty, no
sweep range and no units. Forcing them into a numeric schema would have meant a `value`
field that is sometimes a string, which is the kind of accommodation that ends with the
schema meaning nothing.
"""

from vpl.core.params.catalogue import (
    DEFAULT_REGISTRY_DIR,
    ParameterRegistry,
    default_registry,
)
from vpl.core.params.schema import Parameter, ProvenanceClass, Uncertainty

__all__ = [
    "DEFAULT_REGISTRY_DIR",
    "Parameter",
    "ParameterRegistry",
    "ProvenanceClass",
    "Uncertainty",
    "default_registry",
]
