"""Cross-backend interop markers used by capability routing.

``bo-engine`` must never import a concrete backend package: the dependency
arrows point *into* the engine (``bo-engine-baybe`` depends on
``bo-engine``, not the reverse), so importing ``bo_engine_baybe`` here would
invert the graph. Yet the BoTorch capability guardrail in
:mod:`bo_engine.botorch_backend` has to recognize one BayBE-only marker — a
``role=substance`` parameter encodes a molecular (SMILES → cheminformatics
descriptor) representation. BoTorch has no chemistry kernel and would treat
the SMILES category labels as opaque one-hot categories, silently dropping
the chemistry. To let ``backend="auto"`` route such specs to BayBE (and to
fail a pinned ``backend="botorch"`` loudly at intake) the guardrail reads the
small set of string markers defined here.

These values mirror the BayBE-side source of truth
(:class:`bo_engine_baybe.options.BayBEParameterRole` and the BayBE backend
name). A pin test in the ``bo-engine-baybe`` test suite — the only suite
permitted to import both packages — asserts they stay equal, so a future
BayBE-side rename fails CI instead of silently disabling the guardrail. The
markers are an intentional, named cross-backend contract, not stray
hardcoded literals.
"""

from __future__ import annotations

# Name BayBE registers under the ``bo_mcp.backends`` entry-point group, and
# the key under which BayBE-native per-parameter options are nested inside
# ``ParameterSpec.parameter_options`` (``parameter_options["baybe"]``).
BAYBE_BACKEND_NAME = "baybe"

# Field inside the BayBE option blob selecting the categorical-family role
# (``categorical`` / ``task`` / ``substance``).
BAYBE_PARAMETER_ROLE_KEY = "role"

# Role value marking a molecular ``SubstanceParameter`` (SMILES → descriptor
# encoding). The BoTorch guardrail must veto it.
BAYBE_SUBSTANCE_ROLE = "substance"

# Role value marking a ``CustomDiscreteParameter`` (labels → user-supplied
# numeric representation, e.g. quantum-chemistry descriptors). Like substance,
# BoTorch cannot reproduce the representation — it would one-hot the labels and
# silently drop the supplied encoding — so the guardrail vetoes it too.
BAYBE_CUSTOM_ROLE = "custom"
