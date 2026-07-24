"""BayBE backend for Bayesian Optimization.

Provides a BayBEBackend that implements the BOBackend protocol.
BayBE is BO-MCP's default backend; BoTorch remains available as the
legacy fallback.
"""

from bo_engine_baybe.backend import BayBEBackend

__all__ = ["BayBEBackend"]
__version__ = "0.1.0"
