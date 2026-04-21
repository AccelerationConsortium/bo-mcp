"""BayBE backend for Bayesian Optimization.

Provides a BayBEBackend that implements the BOBackend protocol,
enabling BayBE as an alternative to the default BoTorch backend.
"""

from bo_engine_baybe.backend import BayBEBackend

__all__ = ["BayBEBackend"]
__version__ = "0.1.0"
