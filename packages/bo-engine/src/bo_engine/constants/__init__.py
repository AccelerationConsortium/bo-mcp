"""Constants for Bayesian Optimization configuration.

This package centralizes magic numbers and threshold values used throughout
the bo-engine package, making them easy to understand, tune, and override.
Split by domain (see the submodules below) once the single-file version grew
to ~120 constants across ~40 unrelated sections; every name is re-exported
here so ``from bo_engine.constants import X`` keeps working unchanged
regardless of which submodule now owns ``X``.
"""

from bo_engine.constants import acquisition as _acquisition
from bo_engine.constants import constraints_and_cv as _constraints_and_cv
from bo_engine.constants import convergence as _convergence
from bo_engine.constants import diagnostics as _diagnostics
from bo_engine.constants import gp_model as _gp_model
from bo_engine.constants import numerical as _numerical
from bo_engine.constants import objectives as _objectives
from bo_engine.constants import search_space as _search_space
from bo_engine.constants import transfer_learning as _transfer_learning
from bo_engine.constants import turbo as _turbo
from bo_engine.constants import uncertainty as _uncertainty
from bo_engine.constants.acquisition import *  # noqa: F403
from bo_engine.constants.constraints_and_cv import *  # noqa: F403
from bo_engine.constants.convergence import *  # noqa: F403
from bo_engine.constants.diagnostics import *  # noqa: F403
from bo_engine.constants.gp_model import *  # noqa: F403
from bo_engine.constants.numerical import *  # noqa: F403
from bo_engine.constants.objectives import *  # noqa: F403
from bo_engine.constants.search_space import *  # noqa: F403
from bo_engine.constants.transfer_learning import *  # noqa: F403
from bo_engine.constants.turbo import *  # noqa: F403
from bo_engine.constants.uncertainty import *  # noqa: F403

__all__ = [
    *_acquisition.__all__,
    *_constraints_and_cv.__all__,
    *_convergence.__all__,
    *_diagnostics.__all__,
    *_gp_model.__all__,
    *_numerical.__all__,
    *_objectives.__all__,
    *_search_space.__all__,
    *_transfer_learning.__all__,
    *_turbo.__all__,
    *_uncertainty.__all__,
]
