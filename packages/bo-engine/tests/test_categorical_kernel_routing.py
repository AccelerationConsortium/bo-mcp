"""Tests for the optional CategoricalKernel routing on the GP itself.

The acquisition path already enumerates one-hot categorical combinations
via ``optimize_acqf_mixed``, so projection drift is not the issue here.
What was: the GP's kernel treats each one-hot column as an independent
continuous dimension under the default RBF, producing slower kernel fits
and lower posterior quality on categorical-heavy specs. With
``OptimizationSpec.use_categorical_kernel=True`` the GP gets an additive
``RBF(continuous) + CategoricalKernel(one_hot_blocks)`` kernel via
:func:`bo_engine.models.build_mixed_kernel`.

References:
    - Wan et al., "Think Global and Act Local: Bayesian Optimisation over
      High-Dimensional Categorical and Mixed Search Spaces", ICML 2021 —
      Hamming-style kernels recover better lengthscales than RBF on
      one-hot expansions.
    - BoTorch ``CategoricalKernel`` reference — Hamming distance kernel.
"""

from __future__ import annotations

import torch
from botorch.models.kernels import CategoricalKernel
from gpytorch.kernels import RBFKernel, ScaleKernel

from bo_engine.models import (
    build_mixed_kernel,
    create_single_task_model,
)
from bo_engine.transforms import get_categorical_dim_indices
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

N_OBS = 6


def _make_mixed_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x_cont", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(
                name="x_cat",
                type=ParameterType.CATEGORICAL,
                categories=["a", "b", "c"],
            ),
            ParameterSpec(name="x_cont2", type=ParameterType.CONTINUOUS, bounds=(-1.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        use_categorical_kernel=True,
    )


class TestGetCategoricalDimIndices:
    """Returns indices in the encoded one-hot expanded space."""

    def test_indices_cover_only_one_hot_columns(self) -> None:
        spec = _make_mixed_spec()
        # Layout: x_cont (1), x_cat one-hot (3), x_cont2 (1) → total 5 dims;
        # cat block sits at indices 1, 2, 3.
        assert get_categorical_dim_indices(spec) == [1, 2, 3]

    def test_no_categoricals_returns_empty(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert get_categorical_dim_indices(spec) == []


class TestBuildMixedKernel:
    """Additive RBF + Categorical kernel construction."""

    def test_kernel_is_additive_rbf_plus_categorical(self) -> None:
        kernel = build_mixed_kernel(n_total_dims=5, categorical_dim_indices=[1, 2, 3])
        assert isinstance(kernel, ScaleKernel)
        # Inner additive kernel: gpytorch represents `k1 + k2` via AdditiveKernel
        # which exposes the operands through `.kernels` or via the addition
        # operator structure. We assert the inner kernel contains exactly the
        # expected component types.
        sub_kernels = list(kernel.base_kernel.sub_kernels())
        types = {type(k) for k in sub_kernels}
        assert RBFKernel in types
        assert CategoricalKernel in types

    def test_invalid_indices_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="categorical_dim_indices"):
            build_mixed_kernel(n_total_dims=3, categorical_dim_indices=[5])


class TestModelFactoryRouting:
    """Passing ``categorical_dim_indices`` swaps the covariance module."""

    def test_factory_installs_mixed_kernel(self) -> None:
        torch.manual_seed(0)
        train_x = torch.rand(N_OBS, 5, dtype=torch.float64)
        # Force valid one-hot rows: first 3 cols continuous, next 3 one-hot,
        # last col continuous. Make positions 1-3 a valid one-hot encoding.
        train_x[:, 1:4] = 0.0
        for i in range(N_OBS):
            train_x[i, 1 + (i % 3)] = 1.0
        train_y = torch.randn(N_OBS, 1, dtype=torch.float64)
        bounds = torch.stack(
            [torch.zeros(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64)]
        )

        model = create_single_task_model(
            train_x, train_y, bounds, categorical_dim_indices=[1, 2, 3]
        )

        covar = model.covar_module
        assert isinstance(covar, ScaleKernel)
        sub_kernels = list(covar.base_kernel.sub_kernels())
        assert any(isinstance(k, CategoricalKernel) for k in sub_kernels)
        assert any(isinstance(k, RBFKernel) for k in sub_kernels)

    def test_factory_without_categoricals_keeps_default_kernel(self) -> None:
        torch.manual_seed(1)
        train_x = torch.rand(N_OBS, 3, dtype=torch.float64)
        train_y = torch.randn(N_OBS, 1, dtype=torch.float64)
        bounds = torch.stack(
            [torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)]
        )

        model = create_single_task_model(train_x, train_y, bounds)
        # Default SingleTaskGP wraps an RBFKernel in ScaleKernel; we
        # specifically don't have a CategoricalKernel anywhere in the chain.
        for module in model.modules():
            assert not isinstance(module, CategoricalKernel)
