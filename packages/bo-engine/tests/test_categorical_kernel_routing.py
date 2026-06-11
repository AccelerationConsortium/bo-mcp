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

import pytest
import torch
from botorch.models.kernels import CategoricalKernel
from gpytorch.kernels import RBFKernel, ScaleKernel

from bo_engine.models import (
    build_mixed_kernel,
    create_single_task_model,
)
from bo_engine.transforms import get_categorical_blocks, get_categorical_dim_indices
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


class TestGetCategoricalBlocks:
    """Groups one-hot columns per categorical parameter (preserves boundaries)."""

    def test_single_block(self) -> None:
        spec = _make_mixed_spec()
        # Layout: x_cont (1), x_cat one-hot (3), x_cont2 (1); the cat block is
        # one parameter at indices 1, 2, 3.
        assert get_categorical_blocks(spec) == [[1, 2, 3]]

    def test_adjacent_categoricals_keep_separate_blocks(self) -> None:
        """Two adjacent categorical params must not be merged into one block.

        Their one-hot columns are contiguous, so the flat-index view cannot
        recover the boundary; the grouped view must.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="c1", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
                ParameterSpec(
                    name="c2", type=ParameterType.CATEGORICAL, categories=["x", "y", "z"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            use_categorical_kernel=True,
        )
        assert get_categorical_blocks(spec) == [[0, 1], [2, 3, 4]]

    def test_no_categoricals_returns_empty(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert get_categorical_blocks(spec) == []


class TestBuildMixedKernel:
    """Additive RBF + Categorical kernel construction."""

    def test_kernel_is_additive_rbf_plus_categorical(self) -> None:
        kernel = build_mixed_kernel(n_total_dims=5, categorical_blocks=[[1, 2, 3]])
        assert isinstance(kernel, ScaleKernel)
        # Inner additive kernel: gpytorch represents `k1 + k2` via AdditiveKernel
        # which exposes the operands through `.kernels` or via the addition
        # operator structure. We assert the inner kernel contains exactly the
        # expected component types (the compensated kernel is a CategoricalKernel
        # subclass, so ``isinstance`` still holds).
        sub_kernels = list(kernel.base_kernel.sub_kernels())
        assert any(isinstance(k, RBFKernel) for k in sub_kernels)
        assert any(isinstance(k, CategoricalKernel) for k in sub_kernels)

    def test_one_lengthscale_per_block_not_per_bit(self) -> None:
        """Each categorical block gets a single shared lengthscale (L4).

        The previous kernel passed ``ard_num_dims=len(all one-hot columns)``,
        so a 3-category parameter carried 3 lengthscales (one per one-hot bit),
        over-parameterizing the kernel. The fix gives each block a single
        shared lengthscale.
        """
        kernel = build_mixed_kernel(n_total_dims=6, categorical_blocks=[[1, 2, 3], [4, 5]])
        cat_kernels = [k for k in kernel.modules() if isinstance(k, CategoricalKernel)]
        # One CategoricalKernel per categorical parameter (block), not one
        # shared across all columns.
        assert len(cat_kernels) == 2
        for cat in cat_kernels:
            # A single shared lengthscale → trailing lengthscale dim is 1,
            # regardless of how many categories the block has.
            assert cat.lengthscale.shape[-1] == 1

    def test_one_category_change_decays_as_exp_minus_one(self) -> None:
        """A single differing category yields ``exp(-1/ls)`` for any block size (L4).

        One-hot toggles two columns per category change, so BoTorch's averaged
        ``CategoricalKernel`` would report ``exp(-2/(k*ls))`` (k = category
        count). The compensated kernel divides the summed mismatch by two so
        the decay is the ordinal Hamming ``exp(-1/ls)`` independent of k.
        """
        for n_categories in (2, 3, 5):
            # Column 0 is continuous; the categorical block occupies the rest,
            # so the mixed kernel is an additive RBF + Categorical structure.
            block = list(range(1, n_categories + 1))
            kernel = build_mixed_kernel(n_total_dims=n_categories + 1, categorical_blocks=[block])
            cat = next(k for k in kernel.modules() if isinstance(k, CategoricalKernel))
            with torch.no_grad():
                cat.lengthscale = torch.ones(1, 1, dtype=torch.float64)
            # Two distinct categories: one-hot vectors differing in 2 columns
            # within the block (the continuous column is held equal).
            cat_a = torch.zeros(1, n_categories + 1, dtype=torch.float64)
            cat_a[0, 1] = 1.0
            cat_b = torch.zeros(1, n_categories + 1, dtype=torch.float64)
            cat_b[0, 2] = 1.0
            value = float(cat(cat_a, cat_b).to_dense().item())
            expected = float(torch.exp(torch.tensor(-1.0)).item())
            assert value == pytest.approx(expected, rel=1e-6), (
                f"k={n_categories}: K(catA, catB)={value}, expected exp(-1)={expected}"
            )
            # Same category → similarity 1.
            assert float(cat(cat_a, cat_a).to_dense().item()) == pytest.approx(1.0)

    def test_invalid_indices_rejected(self) -> None:
        with pytest.raises(ValueError, match="categorical_blocks"):
            build_mixed_kernel(n_total_dims=3, categorical_blocks=[[5]])


class TestModelFactoryRouting:
    """Passing ``categorical_blocks`` swaps the covariance module."""

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

        model = create_single_task_model(train_x, train_y, bounds, categorical_blocks=[[1, 2, 3]])

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
