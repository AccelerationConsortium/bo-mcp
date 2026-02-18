"""Unit tests for search space classification and discrete enumeration helpers.

Tests the helper functions added to transforms.py for classifying parameter
spaces and building tensors/dicts for BoTorch's discrete and mixed optimizers.

References:
    - BoTorch optimize_acqf_discrete:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_discrete
    - BoTorch optimize_acqf_mixed:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_mixed
"""

import pytest
import torch

from bo_engine.constants import DISCRETE_ENUMERATION_MAX_POINTS
from bo_engine.transforms import (
    SearchSpaceType,
    build_fixed_features_list,
    classify_search_space,
    count_categorical_combinations,
    enumerate_discrete_choices,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def continuous_spec() -> OptimizationSpec:
    """Purely continuous spec (2 continuous parameters)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


@pytest.fixture
def categorical_spec() -> OptimizationSpec:
    """Purely categorical spec (2 categorical parameters, 3 categories each)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="color", type=ParameterType.CATEGORICAL, categories=["red", "green", "blue"]
            ),
            ParameterSpec(name="size", type=ParameterType.CATEGORICAL, categories=["S", "M", "L"]),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


@pytest.fixture
def mixed_spec() -> OptimizationSpec:
    """Mixed spec (1 continuous + 1 categorical with 3 categories)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(
                name="color", type=ParameterType.CATEGORICAL, categories=["red", "green", "blue"]
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


@pytest.fixture
def discrete_numeric_spec() -> OptimizationSpec:
    """Spec with discrete numeric parameter (treated as continuous)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="n_layers", type=ParameterType.DISCRETE, bounds=(1.0, 10.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


@pytest.fixture
def donor_acceptor_spec() -> OptimizationSpec:
    """The donor/acceptor fragment spec from the example (6 x 6 = 36 combos)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="donor",
                type=ParameterType.CATEGORICAL,
                categories=[
                    "phenyl",
                    "anisole",
                    "aniline",
                    "thiophene",
                    "carbazole",
                    "phenothiazine",
                ],
            ),
            ParameterSpec(
                name="acceptor",
                type=ParameterType.CATEGORICAL,
                categories=[
                    "phenyl",
                    "benzonitrile",
                    "pyridine",
                    "pyrimidine",
                    "benzothiadiazole",
                    "triazine",
                ],
            ),
        ],
        objectives=[ObjectiveSpec(name="gap_eV", minimize=True)],
        batch_size=2,
    )


# =============================================================================
# TestClassifySearchSpace
# =============================================================================


class TestClassifySearchSpace:
    """Test classify_search_space correctly identifies space types."""

    def test_purely_continuous(self, continuous_spec: OptimizationSpec) -> None:
        """Purely continuous parameters should classify as CONTINUOUS."""
        assert classify_search_space(continuous_spec) == SearchSpaceType.CONTINUOUS

    def test_purely_categorical(self, categorical_spec: OptimizationSpec) -> None:
        """All categorical parameters should classify as PURELY_CATEGORICAL."""
        assert classify_search_space(categorical_spec) == SearchSpaceType.PURELY_CATEGORICAL

    def test_mixed_space(self, mixed_spec: OptimizationSpec) -> None:
        """Mix of continuous and categorical should classify as MIXED."""
        assert classify_search_space(mixed_spec) == SearchSpaceType.MIXED

    def test_discrete_numeric_treated_as_continuous(
        self, discrete_numeric_spec: OptimizationSpec
    ) -> None:
        """Discrete numeric parameters should be treated as continuous."""
        assert classify_search_space(discrete_numeric_spec) == SearchSpaceType.CONTINUOUS

    def test_single_categorical(self) -> None:
        """A single categorical parameter should classify as PURELY_CATEGORICAL."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert classify_search_space(spec) == SearchSpaceType.PURELY_CATEGORICAL

    def test_discrete_plus_categorical_is_mixed(self) -> None:
        """Discrete numeric + categorical should classify as MIXED."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="n", type=ParameterType.DISCRETE, bounds=(1.0, 5.0)),
                ParameterSpec(name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert classify_search_space(spec) == SearchSpaceType.MIXED


# =============================================================================
# TestCountCategoricalCombinations
# =============================================================================


class TestCountCategoricalCombinations:
    """Test count_categorical_combinations returns correct product."""

    def test_zero_categoricals(self, continuous_spec: OptimizationSpec) -> None:
        """No categoricals should return 0."""
        assert count_categorical_combinations(continuous_spec) == 0

    def test_single_categorical(self) -> None:
        """Single categorical with 4 categories should return 4."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b", "c", "d"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert count_categorical_combinations(spec) == 4

    def test_two_categoricals_product(self, categorical_spec: OptimizationSpec) -> None:
        """Two categoricals with 3 categories each should return 3 * 3 = 9."""
        assert count_categorical_combinations(categorical_spec) == 9

    def test_donor_acceptor_product(self, donor_acceptor_spec: OptimizationSpec) -> None:
        """6 donors x 6 acceptors = 36 combinations."""
        assert count_categorical_combinations(donor_acceptor_spec) == 36

    def test_three_categoricals_product(self) -> None:
        """3 * 4 * 5 = 60 combinations."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="a", type=ParameterType.CATEGORICAL, categories=["a1", "a2", "a3"]
                ),
                ParameterSpec(
                    name="b",
                    type=ParameterType.CATEGORICAL,
                    categories=["b1", "b2", "b3", "b4"],
                ),
                ParameterSpec(
                    name="c",
                    type=ParameterType.CATEGORICAL,
                    categories=["c1", "c2", "c3", "c4", "c5"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert count_categorical_combinations(spec) == 60

    def test_mixed_counts_only_categoricals(self, mixed_spec: OptimizationSpec) -> None:
        """Mixed spec should count only the categorical combos (3)."""
        assert count_categorical_combinations(mixed_spec) == 3


# =============================================================================
# TestEnumerateDiscreteChoices
# =============================================================================


class TestEnumerateDiscreteChoices:
    """Test enumerate_discrete_choices builds correct one-hot tensors."""

    def test_single_categorical_shape(self) -> None:
        """Single categorical with 3 categories: 3 rows, 3 dims."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b", "c"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        choices = enumerate_discrete_choices(spec)
        assert choices.shape == (3, 3)

    def test_single_categorical_valid_one_hot(self) -> None:
        """Each row should have exactly one 1.0 and rest 0.0."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b", "c"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        choices = enumerate_discrete_choices(spec)
        # Each row sums to 1.0
        assert torch.allclose(choices.sum(dim=1), torch.ones(3, dtype=choices.dtype))
        # Each row has exactly one 1.0
        assert (choices.max(dim=1).values == 1.0).all()
        assert (choices.min(dim=1).values == 0.0).all()

    def test_two_categoricals_cartesian_product(self, categorical_spec: OptimizationSpec) -> None:
        """Two categoricals (3 x 3): 9 rows, 6 dims (3 + 3)."""
        choices = enumerate_discrete_choices(categorical_spec)
        assert choices.shape == (9, 6)

    def test_donor_acceptor_example(self, donor_acceptor_spec: OptimizationSpec) -> None:
        """6 donors x 6 acceptors = 36 rows, 12 dims (6 + 6)."""
        choices = enumerate_discrete_choices(donor_acceptor_spec)
        assert choices.shape == (36, 12)

    def test_all_rows_unique(self, donor_acceptor_spec: OptimizationSpec) -> None:
        """All rows should be unique (no duplicate combinations)."""
        choices = enumerate_discrete_choices(donor_acceptor_spec)
        unique_rows = torch.unique(choices, dim=0)
        assert unique_rows.shape[0] == choices.shape[0]

    def test_valid_one_hot_per_block(self, donor_acceptor_spec: OptimizationSpec) -> None:
        """Each categorical block should be a valid one-hot vector."""
        choices = enumerate_discrete_choices(donor_acceptor_spec)
        # Donor block: dims 0-5 (6 categories)
        donor_block = choices[:, :6]
        assert torch.allclose(donor_block.sum(dim=1), torch.ones(36, dtype=choices.dtype))
        assert ((donor_block == 0.0) | (donor_block == 1.0)).all()

        # Acceptor block: dims 6-11 (6 categories)
        acceptor_block = choices[:, 6:]
        assert torch.allclose(acceptor_block.sum(dim=1), torch.ones(36, dtype=choices.dtype))
        assert ((acceptor_block == 0.0) | (acceptor_block == 1.0)).all()

    def test_exceeds_enumeration_limit_raises(self) -> None:
        """Should raise ValueError when combinations exceed DISCRETE_ENUMERATION_MAX_POINTS."""
        # Create a spec that exceeds the limit: 20 categories each for 4 params = 160,000
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name=f"cat{i}",
                    type=ParameterType.CATEGORICAL,
                    categories=[f"c{j}" for j in range(20)],
                )
                for i in range(4)
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        n_combos = count_categorical_combinations(spec)
        assert n_combos > DISCRETE_ENUMERATION_MAX_POINTS

        with pytest.raises(ValueError, match="enumeration limit"):
            enumerate_discrete_choices(spec)


# =============================================================================
# TestBuildFixedFeaturesList
# =============================================================================


class TestBuildFixedFeaturesList:
    """Test build_fixed_features_list for optimize_acqf_mixed."""

    def test_single_categorical(self) -> None:
        """Single categorical with 3 categories: 3 dicts."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(
                    name="cat", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        ffl = build_fixed_features_list(spec)
        assert len(ffl) == 3

        # Continuous dim 0 should NOT appear in any dict
        for d in ffl:
            assert 0 not in d

        # Categorical dims are 1, 2, 3
        # First combo: A -> {1: 1.0, 2: 0.0, 3: 0.0}
        assert ffl[0] == {1: 1.0, 2: 0.0, 3: 0.0}
        assert ffl[1] == {1: 0.0, 2: 1.0, 3: 0.0}
        assert ffl[2] == {1: 0.0, 2: 0.0, 3: 1.0}

    def test_two_categoricals_with_continuous(self) -> None:
        """Mixed: 1 continuous + 2 categoricals (2 x 3 = 6 dicts)."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="cat1", type=ParameterType.CATEGORICAL, categories=["A", "B"]),
                ParameterSpec(
                    name="cat2", type=ParameterType.CATEGORICAL, categories=["X", "Y", "Z"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        ffl = build_fixed_features_list(spec)
        assert len(ffl) == 6  # 2 * 3

        # Continuous dim 0 should NOT appear in any dict
        for d in ffl:
            assert 0 not in d
            # All keys should be in range 1-5 (dims for 2 + 3 = 5 one-hot dims)
            for key in d:
                assert 1 <= key <= 5

    def test_correct_dimension_indices(self) -> None:
        """Verify dimension indices match the one-hot layout in transforms."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat1", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"]
                ),
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="cat2", type=ParameterType.CATEGORICAL, categories=["X", "Y"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        ffl = build_fixed_features_list(spec)
        # cat1 occupies dims 0, 1, 2 (3 categories)
        # x1 occupies dim 3 (continuous)
        # cat2 occupies dims 4, 5 (2 categories)
        assert len(ffl) == 6  # 3 * 2

        # x1 dim (3) should NOT appear
        for d in ffl:
            assert 3 not in d

        # First combo: cat1=A, cat2=X
        assert ffl[0] == {0: 1.0, 1: 0.0, 2: 0.0, 4: 1.0, 5: 0.0}

    def test_product_of_categorical_sizes(self, mixed_spec: OptimizationSpec) -> None:
        """Number of dicts should equal the product of categorical sizes."""
        ffl = build_fixed_features_list(mixed_spec)
        n_expected = count_categorical_combinations(mixed_spec)
        assert len(ffl) == n_expected
