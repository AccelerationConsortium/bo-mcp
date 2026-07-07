"""Tests for BO engine transforms."""

import pytest
import torch

from bo_engine.constants import DISCRETE_ENUMERATION_MAX_POINTS
from bo_engine.transforms import (
    count_discrete_combinations,
    decode_categorical,
    discrete_enumeration_limit_error,
    encode_categorical,
    get_bounds_tensor,
    get_n_dims,
    normalize_inputs,
    unnormalize_inputs,
)
from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType


def _spec(parameters: list[ParameterSpec]) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


class TestCountDiscreteCombinations:
    """Product counting across every enumerable parameter kind.

    The count is what a backend materializes when it enumerates the full
    discrete portion of the search space (e.g. BayBE's
    ``SearchSpace.from_product``), so values grids, bounds-only integer
    grids, and category lists must all multiply into one product.
    """

    def test_mixed_enumerable_kinds_multiply(self):
        spec = _spec(
            [
                ParameterSpec(name="grid", type=ParameterType.DISCRETE, values=[1.0, 2.0, 3.0]),
                ParameterSpec(name="n", type=ParameterType.DISCRETE, bounds=(0.0, 4.0)),
                ParameterSpec(name="cat", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
            ]
        )
        assert count_discrete_combinations(spec) == 3 * 5 * 2

    def test_continuous_parameters_do_not_contribute(self):
        spec = _spec(
            [
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="grid", type=ParameterType.DISCRETE, values=[1.0, 2.0]),
            ]
        )
        assert count_discrete_combinations(spec) == 2

    def test_no_enumerable_parameters_counts_one(self):
        spec = _spec([ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))])
        assert count_discrete_combinations(spec) == 1

    def test_bounds_only_grid_counts_contained_integers(self):
        """Bounds (0.5, 4.2) contain the integers 1..4 — the materialized grid."""
        spec = _spec([ParameterSpec(name="n", type=ParameterType.DISCRETE, bounds=(0.5, 4.2))])
        assert count_discrete_combinations(spec) == 4

    def test_limit_error_names_count_and_limit(self):
        error = discrete_enumeration_limit_error(123_456)
        assert "123456" in str(error)
        assert str(DISCRETE_ENUMERATION_MAX_POINTS) in str(error)


class TestTransforms:
    """Tests for input/output transformations."""

    def test_get_bounds_tensor_continuous(self):
        """get_bounds_tensor works for continuous parameters."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x1",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="x2",
                    type=ParameterType.CONTINUOUS,
                    bounds=(-5.0, 5.0),
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        bounds = get_bounds_tensor(spec)
        assert bounds.shape == (2, 2)
        assert bounds[0, 0].item() == 0.0
        assert bounds[1, 0].item() == 1.0
        assert bounds[0, 1].item() == -5.0
        assert bounds[1, 1].item() == 5.0

    def test_get_bounds_tensor_categorical(self):
        """get_bounds_tensor expands categorical to one-hot."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        bounds = get_bounds_tensor(spec)
        # 3 categories = 3 one-hot dimensions
        assert bounds.shape == (2, 3)
        # All one-hot bounds are [0, 1]
        assert (bounds[0] == 0).all()
        assert (bounds[1] == 1).all()

    def test_get_bounds_tensor_discrete_without_bounds_or_values_raises(self):
        """A discrete parameter with neither bounds nor values fails loudly.

        ``_encode_param_value`` emits exactly one column for a discrete
        parameter, so a zero-column bounds contribution would silently shift
        every subsequent parameter's bounds (a wrong-domain optimization that
        need not even crash). The bounds builder must raise with the offending
        parameter name, mirroring the continuous/categorical branches.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="bad", type=ParameterType.DISCRETE),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        with pytest.raises(ValueError, match="Discrete parameter 'bad'"):
            get_bounds_tensor(spec)

    def test_bounds_tensor_width_matches_n_dims(self):
        """Bounds-tensor width and ``get_n_dims`` agree for every parameter type.

        Both helpers must report the same encoded width, otherwise the bounds
        tensor and ``train_x`` columns misalign downstream. This pins the
        invariant across a mixed continuous/discrete/categorical spec.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="d_bounds", type=ParameterType.DISCRETE, bounds=(1.0, 10.0)),
                ParameterSpec(name="d_values", type=ParameterType.DISCRETE, values=[2.0, 4.0, 8.0]),
                ParameterSpec(
                    name="cat", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        assert get_bounds_tensor(spec).shape[1] == get_n_dims(spec)

    def test_get_n_dims(self):
        """get_n_dims counts dimensions correctly."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x1",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),
                ),
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        # 1 continuous + 3 one-hot = 4
        assert get_n_dims(spec) == 4

    def test_encode_decode_categorical(self):
        """encode_categorical and decode_categorical are inverses."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x1",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 10.0),
                ),
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        original = {"x1": 5.0, "cat": "B"}
        encoded = encode_categorical(original, spec)

        # Check encoding shape
        assert encoded.shape == (4,)  # 1 + 3

        # Check one-hot encoding
        assert encoded[0].item() == 5.0  # x1
        assert encoded[1].item() == 0.0  # A
        assert encoded[2].item() == 1.0  # B
        assert encoded[3].item() == 0.0  # C

        # Decode back
        decoded = decode_categorical(encoded, spec)
        assert decoded["x1"] == 5.0
        assert decoded["cat"] == "B"

    def test_decode_categorical_argmax_independent_of_relaxation(self):
        """Decoding picks the argmax over the one-hot block deterministically.

        ``softmax`` is monotone, so removing it must not change the picked
        category for any continuous relaxation. We sweep a few representative
        relaxed encodings (logits, soft-positive, near-uniform) to pin the
        invariant: the largest entry's category is returned.

        Reference: BoTorch tutorial on mixed-integer / categorical search
        spaces (https://botorch.org/tutorials/) treats categorical decoding
        as argmax over the one-hot block.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        for encoded, expected in [
            (torch.tensor([0.1, 0.9, 0.0]), "B"),
            (torch.tensor([0.34, 0.33, 0.33]), "A"),
            (torch.tensor([-1.0, -0.5, 0.7]), "C"),
            (torch.tensor([5.0, 1.0, 1.0]), "A"),
        ]:
            decoded = decode_categorical(encoded, spec)
            assert decoded["cat"] == expected

    def test_decode_categorical_ties_resolve_to_first_index(self):
        """Equal one-hot entries decode to the *first* listed category.

        ``torch.argmax`` returns the lowest index on ties, matching the
        encoder's ordering. This is the deterministic tie-breaking rule the
        new comment in ``_decode_param_value`` documents.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B", "C"],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        # All three categories tied; argmax must pick index 0 == "A".
        encoded = torch.tensor([0.5, 0.5, 0.5])
        assert decode_categorical(encoded, spec)["cat"] == "A"

    def test_normalize_unnormalize(self):
        """normalize_inputs and unnormalize_inputs are inverses."""
        bounds = torch.tensor([[0.0, -5.0], [10.0, 5.0]])
        x = torch.tensor([[5.0, 0.0], [10.0, 5.0]])

        normalized = normalize_inputs(x, bounds)
        assert normalized[0, 0].item() == pytest.approx(0.5, rel=1e-5)
        assert normalized[0, 1].item() == pytest.approx(0.5, rel=1e-5)
        assert normalized[1, 0].item() == pytest.approx(1.0, rel=1e-5)
        assert normalized[1, 1].item() == pytest.approx(1.0, rel=1e-5)

        recovered = unnormalize_inputs(normalized, bounds)
        assert torch.allclose(x, recovered)


class TestUnknownCategoryRejected:
    """A typo'd category value must fail loudly, not zero-encode.

    An unknown value silently produced an all-zero one-hot block, which
    argmax-decodes back to the *first* category — silent training-data
    corruption for direct engine callers that bypass the MCP intake
    validation (``bo_engine`` is documented standalone, and the pending-
    point encoder feeds this path too).
    """

    def test_unknown_category_raises_value_error_naming_parameter(self) -> None:
        import pytest

        from bo_engine.transforms import encode_categorical
        from bo_engine.types import (
            ObjectiveSpec,
            OptimizationSpec,
            ParameterSpec,
            ParameterType,
        )

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["etoh", "meoh"],
                )
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        with pytest.raises(ValueError, match="solvent") as excinfo:
            encode_categorical({"solvent": "EtOH"}, spec)
        assert "EtOH" in str(excinfo.value)

    def test_known_category_still_one_hot_encodes(self) -> None:
        from bo_engine.transforms import encode_categorical
        from bo_engine.types import (
            ObjectiveSpec,
            OptimizationSpec,
            ParameterSpec,
            ParameterType,
        )

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="solvent",
                    type=ParameterType.CATEGORICAL,
                    categories=["etoh", "meoh"],
                )
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        encoded = encode_categorical({"solvent": "meoh"}, spec)
        assert encoded.tolist() == [0.0, 1.0]
