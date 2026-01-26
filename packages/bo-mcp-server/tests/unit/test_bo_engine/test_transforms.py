"""Tests for BO engine transforms."""

import pytest
import torch
from bo_engine.transforms import (
    decode_categorical,
    encode_categorical,
    get_bounds_tensor,
    get_n_dims,
    normalize_inputs,
    unnormalize_inputs,
)
from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType


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
