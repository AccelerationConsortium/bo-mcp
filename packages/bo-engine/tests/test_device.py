"""Tests for device management and GPU auto-detection."""

import os

import torch

from bo_engine.device import (
    clear_cache,
    ensure_device,
    get_device,
    get_device_info,
    get_dtype,
    reset_device_cache,
    to_device,
)


class TestDeviceManagement:
    """Tests for device detection and tensor management."""

    def setup_method(self):
        """Reset device cache before each test."""
        reset_device_cache()
        # Clear env vars
        os.environ.pop("BO_ENGINE_DEVICE", None)
        os.environ.pop("BO_ENGINE_DISABLE_GPU", None)

    def teardown_method(self):
        """Clean up after each test."""
        reset_device_cache()
        os.environ.pop("BO_ENGINE_DEVICE", None)
        os.environ.pop("BO_ENGINE_DISABLE_GPU", None)

    def test_get_device_returns_valid_device(self):
        """get_device returns a valid torch.device (CUDA or CPU by default)."""
        device = get_device()
        assert isinstance(device, torch.device)
        # MPS is skipped by default because BoTorch needs float64
        assert device.type in ("cpu", "cuda")

    def test_get_device_caches_result(self):
        """get_device caches the result for performance."""
        device1 = get_device()
        device2 = get_device()
        assert device1 == device2

    def test_reset_device_cache_clears_cache(self):
        """reset_device_cache clears the cached device."""
        _ = get_device()
        reset_device_cache()
        # Force CPU
        os.environ["BO_ENGINE_DEVICE"] = "cpu"
        device = get_device()
        assert device.type == "cpu"

    def test_force_cpu_via_env(self):
        """BO_ENGINE_DEVICE=cpu forces CPU."""
        os.environ["BO_ENGINE_DEVICE"] = "cpu"
        device = get_device()
        assert device.type == "cpu"

    def test_disable_gpu_via_env(self):
        """BO_ENGINE_DISABLE_GPU=1 disables GPU."""
        os.environ["BO_ENGINE_DISABLE_GPU"] = "1"
        device = get_device()
        assert device.type == "cpu"

    def test_get_dtype_returns_float64_by_default(self):
        """get_dtype returns float64 on CPU/CUDA (default devices)."""
        dtype = get_dtype()
        # Default device is CUDA or CPU, both support float64
        assert dtype == torch.float64

    def test_get_dtype_returns_float32_on_mps(self):
        """get_dtype returns float32 when MPS is forced."""
        os.environ["BO_ENGINE_DEVICE"] = "mps"
        reset_device_cache()
        dtype = get_dtype()
        assert dtype == torch.float32

    def test_to_device_moves_tensor(self):
        """to_device moves tensor to selected device."""
        tensor = torch.randn(3, 3)
        result = to_device(tensor)
        assert result.device.type == get_device().type
        assert result.dtype == get_dtype()

    def test_to_device_preserves_data(self):
        """to_device preserves tensor data."""
        original = torch.tensor([1.0, 2.0, 3.0])
        result = to_device(original)
        assert torch.allclose(result.cpu().float(), original.float())

    def test_ensure_device_multiple_tensors(self):
        """ensure_device handles multiple tensors."""
        t1 = torch.randn(2, 2)
        t2 = torch.randn(3, 3)
        t3 = torch.randn(4, 4)

        r1, r2, r3 = ensure_device(t1, t2, t3)

        device = get_device()
        assert r1.device.type == device.type
        assert r2.device.type == device.type
        assert r3.device.type == device.type

    def test_get_device_info_returns_dict(self):
        """get_device_info returns a dictionary with device info."""
        info = get_device_info()
        assert isinstance(info, dict)
        assert "device" in info
        assert "type" in info

    def test_get_device_info_cpu_fields(self):
        """get_device_info includes cpu_only field for CPU."""
        os.environ["BO_ENGINE_DEVICE"] = "cpu"
        reset_device_cache()
        info = get_device_info()
        assert info["type"] == "cpu"
        assert info.get("cpu_only", False) is True

    def test_clear_cache_runs_without_error(self):
        """clear_cache runs without error on any device."""
        clear_cache()  # Should not raise


class TestDeviceWithBOOperations:
    """Tests for device awareness in BO operations."""

    def setup_method(self):
        """Reset device cache before each test."""
        reset_device_cache()
        os.environ.pop("BO_ENGINE_DEVICE", None)
        os.environ.pop("BO_ENGINE_DISABLE_GPU", None)

    def teardown_method(self):
        """Clean up after each test."""
        reset_device_cache()
        os.environ.pop("BO_ENGINE_DEVICE", None)
        os.environ.pop("BO_ENGINE_DISABLE_GPU", None)

    def test_bounds_tensor_on_device(self):
        """get_bounds_tensor creates tensors on the correct device."""
        from bo_engine.transforms import get_bounds_tensor
        from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        bounds = get_bounds_tensor(spec)
        assert bounds.device.type == get_device().type
        assert bounds.dtype == get_dtype()

    def test_encode_categorical_on_device(self):
        """encode_categorical creates tensors on the correct device."""
        from bo_engine.transforms import encode_categorical
        from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        encoded = encode_categorical({"x": 0.5}, spec)
        assert encoded.device.type == get_device().type
        assert encoded.dtype == get_dtype()
