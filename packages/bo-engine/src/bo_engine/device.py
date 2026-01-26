"""Device management for GPU auto-detection and acceleration.

Provides centralized device selection with automatic GPU detection.
Priority: CUDA > CPU (MPS skipped by default due to float64 incompatibility)

BoTorch requires float64 for numerical stability, which MPS doesn't support.
Use BO_ENGINE_DEVICE=mps to force MPS if you accept float32 precision.

Environment variables for control:
- BO_ENGINE_DEVICE: Force specific device (cuda, mps, cpu)
- BO_ENGINE_DISABLE_GPU: Set to '1' to disable GPU
- CUDA_VISIBLE_DEVICES: Standard CUDA device control
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

_DEVICE_CACHE: torch.device | None = None


def get_device() -> torch.device:
    """Get the best available compute device.

    Priority: CUDA > CPU
    MPS is skipped by default because BoTorch requires float64
    which MPS doesn't support. Use BO_ENGINE_DEVICE=mps to force it.

    Returns:
        torch.device for computation
    """
    global _DEVICE_CACHE

    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    forced_device = os.environ.get("BO_ENGINE_DEVICE", "").lower()
    if forced_device in ("cuda", "mps", "cpu"):
        _DEVICE_CACHE = torch.device(forced_device)
        return _DEVICE_CACHE

    if os.environ.get("BO_ENGINE_DISABLE_GPU", "").lower() in ("1", "true"):
        _DEVICE_CACHE = torch.device("cpu")
        return _DEVICE_CACHE

    # Only use CUDA for GPU acceleration (MPS doesn't support float64)
    if torch.cuda.is_available():
        _DEVICE_CACHE = torch.device("cuda")
    else:
        _DEVICE_CACHE = torch.device("cpu")

    return _DEVICE_CACHE


def get_dtype() -> torch.dtype:
    """Get the optimal dtype for BO computations.

    Returns float64 for numerical stability on CPU/CUDA.
    Returns float32 on MPS (Apple Silicon) as it doesn't support float64.
    """
    device = get_device()
    if device.type == "mps":
        return torch.float32
    return torch.float64


def to_device(tensor: Tensor) -> Tensor:
    """Move tensor to the selected compute device.

    Args:
        tensor: Input tensor

    Returns:
        Tensor on the selected device with correct dtype
    """
    device = get_device()
    dtype = get_dtype()
    if tensor.device != device or tensor.dtype != dtype:
        return tensor.to(device=device, dtype=dtype)
    return tensor


def ensure_device(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Ensure all tensors are on the selected device.

    Args:
        *tensors: Variable number of tensors

    Returns:
        Tuple of tensors on the selected device
    """
    return tuple(to_device(t) for t in tensors)


def clear_cache() -> None:
    """Clear GPU memory cache if applicable."""
    device = get_device()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def reset_device_cache() -> None:
    """Reset the cached device (useful for testing)."""
    global _DEVICE_CACHE
    _DEVICE_CACHE = None


def get_device_info() -> dict[str, str | bool | int]:
    """Get information about the current compute device.

    Returns:
        Dictionary with device information
    """
    device = get_device()
    info: dict[str, str | bool | int] = {
        "device": str(device),
        "type": device.type,
    }

    if device.type == "cuda":
        info["cuda_available"] = True
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(device)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(device)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(device)
    elif device.type == "mps":
        info["mps_available"] = True
    else:
        info["cpu_only"] = True

    return info
