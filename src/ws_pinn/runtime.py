"""Runtime selection and reproducible random-number initialization."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


device = _resolve_device(os.environ.get("WS_PINN_DEVICE", "auto"))


def configure_runtime(
    requested_device: str = "auto",
    seed: int = 0,
    deterministic: bool = False,
) -> torch.device:
    """Configure the runtime before importing model/training modules."""
    global device
    device = _resolve_device(requested_device)
    os.environ["WS_PINN_DEVICE"] = requested_device

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return device


def runtime_metadata() -> dict[str, object]:
    """Return software and hardware metadata saved with every run."""
    return {
        "device": str(device),
        "python_torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "numpy_version": np.__version__,
    }

