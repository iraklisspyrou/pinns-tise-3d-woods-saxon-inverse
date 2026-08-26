"""Checkpoint loading without starting a training run."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import training_kwargs
from .data_loader import build_multinucleus_samples
from .models import GlobalContextParamNet, WaveNet3D
from .runtime import device


def load_models_and_samples(config: dict, checkpoint_path: str | Path):
    """Instantiate the configured networks and restore compatible state dicts."""
    kwargs = training_kwargs(config)
    samples, skipped = build_multinucleus_samples(
        dataset_path=kwargs["dataset_path"],
        cases=kwargs["cases"],
        max_states=kwargs["max_states"],
        require_exact_states=True,
    )
    if not samples:
        raise RuntimeError("No valid systems were loaded from the configured dataset.")

    input_dim = (
        kwargs["max_states"] * kwargs["n_r_points"]
        + kwargs["max_states"]
        + 3
        + 3 * kwargs["max_states"]
    )
    wave_net = WaveNet3D(
        n_states=kwargs["max_states"],
        hidden=kwargs["hidden_wave"],
        depth=kwargs["wave_depth"],
        beta=kwargs["beta"],
    ).to(device)
    param_net = GlobalContextParamNet(
        input_dim=input_dim,
        hidden=kwargs["hidden_param"],
        depth=kwargs["param_depth"],
        n_parameters=6,
    ).to(device)

    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a checkpoint dictionary containing state dicts.")

    wave_state = checkpoint.get("wave_net")
    param_state = checkpoint.get("param_net")
    if wave_state is None or param_state is None:
        raise KeyError("Checkpoint must contain 'wave_net' and 'param_net' state dicts.")

    wave_net.load_state_dict(wave_state, strict=True)
    param_net.load_state_dict(param_state, strict=True)
    wave_net.eval()
    param_net.eval()
    return wave_net, param_net, samples, skipped, checkpoint
