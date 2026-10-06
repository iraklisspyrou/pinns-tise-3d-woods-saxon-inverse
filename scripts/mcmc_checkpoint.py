"""Atomic, pickle-free checkpoints for the six-parameter emcee baseline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def save_checkpoint(path, chain, log_probability, random_state, metadata):
    """Replace the previous checkpoint only after the new archive is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(
                stream, chain=chain, log_probability=log_probability,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                rng_name=np.asarray(random_state[0]), rng_keys=random_state[1],
                rng_position=np.asarray(random_state[2]),
                rng_has_gauss=np.asarray(random_state[3]),
                rng_cached_gaussian=np.asarray(random_state[4]),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, settings):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("settings") != settings:
            raise ValueError(
                "Checkpoint settings differ from this run. Keep the same dataset, "
                "prior bounds, likelihood, radial solver, walkers, seed and software. "
                "Use a new output directory for a different inference problem."
            )
        chain = archive["chain"].copy()
        log_probability = archive["log_probability"].copy()
        shape = (settings["walkers"], 6)
        if (chain.ndim != 3 or chain.shape[1:] != shape or len(chain) == 0
                or log_probability.shape != chain.shape[:2]
                or not np.all(np.isfinite(chain))
                or not np.all(np.isfinite(log_probability))):
            raise ValueError("Checkpoint has invalid chain dimensions or non-finite samples")
        random_state = (
            str(archive["rng_name"].item()), archive["rng_keys"].copy(),
            int(archive["rng_position"].item()), int(archive["rng_has_gauss"].item()),
            float(archive["rng_cached_gaussian"].item()),
        )
    return chain, log_probability, random_state, metadata
