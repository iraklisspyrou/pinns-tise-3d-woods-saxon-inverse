#!/usr/bin/env python3
"""Generate a synthetic Woods--Saxon dataset with the independent FD solver."""

from __future__ import annotations

import argparse
from pathlib import Path

from ws_pinn.config import load_config
from ws_pinn.fd_solver import FDParameters, generate_dataset

DEFAULT_NUCLEI = [
    (12, 6), (16, 8), (20, 10), (24, 12), (28, 14), (32, 16),
    (40, 18), (40, 20), (48, 20), (48, 22), (56, 28), (60, 28),
    (72, 30), (90, 40), (100, 50), (132, 50), (208, 82),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Seminole or Wahlborn YAML")
    parser.add_argument("--output", help="Output NPZ path")
    parser.add_argument("--n-grid", type=int, default=2400)
    args = parser.parse_args()

    config = load_config(args.config)
    potential = config["experiment"]["potential"]
    params = FDParameters.from_mapping(config["reference_parameters"])
    output = args.output or f"generated_{potential}_synthetic_dataset.npz"
    path = generate_dataset(
        DEFAULT_NUCLEI,
        params,
        potential,
        Path(output),
        n_grid=args.n_grid,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

