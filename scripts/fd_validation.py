#!/usr/bin/env python3
"""Validate a reference or inferred parameter set with the independent FD solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ws_pinn.config import load_config
from ws_pinn.fd_solver import FDParameters, validate_against_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", help="Override the dataset selected in the YAML")
    parser.add_argument(
        "--parameters",
        help="JSON from training; omit to validate the YAML reference parameters",
    )
    parser.add_argument("--output", default="fd_validation.csv")
    parser.add_argument("--n-grid", type=int, default=2400)
    args = parser.parse_args()

    config = load_config(args.config)
    values = config["reference_parameters"]
    if args.parameters:
        with Path(args.parameters).open("r", encoding="utf-8") as stream:
            values = json.load(stream)

    dataset = args.dataset or config["data"]["path"]
    if not Path(dataset).is_absolute():
        dataset = Path(config["_repository_root"]) / dataset

    rows = validate_against_dataset(
        dataset,
        FDParameters.from_mapping(values),
        config["experiment"]["potential"],
        n_grid=args.n_grid,
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)

    residual = frame["residual"].to_numpy(dtype=float)
    print(f"Matched states: {len(frame)}")
    print(f"MAE:  {np.mean(np.abs(residual)):.6f} MeV")
    print(f"RMSE: {np.sqrt(np.mean(residual**2)):.6f} MeV")
    for species, group in frame.groupby("species"):
        values = group["residual"].to_numpy(dtype=float)
        print(
            f"{species:8s} MAE={np.mean(np.abs(values)):.6f} "
            f"RMSE={np.sqrt(np.mean(values**2)):.6f} MeV"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

