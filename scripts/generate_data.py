#!/usr/bin/env python3
"""Generate a synthetic Woods--Saxon dataset with the independent FD solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ws_pinn.config import load_config
from ws_pinn.fd_solver import FDParameters, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic NPZ archive using the reference parameters "
            "and generation settings recorded in an experiment YAML."
        )
    )
    parser.add_argument("--config", required=True, help="Seminole or Wahlborn YAML")
    parser.add_argument(
        "--parameters",
        help=(
            "Optional parameter JSON from training; omit to use the YAML "
            "reference parameters"
        ),
    )
    parser.add_argument("--output", help="Output NPZ path")
    parser.add_argument("--n-grid", type=int, help="Override generation.n_grid")
    parser.add_argument("--r-max", type=float, help="Override generation.r_max")
    parser.add_argument("--max-l", type=int, help="Override generation.max_l")
    parser.add_argument("--max-nr", type=int, help="Override generation.max_nr")
    args = parser.parse_args()

    config = load_config(args.config)
    potential = config["experiment"]["potential"]
    parameter_values = config["reference_parameters"]
    parameter_source = "reference_parameters in YAML"
    if args.parameters:
        parameter_path = Path(args.parameters).expanduser().resolve()
        with parameter_path.open("r", encoding="utf-8") as stream:
            parameter_values = json.load(stream)
        parameter_source = str(parameter_path)
    params = FDParameters.from_mapping(parameter_values)
    generation = config.get("generation", {})
    if "nuclei" not in generation:
        raise KeyError(
            "The YAML must define generation.nuclei as a list of [A, Z] pairs."
        )

    nuclei = []
    for entry in generation["nuclei"]:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("Each generation.nuclei entry must contain [A, Z].")
        nuclei.append((int(entry[0]), int(entry[1])))

    n_grid = int(args.n_grid or generation.get("n_grid", 2400))
    r_max = float(args.r_max or generation.get("r_max", 25.0))
    max_l = int(args.max_l if args.max_l is not None else generation.get("max_l", 5))
    max_nr = int(
        args.max_nr if args.max_nr is not None else generation.get("max_nr", 3)
    )

    output = Path(
        args.output or f"data/generated_{potential}_synthetic_dataset.npz"
    )
    if not output.is_absolute():
        output = Path(config["_repository_root"]) / output

    print(f"Potential: {potential}")
    print(f"Parameter source: {parameter_source}")
    print(f"Physical parameters: {params}")
    print(
        f"Nuclei: {len(nuclei)}, n_grid={n_grid}, r_max={r_max} fm, "
        f"max_l={max_l}, max_nr={max_nr}"
    )
    path = generate_dataset(
        nuclei,
        params,
        potential,
        output,
        n_grid=n_grid,
        r_max=r_max,
        max_l=max_l,
        max_nr=max_nr,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
