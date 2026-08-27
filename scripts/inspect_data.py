#!/usr/bin/env python3
"""Summarize the supplied NPZ datasets and generate reviewer-facing plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


DEFAULT_DATASETS = (
    Path("data/seminole_synthetic_dataset.npz"),
    Path("data/wahlborn_synthetic_dataset.npz"),
    Path("data/experimental_dataset.npz"),
)

DISPLAY_NAMES = {
    "seminole_synthetic_dataset": "Seminole synthetic",
    "wahlborn_synthetic_dataset": "Wahlborn synthetic",
    "experimental_dataset": "Experimental",
}


def load_levels(path: Path) -> pd.DataFrame:
    """Flatten one object-array NPZ archive into one row per energy level."""
    with np.load(path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise ValueError(f"{path} does not contain an array named 'data'.")
        systems = archive["data"].tolist()

    dataset = DISPLAY_NAMES.get(path.stem, path.stem)
    rows: list[dict] = []
    for system in systems:
        A = int(system["A"])
        Z = int(system["Z"])
        is_proton = bool(system["is_proton"])
        species = "proton" if is_proton else "neutron"
        for state in system["states"]:
            rows.append({
                "dataset": dataset,
                "A": A,
                "Z": Z,
                "species": species,
                "nr": int(state["nr"]),
                "l": int(state["l"]),
                "j": float(state["j"]),
                "energy_MeV": float(state["energy"]),
                "orbital": str(state.get("orbital", "")),
                "state_type": str(state.get("state_type", "")),
            })
    return pd.DataFrame(rows)


def save_state_count_plot(levels: pd.DataFrame, output: Path) -> None:
    datasets = list(levels["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        len(datasets),
        1,
        figsize=(11, 3.2 * len(datasets)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, datasets):
        part = levels[levels["dataset"] == dataset]
        counts = (
            part.groupby(["A", "Z", "species"], as_index=False)
            .size()
            .sort_values(["A", "Z", "species"])
        )
        labels = [
            f"A={row.A}, Z={row.Z}\n{row.species[0]}"
            for row in counts.itertuples()
        ]
        colors = [
            "#d95f02" if species == "proton" else "#1b9e77"
            for species in counts["species"]
        ]
        ax.bar(np.arange(len(counts)), counts["size"], color=colors)
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Bound levels")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.25)

    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_energy_plot(levels: pd.DataFrame, output: Path) -> None:
    datasets = list(levels["dataset"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5.0 * len(datasets), 4.6),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax, dataset in zip(axes, datasets):
        part = levels[levels["dataset"] == dataset]
        for species, color, offset in (
            ("neutron", "#1b9e77", -0.6),
            ("proton", "#d95f02", 0.6),
        ):
            selected = part[part["species"] == species]
            ax.scatter(
                selected["A"] + offset,
                selected["energy_MeV"],
                s=18,
                alpha=0.72,
                color=color,
                label=species,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(dataset)
        ax.set_xlabel("Mass number A")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Single-particle energy (MeV)")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="NPZ archives to inspect (defaults to all supplied datasets)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/data_inspection"),
    )
    args = parser.parse_args()

    missing = [path for path in args.datasets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Dataset files not found: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    levels = pd.concat(
        [load_levels(path) for path in args.datasets],
        ignore_index=True,
    )
    summary = (
        levels.groupby(["dataset", "A", "Z", "species"], as_index=False)
        .agg(
            levels=("energy_MeV", "size"),
            minimum_energy_MeV=("energy_MeV", "min"),
            maximum_energy_MeV=("energy_MeV", "max"),
        )
        .sort_values(["dataset", "A", "Z", "species"])
    )

    levels.to_csv(args.output_dir / "all_levels.csv", index=False)
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    save_state_count_plot(levels, args.output_dir / "dataset_state_counts.png")
    save_energy_plot(levels, args.output_dir / "dataset_energy_overview.png")

    totals = levels.groupby("dataset").size()
    print("Dataset inspection complete:")
    for dataset, count in totals.items():
        systems = summary[summary["dataset"] == dataset].shape[0]
        print(f"  {dataset}: {systems} systems, {count} levels")
    print(f"Saved tables and plots to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
