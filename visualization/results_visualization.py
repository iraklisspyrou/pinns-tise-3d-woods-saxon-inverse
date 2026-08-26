#!/usr/bin/env python3
"""Plot predicted energies and residuals from a result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def choose_columns(frame: pd.DataFrame) -> tuple[str, str]:
    for target, prediction in (
        ("E_target", "E_pred"),
        ("target_energy", "fd_energy"),
        ("energy", "predicted_energy"),
    ):
        if target in frame and prediction in frame:
            return target, prediction
    raise KeyError("CSV must contain E_target/E_pred or target_energy/fd_energy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Energy-table or FD-validation CSV")
    parser.add_argument("--output-dir", default="result_plots")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    args = parser.parse_args()

    frame = pd.read_csv(args.results)
    target_column, prediction_column = choose_columns(frame)
    target = frame[target_column].to_numpy(dtype=float)
    prediction = frame[prediction_column].to_numpy(dtype=float)
    residual = prediction - target
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    if "species" in frame:
        for species, group in frame.groupby("species"):
            ax.scatter(group[target_column], group[prediction_column], label=species, alpha=0.8)
        ax.legend()
    else:
        ax.scatter(target, prediction, alpha=0.8)
    limits = [min(target.min(), prediction.min()), max(target.max(), prediction.max())]
    ax.plot(limits, limits, "k--", linewidth=1)
    ax.set_xlabel("Target energy [MeV]")
    ax.set_ylabel("Predicted energy [MeV]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"predicted_vs_target.{args.format}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.scatter(np.arange(len(residual)), residual, s=18)
    ax.set_xlabel("State index")
    ax.set_ylabel("Residual [MeV]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / f"energy_residuals.{args.format}", bbox_inches="tight")
    plt.close(fig)

    print(f"MAE={np.mean(np.abs(residual)):.6f} MeV")
    print(f"RMSE={np.sqrt(np.mean(residual**2)):.6f} MeV")
    print(f"Saved plots in {out}")


if __name__ == "__main__":
    main()

