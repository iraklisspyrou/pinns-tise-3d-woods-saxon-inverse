#!/usr/bin/env python3
"""Create loss and parameter-history figures from training_history.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PARAMETERS = ("V0", "kappa", "r0", "a", "lam_so", "r0_so")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", help="training_history.csv")
    parser.add_argument("--output-dir", default="training_plots")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf")
    args = parser.parse_args()

    frame = pd.read_csv(args.history)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = frame["epoch"] if "epoch" in frame else frame.index

    loss_columns = [
        name for name in ("loss", "LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LSO", "LKL")
        if name in frame
    ]
    if loss_columns:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for name in loss_columns:
            ax.plot(x, frame[name].clip(lower=1.0e-30), label=name)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"loss_history.{args.format}", bbox_inches="tight")
        plt.close(fig)

    for name in PARAMETERS:
        mean_column, sigma_column = f"{name}_mu", f"{name}_sigma"
        if mean_column not in frame:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        mean = frame[mean_column]
        ax.plot(x, mean, label="distribution mean")
        if sigma_column in frame:
            sigma = frame[sigma_column]
            ax.fill_between(x, mean - sigma, mean + sigma, alpha=0.25, label="model spread")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{name}_history.{args.format}", bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plots in {out}")


if __name__ == "__main__":
    main()

