"""Create spectrum plots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

L_LETTERS = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
    5: "h",
    6: "i",
    7: "j",
    8: "k",
}

ELEMENT_SYMBOLS = {
    8: "O",
    20: "Ca",
    28: "Ni",
    40: "Zr",
    50: "Sn",
    82: "Pb",
}

StateKey = tuple[int, int, bool, int, int, int]


def load_nested_npz(path: str | Path) -> list[dict[str, Any]]:
    """Load the actual nested dictionary format used by the supplied files."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as archive:
        if "data" not in archive.files:
            raise KeyError(
                f"{path} does not contain the required top-level key 'data'. "
                f"Available keys: {archive.files}"
            )
        raw = archive["data"]

    records: list[dict[str, Any]] = []
    for item in raw.tolist():
        if not isinstance(item, dict):
            raise TypeError(f"Unexpected entry in {path}: {type(item)!r}")
        required = {"A", "Z", "is_proton", "states"}
        missing = required.difference(item)
        if missing:
            raise KeyError(f"Entry in {path} is missing fields: {sorted(missing)}")
        records.append(item)
    return records


def normalized_key(A: int, Z: int, is_proton: bool, state: dict[str, Any]) -> StateKey:
    """Use 2j as an integer to avoid floating-point matching problems."""
    return (
        int(A),
        int(Z),
        bool(is_proton),
        int(state["nr"]),
        int(state["l"]),
        int(round(2.0 * float(state["j"]))),
    )


def flatten_records(records: list[dict[str, Any]]) -> dict[StateKey, dict[str, Any]]:
    flat: dict[StateKey, dict[str, Any]] = {}
    for block in records:
        A = int(block["A"])
        Z = int(block["Z"])
        is_proton = bool(block["is_proton"])
        for state in block["states"]:
            key = normalized_key(A, Z, is_proton, state)
            flat[key] = dict(state)
    return flat


def available_nuclei(records: list[dict[str, Any]]) -> set[tuple[int, int]]:
    return {(int(block["A"]), int(block["Z"])) for block in records}


def orbital_label(key: StateKey, exp_state: dict[str, Any] | None = None) -> str:
    if exp_state is not None and exp_state.get("orbital"):
        return str(exp_state["orbital"])
    _, _, _, nr, l, two_j = key
    shell = nr + 1
    letter = L_LETTERS.get(l, f"l{l}")
    return f"{shell}{letter}{two_j}/2"


def energy_of(mapping: dict[StateKey, dict[str, Any]], key: StateKey) -> float | None:
    state = mapping.get(key)
    if state is None:
        return None
    value = float(state["energy"])
    return value if np.isfinite(value) else None


def adjusted_label_positions(values: list[float], minimum_spacing: float = 0.28) -> list[float]:
    """Reduce overlapping orbital labels while retaining their energy ordering."""
    if not values:
        return []
    order = np.argsort(values)
    adjusted = np.asarray(values, dtype=float).copy()
    for previous, current in zip(order[:-1], order[1:]):
        if adjusted[current] - adjusted[previous] < minimum_spacing:
            adjusted[current] = adjusted[previous] + minimum_spacing
    return adjusted.tolist()


def plot_nucleus(
    A: int,
    Z: int,
    exp: dict[StateKey, dict[str, Any]],
    seminole: dict[StateKey, dict[str, Any]],
    pinn: dict[StateKey, dict[str, Any]],
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 400,
) -> None:
    neutron_x = (0.0, 1.0, 2.0)
    proton_x = (4.0, 5.0, 6.0)
    column_labels = ("exp", "Seminole", "PINN")
    half_width = 0.24

    exp_keys = [
        key for key in exp
        if key[0] == A and key[1] == Z
    ]
    if not exp_keys:
        raise ValueError(f"No experimental states found for A={A}, Z={Z}")

    # Plot only experimentally tabulated states, matching the logic of the
    # original level-scheme figures.
    neutron_keys = sorted(
        [key for key in exp_keys if not key[2]],
        key=lambda key: energy_of(exp, key) if energy_of(exp, key) is not None else np.inf,
    )
    proton_keys = sorted(
        [key for key in exp_keys if key[2]],
        key=lambda key: energy_of(exp, key) if energy_of(exp, key) is not None else np.inf,
    )

    all_energies: list[float] = []
    for key in exp_keys:
        for mapping in (exp, seminole, pinn):
            value = energy_of(mapping, key)
            if value is not None:
                all_energies.append(value)

    if not all_energies:
        raise ValueError(f"No finite energies found for A={A}, Z={Z}")

    ymin = min(all_energies)
    ymax = max(all_energies)
    span = max(ymax - ymin, 4.0)
    lower = np.floor((ymin - 0.08 * span) / 2.0) * 2.0
    upper = np.ceil((max(ymax, 0.0) + 0.08 * span) / 2.0) * 2.0

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.set_xlim(-0.9, 6.75)
    ax.set_ylim(lower, upper)
    ax.set_ylabel("Energy [MeV]", fontsize=12)

    # Dotted grid, matching the visual logic of the reference figure.
    yticks = np.arange(np.ceil(lower / 2.0) * 2.0, upper + 0.01, 2.0)
    ax.set_yticks(yticks)
    ax.grid(axis="y", linestyle=":", linewidth=0.75, color="0.3")
    for x in neutron_x + proton_x:
        ax.axvline(x, linestyle=":", linewidth=0.75, color="0.45", zorder=0)

    def draw_species(keys: list[StateKey], xs: tuple[float, float, float], title: str) -> None:
        exp_y = [float(energy_of(exp, key)) for key in keys]
        label_y = adjusted_label_positions(exp_y, minimum_spacing=max(0.22, 0.018 * span))

        for key, text_y in zip(keys, label_y):
            energies = (
                energy_of(exp, key),
                energy_of(seminole, key),
                energy_of(pinn, key),
            )

            for x, energy in zip(xs, energies):
                if energy is None:
                    continue
                ax.hlines(
                    energy,
                    x - half_width,
                    x + half_width,
                    linewidth=0.9,
                    color="black",
                    zorder=3,
                )

            for index in range(2):
                first = energies[index]
                second = energies[index + 1]
                if first is None or second is None:
                    continue
                ax.plot(
                    [xs[index] + half_width, xs[index + 1] - half_width],
                    [first, second],
                    linestyle=(0, (2.2, 2.4)),
                    linewidth=0.75,
                    color="0.15",
                    zorder=2,
                )

            label = orbital_label(key, exp[key])
            exp_energy = energies[0]
            ax.text(
                xs[0] - 0.38,
                text_y,
                label,
                ha="right",
                va="center",
                fontsize=8.5,
            )
            # Small leader when a label has been displaced to prevent overlap.
            if exp_energy is not None and abs(text_y - exp_energy) > 1.0e-9:
                ax.plot(
                    [xs[0] - 0.35, xs[0] - half_width],
                    [text_y, exp_energy],
                    linewidth=0.5,
                    color="0.25",
                )

        ax.text(np.mean(xs), upper - 0.11 * (upper - lower), title,
                ha="center", va="top", fontsize=12)

    draw_species(neutron_keys, neutron_x, "neut.")
    draw_species(proton_keys, proton_x, "prot.")

    ax.set_xticks(neutron_x + proton_x)
    ax.set_xticklabels(column_labels + column_labels, fontsize=10)

    # Neutron and proton numbers, as in the original figures.
    N = A - Z
    number_y = lower + 0.33 * (upper - lower)
    ax.text(np.mean(neutron_x), number_y, str(N), ha="center", va="center", fontsize=11)
    ax.text(np.mean(proton_x), number_y, str(Z), ha="center", va="center", fontsize=11)

    symbol = ELEMENT_SYMBOLS.get(Z, f"Z{Z}")
    ax.set_title(rf"$^{{{A}}}\mathrm{{{symbol}}}$", fontsize=15, pad=12)

    # Report missing matches visibly in the terminal, but do not fabricate levels.
    for model_name, mapping in (("Seminole", seminole), ("PINN", pinn)):
        missing = [orbital_label(key, exp[key]) for key in exp_keys if key not in mapping]
        if missing:
            print(f"Warning: {model_name} has no matching state for {A}{symbol}: {', '.join(missing)}")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{A}{symbol}_exp_seminole_pinn"
    for extension in formats:
        out = output_dir / f"{stem}.{extension}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def parse_nucleus_token(token: str) -> tuple[int, int]:
    """Parse forms such as 208:82 or 208,82."""
    for separator in (":", ","):
        if separator in token:
            A_text, Z_text = token.split(separator, 1)
            return int(A_text), int(Z_text)
    raise argparse.ArgumentTypeError(
        f"Invalid nucleus '{token}'. Use A:Z, for example 208:82."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimental", default="data/experimental_dataset.npz")
    parser.add_argument("--seminole", required=True, help="FD spectrum from reference parameters")
    parser.add_argument("--pinn", required=True, help="FD spectrum from inferred parameters")
    parser.add_argument("--output-dir", default="spectrum_plots")
    parser.add_argument(
        "--nucleus",
        action="append",
        type=parse_nucleus_token,
        help="Nucleus as A:Z. Repeat for multiple nuclei. Default: all nuclei common to all three files.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    exp_records = load_nested_npz(args.experimental)
    sem_records = load_nested_npz(args.seminole)
    pinn_records = load_nested_npz(args.pinn)

    exp = flatten_records(exp_records)
    seminole = flatten_records(sem_records)
    pinn = flatten_records(pinn_records)

    exp_nuclei = available_nuclei(exp_records)
    sem_nuclei = available_nuclei(sem_records)
    pinn_nuclei = available_nuclei(pinn_records)
    common_nuclei = sorted(exp_nuclei & sem_nuclei & pinn_nuclei)

    nuclei = args.nucleus if args.nucleus else common_nuclei
    if not nuclei:
        raise RuntimeError("No nuclei are common to the three datasets.")

    unavailable = [nucleus for nucleus in nuclei if nucleus not in common_nuclei]
    if unavailable:
        formatted = ", ".join(f"A={A}, Z={Z}" for A, Z in unavailable)
        raise ValueError(
            "The requested nuclei are not present in all three datasets: " + formatted
        )

    output_dir = Path(args.output_dir)
    print("Common nuclei:", common_nuclei)
    for A, Z in nuclei:
        plot_nucleus(
            A=A,
            Z=Z,
            exp=exp,
            seminole=seminole,
            pinn=pinn,
            output_dir=output_dir,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
