"""Independent finite-difference radial Woods--Saxon solver.

This module is deliberately NumPy/SciPy based and does not call WaveNet or
ParamNet.  It is used both to generate synthetic spectra and to perform the
independent closure validation reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.linalg import eigh_tridiagonal

HC = 197.3269804
U = 931.49410242
E2 = 1.43996448


@dataclass(frozen=True)
class FDParameters:
    V0: float
    kappa: float
    r0: float
    a: float
    lam_so: float
    r0_so: float

    @classmethod
    def from_mapping(cls, values: dict) -> "FDParameters":
        cleaned = {}
        for key in ("V0", "kappa", "r0", "a", "lam_so", "r0_so"):
            value = values[key]
            cleaned[key] = float(value.get("mean", value) if isinstance(value, dict) else value)
        return cls(**cleaned)


def reduced_mass(A: int, is_proton: bool) -> float:
    core = float(A - 1) * U
    nucleon = (1.007276466621 if is_proton else 1.00866491588) * U
    return core * nucleon / (core + nucleon)


def l_dot_s(l: int, j: float) -> float:
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def tdotT_expectation(A: int, Z: int, is_proton: bool) -> float:
    N = A - Z
    if N == Z:
        rhs = 3.0
    elif N > Z:
        rhs = ((N - Z + 1) if is_proton else -(N - Z + 1)) + 2.0
    else:
        rhs = ((N - Z - 1) if is_proton else -(N - Z - 1)) + 2.0
    return -0.25 * float(rhs)


def central_depth(
    A: int, Z: int, is_proton: bool, params: FDParameters, potential: str
) -> float:
    if potential == "wahlborn":
        eta = 1.0 if is_proton else -1.0
        return params.V0 * (1.0 + eta * params.kappa * (A - 2 * Z) / A)
    if potential == "seminole":
        return params.V0 * (
            1.0 - 4.0 * params.kappa * tdotT_expectation(A, Z, is_proton) / A
        )
    raise ValueError("potential must be 'seminole' or 'wahlborn'")


def effective_potential(
    r: np.ndarray,
    A: int,
    Z: int,
    is_proton: bool,
    l: int,
    j: float,
    params: FDParameters,
    potential: str,
) -> np.ndarray:
    mu = reduced_mass(A, is_proton)
    kinetic_factor = HC**2 / (2.0 * mu)
    radius = params.r0 * A ** (1.0 / 3.0)
    radius_so = params.r0_so * A ** (1.0 / 3.0)

    x = np.clip((r - radius) / params.a, -80.0, 80.0)
    f = 1.0 / (1.0 + np.exp(x))
    depth = central_depth(A, Z, is_proton, params, potential)
    central = -depth * f

    coulomb = np.zeros_like(r)
    if is_proton:
        z_core = max(Z - 1, 0)
        inside = r <= radius
        coulomb[inside] = z_core * E2 / (2.0 * radius) * (
            3.0 - (r[inside] / radius) ** 2
        )
        coulomb[~inside] = z_core * E2 / r[~inside]

    x_so = np.clip((r - radius_so) / params.a, -80.0, 80.0)
    exp_so = np.exp(x_so)
    df_dr = -(exp_so / (1.0 + exp_so) ** 2) / params.a
    so_depth = depth if potential == "wahlborn" else params.V0
    spin_orbit = (
        HC**2
        / (2.0 * mu**2 * r)
        * (params.lam_so * so_depth * df_dr)
        * l_dot_s(l, j)
    )

    centrifugal = kinetic_factor * l * (l + 1.0) / r**2
    return central + coulomb + spin_orbit + centrifugal


def radial_eigenvalues(
    A: int,
    Z: int,
    is_proton: bool,
    l: int,
    j: float,
    params: FDParameters,
    potential: str,
    n_eigenvalues: int = 8,
    r_max: float = 25.0,
    n_grid: int = 2400,
) -> np.ndarray:
    """Solve the Dirichlet problem for u(r)=rR(r) on an interior grid."""
    full_grid = np.linspace(0.0, r_max, n_grid)
    r = full_grid[1:-1]
    dr = full_grid[1] - full_grid[0]
    mu = reduced_mass(A, is_proton)
    kinetic_factor = HC**2 / (2.0 * mu)

    potential_values = effective_potential(
        r, A, Z, is_proton, l, j, params, potential
    )
    diagonal = 2.0 * kinetic_factor / dr**2 + potential_values
    off_diagonal = np.full(r.size - 1, -kinetic_factor / dr**2)
    upper = min(max(n_eigenvalues - 1, 0), diagonal.size - 1)
    values = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, upper),
        check_finite=False,
        eigvals_only=True,
    )
    return np.asarray(values, dtype=float)


def allowed_j(l: int) -> tuple[float, ...]:
    return (0.5,) if l == 0 else (l - 0.5, l + 0.5)


def generate_spectrum(
    A: int,
    Z: int,
    is_proton: bool,
    params: FDParameters,
    potential: str,
    max_l: int = 5,
    max_nr: int = 3,
    **solver_kwargs,
) -> list[dict]:
    states = []
    for l in range(max_l + 1):
        for j in allowed_j(l):
            values = radial_eigenvalues(
                A,
                Z,
                is_proton,
                l,
                j,
                params,
                potential,
                n_eigenvalues=max_nr + 4,
                **solver_kwargs,
            )
            bound = values[values < 0.0]
            for nr, energy in enumerate(bound[: max_nr + 1]):
                states.append({"nr": nr, "l": l, "j": j, "energy": float(energy)})
    return sorted(states, key=lambda state: state["energy"])


def generate_dataset(
    nuclei: Iterable[tuple[int, int]],
    params: FDParameters,
    potential: str,
    output: str | Path,
    **solver_kwargs,
) -> Path:
    records = []
    for A, Z in nuclei:
        for is_proton in (False, True):
            records.append(
                {
                    "A": int(A),
                    "Z": int(Z),
                    "is_proton": bool(is_proton),
                    "states": generate_spectrum(
                        A,
                        Z,
                        is_proton,
                        params,
                        potential,
                        **solver_kwargs,
                    ),
                }
            )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, data=np.asarray(records, dtype=object))
    return output


def validate_against_dataset(
    dataset_path: str | Path,
    params: FDParameters,
    potential: str,
    **solver_kwargs,
) -> list[dict]:
    """Recompute every tabulated channel and return matched residual rows."""
    with np.load(dataset_path, allow_pickle=True) as archive:
        records = archive["data"].tolist()

    rows = []
    cache: dict[tuple[int, int, bool, int, float], np.ndarray] = {}
    for block in records:
        A, Z, is_proton = int(block["A"]), int(block["Z"]), bool(block["is_proton"])
        for state in block["states"]:
            key = (A, Z, is_proton, int(state["l"]), float(state["j"]))
            if key not in cache:
                cache[key] = radial_eigenvalues(
                    A,
                    Z,
                    is_proton,
                    key[3],
                    key[4],
                    params,
                    potential,
                    n_eigenvalues=max(int(state["nr"]) + 4, 8),
                    **solver_kwargs,
                )
            nr = int(state["nr"])
            values = cache[key]
            if nr >= len(values):
                continue
            predicted = float(values[nr])
            target = float(state["energy"])
            rows.append(
                {
                    "A": A,
                    "Z": Z,
                    "species": "proton" if is_proton else "neutron",
                    "nr": nr,
                    "l": int(state["l"]),
                    "j": float(state["j"]),
                    "target_energy": target,
                    "fd_energy": predicted,
                    "residual": predicted - target,
                }
            )
    return rows

