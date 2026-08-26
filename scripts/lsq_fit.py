#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

HBARC = 197.3269804
AMU = 931.49410242
E2 = 1.43996448

PARAMETER_NAMES = ("V0", "kappa", "r0", "a", "lambda_so", "r0_so")
SEMINOLE_REFERENCE = np.array([52.06, 0.639, 1.260, 0.662, 24.1, 1.160], dtype=float)

# Used only to draw a physically reasonable random initial point. LM itself is unbounded.
START_LOW = np.array([40.0, 0.30, 1.15, 0.55, 15.0, 0.90], dtype=float)
START_HIGH = np.array([65.0, 1.00, 1.35, 0.75, 40.0, 1.35], dtype=float)

# Broad validity limits. Invalid LM trial points return a large fixed residual.
VALID_LOW = np.array([20.0, -1.0, 0.70, 0.20, 0.0, 0.60], dtype=float)
VALID_HIGH = np.array([90.0, 2.0, 1.80, 1.20, 80.0, 1.80], dtype=float)

IDENTIFICATION_CASES = (
    (40, 20, False),
    (48, 20, False),
    (132, 50, False),
    (208, 82, False),
    (48, 20, True),
    (132, 50, True),
    (208, 82, True),
)


@dataclass(frozen=True)
class SeminoleParameters:
    V0: float
    kappa: float
    r0: float
    a: float
    lambda_so: float
    r0_so: float

    @classmethod
    def from_array(cls, x: np.ndarray) -> "SeminoleParameters":
        return cls(*map(float, x))


@dataclass
class Counters:
    residual_calls: int = 0
    channel_eigensolves: int = 0


def load_npz_dataset(path: Path) -> pd.DataFrame:
    archive = np.load(path, allow_pickle=True)
    if "data" not in archive:
        raise ValueError("NPZ file must contain an object array named 'data'.")

    rows: list[dict] = []
    for item in archive["data"]:
        A = int(item["A"])
        Z = int(item["Z"])
        is_proton = bool(item["is_proton"])
        for st in item["states"]:
            rows.append({
                "A": A,
                "Z": Z,
                "is_proton": is_proton,
                "nr": int(st["nr"]),
                "l": int(st["l"]),
                "j": float(st["j"]),
                "energy": float(st["energy"]),
                "orbital": str(st.get("orbital", "")),
                "state_type": str(st.get("state_type", "")).strip().lower(),
            })
    df = pd.DataFrame(rows)
    if len(df) != 86:
        print(f"Warning: Schwierz bound-state dataset normally has 86 levels; loaded {len(df)}.")
    return df


def pair_preserving_selection(group: pd.DataFrame, max_states: int = 6) -> pd.DataFrame:
    """Exact state-selection logic used by the PINN identification code."""
    if group.empty:
        raise ValueError("Cannot select states from an empty nucleus--species system.")

    records = group.to_dict("records")
    deepest = min(records, key=lambda st: float(st["energy"]))
    selected = [deepest]

    def state_id(st: dict) -> tuple:
        return (int(st["nr"]), int(st["l"]), float(st["j"]), float(st["energy"]))

    selected_ids = {state_id(deepest)}
    grouped: dict[tuple[int, int], list[dict]] = {}
    for st in records:
        grouped.setdefault((int(st["nr"]), int(st["l"])), []).append(st)

    pair_groups = [grp for grp in grouped.values() if len(grp) == 2]
    unpaired = [st for grp in grouped.values() if len(grp) == 1 for st in grp]
    sorted_pairs = sorted(
        pair_groups,
        key=lambda grp: sum(float(s["energy"]) for s in grp) / 2.0,
    )

    remaining_slots = max_states - len(selected)
    max_pairs = remaining_slots // 2
    if max_pairs > 0:
        if len(sorted_pairs) > max_pairs:
            n_deep = max(1, (max_pairs * 2) // 3)
            n_shallow = max_pairs - n_deep
            chosen_pairs = sorted_pairs[:n_deep]
            if n_shallow > 0:
                chosen_pairs += sorted_pairs[-n_shallow:]
        else:
            chosen_pairs = sorted_pairs

        for grp in chosen_pairs:
            for st in grp:
                if len(selected) >= max_states:
                    break
                if state_id(st) not in selected_ids:
                    selected.append(st)
                    selected_ids.add(state_id(st))

    for pool in (
        sorted(unpaired, key=lambda st: float(st["energy"])),
        sorted(records, key=lambda st: float(st["energy"])),
    ):
        for st in pool:
            if len(selected) >= max_states:
                break
            if state_id(st) not in selected_ids:
                selected.append(st)
                selected_ids.add(state_id(st))

    return pd.DataFrame(sorted(selected, key=lambda st: float(st["energy"])))


def build_pinn_identification_set(df: pd.DataFrame, max_states: int = 6) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for A, Z, is_proton in IDENTIFICATION_CASES:
        group = df[(df.A == A) & (df.Z == Z) & (df.is_proton == is_proton)]
        selected = pair_preserving_selection(group, max_states=max_states)
        if len(selected) != max_states:
            raise RuntimeError(
                f"Expected {max_states} states for A={A}, Z={Z}, "
                f"is_proton={is_proton}; found {len(selected)}."
            )
        parts.append(selected)

    result = pd.concat(parts, ignore_index=True)
    if len(result) != 42:
        raise RuntimeError(f"Expected 42 identification levels, found {len(result)}.")
    return result


def reduced_mass(A: int, is_proton: bool) -> float:
    m_core = float(A - 1) * AMU
    m_nucleon = (1.007276466621 if is_proton else 1.00866491588) * AMU
    return (m_core * m_nucleon) / (m_core + m_nucleon)


def l_dot_s(l: int, j: float) -> float:
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def tdotT_expectation(A: int, Z: int, is_proton: bool) -> float:
    """Equation (14) convention in Schwierz et al."""
    N = A - Z
    if N == Z:
        rhs = 3.0
    elif N > Z:
        term = float(N - Z + 1)
        rhs = (term if is_proton else -term) + 2.0
    else:
        term = float(N - Z - 1)
        rhs = (term if is_proton else -term) + 2.0
    return -0.25 * rhs


def ws_form(r: np.ndarray, radius: float, diffuseness: float) -> np.ndarray:
    x = np.clip((r - radius) / diffuseness, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(x))


def ws_derivative(r: np.ndarray, radius: float, diffuseness: float) -> np.ndarray:
    f = ws_form(r, radius, diffuseness)
    return -(f * (1.0 - f)) / diffuseness


def solver_nucleus(row: pd.Series, prescription: str) -> tuple[int, int]:
    """Map a tabulated magic nucleus to the nucleus used in the forward problem.

    Published prescription:
      neutron particle: (N_m+1, Z_m) => (A_m+1, Z_m)
      proton particle:  (N_m, Z_m+1) => (A_m+1, Z_m+1)
      hole state:       (N_m, Z_m)   => (A_m, Z_m)
    """
    A = int(row.A)
    Z = int(row.Z)
    if prescription == "direct" or row.state_type != "particle":
        return A, Z
    if bool(row.is_proton):
        return A + 1, Z + 1
    return A + 1, Z


def effective_potential(
    r: np.ndarray,
    A: int,
    Z: int,
    is_proton: bool,
    l: int,
    j: float,
    p: SeminoleParameters,
) -> np.ndarray:
    mu = reduced_mass(A, is_proton)
    R = p.r0 * A ** (1.0 / 3.0)
    Rso = p.r0_so * A ** (1.0 / 3.0)

    depth = p.V0 * (1.0 - (4.0 * p.kappa / A) * tdotT_expectation(A, Z, is_proton))
    central = -depth * ws_form(r, R, p.a)

    if is_proton:
        z_core = max(Z - 1, 0)
        coulomb = np.where(
            r <= R,
            z_core * E2 * (3.0 * R * R - r * r) / (2.0 * R**3),
            z_core * E2 / r,
        )
    else:
        coulomb = np.zeros_like(r)

    # Eq. (9), Eq. (10), and V_tilde=lambda*V0 from Eq. (16).
    spin_orbit = (
        (HBARC**2 / (2.0 * mu**2))
        * (p.lambda_so * p.V0)
        * (ws_derivative(r, Rso, p.a) / r)
        * l_dot_s(l, j)
    )
    return central + coulomb + spin_orbit


def channel_eigenvalues(
    A: int,
    Z: int,
    is_proton: bool,
    l: int,
    j: float,
    p: SeminoleParameters,
    r_max: float,
    n_grid: int,
    n_eigs: int,
) -> np.ndarray:
    r = np.linspace(0.0, r_max, n_grid)
    dr = r[1] - r[0]
    ri = r[1:-1]
    mu = reduced_mass(A, is_proton)
    K = HBARC**2 / (2.0 * mu)

    potential = effective_potential(ri, A, Z, is_proton, l, j, p)
    diagonal = 2.0 * K / dr**2 + K * l * (l + 1.0) / ri**2 + potential
    off = np.full(ri.size - 1, -K / dr**2)
    H = diags((off, diagonal, off), offsets=(-1, 0, 1), format="csc")

    k = min(max(int(n_eigs), 1), H.shape[0] - 2)
    vals = eigsh(H, k=k, which="SA", return_eigenvectors=False, tol=1e-9)
    return np.sort(np.asarray(vals, dtype=float))


class ForwardModel:
    def __init__(self, r_max: float, n_grid: int, prescription: str):
        self.r_max = float(r_max)
        self.n_grid = int(n_grid)
        self.prescription = prescription
        self.counters = Counters()

    def predict(self, states: pd.DataFrame, x: np.ndarray) -> np.ndarray:
        p = SeminoleParameters.from_array(x)
        predictions = np.empty(len(states), dtype=float)

        groups: dict[tuple, list[tuple[int, int]]] = {}
        for idx, row in states.reset_index(drop=True).iterrows():
            Acalc, Zcalc = solver_nucleus(row, self.prescription)
            key = (Acalc, Zcalc, bool(row.is_proton), int(row.l), float(row.j))
            groups.setdefault(key, []).append((idx, int(row.nr)))

        for (A, Z, is_p, l, j), entries in groups.items():
            max_nr = max(nr for _, nr in entries)
            vals = channel_eigenvalues(
                A, Z, is_p, l, j, p,
                r_max=self.r_max,
                n_grid=self.n_grid,
                n_eigs=max(max_nr + 4, 8),
            )
            self.counters.channel_eigensolves += 1
            for idx, nr in entries:
                if nr >= len(vals):
                    raise RuntimeError(f"Missing nr={nr} eigenvalue for channel {(A,Z,is_p,l,j)}")
                predictions[idx] = vals[nr]
        return predictions

    def residuals(self, x: np.ndarray, states: pd.DataFrame) -> np.ndarray:
        self.counters.residual_calls += 1
        x = np.asarray(x, dtype=float)
        if (not np.all(np.isfinite(x)) or np.any(x < VALID_LOW) or np.any(x > VALID_HIGH)):
            return np.full(len(states), 1.0e4, dtype=float)
        try:
            pred = self.predict(states, x)
            return pred - states.energy.to_numpy(dtype=float)
        except Exception:
            return np.full(len(states), 1.0e4, dtype=float)


def metrics(residual: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual, dtype=float)
    return {
        "N": int(residual.size),
        "MAE_MeV": float(np.mean(np.abs(residual))),
        "RMSE_MeV": float(np.sqrt(np.mean(residual**2))),
        "bias_MeV": float(np.mean(residual)),
        "max_abs_error_MeV": float(np.max(np.abs(residual))),
        "chi2_unweighted": float(np.sum(residual**2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experimental_dataset.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lsq_fit"))
    parser.add_argument("--r-max", type=float, default=25.0)
    parser.add_argument("--n-grid", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--n-starts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=1000)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--xtol", type=float, default=1e-10)
    parser.add_argument("--gtol", type=float, default=1e-10)
    parser.add_argument("--prescription", choices=("schwierz", "direct"), default="schwierz")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()

    if args.n_starts < 1:
        raise ValueError("--n-starts must be >= 1")

    full_states = load_npz_dataset(args.dataset)
    states = build_pinn_identification_set(full_states, max_states=6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loaded {len(full_states)} available bound orbital energies from {args.dataset}")
    print(f"Selected exactly {len(states)} PINN identification levels")
    print(states[["A", "Z", "is_proton", "orbital", "nr", "l", "j", "energy", "state_type"]].to_string(index=False))
    print(f"Forward prescription: {args.prescription}")
    print("Algorithm: Levenberg-Marquardt (scipy.optimize.least_squares, method='lm')")
    print("Objective: unweighted sum of squared energy residuals")

    run_summaries = []
    best = None
    total_start = perf_counter()

    for start_id in range(args.n_starts):
        x0 = rng.uniform(START_LOW, START_HIGH)
        model = ForwardModel(args.r_max, args.n_grid, args.prescription)
        print(f"\nStart {start_id + 1}/{args.n_starts}: {dict(zip(PARAMETER_NAMES, x0))}")
        t0 = perf_counter()
        result = least_squares(
            model.residuals,
            x0=x0,
            args=(states,),
            method="lm",
            jac="2-point",
            x_scale="jac",
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
            max_nfev=args.max_nfev,
            verbose=args.verbose,
        )
        elapsed = perf_counter() - t0
        residual = model.predict(states, result.x) - states.energy.to_numpy(float)
        m = metrics(residual)
        summary = {
            "start_id": start_id,
            "initial_parameters": dict(zip(PARAMETER_NAMES, map(float, x0))),
            "fitted_parameters": dict(zip(PARAMETER_NAMES, map(float, result.x))),
            "metrics": m,
            "runtime_seconds": elapsed,
            "nfev": int(result.nfev),
            "njev": None if result.njev is None else int(result.njev),
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
            "optimality": float(result.optimality),
            "residual_calls": model.counters.residual_calls,
            "channel_eigensolves": model.counters.channel_eigensolves,
        }
        run_summaries.append(summary)
        if best is None or m["chi2_unweighted"] < best[0]:
            best = (m["chi2_unweighted"], result.x.copy(), residual.copy(), summary)

    assert best is not None
    _, best_x, best_residual, best_summary = best
    total_runtime = perf_counter() - total_start

    out = states.copy()
    out["E_fd_MeV"] = out.energy.to_numpy(float) + best_residual
    out["residual_MeV"] = best_residual
    out["abs_error_MeV"] = np.abs(best_residual)
    out.to_csv(args.output_dir / "best_fit_state_table.csv", index=False)

    report = {
        "interpretation": "Schwierz-style LM baseline on the exact 42-level PINN identification set; not the original Seminole calibration",
        "dataset": str(args.dataset),
        "n_available_levels": len(full_states),
        "n_identification_levels": len(states),
        "algorithm": "Levenberg-Marquardt via scipy.optimize.least_squares(method='lm')",
        "objective": "unweighted chi-square = sum_k (E_FD - E_exp)^2",
        "particle_hole_prescription": args.prescription,
        "r_max_fm": args.r_max,
        "n_grid": args.n_grid,
        "seed": args.seed,
        "n_starts": args.n_starts,
        "total_runtime_seconds": total_runtime,
        "seminole_reference": dict(zip(PARAMETER_NAMES, map(float, SEMINOLE_REFERENCE))),
        "best_run": best_summary,
        "all_runs": run_summaries,
    }
    with open(args.output_dir / "fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nBest fitted parameters:")
    for name, value in zip(PARAMETER_NAMES, best_x):
        print(f"  {name:10s} = {value:.8f}")
    print("\nBest metrics:")
    for key, value in metrics(best_residual).items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
