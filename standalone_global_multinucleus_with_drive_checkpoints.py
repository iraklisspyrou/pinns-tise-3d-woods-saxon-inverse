"""
Standalone Google Colab script for the six-parameter global multi-nucleus PINN.

This file contains:
  1. the complete global-context PINN,
  2. global multi-nucleus training,
  3. Weights & Biases logging,
  4. training-history plots,
  5. global parameter plots,
  6. energy reconstruction diagnostics,
  7. radial wavefunction plots,
  8. angular heatmaps,
  9. overlap diagnostics,
  10. checkpoints, CSV files and JSON summaries.

No second Python file or imported user module is required.

The only external input is the NPZ dataset.

Google Colab use:
    1. Upload this file and your dataset to /content.
    2. Run:
           !pip -q install wandb pandas matplotlib
           import wandb
           wandb.login()
           %run /content/standalone_global_multinucleus_wandb_colab.py

Edit only the RUN CONFIGURATION section at the bottom when needed.
"""

# ============================================================
# Global-context six-parameter Wahlborn-like Woods-Saxon PINN
#
# Main idea:
#   - WaveNet predicts R(r), Theta(theta), Phi(phi) per selected state.
#   - ParamNet receives WaveNet radial samples from ALL nuclei/species.
#   - ParamNet mean-pools those sample encodings and outputs ONE global
#     posterior q(z) for one shared parameter set:
#       V0, kappa, r0, a, lambda_SO, r0_SO.
#
# This is not the old per-nucleus conditional ParamNet.
# It is also not a free global vector with no input.
# It is a global, context-informed parameter network.
# ============================================================

import math
import json
import time
import csv
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    PLOTTING_AVAILABLE = False

# ============================================================
# Device / seeds
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Constants
# ============================================================
hc = 197.3269804
u = 931.49410242
e2 = 1.43996448
R_MAX = 25.0
PI = math.pi
TWOPI = 2.0 * math.pi

# Prior in latent space:
#     p(z) = N(0, PRIOR_STD^2 I)
PRIOR_STD = 2.0

# Edit these only for printing/comparison if your FD generator used
# slightly different target values.
WAHLBORN_REFERENCE = {
    "V0": 51.0,
    "kappa": 0.67,
    "r0": 1.27,
    "a": 0.67,
    "lam_so": 32.0,
    "r0_so": 1.27,
}

# ============================================================
# Dataset utilities
# ============================================================
def load_fd_dataset(path="ws_fd_dataset_2.npz"):
    raw = np.load(path, allow_pickle=True)["data"]
    return list(raw)


def get_sample_by_nucleus(dataset, A, Z, is_proton, max_states=6):
    """
    Pair-preserving state selector.

    Selection priority:
      1. deepest bound state,
      2. spin-orbit pairs grouped by (nr,l),
      3. unpaired states,
      4. remaining unused states.

    No fake states are created.
    """
    for item in dataset:
        if (
            int(item["A"]) == int(A)
            and int(item["Z"]) == int(Z)
            and bool(item["is_proton"]) == bool(is_proton)
        ):
            all_states = [
                {
                    "nr": int(st["nr"]),
                    "l": int(st["l"]),
                    "j": float(st["j"]),
                    "energy": float(st["energy"]),
                }
                for st in item["states"]
            ]

            if len(all_states) == 0:
                raise ValueError(
                    f"No states found for A={A}, Z={Z}, is_proton={is_proton}"
                )

            deepest_state = sorted(all_states, key=lambda x: float(x["energy"]))[0]
            selected_states = [deepest_state]

            def state_id(st):
                return (
                    int(st["nr"]),
                    int(st["l"]),
                    float(st["j"]),
                    float(st["energy"]),
                )

            selected_ids = {state_id(deepest_state)}

            groups = {}
            for st in all_states:
                key = (int(st["nr"]), int(st["l"]))
                groups.setdefault(key, []).append(st)

            pair_groups = [grp for grp in groups.values() if len(grp) == 2]
            unpaired_states = [
                st for grp in groups.values() if len(grp) == 1 for st in grp
            ]

            sorted_pairs = sorted(
                pair_groups,
                key=lambda grp: sum(float(s["energy"]) for s in grp) / 2.0,
            )

            remaining_slots = max_states - len(selected_states)
            max_pairs = remaining_slots // 2

            if max_pairs > 0:
                if len(sorted_pairs) > max_pairs:
                    target_deep_pairs = (max_pairs * 2) // 3
                    if target_deep_pairs == 0 and max_pairs > 0:
                        target_deep_pairs = 1

                    shallow_pairs_count = max_pairs - target_deep_pairs
                    deep_pairs = sorted_pairs[:target_deep_pairs]
                    shallow_pairs = (
                        sorted_pairs[-shallow_pairs_count:]
                        if shallow_pairs_count > 0
                        else []
                    )
                    chosen_pairs = deep_pairs + shallow_pairs
                else:
                    chosen_pairs = sorted_pairs

                for grp in chosen_pairs:
                    for st in grp:
                        if len(selected_states) >= max_states:
                            break
                        if state_id(st) not in selected_ids:
                            selected_states.append(st)
                            selected_ids.add(state_id(st))

            needed_states = max_states - len(selected_states)

            if needed_states > 0 and unpaired_states:
                unpaired_states = sorted(
                    unpaired_states,
                    key=lambda x: float(x["energy"]),
                )
                for st in unpaired_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))

            if len(selected_states) < max_states:
                remaining_states = sorted(
                    all_states,
                    key=lambda x: float(x["energy"]),
                )
                for st in remaining_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))

            selected_states = sorted(
                selected_states,
                key=lambda x: float(x["energy"]),
            )

            species_name = "proton" if is_proton else "neutron"
            print(f"\nSelected states for A={A}, Z={Z}, species={species_name}:")
            for idx, state in enumerate(selected_states, start=1):
                print(
                    f"  {idx:2d}. "
                    f"nr={int(state['nr'])}, "
                    f"l={int(state['l'])}, "
                    f"j={float(state['j']):.1f}, "
                    f"E={float(state['energy']):.6f} MeV"
                )

            return {
                "A": int(A),
                "Z": int(Z),
                "is_proton": bool(is_proton),
                "states": selected_states,
            }

    raise ValueError(f"No sample found for A={A}, Z={Z}, is_proton={is_proton}")


def list_available_cases(dataset_path="ws_fd_dataset_2.npz"):
    dataset = load_fd_dataset(dataset_path)
    return [(int(x["A"]), int(x["Z"]), bool(x["is_proton"])) for x in dataset]


def build_multinucleus_samples(
    dataset_path="ws_fd_dataset_2.npz",
    cases=None,
    max_states=6,
    require_exact_states=True,
):
    dataset = load_fd_dataset(dataset_path)
    if cases is None:
        cases = list_available_cases(dataset_path)

    samples, skipped = [], []
    for A, Z, is_proton in cases:
        try:
            sample = get_sample_by_nucleus(
                dataset,
                A=A,
                Z=Z,
                is_proton=is_proton,
                max_states=max_states,
            )
            if require_exact_states and len(sample["states"]) != max_states:
                skipped.append((A, Z, is_proton, len(sample["states"])))
                continue
            samples.append(sample)
        except Exception as e:
            skipped.append((A, Z, is_proton, str(e)))

    return samples, skipped


# ============================================================
# Physics helpers
# ============================================================
def reduced_mass(A: int, is_proton: bool):
    m_core = float(A - 1) * u
    m_nucl = (1.007276466621 * u) if is_proton else (1.00866491588 * u)
    return (m_core * m_nucl) / (m_core + m_nucl)


def K_value(mu):
    return hc**2 / (2.0 * mu)


def l_dot_s(l: int, j: float):
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


@dataclass
class WSParams:
    V0: torch.Tensor
    kappa: torch.Tensor
    r0: torch.Tensor
    a: torch.Tensor
    lam_so: torch.Tensor
    r0_so: torch.Tensor


# ============================================================
# Scaling
# ============================================================
def scale_energy(E):
    return E / 100.0


def scale_r(r):
    return r / R_MAX


def scale_theta(th):
    return th / PI


def scale_phi(ph):
    return ph / TWOPI


def scale_nucleus(A, Z, is_proton):
    return torch.tensor(
        [float(A) / 250.0, float(Z) / 100.0, 1.0 if is_proton else 0.0],
        dtype=torch.float32,
        device=device,
    )


def scale_quantum(nr, l, j):
    return torch.tensor(
        [float(nr) / 5.0, float(l) / 8.0, float(j) / 8.0],
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# Six-parameter physical map
# ============================================================
def params_from_raw_six(raw_params):
    """
    Latent-to-physical bounded map.

    Bounds are chosen to include the usual Wahlborn-like values.
    """
    V0 = 40.0 + 25.0 * torch.sigmoid(raw_params[0])
    kappa = 0.30 + 0.70 * torch.sigmoid(raw_params[1])
    r0 = 1.15 + 0.20 * torch.sigmoid(raw_params[2])
    a = 0.55 + 0.20 * torch.sigmoid(raw_params[3])
    lam_so = 15.0 + 20.0 * torch.sigmoid(raw_params[4])
    r0_so = 1.05 + 0.25 * torch.sigmoid(raw_params[5])

    return WSParams(
        V0=V0,
        kappa=kappa,
        r0=r0,
        a=a,
        lam_so=lam_so,
        r0_so=r0_so,
    )


@torch.no_grad()
def physical_stats_from_mu_sigma(mu, sigma, n_samples=1000):
    eps = torch.randn(n_samples, 6, device=mu.device)
    raw = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    V0 = 40.0 + 25.0 * torch.sigmoid(raw[:, 0])
    kappa = 0.30 + 0.70 * torch.sigmoid(raw[:, 1])
    r0 = 1.15 + 0.20 * torch.sigmoid(raw[:, 2])
    a = 0.55 + 0.20 * torch.sigmoid(raw[:, 3])
    lam_so = 15.0 + 20.0 * torch.sigmoid(raw[:, 4])
    r0_so = 1.05 + 0.25 * torch.sigmoid(raw[:, 5])

    return {
        "V0": (float(V0.mean()), float(V0.std(unbiased=False))),
        "kappa": (float(kappa.mean()), float(kappa.std(unbiased=False))),
        "r0": (float(r0.mean()), float(r0.std(unbiased=False))),
        "a": (float(a.mean()), float(a.std(unbiased=False))),
        "lam_so": (float(lam_so.mean()), float(lam_so.std(unbiased=False))),
        "r0_so": (float(r0_so.mean()), float(r0_so.std(unbiased=False))),
    }


# =========================================================
# Potential: six-parameter Seminole / Schwierz Woods-Saxon
# =========================================================

def R_central(A, params: WSParams):
    return params.r0 * (float(A) ** (1.0 / 3.0))


def R_spin_orbit(A, params: WSParams):
    return params.r0_so * (float(A) ** (1.0 / 3.0))


def f_ws(r, R, a):
    x = ((r - R) / a).clamp(-80.0, 80.0)
    return 1.0 / (1.0 + torch.exp(x))


def df_ws_dr(r, R, a):
    x = ((r - R) / a).clamp(-80.0, 80.0)
    ex = torch.exp(x)

    return -(ex / (1.0 + ex) ** 2) * (1.0 / a)


def tdotT_expectation(A: int, Z: int, is_proton: bool):
    """
    Seminole / Schwierz expectation value:

        <t . T_core> = -1/4 * rhs

    with the proton/neutron dependence defined through the
    isospin of the core.
    """
    N = int(A) - int(Z)

    if N == Z:
        rhs = 3.0

    elif N > Z:
        rhs = (
            (N - Z + 1)
            if is_proton
            else -(N - Z + 1)
        ) + 2.0

    else:
        rhs = (
            (N - Z - 1)
            if is_proton
            else -(N - Z - 1)
        ) + 2.0

    return -0.25 * float(rhs)


def V_depth(A, Z, is_proton, params: WSParams):
    """
    Seminole / Schwierz isospin-dependent central depth:

        V_tau = V0 * [
            1 - (4*kappa/A) * <t . T_core>
        ]
    """
    tdotT = tdotT_expectation(
        A,
        Z,
        is_proton,
    )

    return params.V0 * (
        1.0
        - (
            4.0
            * params.kappa
            / float(A)
        )
        * tdotT
    )


def V_central(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    """
    Central Woods-Saxon potential:

        Vcentral(r) = -V_tau * f(r, R, a)
    """
    Vtau = V_depth(
        A,
        Z,
        is_proton,
        params,
    )

    R = R_central(A, params)

    return -Vtau * f_ws(
        r,
        R,
        params.a,
    )


def Vcentral_r2(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    return (r**2) * V_central(
        r,
        A,
        Z,
        is_proton,
        params,
    )


def Vc_r2(
    r,
    A,
    Z,
    is_proton,
    params: WSParams,
):
    """
    r^2-scaled Coulomb potential for a uniformly charged sphere.
    """
    if not is_proton:
        return torch.zeros_like(r)

    Z_core = max(int(Z) - 1, 0)
    Zc = float(Z_core)

    R = R_central(A, params)
    rc = r.clamp_min(0.0)

    inside = rc <= R
    outside = ~inside

    out = torch.empty_like(rc)

    out[inside] = (
        Zc
        * e2
        / (2.0 * R)
        * (
            3.0 * rc[inside] ** 2
            - rc[inside] ** 4 / R**2
        )
    )

    out[outside] = (
        Zc
        * e2
        * rc[outside]
    )

    return out


def Vso_r2(
    r,
    A,
    Z,
    is_proton,
    l,
    j,
    params: WSParams,
):
    """
    r^2-scaled Seminole / Schwierz spin-orbit potential.

    Seminole convention:

        Vtilde_SO = lambda_SO * V0

    The spin-orbit strength does not use V_tau.
    """
    mu = reduced_mass(
        A,
        is_proton,
    )

    rc = r.clamp_min(1e-10)
    Rso = R_spin_orbit(A, params)

    Vtilde = (
        params.lam_so
        * params.V0
    )

    dVt_dr = (
        Vtilde
        * df_ws_dr(
            rc,
            Rso,
            params.a,
        )
    )

    pref_r2 = (
        rc
        * hc**2
        / (2.0 * mu**2)
    )

    return (
        pref_r2
        * dVt_dr
        * float(
            l_dot_s(l, j)
        )
    )


def Veff_r2(
    r,
    A,
    Z,
    is_proton,
    l,
    j,
    params: WSParams,
):
    """
    Full r^2-scaled effective potential:

        r^2 Veff
        =
        r^2 Vcentral
        + r^2 VC
        + r^2 VSO
    """
    return (
        Vcentral_r2(
            r,
            A,
            Z,
            is_proton,
            params,
        )
        + Vc_r2(
            r,
            A,
            Z,
            is_proton,
            params,
        )
        + Vso_r2(
            r,
            A,
            Z,
            is_proton,
            l,
            j,
            params,
        )
    )

# ============================================================
# Neural networks
# ============================================================
class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=128, depth=5):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, output_dim)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class WaveNet3D(nn.Module):
    def __init__(self, n_states, hidden=128, depth=5, beta=0.6):
        super().__init__()
        self.n_states = n_states
        self.beta = beta
        input_dim = 1 + n_states + 3 + 3
        self.R_net = ConditionalMLP(input_dim, 1, hidden, depth)
        self.Th_net = ConditionalMLP(input_dim, 1, hidden, depth)
        self.Ph_net = ConditionalMLP(input_dim, 2, hidden, depth)

    def _context(self, x_coord, E_vec, sample, st, coord_scaler):
        N = x_coord.shape[0]
        x_scaled = coord_scaler(x_coord)
        E_scaled = scale_energy(E_vec).unsqueeze(0).repeat(N, 1)
        nuc = scale_nucleus(
            sample["A"],
            sample["Z"],
            sample["is_proton"],
        ).unsqueeze(0).repeat(N, 1)
        q = scale_quantum(st["nr"], st["l"], st["j"]).unsqueeze(0).repeat(N, 1)
        return torch.cat([x_scaled, E_scaled, nuc, q], dim=1)

    def R(self, r, E_vec, sample, st):
        x = self._context(r, E_vec, sample, st, scale_r)
        raw = self.R_net(x)
        rc = r.clamp_min(1e-8)
        l = int(st["l"])
        return (rc**l) * torch.exp(-self.beta * rc) * raw

    def Theta(self, th, E_vec, sample, st):
        x = self._context(th, E_vec, sample, st, scale_theta)
        return self.Th_net(x)

    def Phi(self, ph, E_vec, sample, st):
        x = self._context(ph, E_vec, sample, st, scale_phi)
        out = self.Ph_net(x)
        return out[:, :1], out[:, 1:]


class GlobalContextParamNet(nn.Module):
    """
    Context-informed global parameter network.

    Input:
        X : shape (N_samples, input_dim)
            one row per nucleus/species sample.

    Output:
        one global q(z), not one q(z|sample).
    """

    def __init__(self, input_dim, hidden=256, depth=3, n_parameters=6):
        super().__init__()
        if n_parameters != 6:
            raise ValueError("Expected six Woods-Saxon parameters.")

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]

        self.encoder = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, n_parameters)
        self.logvar_head = nn.Linear(hidden, n_parameters)

        self.apply(self._init_weights)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -3.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, X):
        h = self.encoder(X)
        h_global = h.mean(dim=0)  # permutation-invariant pooling over nuclei
        mu = self.mu_head(h_global)
        logvar = self.logvar_head(h_global)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma, logvar


# ============================================================
# Autograd and wavefunction helpers
# ============================================================
def d(f, x):
    return torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


def make_grids(Nr=512, Nth=256, Nph=256, requires_grad=True):
    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    th = torch.linspace(1e-5, PI - 1e-5, Nth, device=device).unsqueeze(1)
    ph = torch.linspace(0.0, TWOPI, Nph + 1, device=device)[:-1].unsqueeze(1)
    r.requires_grad_(requires_grad)
    th.requires_grad_(requires_grad)
    ph.requires_grad_(requires_grad)
    return r, th, ph


def eval_R(wave_net, r, E_vec, sample, st):
    return wave_net.R(r, E_vec, sample, st)


def eval_Theta(wave_net, th, E_vec, sample, st):
    return wave_net.Theta(th, E_vec, sample, st)


def eval_Phi(wave_net, ph, E_vec, sample, st):
    return wave_net.Phi(ph, E_vec, sample, st)


def norm_parts(wave_net, E_vec, sample, st, Nr=512, Nth=256, Nph=256, eps=1e-12):
    r, th, ph = make_grids(Nr, Nth, Nph, requires_grad=True)

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)

    r1, th1, ph1 = r.squeeze(), th.squeeze(), ph.squeeze()

    IR = torch.trapz((Rv.squeeze() ** 2) * (r1**2), r1)
    sinth = torch.sin(th1).clamp_min(1e-8)
    Ith = torch.trapz((Thv.squeeze() ** 2) * sinth, th1)
    Phi2 = a_re.squeeze() ** 2 + b_im.squeeze() ** 2
    Iphi = torch.trapz(Phi2, ph1)

    Iang = Ith * Iphi
    I = (IR * Iang).clamp_min(eps)
    s = torch.rsqrt(I)

    return IR, Ith, Iphi, Iang, I, s, r, th, ph


def psi_scale_only(wave_net, E_vec, sample, st, Nr=512, Nth=256, Nph=256):
    IR, Ith, Iphi, Iang, I, s, *_ = norm_parts(
        wave_net,
        E_vec,
        sample,
        st,
        Nr,
        Nth,
        Nph,
    )
    return IR, Ith, Iphi, Iang, I, s


def eval_R_norm(wave_net, r, E_vec, sample, st, Nr=512, Nth=256, Nph=256):
    _, _, _, _, _, s = psi_scale_only(wave_net, E_vec, sample, st, Nr, Nth, Nph)
    return s * eval_R(wave_net, r, E_vec, sample, st), s


def eval_psi_norm(wave_net, r, th, ph, E_vec, sample, st, Nr=512, Nth=256, Nph=256):
    _, _, _, _, _, s = psi_scale_only(wave_net, E_vec, sample, st, Nr, Nth, Nph)
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    return s * Rv * Thv * a_re, s * Rv * Thv * b_im, s


# ============================================================
# ParamNet context from WaveNet radial points
# ============================================================
def make_param_input_from_radial_psi(
    wave_net,
    sample,
    E_vec,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    """
    One sample-level context vector:
      normalized radial values for all selected states
      + target energies
      + nucleus descriptors
      + quantum numbers.
    """
    r = torch.linspace(0.0, R_MAX, n_r_points, device=device).unsqueeze(1)
    radial_parts = []

    for st in sample["states"]:
      IR, _, _, _, _, _ = psi_scale_only(
          wave_net,
          E_vec,
          sample,
          st,
          Nr_norm,
          Nth_norm,
          Nph_norm,
          )
      R_raw = eval_R(
          wave_net,
          r,
          E_vec,
          sample,
          st,
          )
      Rn = (
          R_raw
          / torch.sqrt(IR.clamp_min(1e-12))
      ).squeeze()

      idx = min(5, Rn.numel() - 1)
      sign = torch.sign(Rn[idx].detach())
      if sign.item() == 0.0:
        sign = torch.tensor(1.0, device=device)

      radial_parts.append(sign * Rn)

    psi_vec = torch.cat(radial_parts)
    E_scaled = scale_energy(E_vec)
    nuc = scale_nucleus(sample["A"], sample["Z"], sample["is_proton"])
    q_vec = torch.cat(
        [scale_quantum(st["nr"], st["l"], st["j"]) for st in sample["states"]]
    )

    return torch.cat([psi_vec, E_scaled, nuc, q_vec])


def make_global_param_context(
    wave_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    """
    Matrix of sample-level contexts.

    Shape:
        (N_samples, input_dim)

    ParamNet will pool this matrix and output one global q(z).
    """
    x_list = []

    for sample in samples:
        E_vec = torch.tensor(
            [float(st["energy"]) for st in sample["states"]],
            dtype=torch.float32,
            device=device,
        )
        x_sample = make_param_input_from_radial_psi(
            wave_net=wave_net,
            sample=sample,
            E_vec=E_vec,
            n_r_points=n_r_points,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )
        x_list.append(x_sample)

    return torch.stack(x_list, dim=0)


# ============================================================
# Rayleigh energy and losses
# ============================================================
def energy_rayleigh_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    params,
    Nr=512,
    Nth=256,
    Nph=256,
    eps=1e-12,
):
    A, Z, is_proton = sample["A"], sample["Z"], sample["is_proton"]
    ell, j = int(st["l"]), float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    IR, Ith, Iphi, Iang, _, s, r, th, ph = norm_parts(
        wave_net,
        E_vec,
        sample,
        st,
        Nr,
        Nth,
        Nph,
        eps,
    )

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)

    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)

    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = d(a_re, ph), d(b_im, ph)

    r1, th1, ph1 = r.squeeze(), th.squeeze(), ph.squeeze()

    IRprime = torch.trapz((Rr.squeeze() ** 2) * (r1**2), r1)
    IR0 = torch.trapz(Rv.squeeze() ** 2, r1)

    Veff_r2_vals = Veff_r2(r, A, Z, is_proton, ell, j, params).squeeze()
    IVR = torch.trapz(Veff_r2_vals * (Rv.squeeze() ** 2), r1)

    sinth = torch.sin(th1).clamp_min(1e-8)
    Ith = torch.trapz((Thv.squeeze() ** 2) * sinth, th1)
    Ithprime = torch.trapz((Th_th.squeeze() ** 2) * sinth, th1)
    Ith_over_sin = torch.trapz((Thv.squeeze() ** 2) / sinth, th1)

    Phi2 = a_re.squeeze() ** 2 + b_im.squeeze() ** 2
    Iphi = torch.trapz(Phi2, ph1)
    PhiP2 = ap.squeeze() ** 2 + bp.squeeze() ** 2
    Iphi_prime = torch.trapz(PhiP2, ph1)

    A0 = Ith * Iphi

    T_r = s * s * IRprime * A0
    T_th = s * s * IR0 * Ithprime * Iphi
    T_ph = s * s * IR0 * Ith_over_sin * Iphi_prime
    U = s * s * IVR * A0

    num = K * (T_r + T_th + T_ph) + U
    den = (s * s * IR * Iang).clamp_min(eps)

    return (num / den).squeeze()


def loss_pde_radial_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    params,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    N=256,
):
    A, Z, is_proton = sample["A"], sample["Z"], sample["is_proton"]
    ell, j = int(st["l"]), float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    r = torch.rand(N, 1, device=device) * R_MAX
    r.requires_grad_(True)

    _, _, _, _, _, s = psi_scale_only(
        wave_net,
        E_vec,
        sample,
        st,
        Nr_norm,
        Nth_norm,
        Nph_norm,
    )

    E_det = energy_rayleigh_full3d(
        wave_net,
        E_vec,
        sample,
        st,
        params,
        Nr_norm,
        Nth_norm,
        Nph_norm,
    ).detach()

    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)
    Rrr = d(Rr, r)

    rc = r.clamp_min(1e-8)
    Rn, Rnr, Rnrr = s * Rv, s * Rr, s * Rrr

    Veff_r2_vals = Veff_r2(rc, A, Z, is_proton, ell, j, params)

    residual = (
        -K * ((rc**2) * Rnrr + 2.0 * rc * Rnr - ell * (ell + 1.0) * Rn)
        + (Veff_r2_vals - (rc**2) * E_det) * Rn
    )

    return torch.mean(residual**2)


def loss_pde_theta_full3d(wave_net, E_vec, sample, st, emm=0, N=256):
    ell = int(st["l"])

    th = (torch.rand(N, 1, device=device) * (1.0 - 2e-3) + 1e-3) * PI
    th.requires_grad_(True)

    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)
    Th_thth = d(Th_th, th)

    sinth, costh = torch.sin(th), torch.cos(th)
    sin2 = sinth**2

    op = (
        sin2 * Th_thth
        + sinth * costh * Th_th
        + (ell * (ell + 1.0) * sin2 - float(emm**2)) * Thv
    )

    return torch.mean(op**2)


def loss_pde_phi_full3d(wave_net, E_vec, sample, st, emm=0, N=256, Np_bc=64):
    ph = torch.rand(N, 1, device=device) * TWOPI
    ph.requires_grad_(True)

    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = d(a_re, ph), d(b_im, ph)
    app, bpp = d(ap, ph), d(bp, ph)

    L_ode = ((app + float(emm**2) * a_re) ** 2 + (bpp + float(emm**2) * b_im) ** 2).mean()

    z0 = torch.zeros(Np_bc, 1, device=device, requires_grad=True)
    z2 = torch.full_like(z0, TWOPI, requires_grad=True)

    a0, b0 = eval_Phi(wave_net, z0, E_vec, sample, st)
    a2, b2 = eval_Phi(wave_net, z2, E_vec, sample, st)

    ap0, bp0 = d(a0, z0), d(b0, z0)
    ap2, bp2 = d(a2, z2), d(b2, z2)

    L_per = ((a0 - a2) ** 2 + (b0 - b2) ** 2 + (ap0 - ap2) ** 2 + (bp0 - bp2) ** 2).mean()
    L_gauge = ((a0 - 1.0) ** 2 + b0**2 + ap0**2 + (bp0 - float(emm)) ** 2).mean()

    return L_ode + L_per + 2.0 * L_gauge


def loss_bc_full3d(
    wave_net,
    E_vec,
    sample,
    st,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    Nb=64,
):
    ell = int(st["l"])

    rR = torch.full((Nb, 1), R_MAX, device=device, requires_grad=True)
    RnR, _ = eval_R_norm(wave_net, rR, E_vec, sample, st, Nr_norm, Nth_norm, Nph_norm)
    L_Rmax = torch.mean(RnR**2)

    r0 = torch.zeros((Nb, 1), device=device, requires_grad=True)
    Rn0, _ = eval_R_norm(wave_net, r0, E_vec, sample, st, Nr_norm, Nth_norm, Nph_norm)

    if ell == 0:
        dRn0 = d(Rn0, r0)
        L_0 = torch.mean(dRn0**2)
    elif ell == 1:
        L_0 = torch.mean(Rn0**2)
    else:
        dRn0 = d(Rn0, r0)
        L_0 = torch.mean(Rn0**2) + torch.mean(dRn0**2)

    return L_Rmax + L_0


def loss_orthogonality_full3d(
    wave_net,
    sample,
    E_vec,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    N=512,
):
    """
    Selective orthogonality only for same (l,j) and different nr.
    Spin-orbit partners are not penalized here.
    """
    states = sample["states"]
    K_states = len(states)

    if K_states < 2:
        return torch.tensor(0.0, device=device)

    r = torch.rand(N, 1, device=device) * R_MAX
    th = torch.rand(N, 1, device=device) * PI
    ph = torch.rand(N, 1, device=device) * TWOPI

    weight = (r.squeeze() ** 2) * torch.sin(th.squeeze()).clamp_min(1e-12)
    volume = R_MAX * PI * TWOPI

    psis = []
    for st in states:
        Re, Im, _ = eval_psi_norm(
            wave_net,
            r,
            th,
            ph,
            E_vec,
            sample,
            st,
            Nr=Nr_norm,
            Nth=Nth_norm,
            Nph=Nph_norm,
        )
        psis.append((Re.squeeze(), Im.squeeze()))

    total = torch.tensor(0.0, device=device)
    n_pairs = 0

    for i in range(K_states):
        st_i = states[i]
        nr_i = int(st_i["nr"])
        l_i = int(st_i["l"])
        j_i = float(st_i["j"])
        Re_i, Im_i = psis[i]

        for j_idx in range(i):
            st_j = states[j_idx]
            nr_j = int(st_j["nr"])
            l_j = int(st_j["l"])
            j_j = float(st_j["j"])

            same_lj_different_nr = (
                l_i == l_j
                and abs(j_i - j_j) < 1e-6
                and nr_i != nr_j
            )

            if not same_lj_different_nr:
                continue

            Re_j, Im_j = psis[j_idx]

            overlap_re_density = Re_j * Re_i + Im_j * Im_i
            overlap_im_density = Re_j * Im_i - Im_j * Re_i

            overlap_re = volume * torch.mean(overlap_re_density * weight)
            overlap_im = volume * torch.mean(overlap_im_density * weight)

            total = total + overlap_re**2 + overlap_im**2
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    return total / n_pairs


def loss_spin_orbit_splitting(pred_energy_map, target_energy_map, states):
    """
    Explicit splitting loss for pairs:
        (nr, l, j=l-1/2) and (nr, l, j=l+1/2).

    This helps constrain lambda_SO and r0_SO.
    """
    LSO = torch.tensor(0.0, device=device)
    n_pairs = 0
    seen = set()

    for st in states:
        nr = int(st["nr"])
        l = int(st["l"])

        if l == 0:
            continue

        jm = round(float(l) - 0.5, 6)
        jp = round(float(l) + 0.5, 6)

        key_m = (nr, l, jm)
        key_p = (nr, l, jp)
        pair_id = (nr, l)

        if pair_id in seen:
            continue

        if key_m in pred_energy_map and key_p in pred_energy_map:
            pred_split = pred_energy_map[key_p] - pred_energy_map[key_m]
            target_split = target_energy_map[key_p] - target_energy_map[key_m]
            LSO = LSO + (pred_split - target_split) ** 2
            n_pairs += 1
            seen.add(pair_id)

    if n_pairs == 0:
        return torch.tensor(0.0, device=device)

    return LSO / n_pairs


# ============================================================
# KL prior
# ============================================================
def gaussian_kl_to_isotropic_prior(mu, logvar, prior_std=PRIOR_STD):
    prior_std_tensor = torch.as_tensor(
        prior_std,
        dtype=mu.dtype,
        device=mu.device,
    )

    if prior_std_tensor.item() <= 0.0:
        raise ValueError("prior_std must be strictly positive.")

    return 0.5 * torch.sum(
        (logvar.exp() + mu.pow(2)) / prior_std_tensor.pow(2)
        - 1.0
        + 2.0 * torch.log(prior_std_tensor)
        - logvar
    )


# ============================================================
# Losses using one global parameter vector
# ============================================================
def compute_sample_physics_loss_full3d(
    wave_net,
    sample,
    params,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    emm=0,
    wE=1.0,
    wR=10.0,
    wTh=5.0,
    wPh=5.0,
    wBC=5.0,
    wORTH=0.0,
    wSO=0.0,
    orth_points=512,
):
    states = sample["states"]
    K_states = len(states)

    E_vec = torch.tensor(
        [float(st["energy"]) for st in states],
        dtype=torch.float32,
        device=device,
    )

    LE = torch.tensor(0.0, device=device)
    LR = torch.tensor(0.0, device=device)
    LTH = torch.tensor(0.0, device=device)
    LPH = torch.tensor(0.0, device=device)
    LBC = torch.tensor(0.0, device=device)

    rows = []
    pred_energy_map = {}
    target_energy_map = {}

    for i, st in enumerate(states):
        E_target = E_vec[i]

        E_pred = energy_rayleigh_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            params,
            Nr_norm,
            Nth_norm,
            Nph_norm,
        )

        LE = LE + (E_pred - E_target) ** 2

        LR = LR + loss_pde_radial_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            params,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            N=256,
        )

        LTH = LTH + loss_pde_theta_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            emm=emm,
            N=256,
        )

        LPH = LPH + loss_pde_phi_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            emm=emm,
            N=256,
            Np_bc=64,
        )

        LBC = LBC + loss_bc_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            Nb=64,
        )

        key = (int(st["nr"]), int(st["l"]), round(float(st["j"]), 6))
        pred_energy_map[key] = E_pred
        target_energy_map[key] = E_target

        rows.append(
            (
                st,
                float(E_pred.detach().cpu()),
                float(E_target.detach().cpu()),
            )
        )

    LE = LE / K_states
    LR = LR / K_states
    LTH = LTH / K_states
    LPH = LPH / K_states
    LBC = LBC / K_states

    if wORTH > 0.0:
        LORTH = loss_orthogonality_full3d(
            wave_net,
            sample,
            E_vec,
            Nr_norm,
            Nth_norm,
            Nph_norm,
            N=orth_points,
        )
    else:
        LORTH = torch.tensor(0.0, device=device)

    if wSO > 0.0:
        LSO = loss_spin_orbit_splitting(pred_energy_map, target_energy_map, states)
    else:
        LSO = torch.tensor(0.0, device=device)

    physics_loss = (
        wE * LE
        + wR * LR
        + wTh * LTH
        + wPh * LPH
        + wBC * LBC
        + wORTH * LORTH
        + wSO * LSO
    )

    return physics_loss, LE, LR, LTH, LPH, LBC, LORTH, LSO, rows


def compute_multinucleus_loss_full3d(
    wave_net,
    param_net,
    samples,
    batch_size=None,
    prior_std=PRIOR_STD,
    wKL=0.0,
    is_training=True,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    **loss_kwargs,
):
    """
    Build global context from all samples, infer one global q(z),
    then use the same physical parameters for every sample in the loss batch.
    """
    X_global = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    mu, sigma, logvar = param_net(X_global)

    raw_params = mu + sigma * torch.randn_like(sigma) if is_training else mu
    params = params_from_raw_six(raw_params)

    L_KL = gaussian_kl_to_isotropic_prior(mu, logvar, prior_std=prior_std)

    if batch_size is None or batch_size >= len(samples):
        batch = samples
    else:
        idx = np.random.choice(len(samples), size=batch_size, replace=False)
        batch = [samples[i] for i in idx]

    total_physics_loss = torch.tensor(0.0, device=device)
    metrics = {k: 0.0 for k in ["LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LSO"]}
    per_sample_rows = []

    for sample in batch:
        (
            sample_loss,
            LE,
            LR,
            LTH,
            LPH,
            LBC,
            LORTH,
            LSO,
            rows,
        ) = compute_sample_physics_loss_full3d(
            wave_net=wave_net,
            sample=sample,
            params=params,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            **loss_kwargs,
        )

        total_physics_loss = total_physics_loss + sample_loss

        values = {
            "LE": LE,
            "LR": LR,
            "LTH": LTH,
            "LPH": LPH,
            "LBC": LBC,
            "LORTH": LORTH,
            "LSO": LSO,
        }
        for key, value in values.items():
            metrics[key] += float(value.detach().cpu())

        per_sample_rows.append(
            {
                "sample": sample,
                "rows": rows,
                "loss": float(sample_loss.detach().cpu()),
            }
        )

    B = len(batch)
    total_physics_loss = total_physics_loss / B
    for key in metrics:
        metrics[key] /= B

    total_loss = total_physics_loss + wKL * L_KL
    metrics["LKL"] = float(L_KL.detach().cpu())

    return total_loss, metrics, per_sample_rows


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def infer_global_parameters(
    wave_net,
    param_net,
    samples,
    n_samples=5000,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    wave_net.eval()
    param_net.eval()

    X_global = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    mu, sigma, _ = param_net(X_global)
    return physical_stats_from_mu_sigma(mu, sigma, n_samples=n_samples)


@torch.no_grad()
def infer_global_raw_latent(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    wave_net.eval()
    param_net.eval()

    X_global = make_global_param_context(
        wave_net,
        samples,
        n_r_points,
        Nr_norm,
        Nth_norm,
        Nph_norm,
    )
    mu, sigma, logvar = param_net(X_global)
    return mu.detach().cpu().tolist(), sigma.detach().cpu().tolist(), logvar.detach().cpu().tolist()


# ============================================================
# Diagnostics
# ============================================================
def print_spin_orbit_pairs(samples):
    for sample in samples:
        A = sample["A"]
        Z = sample["Z"]
        species = "p" if sample["is_proton"] else "n"
        states = sample["states"]

        state_map = {
            (int(st["nr"]), int(st["l"]), round(float(st["j"]), 6)): st
            for st in states
        }

        print(f"\nA={A}, Z={Z}, species={species}")
        n_pairs = 0
        seen = set()

        for st in states:
            nr = int(st["nr"])
            l = int(st["l"])
            if l == 0:
                continue

            jm = round(float(l) - 0.5, 6)
            jp = round(float(l) + 0.5, 6)
            key_m = (nr, l, jm)
            key_p = (nr, l, jp)
            pair_id = (nr, l)

            if pair_id in seen:
                continue

            if key_m in state_map and key_p in state_map:
                Em = float(state_map[key_m]["energy"])
                Ep = float(state_map[key_p]["energy"])
                print(
                    f"  pair nr={nr}, l={l}: "
                    f"j-={jm}, E-={Em:.4f} | "
                    f"j+={jp}, E+={Ep:.4f} | "
                    f"split={Ep - Em:.4f}"
                )
                n_pairs += 1
                seen.add(pair_id)

        if n_pairs == 0:
            print("  No spin-orbit pairs selected.")


# ============================================================
# Plotting
# ============================================================
def plot_history(history, out_dir):
    if not PLOTTING_AVAILABLE or len(history) == 0:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ["loss", "LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LSO", "LKL"]:
        vals = [max(float(h[key]), 1e-30) for h in history]
        ax.plot(epochs, vals, label=key)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out_dir / "global_context_loss_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for p in ["V0", "kappa", "r0", "a", "lam_so", "r0_so"]:
        mu_key = f"{p}_mu"
        sig_key = f"{p}_sigma"
        if mu_key not in history[0]:
            continue

        mu = np.array([h[mu_key] for h in history], dtype=float)
        sig = np.array([h[sig_key] for h in history], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, mu, label=f"{p} mean")
        ax.fill_between(epochs, mu - sig, mu + sig, alpha=0.25, label="±1σ")

        if p in WAHLBORN_REFERENCE:
            ax.axhline(float(WAHLBORN_REFERENCE[p]), linestyle="--", linewidth=1.0, label="reference")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(p)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.savefig(out_dir / f"global_context_{p}_history.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Training
# ============================================================
def train_multinucleus_full3d(
    dataset_path="ws_fd_dataset_2.npz",
    cases=None,
    max_states=8,
    epochs=15000,
    batch_size=None,
    lr_wave=5e-4,
    lr_param=1e-3,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=128,
    hidden_wave=256,
    hidden_param=256,
    emm=0,
    wE=10.0,
    wR=10.0,
    wTh=2.0,
    wPh=2.0,
    wBC=5.0,
    wORTH=0.0,
    wSO=10.0,
    wKL=0.0,
    prior_std=PRIOR_STD,
    orth_points=512,
    use_scheduler_wave=True,
    use_scheduler_param=True,
    gamma_wave=0.6,
    gamma_param=0.5,
    step_size_wave=1000,
    step_size_param=1500,
    print_every=100,
    log_every=100,
    out_dir="wahlborn_global_context_outputs",
    use_wandb=False,
    wandb_project="inverse-ws-pinn",
    wandb_run_name="wahlborn_global_context",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = build_multinucleus_samples(
        dataset_path=dataset_path,
        cases=cases,
        max_states=max_states,
        require_exact_states=True,
    )

    if len(samples) == 0:
        raise RuntimeError("No valid samples found. Check dataset path, cases, and max_states.")

    K_states = max_states
    param_input_dim = K_states * n_r_points + K_states + 3 + K_states * 3

    wave_net = WaveNet3D(
        n_states=K_states,
        hidden=hidden_wave,
        depth=5,
        beta=0.6,
    ).to(device)

    param_net = GlobalContextParamNet(
        input_dim=param_input_dim,
        hidden=hidden_param,
        depth=3,
        n_parameters=6,
    ).to(device)

    optimizer_wave = torch.optim.Adam(wave_net.parameters(), lr=lr_wave)
    optimizer_param = torch.optim.Adam(param_net.parameters(), lr=lr_param)

    scheduler_wave = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_wave,
            step_size=step_size_wave,
            gamma=gamma_wave,
        )
        if use_scheduler_wave
        else None
    )

    scheduler_param = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_param,
            step_size=step_size_param,
            gamma=gamma_param,
        )
        if use_scheduler_param
        else None
    )

    config = {
        "dataset_path": dataset_path,
        "n_samples": len(samples),
        "skipped": skipped,
        "max_states": max_states,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_wave": lr_wave,
        "lr_param": lr_param,
        "n_r_points": n_r_points,
        "Nr_norm": Nr_norm,
        "Nth_norm": Nth_norm,
        "Nph_norm": Nph_norm,
        "hidden_wave": hidden_wave,
        "hidden_param": hidden_param,
        "emm": emm,
        "wE": wE,
        "wR": wR,
        "wTh": wTh,
        "wPh": wPh,
        "wBC": wBC,
        "wORTH": wORTH,
        "wSO": wSO,
        "wKL": wKL,
        "prior_std": prior_std,
        "orth_points": orth_points,
        "param_net_type": "global_context_mean_pooling",
        "param_input_dim": param_input_dim,
        "potential": "wahlborn_like_central_and_spin_orbit",
        "reference": WAHLBORN_REFERENCE,
        "use_scheduler_wave": use_scheduler_wave,
        "use_scheduler_param": use_scheduler_param,
        "gamma_wave": gamma_wave,
        "gamma_param": gamma_param,
        "step_size_wave": step_size_wave,
        "step_size_param": step_size_param,
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=config,
            reinit=True,
        )

    print("\n====================================================")
    print("GLOBAL-CONTEXT SIX-PARAMETER WAHLBORN PINN TRAINING")
    print("====================================================")
    print(f"Device: {device}")
    print(f"Samples used: {len(samples)}")

    if skipped:
        print(f"Skipped cases: {skipped}")

    print("Cases:")
    for s in samples:
        print(
            f"  A={s['A']:3d} Z={s['Z']:3d} "
            f"species={'p' if s['is_proton'] else 'n'} "
            f"states={len(s['states'])}"
        )

    print("\nSpin-orbit pairs selected:")
    print_spin_orbit_pairs(samples)

    history = []
    t0 = time.time()

    loss_kwargs = dict(
        emm=emm,
        wE=wE,
        wR=wR,
        wTh=wTh,
        wPh=wPh,
        wBC=wBC,
        wORTH=wORTH,
        wSO=wSO,
        orth_points=orth_points,
    )

    for ep in range(1, epochs + 1):
        wave_net.train()
        param_net.train()

        optimizer_wave.zero_grad()
        optimizer_param.zero_grad()

        loss, metrics, per_sample_rows = compute_multinucleus_loss_full3d(
            wave_net=wave_net,
            param_net=param_net,
            samples=samples,
            batch_size=batch_size,
            prior_std=prior_std,
            wKL=wKL,
            is_training=True,
            n_r_points=n_r_points,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            **loss_kwargs,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(wave_net.parameters()) + list(param_net.parameters()),
            5.0,
        )

        optimizer_wave.step()
        optimizer_param.step()

        if scheduler_wave is not None:
            scheduler_wave.step()

        if scheduler_param is not None:
            scheduler_param.step()

        should_log = ep == 1 or ep % log_every == 0 or ep == epochs
        should_print = ep == 1 or ep % print_every == 0 or ep == epochs

        if should_log:
            pred_monitor = infer_global_parameters(
                wave_net=wave_net,
                param_net=param_net,
                samples=samples,
                n_samples=1000,
                n_r_points=n_r_points,
                Nr_norm=Nr_norm,
                Nth_norm=Nth_norm,
                Nph_norm=Nph_norm,
            )

            row = {
                "epoch": ep,
                "time_sec": time.time() - t0,
                "loss": float(loss.detach().cpu()),
                **metrics,
                "lr_wave": optimizer_wave.param_groups[0]["lr"],
                "lr_param": optimizer_param.param_groups[0]["lr"],
            }

            for p in ["V0", "kappa", "r0", "a", "lam_so", "r0_so"]:
                row[f"{p}_mu"] = pred_monitor[p][0]
                row[f"{p}_sigma"] = pred_monitor[p][1]

            history.append(row)

            if use_wandb and WANDB_AVAILABLE:
                wandb.log(
                    {
                        "epoch": ep,
                        "loss/total": row["loss"],
                        "loss/LE": row["LE"],
                        "loss/LR": row["LR"],
                        "loss/LTH": row["LTH"],
                        "loss/LPH": row["LPH"],
                        "loss/LBC": row["LBC"],
                        "loss/LORTH": row["LORTH"],
                        "loss/LSO": row["LSO"],
                        "loss/LKL": row["LKL"],
                        "lr/wave": row["lr_wave"],
                        "lr/param": row["lr_param"],
                        "params_global/V0_mu": row["V0_mu"],
                        "params_global/V0_sigma": row["V0_sigma"],
                        "params_global/kappa_mu": row["kappa_mu"],
                        "params_global/kappa_sigma": row["kappa_sigma"],
                        "params_global/r0_mu": row["r0_mu"],
                        "params_global/r0_sigma": row["r0_sigma"],
                        "params_global/a_mu": row["a_mu"],
                        "params_global/a_sigma": row["a_sigma"],
                        "params_global/lam_so_mu": row["lam_so_mu"],
                        "params_global/lam_so_sigma": row["lam_so_sigma"],
                        "params_global/r0_so_mu": row["r0_so_mu"],
                        "params_global/r0_so_sigma": row["r0_so_sigma"],
                    },
                    step=ep,
                )

        if should_print:
            print(
                f"[ep={ep:6d}] "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"LE={metrics['LE']:.3e} "
                f"LR={metrics['LR']:.3e} "
                f"LTH={metrics['LTH']:.3e} "
                f"LPH={metrics['LPH']:.3e} "
                f"LBC={metrics['LBC']:.3e} "
                f"LORTH={metrics['LORTH']:.3e} "
                f"LSO={metrics['LSO']:.3e} "
                f"LKL={metrics['LKL']:.3e} "
                f"lr_wave={optimizer_wave.param_groups[0]['lr']:.2e} "
                f"lr_param={optimizer_param.param_groups[0]['lr']:.2e}"
            )

            if len(history) > 0:
                h = history[-1]
                print(
                    f" global params: "
                    f"V0={h['V0_mu']:.3f}±{h['V0_sigma']:.3f}, "
                    f"kappa={h['kappa_mu']:.3f}±{h['kappa_sigma']:.3f}, "
                    f"r0={h['r0_mu']:.3f}±{h['r0_sigma']:.3f}, "
                    f"a={h['a_mu']:.3f}±{h['a_sigma']:.3f}, "
                    f"lambda={h['lam_so_mu']:.3f}±{h['lam_so_sigma']:.3f}, "
                    f"r0_so={h['r0_so_mu']:.3f}±{h['r0_so_sigma']:.3f}"
                )

    if history:
        keys = list(history[0].keys())
        with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)

    final_global_prediction = infer_global_parameters(
        wave_net=wave_net,
        param_net=param_net,
        samples=samples,
        n_samples=10000,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    final_predictions = {
        parameter: {
            "mean": float(final_global_prediction[parameter][0]),
            "sigma": float(final_global_prediction[parameter][1]),
            "reference": float(WAHLBORN_REFERENCE.get(parameter, float("nan"))),
        }
        for parameter in ["V0", "kappa", "r0", "a", "lam_so", "r0_so"]
    }

    mu_raw, sigma_raw, logvar_raw = infer_global_raw_latent(
        wave_net=wave_net,
        param_net=param_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    with open(out_dir / "final_global_parameter_prediction.json", "w", encoding="utf-8") as f:
        json.dump(final_predictions, f, indent=2)

    with open(out_dir / "final_global_latent_prediction.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mu_raw": mu_raw,
                "sigma_raw": sigma_raw,
                "logvar_raw": logvar_raw,
            },
            f,
            indent=2,
        )

    print("\n====================================================")
    print("FINAL GLOBAL WAHLBORN-LIKE PARAMETERIZATION")
    print("====================================================")
    print(
        f"V0={final_predictions['V0']['mean']:.4f}±{final_predictions['V0']['sigma']:.4f}, "
        f"kappa={final_predictions['kappa']['mean']:.4f}±{final_predictions['kappa']['sigma']:.4f}, "
        f"r0={final_predictions['r0']['mean']:.4f}±{final_predictions['r0']['sigma']:.4f}, "
        f"a={final_predictions['a']['mean']:.4f}±{final_predictions['a']['sigma']:.4f}, "
        f"lambda={final_predictions['lam_so']['mean']:.4f}±{final_predictions['lam_so']['sigma']:.4f}, "
        f"r0_so={final_predictions['r0_so']['mean']:.4f}±{final_predictions['r0_so']['sigma']:.4f}"
    )

    torch.save(wave_net.state_dict(), out_dir / "wave_net_global_context.pt")
    torch.save(param_net.state_dict(), out_dir / "global_context_param_net.pt")

    plot_history(history, out_dir)

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return wave_net, param_net, history, samples, final_predictions


# ============================================================
# Standalone diagnostics configuration
# ============================================================
# Prefer the Seminole/Schwierz reference when present. The reference is used
# only in plots and error logging; it does not alter the loss or KL prior.
REFERENCE = globals().get(
    "SCHWIERZ_REFERENCE",
    globals().get("WAHLBORN_REFERENCE", None),
)

MODULE_NAME = "standalone_colab_script"

PARAM_NAMES = ["V0", "kappa", "r0", "a", "lam_so", "r0_so"]

# ============================================================
# Generic utilities
# ============================================================
PARAM_LABELS = {
    "V0": r"$V_0$",
    "kappa": r"$\kappa$",
    "r0": r"$r_0$",
    "a": r"$a$",
    "lam_so": r"$\lambda_{\mathrm{SO}}$",
    "r0_so": r"$r_{0,\mathrm{SO}}$",
}

PARAM_UNITS = {
    "V0": "MeV",
    "kappa": "",
    "r0": "fm",
    "a": "fm",
    "lam_so": "",
    "r0_so": "fm",
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def wb_log(payload, step=None):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(payload, step=step)


def species_tag(is_proton):
    return "p" if bool(is_proton) else "n"


def nucleus_tag(sample):
    return (
        f"A{int(sample['A'])}_Z{int(sample['Z'])}_"
        f"{species_tag(sample['is_proton'])}"
    )


def state_label(st):
    return (
        f"nr={int(st['nr'])}, "
        f"l={int(st['l'])}, "
        f"j={float(st['j']):.1f}"
    )


def state_short_label(st):
    return f"{int(st['nr'])},{int(st['l'])},{float(st['j']):.1f}"


def save_history_csv(history, path):
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# Global parameter inference
# ============================================================
@torch.no_grad()
def infer_global_stats(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    n_mc_samples=3000,
):
    return infer_global_parameters(
        wave_net=wave_net,
        param_net=param_net,
        samples=samples,
        n_samples=n_mc_samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )


@torch.no_grad()
def global_mean_params(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    """
    Deterministic physical parameter set obtained by mapping latent posterior
    mean mu through the bounded physical map.
    """
    wave_net.eval()
    param_net.eval()

    X_global = make_global_param_context(
        wave_net=wave_net,
        samples=samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )
    mu, _, _ = param_net(X_global)
    return params_from_raw_six(mu)


def wsparams_to_dict(params):
    return {
        "V0": float(params.V0.detach().cpu()),
        "kappa": float(params.kappa.detach().cpu()),
        "r0": float(params.r0.detach().cpu()),
        "a": float(params.a.detach().cpu()),
        "lam_so": float(params.lam_so.detach().cpu()),
        "r0_so": float(params.r0_so.detach().cpu()),
    }


# ============================================================
# Training-history plots
# ============================================================
def plot_loss_history(history, out_dir, log_wandb=True):
    if not history:
        return None

    out_dir = ensure_dir(out_dir)
    df = pd.DataFrame(history)

    available = [
        c for c in
        ["loss", "LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LSO", "LKL"]
        if c in df.columns
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for col in available:
        y = np.maximum(df[col].to_numpy(dtype=float), 1e-30)
        ax.plot(df["epoch"], y, linewidth=1.6, label=col)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Global multi-nucleus training losses")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    path = save_fig(fig, out_dir / "global_loss_components.png")

    if log_wandb:
        wb_log({"plots/training/loss_components": wandb.Image(path)})

    return path


def plot_learning_rates(history, out_dir, log_wandb=True):
    if not history:
        return None

    df = pd.DataFrame(history)
    if "lr_wave" not in df or "lr_param" not in df:
        return None

    out_dir = ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(df["epoch"], df["lr_wave"], label="WaveNet learning rate")
    ax.plot(df["epoch"], df["lr_param"], label="ParamNet learning rate")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedules")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path = save_fig(fig, out_dir / "learning_rates.png")

    if log_wandb:
        wb_log({"plots/training/learning_rates": wandb.Image(path)})

    return path


def plot_parameter_history(history, out_dir, reference=None, log_wandb=True):
    if not history:
        return []

    out_dir = ensure_dir(out_dir)
    df = pd.DataFrame(history)
    paths = []

    for p in PARAM_NAMES:
        mu_col = f"{p}_mu"
        sigma_col = f"{p}_sigma"
        if mu_col not in df:
            continue

        x = df["epoch"].to_numpy()
        mu = df[mu_col].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        ax.plot(x, mu, linewidth=1.9, label="Posterior mean")

        if sigma_col in df:
            sigma = df[sigma_col].to_numpy(dtype=float)
            ax.fill_between(
                x,
                mu - sigma,
                mu + sigma,
                alpha=0.25,
                label=r"$\pm1\sigma$",
            )

        if reference is not None and p in reference:
            ax.axhline(
                float(reference[p]),
                linestyle="--",
                linewidth=1.3,
                label="Reference",
            )

        unit = PARAM_UNITS[p]
        ylabel = PARAM_LABELS[p] + (f" ({unit})" if unit else "")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Global {p} evolution")
        ax.grid(True, alpha=0.3)
        ax.legend()

        path = save_fig(fig, out_dir / f"global_{p}_history.png")
        paths.append(path)

        if log_wandb:
            wb_log({f"plots/parameters/{p}_history": wandb.Image(path)})

    return paths


def plot_parameter_summary(
    stats,
    out_dir,
    reference=None,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)

    rows = []
    for p in PARAM_NAMES:
        mean, sigma = stats[p]
        rows.append({
            "parameter": p,
            "mean": float(mean),
            "sigma": float(sigma),
            "reference": (
                float(reference[p])
                if reference is not None and p in reference
                else np.nan
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "final_global_parameter_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.7))
    axes = axes.ravel()

    for ax, row in zip(axes, rows):
        p = row["parameter"]
        mean = row["mean"]
        sigma = row["sigma"]

        ax.errorbar(
            [0],
            [mean],
            yerr=[sigma],
            fmt="o",
            capsize=5,
            label="PINN",
        )

        if np.isfinite(row["reference"]):
            ax.axhline(
                row["reference"],
                linestyle="--",
                linewidth=1.3,
                label="Reference",
            )

        ax.set_xticks([])
        unit = PARAM_UNITS[p]
        ax.set_ylabel(PARAM_LABELS[p] + (f" ({unit})" if unit else ""))
        ax.set_title(p)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Final global parameter posterior", y=1.01)
    fig.tight_layout()
    path = save_fig(fig, out_dir / "final_global_parameter_summary.png")

    if log_wandb:
        wb_log({
            "plots/parameters/final_summary": wandb.Image(path),
            "tables/final_global_parameters": wandb.Table(dataframe=df),
        })

    return {"path": path, "table": df}


# ============================================================
# Energy diagnostics
# ============================================================
def compute_all_energy_tables(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    """
    Evaluate Rayleigh energies for every selected state using the single
    deterministic global parameter set obtained from latent posterior mean.
    """
    params = global_mean_params(
        wave_net,
        param_net,
        samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    wave_net.eval()
    param_net.eval()

    rows = []
    for sample in samples:
        E_vec = torch.tensor(
            [float(st["energy"]) for st in sample["states"]],
            dtype=torch.float32,
            device=device,
        )

        for st in sample["states"]:
            E_pred = energy_rayleigh_full3d(
                wave_net,
                E_vec,
                sample,
                st,
                params,
                Nr=Nr_norm,
                Nth=Nth_norm,
                Nph=Nph_norm,
            )

            E_target = float(st["energy"])
            E_pred_float = float(E_pred.detach().cpu())

            rows.append({
                "A": int(sample["A"]),
                "Z": int(sample["Z"]),
                "species": species_tag(sample["is_proton"]),
                "is_proton": bool(sample["is_proton"]),
                "nr": int(st["nr"]),
                "l": int(st["l"]),
                "j": float(st["j"]),
                "E_target": E_target,
                "E_pred": E_pred_float,
                "residual": E_pred_float - E_target,
                "abs_error": abs(E_pred_float - E_target),
            })

    return pd.DataFrame(rows), wsparams_to_dict(params)


def plot_global_energy_diagnostics(
    wave_net,
    param_net,
    samples,
    out_dir,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)

    df, mean_params = compute_all_energy_tables(
        wave_net,
        param_net,
        samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    df.to_csv(out_dir / "all_nuclei_energy_table.csv", index=False)

    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(np.mean(df["residual"].to_numpy() ** 2)))
    bias = float(df["residual"].mean())

    # Predicted against target
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for species, group in df.groupby("species"):
        ax.scatter(
            group["E_target"],
            group["E_pred"],
            s=38,
            alpha=0.8,
            label=species,
        )

    lo = min(float(df["E_target"].min()), float(df["E_pred"].min()))
    hi = max(float(df["E_target"].max()), float(df["E_pred"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3)
    ax.set_xlabel(r"Target energy $E_{\mathrm{target}}$ (MeV)")
    ax.set_ylabel(r"PINN Rayleigh energy $E_{\mathrm{PINN}}$ (MeV)")
    ax.set_title("All selected states: predicted vs target")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Species")
    scatter_path = save_fig(fig, out_dir / "energy_predicted_vs_target.png")

    # Residual distributions by nucleus/species
    group_df = (
        df.groupby(["A", "Z", "species"], as_index=False)
        .agg(
            MAE=("abs_error", "mean"),
            RMSE=("residual", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            Bias=("residual", "mean"),
            N_states=("residual", "size"),
        )
    )
    group_df["case"] = group_df.apply(
        lambda r: f"A{int(r.A)} Z{int(r.Z)} {r.species}",
        axis=1,
    )
    group_df.to_csv(out_dir / "energy_metrics_by_nucleus.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, max(4.5, 0.38 * len(group_df))))
    y = np.arange(len(group_df))
    ax.barh(y, group_df["MAE"])
    ax.set_yticks(y)
    ax.set_yticklabels(group_df["case"])
    ax.set_xlabel("MAE (MeV)")
    ax.set_title("Energy reconstruction MAE by nucleus/species")
    ax.grid(True, axis="x", alpha=0.3)
    mae_path = save_fig(fig, out_dir / "energy_mae_by_nucleus.png")

    # State residuals
    labels = [
        f"A{int(r.A)} {r.species} ({int(r.nr)},{int(r.l)},{r.j:.1f})"
        for _, r in df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.22 * len(df)), 4.8))
    ax.bar(np.arange(len(df)), df["residual"])
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel(r"$E_{\mathrm{PINN}}-E_{\mathrm{target}}$ (MeV)")
    ax.set_title("Statewise energy residuals")
    ax.grid(True, axis="y", alpha=0.3)
    residual_path = save_fig(fig, out_dir / "statewise_energy_residuals.png")

    if log_wandb:
        wb_log({
            "final/energy_MAE": mae,
            "final/energy_RMSE": rmse,
            "final/energy_bias": bias,
            "plots/energy/predicted_vs_target": wandb.Image(scatter_path),
            "plots/energy/mae_by_nucleus": wandb.Image(mae_path),
            "plots/energy/statewise_residuals": wandb.Image(residual_path),
            "tables/all_nuclei_energies": wandb.Table(dataframe=df),
            "tables/energy_metrics_by_nucleus": wandb.Table(dataframe=group_df),
        })

    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "energy_table": df,
        "group_table": group_df,
        "mean_params": mean_params,
        "paths": [scatter_path, mae_path, residual_path],
    }


# ============================================================
# Wavefunction diagnostics
# ============================================================
@torch.no_grad()
def collect_radial_probability_data(
    wave_net,
    sample,
    n_r=800,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    wave_net.eval()

    E_vec = torch.tensor(
        [float(st["energy"]) for st in sample["states"]],
        dtype=torch.float32,
        device=device,
    )
    r = torch.linspace(0.0, R_MAX, n_r, device=device).unsqueeze(1)
    r1 = r.squeeze()

    records = []
    for st in sample["states"]:
        _, Ith, Iphi, _, _, s = psi_scale_only(
            wave_net,
            E_vec,
            sample,
            st,
            Nr=Nr_norm,
            Nth=Nth_norm,
            Nph=Nph_norm,
        )

        Rn = (s * eval_R(wave_net, r, E_vec, sample, st)).squeeze()
        P_r = Rn.pow(2) * r1.pow(2) * Ith * Iphi

        records.append({
            "state": dict(st),
            "label": state_label(st),
            "r": r1.detach().cpu().numpy(),
            "R_norm": Rn.detach().cpu().numpy(),
            "P_r": P_r.detach().cpu().numpy(),
        })

    return records


def plot_radial_wavefunctions(
    wave_net,
    samples,
    out_dir,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    max_cases=None,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)
    selected_samples = samples if max_cases is None else samples[:max_cases]
    paths = []

    for sample in selected_samples:
        tag = nucleus_tag(sample)
        records = collect_radial_probability_data(
            wave_net,
            sample,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for record in records:
            ax.plot(record["r"], record["P_r"], label=record["label"])
        ax.set_xlabel("r (fm)")
        ax.set_ylabel(r"$P(r)$")
        ax.set_title(f"{tag}: radial probability densities")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = save_fig(fig, out_dir / f"{tag}_radial_probability.png")
        paths.append(path)

        if log_wandb:
            wb_log({
                f"plots/wavefunctions/{tag}/radial_probability":
                    wandb.Image(path)
            })

    return paths


@torch.no_grad()
def plot_theta_phi_heatmaps(
    wave_net,
    samples,
    out_dir,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    max_cases=2,
    max_states_per_case=3,
    n_theta=160,
    n_phi=220,
    log_wandb=True,
):
    """
    Heavy final diagnostic. To control runtime, only the first max_cases and
    first max_states_per_case are plotted by default.
    """
    out_dir = ensure_dir(out_dir)
    paths = []

    for sample in samples[:max_cases]:
        tag = nucleus_tag(sample)
        E_vec = torch.tensor(
            [float(st["energy"]) for st in sample["states"]],
            dtype=torch.float32,
            device=device,
        )

        radial = collect_radial_probability_data(
            wave_net,
            sample,
            n_r=600,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )

        theta = torch.linspace(
            1e-5, PI - 1e-5, n_theta, device=device
        )
        phi = torch.linspace(
            0.0, TWOPI, n_phi + 1, device=device
        )[:-1]

        TH, PH = torch.meshgrid(theta, phi, indexing="ij")
        th_flat = TH.reshape(-1, 1)
        ph_flat = PH.reshape(-1, 1)

        for state_idx, st in enumerate(sample["states"][:max_states_per_case]):
            r_values = radial[state_idx]["r"]
            P_r = radial[state_idx]["P_r"]
            r_peak = float(r_values[int(np.argmax(P_r))])
            r_flat = torch.full_like(th_flat, r_peak)

            Re, Im, _ = eval_psi_norm(
                wave_net,
                r_flat,
                th_flat,
                ph_flat,
                E_vec,
                sample,
                st,
                Nr=Nr_norm,
                Nth=Nth_norm,
                Nph=Nph_norm,
            )
            psi2 = (
                Re.squeeze().pow(2) + Im.squeeze().pow(2)
            ).reshape(n_theta, n_phi).detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(6.8, 5.0))
            im = ax.imshow(
                psi2,
                origin="lower",
                aspect="auto",
                extent=[0.0, 2.0 * math.pi, 0.0, math.pi],
            )
            ax.set_xlabel(r"$\phi$ (rad)")
            ax.set_ylabel(r"$\theta$ (rad)")
            ax.set_title(
                f"{tag}: {state_label(st)}, r={r_peak:.2f} fm"
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(r"$|\Psi(r_{\mathrm{peak}},\theta,\phi)|^2$")

            path = save_fig(
                fig,
                out_dir / f"{tag}_state{state_idx}_theta_phi_heatmap.png",
            )
            paths.append(path)

            if log_wandb:
                wb_log({
                    f"plots/wavefunctions/{tag}/heatmap_state_{state_idx}":
                        wandb.Image(path)
                })

    return paths

@torch.no_grad()
def plot_full_psi2_slice_vs_theta(
    wave_net,
    samples,
    out_dir,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    max_cases=None,
    n_theta=400,
    phi0=0.0,
    use_common_r0=True,
    log_wandb=True,
):
    """
    Plot the normalized full wavefunction density

        |Psi(r0, theta, phi0)|^2

    versus theta for all selected states of each nuclear system.

    Parameters
    ----------
    use_common_r0:
        If True, one common r0 is used for all states of a nucleus.
        It is chosen as the peak position of the summed radial
        probability densities.

        If False, each state is evaluated at its own radial
        probability-density peak.
    """
    if not PLOTTING_AVAILABLE:
        return []

    wave_net.eval()
    out_dir = ensure_dir(out_dir)

    selected_samples = (
        samples if max_cases is None else samples[:max_cases]
    )

    paths = []

    for sample in selected_samples:
        tag = nucleus_tag(sample)

        states = sample["states"]
        E_vec = torch.tensor(
            [float(st["energy"]) for st in states],
            dtype=torch.float32,
            device=device,
        )

        radial_records = collect_radial_probability_data(
            wave_net,
            sample,
            n_r=600,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )

        if use_common_r0:
            r_grid = radial_records[0]["r"]

            summed_probability = np.zeros_like(r_grid)
            for record in radial_records:
                summed_probability += record["P_r"]

            common_r0 = float(
                r_grid[int(np.argmax(summed_probability))]
            )
        else:
            common_r0 = None

        theta = torch.linspace(
            1e-5,
            PI - 1e-5,
            n_theta,
            device=device,
        ).unsqueeze(1)

        phi = torch.full_like(theta, float(phi0))

        fig, ax = plt.subplots(figsize=(8.0, 5.2))

        for state_idx, st in enumerate(states):
            if use_common_r0:
                r0 = common_r0
            else:
                r_values = radial_records[state_idx]["r"]
                P_r = radial_records[state_idx]["P_r"]
                r0 = float(r_values[int(np.argmax(P_r))])

            r = torch.full_like(theta, r0)

            psi_re, psi_im, _ = eval_psi_norm(
                wave_net,
                r,
                theta,
                phi,
                E_vec,
                sample,
                st,
                Nr=Nr_norm,
                Nth=Nth_norm,
                Nph=Nph_norm,
            )

            psi2 = (
                psi_re.squeeze().pow(2)
                + psi_im.squeeze().pow(2)
            ).detach().cpu().numpy()

            ax.plot(
                theta.squeeze().detach().cpu().numpy(),
                psi2,
                label=state_label(st),
            )

        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_ylabel(
            r"$|\Psi(r_0,\theta,\phi_0)|^2$"
        )

        if use_common_r0:
            ax.set_title(
                f"{tag}: full wavefunction slice vs theta\n"
                f"$r_0={common_r0:.2f}$ fm, "
                f"$\\phi_0={phi0:.2f}$ rad"
            )
        else:
            ax.set_title(
                f"{tag}: full wavefunction slice vs theta\n"
                "each state evaluated at its radial peak"
            )

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        path = save_fig(
            fig,
            out_dir / f"{tag}_full_psi2_slice_vs_theta.png",
        )
        paths.append(path)

        if log_wandb:
            wb_log({
                f"plots/wavefunctions/{tag}/"
                "full_psi2_slice_vs_theta":
                    wandb.Image(path)
            })

    return paths


# ============================================================
# Overlap diagnostics
# ============================================================
@torch.no_grad()
def compute_overlap_matrix(
    wave_net,
    sample,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    n_points=8000,
):
    wave_net.eval()
    states = sample["states"]
    E_vec = torch.tensor(
        [float(st["energy"]) for st in states],
        dtype=torch.float32,
        device=device,
    )

    r = torch.rand(n_points, 1, device=device) * R_MAX
    theta = torch.rand(n_points, 1, device=device) * PI
    phi = torch.rand(n_points, 1, device=device) * TWOPI

    weight = r.squeeze().pow(2) * torch.sin(theta.squeeze()).clamp_min(1e-12)
    volume = R_MAX * PI * TWOPI

    psi = []
    for st in states:
        Re, Im, _ = eval_psi_norm(
            wave_net,
            r,
            theta,
            phi,
            E_vec,
            sample,
            st,
            Nr=Nr_norm,
            Nth=Nth_norm,
            Nph=Nph_norm,
        )
        psi.append((Re.squeeze(), Im.squeeze()))

    K = len(states)
    O = torch.zeros(K, K, device=device)

    for i in range(K):
        Re_i, Im_i = psi[i]
        for j in range(K):
            Re_j, Im_j = psi[j]
            overlap_re = volume * torch.mean(
                (Re_i * Re_j + Im_i * Im_j) * weight
            )
            overlap_im = volume * torch.mean(
                (Re_i * Im_j - Im_i * Re_j) * weight
            )
            O[i, j] = torch.sqrt(overlap_re.pow(2) + overlap_im.pow(2))

    return O.detach().cpu().numpy()


def plot_overlap_matrices(
    wave_net,
    samples,
    out_dir,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    max_cases=3,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)
    results = []

    for sample in samples[:max_cases]:
        tag = nucleus_tag(sample)
        O = compute_overlap_matrix(
            wave_net,
            sample,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )

        labels = [state_short_label(st) for st in sample["states"]]

        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        im = ax.imshow(O, origin="lower", vmin=0.0)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(r"State $(n_r,l,j)$")
        ax.set_ylabel(r"State $(n_r,l,j)$")
        ax.set_title(f"{tag}: absolute overlap matrix")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$|\langle\psi_i|\psi_j\rangle|$")

        path = save_fig(fig, out_dir / f"{tag}_overlap_matrix.png")

        offdiag = O.copy()
        np.fill_diagonal(offdiag, 0.0)
        max_offdiag = float(np.max(offdiag))

        results.append({
            "case": tag,
            "max_abs_offdiag_overlap": max_offdiag,
            "path": path,
        })

        if log_wandb:
            wb_log({
                f"final/overlap/{tag}_max_offdiag": max_offdiag,
                f"plots/overlap/{tag}": wandb.Image(path),
            })

    return results


# ============================================================
# W&B tables
# ============================================================
def build_samples_table(samples):
    rows = []
    for sample in samples:
        for st in sample["states"]:
            rows.append({
                "A": int(sample["A"]),
                "Z": int(sample["Z"]),
                "species": species_tag(sample["is_proton"]),
                "nr": int(st["nr"]),
                "l": int(st["l"]),
                "j": float(st["j"]),
                "energy": float(st["energy"]),
            })
    return pd.DataFrame(rows)


# ============================================================
# Instrumented global multi-nucleus training
# ============================================================
def train_global_multinucleus_instrumented(
    dataset_path="ws_fd_dataset_2.npz",
    cases=None,
    max_states=8,
    epochs=15000,
    batch_size=None,
    lr_wave=5e-4,
    lr_param=1e-3,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=128,
    hidden_wave=256,
    hidden_param=256,
    emm=0,
    wE=10.0,
    wR=10.0,
    wTh=2.0,
    wPh=2.0,
    wBC=5.0,
    wORTH=0.0,
    wSO=10.0,
    wKL=0.0,
    prior_std=PRIOR_STD,
    orth_points=512,
    use_scheduler_wave=True,
    use_scheduler_param=True,
    gamma_wave=0.6,
    gamma_param=0.5,
    step_size_wave=1000,
    step_size_param=1500,
    print_every=100,
    log_every=100,
    plot_every=1000,
    checkpoint_every=100,
    resume=True,
    out_dir="global_multinucleus_instrumented_outputs",
    use_wandb=True,
    wandb_project="inverse-ws-pinn",
    wandb_run_name="global_multinucleus_six_parameter",
    wandb_group=None,
    log_model_watch=False,
    final_wavefunction_cases=4,
    final_heatmap_cases=2,
    final_overlap_cases=3,
):
    out_dir = ensure_dir(out_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    checkpoints_dir = ensure_dir(out_dir / "checkpoints")

    samples, skipped = build_multinucleus_samples(
        dataset_path=dataset_path,
        cases=cases,
        max_states=max_states,
        require_exact_states=True,
    )

    if not samples:
        raise RuntimeError(
            "No valid samples found. Check dataset_path, cases, and max_states."
        )

    K_states = max_states
    param_input_dim = (
        K_states * n_r_points
        + K_states
        + 3
        + K_states * 3
    )

    wave_net = WaveNet3D(
        n_states=K_states,
        hidden=hidden_wave,
        depth=5,
        beta=0.6,
    ).to(device)

    param_net = GlobalContextParamNet(
        input_dim=param_input_dim,
        hidden=hidden_param,
        depth=3,
        n_parameters=6,
    ).to(device)

    optimizer_wave = torch.optim.Adam(
        wave_net.parameters(),
        lr=lr_wave,
    )
    optimizer_param = torch.optim.Adam(
        param_net.parameters(),
        lr=lr_param,
    )

    scheduler_wave = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_wave,
            step_size=step_size_wave,
            gamma=gamma_wave,
        )
        if use_scheduler_wave else None
    )
    scheduler_param = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_param,
            step_size=step_size_param,
            gamma=gamma_param,
        )
        if use_scheduler_param else None
    )

    config = {
        "base_module": MODULE_NAME,
        "dataset_path": dataset_path,
        "cases": cases,
        "n_samples": len(samples),
        "skipped": skipped,
        "max_states": max_states,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_wave": lr_wave,
        "lr_param": lr_param,
        "n_r_points": n_r_points,
        "Nr_norm": Nr_norm,
        "Nth_norm": Nth_norm,
        "Nph_norm": Nph_norm,
        "hidden_wave": hidden_wave,
        "hidden_param": hidden_param,
        "emm": emm,
        "wE": wE,
        "wR": wR,
        "wTh": wTh,
        "wPh": wPh,
        "wBC": wBC,
        "wORTH": wORTH,
        "wSO": wSO,
        "wKL": wKL,
        "prior_std": prior_std,
        "orth_points": orth_points,
        "param_input_dim": param_input_dim,
        "parameter_mode": "one_global_context_informed_posterior",
        "reference": REFERENCE,
        "checkpoint_every": checkpoint_every,
        "resume": resume,
    }

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError(
                "use_wandb=True, but wandb is not installed."
            )

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config=config,
            reinit=True,
        )

        sample_df = build_samples_table(samples)
        wb_log({"tables/states_used": wandb.Table(dataframe=sample_df)})

        if log_model_watch:
            wandb.watch(
                (wave_net, param_net),
                log="gradients",
                log_freq=max(log_every, 1),
            )

    print("\n========================================================")
    print("GLOBAL MULTI-NUCLEUS SIX-PARAMETER INSTRUMENTED TRAINING")
    print("========================================================")
    print(f"Base module: {MODULE_NAME}")
    print(f"Device: {device}")
    print(f"Samples used: {len(samples)}")
    if skipped:
        print(f"Skipped cases: {skipped}")

    for sample in samples:
        print(
            f"A={sample['A']:3d} Z={sample['Z']:3d} "
            f"species={species_tag(sample['is_proton'])} "
            f"states={len(sample['states'])}"
        )

    history = []
    best_loss = float("inf")
    best_epoch = None
    start_epoch = 1
    t0 = time.time()

    latest_checkpoint_path = checkpoints_dir / "latest_checkpoint.pt"

    def save_training_checkpoint(path, epoch, current_loss):
        """Atomically save everything needed to continue training."""
        checkpoint = {
            "epoch": int(epoch),
            "wave_net": wave_net.state_dict(),
            "param_net": param_net.state_dict(),
            "optimizer_wave": optimizer_wave.state_dict(),
            "optimizer_param": optimizer_param.state_dict(),
            "scheduler_wave": (
                scheduler_wave.state_dict() if scheduler_wave is not None else None
            ),
            "scheduler_param": (
                scheduler_param.state_dict() if scheduler_param is not None else None
            ),
            "loss": None if current_loss is None else float(current_loss),
            "best_loss": float(best_loss),
            "best_epoch": best_epoch,
            "history": history,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

    if resume and latest_checkpoint_path.exists():
        print(f"Resuming from {latest_checkpoint_path}")
        checkpoint = torch.load(
            latest_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        wave_net.load_state_dict(checkpoint["wave_net"])
        param_net.load_state_dict(checkpoint["param_net"])
        optimizer_wave.load_state_dict(checkpoint["optimizer_wave"])
        optimizer_param.load_state_dict(checkpoint["optimizer_param"])

        if scheduler_wave is not None and checkpoint.get("scheduler_wave") is not None:
            scheduler_wave.load_state_dict(checkpoint["scheduler_wave"])
        if scheduler_param is not None and checkpoint.get("scheduler_param") is not None:
            scheduler_param.load_state_dict(checkpoint["scheduler_param"])

        history = checkpoint.get("history", [])
        best_loss = float(checkpoint.get("best_loss", float("inf")))
        best_epoch = checkpoint.get("best_epoch")
        start_epoch = int(checkpoint["epoch"]) + 1

        if checkpoint.get("torch_rng_state") is not None:
            cpu_rng_state = torch.as_tensor(
                checkpoint["torch_rng_state"],
                dtype=torch.uint8,
                device="cpu",
            )
            torch.set_rng_state(cpu_rng_state)

        if checkpoint.get("numpy_rng_state") is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])

        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            cuda_rng_states = [
                torch.as_tensor(
                    state,
                    dtype=torch.uint8,
                    device="cpu",
                )
                for state in checkpoint["cuda_rng_state_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_rng_states)

        print(
            f"Checkpoint loaded: completed epoch {start_epoch - 1}; "
            f"continuing at epoch {start_epoch}."
        )
    elif resume:
        print(f"No existing checkpoint at {latest_checkpoint_path}; starting from epoch 1.")

    loss_kwargs = {
        "emm": emm,
        "wE": wE,
        "wR": wR,
        "wTh": wTh,
        "wPh": wPh,
        "wBC": wBC,
        "wORTH": wORTH,
        "wSO": wSO,
        "orth_points": orth_points,
    }

    for epoch in range(start_epoch, epochs + 1):
        wave_net.train()
        param_net.train()

        optimizer_wave.zero_grad(set_to_none=True)
        optimizer_param.zero_grad(set_to_none=True)

        loss, metrics, per_sample_rows = compute_multinucleus_loss_full3d(
            wave_net=wave_net,
            param_net=param_net,
            samples=samples,
            batch_size=batch_size,
            prior_std=prior_std,
            wKL=wKL,
            is_training=True,
            n_r_points=n_r_points,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            **loss_kwargs,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at epoch {epoch}: {loss}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(wave_net.parameters()) + list(param_net.parameters()),
            max_norm=5.0,
        )

        optimizer_wave.step()
        optimizer_param.step()

        if scheduler_wave is not None:
            scheduler_wave.step()
        if scheduler_param is not None:
            scheduler_param.step()

        loss_value = float(loss.detach().cpu())

        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "wave_net": wave_net.state_dict(),
                    "param_net": param_net.state_dict(),
                    "optimizer_wave": optimizer_wave.state_dict(),
                    "optimizer_param": optimizer_param.state_dict(),
                    "scheduler_wave": (scheduler_wave.state_dict() if scheduler_wave is not None else None),
                    "scheduler_param": (scheduler_param.state_dict() if scheduler_param is not None else None),
                    "loss": loss_value,
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": config,
                },
                checkpoints_dir / "best_training_loss.pt",
            )

        should_log = (
            epoch == 1
            or epoch % log_every == 0
            or epoch == epochs
        )
        should_print = (
            epoch == 1
            or epoch % print_every == 0
            or epoch == epochs
        )
        should_plot = (
            plot_every is not None
            and (
                epoch == 1
                or epoch % plot_every == 0
                or epoch == epochs
            )
        )

        if should_log:
            stats = infer_global_stats(
                wave_net,
                param_net,
                samples,
                n_r_points=n_r_points,
                Nr_norm=Nr_norm,
                Nth_norm=Nth_norm,
                Nph_norm=Nph_norm,
                n_mc_samples=1000,
            )

            row = {
                "epoch": epoch,
                "time_sec": time.time() - t0,
                "loss": loss_value,
                **metrics,
                "lr_wave": optimizer_wave.param_groups[0]["lr"],
                "lr_param": optimizer_param.param_groups[0]["lr"],
                "grad_norm": float(grad_norm.detach().cpu()),
            }

            for p in PARAM_NAMES:
                row[f"{p}_mu"] = float(stats[p][0])
                row[f"{p}_sigma"] = float(stats[p][1])

            history.append(row)
            save_history_csv(history, out_dir / "training_history.csv")

            payload = {
                "epoch": epoch,
                "loss/total": row["loss"],
                "loss/energy": row.get("LE", np.nan),
                "loss/radial_pde": row.get("LR", np.nan),
                "loss/theta_pde": row.get("LTH", np.nan),
                "loss/phi_pde": row.get("LPH", np.nan),
                "loss/boundary": row.get("LBC", np.nan),
                "loss/orthogonality": row.get("LORTH", np.nan),
                "loss/spin_orbit": row.get("LSO", np.nan),
                "loss/KL": row.get("LKL", np.nan),
                "optimization/lr_wave": row["lr_wave"],
                "optimization/lr_param": row["lr_param"],
                "optimization/gradient_norm": row["grad_norm"],
                "optimization/best_training_loss": best_loss,
            }

            for p in PARAM_NAMES:
                payload[f"parameters/{p}_mean"] = row[f"{p}_mu"]
                payload[f"parameters/{p}_sigma"] = row[f"{p}_sigma"]

                if REFERENCE is not None and p in REFERENCE:
                    payload[f"parameters/{p}_error_to_reference"] = (
                        row[f"{p}_mu"] - float(REFERENCE[p])
                    )
                    payload[f"parameters/{p}_abs_error_to_reference"] = abs(
                        row[f"{p}_mu"] - float(REFERENCE[p])
                    )

            wb_log(payload, step=epoch)

        if should_print:
            if history:
                h = history[-1]
                param_text = ", ".join(
                    f"{p}={h[f'{p}_mu']:.4f}±{h[f'{p}_sigma']:.4f}"
                    for p in PARAM_NAMES
                )
            else:
                param_text = "parameter statistics not logged yet"

            print(
                f"[epoch={epoch:6d}] "
                f"loss={loss_value:.3e} "
                f"LE={metrics.get('LE', float('nan')):.3e} "
                f"LR={metrics.get('LR', float('nan')):.3e} "
                f"LTH={metrics.get('LTH', float('nan')):.3e} "
                f"LPH={metrics.get('LPH', float('nan')):.3e} "
                f"LBC={metrics.get('LBC', float('nan')):.3e} "
                f"LORTH={metrics.get('LORTH', float('nan')):.3e} "
                f"LSO={metrics.get('LSO', float('nan')):.3e} "
                f"LKL={metrics.get('LKL', float('nan')):.3e}"
            )
            print("  global params:", param_text)

        should_checkpoint = (
            checkpoint_every is not None
            and checkpoint_every > 0
            and (epoch % checkpoint_every == 0 or epoch == epochs)
        )
        if should_checkpoint:
            save_training_checkpoint(
                latest_checkpoint_path,
                epoch=epoch,
                current_loss=loss_value,
            )
            print(f"Checkpoint saved: {latest_checkpoint_path}")

        if should_plot and history:
            live_dir = ensure_dir(plots_dir / "live")
            plot_loss_history(
                history,
                live_dir,
                log_wandb=use_wandb,
            )
            plot_learning_rates(
                history,
                live_dir,
                log_wandb=use_wandb,
            )
            plot_parameter_history(
                history,
                live_dir,
                reference=REFERENCE,
                log_wandb=use_wandb,
            )

    # Final resumable checkpoint
    final_loss = float(history[-1]["loss"]) if history else None
    save_training_checkpoint(
        checkpoints_dir / "final_checkpoint.pt",
        epoch=epochs,
        current_loss=final_loss,
    )
    save_training_checkpoint(
        latest_checkpoint_path,
        epoch=epochs,
        current_loss=final_loss,
    )

    torch.save(
        wave_net.state_dict(),
        out_dir / "wave_net_global_multinucleus.pt",
    )
    torch.save(
        param_net.state_dict(),
        out_dir / "param_net_global_multinucleus.pt",
    )

    final_stats = infer_global_stats(
        wave_net,
        param_net,
        samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        n_mc_samples=10000,
    )

    final_json = {
        p: {
            "mean": float(final_stats[p][0]),
            "sigma": float(final_stats[p][1]),
        }
        for p in PARAM_NAMES
    }

    latent_mu, latent_sigma, latent_logvar = infer_global_raw_latent(
        wave_net,
        param_net,
        samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    with open(
        out_dir / "final_global_parameter_prediction.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_json, f, indent=2)

    with open(
        out_dir / "final_global_latent_prediction.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "mu": latent_mu,
                "sigma": latent_sigma,
                "logvar": latent_logvar,
            },
            f,
            indent=2,
        )

    # Final plots and diagnostics
    final_plots_dir = ensure_dir(plots_dir / "final")

    plot_loss_history(
        history,
        final_plots_dir,
        log_wandb=use_wandb,
    )
    plot_learning_rates(
        history,
        final_plots_dir,
        log_wandb=use_wandb,
    )
    plot_parameter_history(
        history,
        final_plots_dir,
        reference=REFERENCE,
        log_wandb=use_wandb,
    )
    plot_parameter_summary(
        final_stats,
        final_plots_dir,
        reference=REFERENCE,
        log_wandb=use_wandb,
    )

    energy_results = plot_global_energy_diagnostics(
        wave_net,
        param_net,
        samples,
        final_plots_dir / "energies",
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        log_wandb=use_wandb,
    )

    plot_radial_wavefunctions(
        wave_net,
        samples,
        final_plots_dir / "radial_wavefunctions",
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        max_cases=final_wavefunction_cases,
        log_wandb=use_wandb,
    )

    plot_theta_phi_heatmaps(
        wave_net,
        samples,
        final_plots_dir / "theta_phi_heatmaps",
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        max_cases=final_heatmap_cases,
        max_states_per_case=3,
        log_wandb=use_wandb,
    )

    plot_full_psi2_slice_vs_theta(
        wave_net,
        samples,
        final_plots_dir / "full_psi2_theta_slices",
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        max_cases=final_wavefunction_cases,
        n_theta=400,
        phi0=0.0,
        use_common_r0=True,
        log_wandb=use_wandb,
    )

    overlap_results = plot_overlap_matrices(
        wave_net,
        samples,
        final_plots_dir / "overlaps",
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        max_cases=final_overlap_cases,
        log_wandb=use_wandb,
    )

    summary = {
        "best_training_loss": best_loss,
        "best_training_epoch": best_epoch,
        "final_parameters": final_json,
        "final_energy_MAE": energy_results["MAE"],
        "final_energy_RMSE": energy_results["RMSE"],
        "final_energy_bias": energy_results["Bias"],
        "overlap_results": overlap_results,
    }

    with open(
        out_dir / "final_run_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"{wandb_run_name}-outputs",
            type="model-and-diagnostics",
        )
        artifact.add_dir(str(out_dir))
        wandb.log_artifact(artifact)
        wandb.finish()

    print("\n========================================================")
    print("TRAINING COMPLETE")
    print("========================================================")
    print(f"Best training loss: {best_loss:.6e} at epoch {best_epoch}")
    print(
        f"Final energy MAE={energy_results['MAE']:.6f} MeV, "
        f"RMSE={energy_results['RMSE']:.6f} MeV"
    )
    print("Final global parameters:")
    for p in PARAM_NAMES:
        print(
            f"  {p}: "
            f"{final_json[p]['mean']:.6f} ± {final_json[p]['sigma']:.6f}"
        )

    return {
        "wave_net": wave_net,
        "param_net": param_net,
        "history": history,
        "samples": samples,
        "final_parameters": final_json,
        "energy_results": energy_results,
        "summary": summary,
    }


# ============================================================
# RUN CONFIGURATION — edit this section in Colab if needed
# ============================================================
if __name__ == "__main__":
    # Colab's /content filesystem is temporary. Google Drive persists after
    # runtime disconnections and resets, so checkpoints and outputs go there.
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        print("Not running in Colab; using the configured output path.")

    if not WANDB_AVAILABLE:
        raise ImportError(
            "wandb is not installed. In a Colab cell run:\n"
            "!pip -q install wandb pandas matplotlib"
        )

    # W&B will use the account authenticated with wandb.login().
    if wandb.api.api_key is None:
        wandb.login()

    CASES = [
        (208, 82, True),
        (132, 50, True),
        (40, 20, True),
        #(100, 50, True),
        (48, 20, True),
        (208, 82, False),
        (132, 50, False),
        (40, 20, False),
        #(100, 50, False),
        (48, 20, False),
    ]

    result = train_global_multinucleus_instrumented(
        dataset_path="experimental_dataset.npz",
        cases=CASES,
        max_states=6,
        epochs=30000,
        batch_size=2,
        lr_wave=5e-4,
        lr_param=1e-3,
        n_r_points=96,
        Nr_norm=1024,
        Nth_norm=512,
        Nph_norm=256,
        hidden_wave=256,
        hidden_param=256,
        emm=0,
        wE=0.5,
        wR=10.0,
        wTh=5.0,
        wPh=5.0,
        wBC=5.0,
        wORTH=5.0,
        wSO=0.2,
        wKL=1e-4,
        prior_std=PRIOR_STD,
        orth_points=512,
        use_scheduler_wave=True,
        use_scheduler_param=True,
        gamma_wave=0.6,
        gamma_param=0.5,
        step_size_wave=1500,
        step_size_param=2000,
        print_every=100,
        log_every=100,
        plot_every=1000,
        checkpoint_every=100,
        resume=True,
        out_dir="/content/drive/MyDrive/PINN_runs/seminole_global_multinucleus_wandb_outputs",
        use_wandb=True,
        wandb_project="inverse-ws-pinn-wahlborn-experimental",
        wandb_run_name="seminole_global_multinucleus_six_parameter",
        wandb_group="global-context",
        log_model_watch=False,
        final_wavefunction_cases=4,
        final_heatmap_cases=2,
        final_overlap_cases=3,
    )