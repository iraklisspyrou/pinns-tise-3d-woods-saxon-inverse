"""Inverse Woods–Saxon PINN with probabilistic parameter inference.

The model learns separable radial and angular wavefunction components and
infers V0, r0, a and lambda_so from finite-difference spectra.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import yaml
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda")  # Windows/Linux
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # MacOS
else:
    device = torch.device("cpu")


config_path = Path("config.yaml")
config = yaml.load(config_path.open(), Loader=yaml.FullLoader)


torch.manual_seed(0)
np.random.seed(0)

# Quantum constants
const = config["const"]
hc = const["hc"]
u = const["u"]
e2 = const["e2"]
R_MAX = const["R_MAX"]
kappa_fixed = const["kappa_fixed"]
r0_so_fixed = const["r0_so_fixed"]
a_so_fixed = const["a_so_fixed"]

# global constants
PI = math.pi
TWOPI = 2.0 * math.pi



# Dataset loading and state selection

def load_fd_dataset(path="ws_fd_dataset.npz"):
    """Load finite-difference spectra stored in the project NPZ format."""
    raw = np.load(path, allow_pickle=True)["data"]
    return list(raw)


def get_sample_by_nucleus(dataset, A, Z, is_proton, max_states=6):
    """Select a representative set of bound states for one nucleus and species."""
    for item in dataset:
        if (
            int(item["A"]) == int(A)
            and int(item["Z"]) == int(Z)
            and (bool(item["is_proton"]) == bool(is_proton))
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
            # Always retain the deepest state, then add representative spin-orbit pairs.
            deepest_state = sorted(all_states, key=lambda x: x["energy"])[0]
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
                if key not in groups:
                    groups[key] = []
                groups[key].append(st)
            pair_groups = [grp for grp in groups.values() if len(grp) == 2]
            unpaired_states = [
                st for grp in groups.values() if len(grp) == 1 for st in grp
            ]
            sorted_pairs = sorted(
                pair_groups,
                key=lambda grp: sum((float(s["energy"]) for s in grp)) / 2.0,
            )
            remaining_slots = max_states - len(selected_states)
            max_pairs = remaining_slots // 2
            if max_pairs > 0:
                if len(sorted_pairs) > max_pairs:
                    target_deep_pairs = max_pairs * 2 // 3
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
                    unpaired_states, key=lambda x: float(x["energy"])
                )
                for st in unpaired_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))
            if len(selected_states) < max_states:
                remaining_states = sorted(all_states, key=lambda x: float(x["energy"]))
                for st in remaining_states:
                    if len(selected_states) >= max_states:
                        break
                    if state_id(st) not in selected_ids:
                        selected_states.append(st)
                        selected_ids.add(state_id(st))
            selected_states = sorted(selected_states, key=lambda x: float(x["energy"]))
            return {
                "A": int(A),
                "Z": int(Z),
                "is_proton": bool(is_proton),
                "states": selected_states,
            }
    raise ValueError(f"No sample found for A={A}, Z={Z}, is_proton={is_proton}")


# Nuclear physics helpers

def reduced_mass(A: int, is_proton: bool):
    """Return the reduced mass of the valence nucleon and the residual core."""
    m_core = float(A - 1) * u
    m_nucl = 1.007276466621 * u if is_proton else 1.00866491588 * u
    return m_core * m_nucl / (m_core + m_nucl)


def K_value(mu):
    return hc**2 / (2.0 * mu)


def l_dot_s(l: int, j: float):
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def tdotT_expectation(A: int, Z: int, is_proton: bool):
    N = A - Z
    if N == Z:
        rhs = 3.0
    elif N > Z:
        rhs = (N - Z + 1 if is_proton else -(N - Z + 1)) + 2.0
    else:
        rhs = (N - Z - 1 if is_proton else -(N - Z - 1)) + 2.0
    return -0.25 * rhs


@dataclass
class WSParams:
    """Trainable Woods–Saxon parameter set used by the inverse model."""
    V0: torch.Tensor
    r0: torch.Tensor
    a: torch.Tensor
    lam_so: torch.Tensor


# Input scaling

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


# Woods–Saxon mean-field potential

def R_central(A, params: WSParams):
    return params.r0 * float(A) ** (1.0 / 3.0)


def R_spin_orbit(A):
    return r0_so_fixed * float(A) ** (1.0 / 3.0)


def f_ws(r, R, a):
    x = (r - R) / a
    return 1.0 / (1.0 + torch.exp(x))


def df_ws_dr(r, R, a):
    x = (r - R) / a
    ex = torch.exp(x)
    return -(ex / (1.0 + ex) ** 2) * (1.0 / a)


def V_depth(A, Z, is_proton, params: WSParams):
    tdotT = tdotT_expectation(A, Z, is_proton)
    return params.V0 * (1.0 - 4.0 * kappa_fixed / float(A) * tdotT)


def V_central(r, A, Z, is_proton, params: WSParams):
    V = V_depth(A, Z, is_proton, params)
    R = R_central(A, params)
    return -V * f_ws(r, R, params.a)


def Vcentral_r2(r, A, Z, is_proton, params: WSParams):
    return r**2 * V_central(r, A, Z, is_proton, params)


def Vc_r2(r, A, Z, is_proton, params: WSParams):
    if not is_proton:
        return torch.zeros_like(r)
    Z_core = max(int(Z) - 1, 0)
    Zc = float(Z_core)
    R = R_central(A, params)
    rc = r.clamp_min(0.0)
    inside = rc <= R
    outside = ~inside
    out = torch.empty_like(rc)
    out[inside] = Zc * e2 / (2.0 * R) * (3.0 * rc[inside] ** 2 - rc[inside] ** 4 / R**2)
    out[outside] = Zc * e2 * rc[outside]
    return out


def Vso_r2(r, A, Z, is_proton, l, j, params: WSParams):
    mu = reduced_mass(A, is_proton)
    rc = r.clamp_min(1e-10)
    Rso = R_spin_orbit(A)
    Vtilde = params.lam_so * params.V0
    dVt_dr = Vtilde * df_ws_dr(rc, Rso, a_so_fixed)
    pref_r2 = rc * hc**2 / (2.0 * mu**2)
    return pref_r2 * dVt_dr * float(l_dot_s(l, j))


def Veff_r2(r, A, Z, is_proton, l, j, params: WSParams):
    return (
        Vcentral_r2(r, A, Z, is_proton, params)
        + Vc_r2(r, A, Z, is_proton, params)
        + Vso_r2(r, A, Z, is_proton, l, j, params)
    )


def V_eff(r, A, Z, is_proton, l, j, params: WSParams):
    rc = r.clamp_min(1e-10)
    return Veff_r2(rc, A, Z, is_proton, l, j, params) / rc**2


# Neural network models

class ConditionalMLP(nn.Module):
    """Fully connected network used for each conditional wavefunction component."""

    def __init__(self, input_dim, output_dim=1, hidden=256, depth=5):
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
    """Learn the separable wavefunction components R(r), Theta(theta) and Phi(phi)."""

    def __init__(self, n_states, hidden=256, depth=5, beta=0.6):
        super().__init__()
        self.n_states = n_states
        self.beta = beta
        input_dim = 1 + n_states + 3 + 3
        self.R_net = ConditionalMLP(input_dim, output_dim=1, hidden=hidden, depth=depth)
        self.Th_net = ConditionalMLP(
            input_dim, output_dim=1, hidden=hidden, depth=depth
        )
        self.Ph_net = ConditionalMLP(
            input_dim, output_dim=2, hidden=hidden, depth=depth
        )

    def _context(self, x_coord, E_vec, sample, st, coord_scaler):
        N = x_coord.shape[0]
        x_scaled = coord_scaler(x_coord)
        E_scaled = scale_energy(E_vec).unsqueeze(0).repeat(N, 1)
        nuc = (
            scale_nucleus(sample["A"], sample["Z"], sample["is_proton"])
            .unsqueeze(0)
            .repeat(N, 1)
        )
        q = scale_quantum(st["nr"], st["l"], st["j"]).unsqueeze(0).repeat(N, 1)
        return torch.cat([x_scaled, E_scaled, nuc, q], dim=1)

    def R(self, r, E_vec, sample, st):
        x = self._context(r, E_vec, sample, st, scale_r)
        raw = self.R_net(x)
        rc = r.clamp_min(1e-08)
        l = int(st["l"])
        return rc**l * torch.exp(-self.beta * rc) * raw

    def Theta(self, th, E_vec, sample, st):
        x = self._context(th, E_vec, sample, st, scale_theta)
        return self.Th_net(x)

    def Phi(self, ph, E_vec, sample, st):
        x = self._context(ph, E_vec, sample, st, scale_phi)
        out = self.Ph_net(x)
        return (out[:, :1], out[:, 1:])


class ProbabilisticParamNet(nn.Module):
    """Infer Gaussian latent distributions for the Woods–Saxon parameters."""

    def __init__(self, input_dim, hidden=256, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        self.feature_extractor = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden, 4)
        self.logvar_head = nn.Linear(hidden, 4)
        self.apply(self._init_weights)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -3.0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        sigma = torch.exp(0.5 * logvar)
        return (mu, sigma, logvar)


# Differentiation, quadrature and wavefunction normalization

def d(f, x):
    """Differentiate a tensor with respect to its input using autograd."""
    return torch.autograd.grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True
    )[0]


def make_grids(Nr=1024, Nth=512, Nph=512, requires_grad=True):
    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    th = torch.linspace(1e-05, PI - 1e-05, Nth, device=device).unsqueeze(1)
    ph = torch.linspace(0.0, TWOPI, Nph + 1, device=device)[:-1].unsqueeze(1)
    r.requires_grad_(requires_grad)
    th.requires_grad_(requires_grad)
    ph.requires_grad_(requires_grad)
    return (r, th, ph)


def eval_R(wave_net, r, E_vec, sample, st):
    return wave_net.R(r, E_vec, sample, st)


def eval_Theta(wave_net, th, E_vec, sample, st):
    return wave_net.Theta(th, E_vec, sample, st)


def eval_Phi(wave_net, ph, E_vec, sample, st):
    return wave_net.Phi(ph, E_vec, sample, st)


def norm_parts(wave_net, E_vec, sample, st, Nr=1024, Nth=512, Nph=512, eps=1e-12):
    """Compute separated normalization integrals with trapezoidal quadrature."""
    r, th, ph = make_grids(Nr=Nr, Nth=Nth, Nph=Nph, requires_grad=True)
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    r1 = r.squeeze()
    th1 = th.squeeze()
    ph1 = ph.squeeze()
    IR = torch.trapz(Rv.squeeze() ** 2 * r1**2, r1)
    sinth = torch.sin(th1).clamp_min(1e-08)
    Ith = torch.trapz(Thv.squeeze() ** 2 * sinth, th1)
    Phi2 = a_re.squeeze() ** 2 + b_im.squeeze() ** 2
    Iphi = torch.trapz(Phi2, ph1)
    Iang = Ith * Iphi
    I = (IR * Iang).clamp_min(eps)
    s = torch.rsqrt(I)
    return (IR, Ith, Iphi, Iang, I, s, r, th, ph)


def psi_scale_only(wave_net, E_vec, sample, st, Nr=1024, Nth=512, Nph=512):
    IR, Ith, Iphi, Iang, I, s, *_ = norm_parts(
        wave_net, E_vec, sample, st, Nr=Nr, Nth=Nth, Nph=Nph
    )
    return (IR, Ith, Iphi, Iang, I, s)


def eval_R_norm(wave_net, r, E_vec, sample, st, Nr=1024, Nth=512, Nph=512):
    _, _, _, _, _, s = psi_scale_only(
        wave_net, E_vec, sample, st, Nr=Nr, Nth=Nth, Nph=Nph
    )
    return (s * eval_R(wave_net, r, E_vec, sample, st), s)


def eval_psi_norm(wave_net, r, th, ph, E_vec, sample, st, Nr=1024, Nth=512, Nph=512):
    _, _, _, _, _, s = psi_scale_only(
        wave_net, E_vec, sample, st, Nr=Nr, Nth=Nth, Nph=Nph
    )
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    Re = s * Rv * Thv * a_re
    Im = s * Rv * Thv * b_im
    return (Re, Im, s)


# ParamNet input construction

def make_param_input_from_radial_psi(
    wave_net, sample, E_vec, n_r_points=96, Nr_norm=1024, Nth_norm=512, Nph_norm=512
):
    """Build ParamNet features from normalized radial wavefunction samples."""
    r = torch.linspace(0.0, R_MAX, n_r_points, device=device).unsqueeze(1)
    radial_parts = []
    for st in sample["states"]:
        _, _, _, _, _, s = psi_scale_only(
            wave_net, E_vec, sample, st, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
        )
        Rn = (s * eval_R(wave_net, r, E_vec, sample, st)).squeeze()
        idx = min(5, Rn.numel() - 1)
        sign = torch.sign(Rn[idx].detach())
        if sign.item() == 0.0:
            sign = torch.tensor(1.0, device=device)
        Rn = sign * Rn
        radial_parts.append(Rn)
    psi_vec = torch.cat(radial_parts)
    E_scaled = scale_energy(E_vec)
    nuc = scale_nucleus(sample["A"], sample["Z"], sample["is_proton"])
    q_vec = torch.cat(
        [scale_quantum(st["nr"], st["l"], st["j"]) for st in sample["states"]]
    )
    return torch.cat([psi_vec, E_scaled, nuc, q_vec])


# Physics-informed objectives

def energy_rayleigh_full3d(
    wave_net, E_vec, sample, st, params, Nr=1024, Nth=512, Nph=512, eps=1e-12
):
    """Compute the state energy through the full separable Rayleigh quotient."""
    A, Z, is_proton = (sample["A"], sample["Z"], sample["is_proton"])
    ell, j = (int(st["l"]), float(st["j"]))
    mu = reduced_mass(A, is_proton)
    K = K_value(mu)
    IR, Ith, Iphi, Iang, _, s, r, th, ph = norm_parts(
        wave_net, E_vec, sample, st, Nr=Nr, Nth=Nth, Nph=Nph, eps=eps
    )
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = (d(a_re, ph), d(b_im, ph))
    r1, th1, ph1 = (r.squeeze(), th.squeeze(), ph.squeeze())
    IRprime = torch.trapz(Rr.squeeze() ** 2 * r1**2, r1)
    IR0 = torch.trapz(Rv.squeeze() ** 2, r1)
    Veff_r2_vals = Veff_r2(r, A, Z, is_proton, ell, j, params).squeeze()
    IVR = torch.trapz(Veff_r2_vals * Rv.squeeze() ** 2, r1)
    sinth = torch.sin(th1).clamp_min(1e-08)
    Ith = torch.trapz(Thv.squeeze() ** 2 * sinth, th1)
    Ithprime = torch.trapz(Th_th.squeeze() ** 2 * sinth, th1)
    Ith_over_sin = torch.trapz(Thv.squeeze() ** 2 / sinth, th1)
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
    wave_net, E_vec, sample, st, params, Nr_norm=1024, Nth_norm=512, Nph_norm=512, N=512
):
    """Mean-squared radial Schrödinger residual at random collocation points."""
    A, Z, is_proton = (sample["A"], sample["Z"], sample["is_proton"])
    ell, j = (int(st["l"]), float(st["j"]))
    mu = reduced_mass(A, is_proton)
    K = K_value(mu)
    Ns = N // 2
    r = torch.rand(N, 1, device=device) * R_MAX
    r.requires_grad_(True)
    _, _, _, _, _, s = psi_scale_only(
        wave_net, E_vec, sample, st, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
    )
    E_det = energy_rayleigh_full3d(
        wave_net, E_vec, sample, st, params, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
    ).detach()
    Rv = eval_R(wave_net, r, E_vec, sample, st)
    Rr = d(Rv, r)
    Rrr = d(Rr, r)
    rc = r.clamp_min(1e-08)
    Rn, Rnr, Rnrr = (s * Rv, s * Rr, s * Rrr)
    Veff_r2_vals = Veff_r2(rc, A, Z, is_proton, ell, j, params)
    residual = (
        -K * (rc**2 * Rnrr + 2.0 * rc * Rnr - ell * (ell + 1.0) * Rn)
        + (Veff_r2_vals - rc**2 * E_det) * Rn
    )
    return torch.mean(residual**2)


def loss_pde_theta_full3d(wave_net, E_vec, sample, st, emm=0, N=512):
    """Mean-squared residual of the polar angular equation."""
    ell = int(st["l"])
    th = (torch.rand(N, 1, device=device) * (1.0 - 0.002) + 0.001) * PI
    th.requires_grad_(True)
    Thv = eval_Theta(wave_net, th, E_vec, sample, st)
    Th_th = d(Thv, th)
    Th_thth = d(Th_th, th)
    sinth, costh = (torch.sin(th), torch.cos(th))
    sin2 = sinth**2
    op = (
        sin2 * Th_thth
        + sinth * costh * Th_th
        + (ell * (ell + 1.0) * sin2 - float(emm**2)) * Thv
    )
    return torch.mean(op**2)


def loss_pde_phi_full3d(wave_net, E_vec, sample, st, emm=0, N=512, Np_bc=64):
    """Azimuthal ODE loss with periodicity and phase-fixing constraints."""
    ph = torch.rand(N, 1, device=device) * TWOPI
    ph.requires_grad_(True)
    a_re, b_im = eval_Phi(wave_net, ph, E_vec, sample, st)
    ap, bp = (d(a_re, ph), d(b_im, ph))
    app, bpp = (d(ap, ph), d(bp, ph))
    L_ode = (
        (app + float(emm**2) * a_re) ** 2 + (bpp + float(emm**2) * b_im) ** 2
    ).mean()
    z0 = torch.zeros(Np_bc, 1, device=device, requires_grad=True)
    z2 = torch.full_like(z0, TWOPI, requires_grad=True)
    a0, b0 = eval_Phi(wave_net, z0, E_vec, sample, st)
    a2, b2 = eval_Phi(wave_net, z2, E_vec, sample, st)
    ap0, bp0 = (d(a0, z0), d(b0, z0))
    ap2, bp2 = (d(a2, z2), d(b2, z2))
    L_per = (
        (a0 - a2) ** 2 + (b0 - b2) ** 2 + (ap0 - ap2) ** 2 + (bp0 - bp2) ** 2
    ).mean()
    L_gauge = ((a0 - 1.0) ** 2 + b0**2 + ap0**2 + (bp0 - float(emm)) ** 2).mean()
    return L_ode + L_per + 2.0 * L_gauge


def loss_bc_full3d(
    wave_net, E_vec, sample, st, Nr_norm=1024, Nth_norm=512, Nph_norm=512, Nb=128
):
    """Enforce radial decay at R_MAX and regularity at the origin."""
    ell = int(st["l"])
    rR = torch.full((Nb, 1), R_MAX, device=device, requires_grad=True)
    RnR, _ = eval_R_norm(
        wave_net, rR, E_vec, sample, st, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
    )
    L_Rmax = torch.mean(RnR**2)
    r0 = torch.zeros((Nb, 1), device=device, requires_grad=True)
    Rn0, _ = eval_R_norm(
        wave_net, r0, E_vec, sample, st, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
    )
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
    wave_net, sample, E_vec, Nr_norm=512, Nth_norm=256, Nph_norm=256, N=2048
):
    """Penalize overlaps between radial excitations sharing the same (l, j)."""
    states = sample["states"]
    K_states = len(states)
    if K_states < 2:
        return torch.tensor(0.0, device=device)
    r = torch.rand(N, 1, device=device) * R_MAX
    th = torch.rand(N, 1, device=device) * PI
    ph = torch.rand(N, 1, device=device) * TWOPI
    weight = r.squeeze() ** 2 * torch.sin(th.squeeze()).clamp_min(1e-12)
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
        for j in range(i):
            st_j = states[j]
            nr_j = int(st_j["nr"])
            l_j = int(st_j["l"])
            j_j = float(st_j["j"])
            same_lj_different_nr = (
                l_i == l_j and abs(j_i - j_j) < 1e-06 and (nr_i != nr_j)
            )
            if not same_lj_different_nr:
                continue
            Re_j, Im_j = psis[j]
            overlap_re_density = Re_j * Re_i + Im_j * Im_i
            overlap_im_density = Re_j * Im_i - Im_j * Re_i
            overlap_re = volume * torch.mean(overlap_re_density * weight)
            overlap_im = volume * torch.mean(overlap_im_density * weight)
            overlap_abs_squared = overlap_re**2 + overlap_im**2
            total = total + overlap_abs_squared
            n_pairs += 1
    if n_pairs == 0:
        return torch.tensor(0.0, device=device)
    return total / n_pairs


# Joint inverse-problem loss

def compute_sample_loss_full3d(
    wave_net,
    param_net,
    sample,
    n_r_points=64,
    Nr_norm=1024,
    Nth_norm=512,
    Nph_norm=512,
    emm=0,
    wE=50.0,
    wR=5.0,
    wTh=5.0,
    wPh=5.0,
    wBC=5.0,
    wORTH=10.0,
    wKL=0.001,
    is_training=True,
):
    """Assemble all physics, data, orthogonality and KL loss terms."""
    states = sample["states"]
    K_states = len(states)
    E_vec = torch.tensor(
        [float(st["energy"]) for st in states], dtype=torch.float32, device=device
    )
    x_param = make_param_input_from_radial_psi(
        wave_net,
        sample,
        E_vec,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )
    mu, sigma, logvar = param_net(x_param)
    L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Reparameterization keeps stochastic sampling differentiable.
    if is_training:
        epsilon = torch.randn_like(sigma)
        raw_params = mu + sigma * epsilon
    else:
        raw_params = mu
    # Map unconstrained latent variables to physically meaningful intervals.
    V0 = 40.0 + 25.0 * torch.sigmoid(raw_params[0])
    r0 = 1.15 + 0.2 * torch.sigmoid(raw_params[1])
    a = 0.55 + 0.2 * torch.sigmoid(raw_params[2])
    lam_so = 15.0 + 20.0 * torch.sigmoid(raw_params[3])
    params = WSParams(V0=V0, r0=r0, a=a, lam_so=lam_so)
    with torch.no_grad():
        phys_mu_V0 = 40.0 + 25.0 * torch.sigmoid(mu[0])
        phys_sig_V0 = (
            25.0 * torch.sigmoid(mu[0] + sigma[0])
            - 25.0 * torch.sigmoid(mu[0] - sigma[0])
        ) / 2.0
        phys_mu_r0 = 1.15 + 0.2 * torch.sigmoid(mu[1])
        phys_sig_r0 = (
            0.2 * torch.sigmoid(mu[1] + sigma[1])
            - 0.2 * torch.sigmoid(mu[1] - sigma[1])
        ) / 2.0
        phys_mu_a = 0.55 + 0.2 * torch.sigmoid(mu[2])
        phys_sig_a = (
            0.2 * torch.sigmoid(mu[2] + sigma[2])
            - 0.2 * torch.sigmoid(mu[2] - sigma[2])
        ) / 2.0
        phys_mu_lam = 15.0 + 20.0 * torch.sigmoid(mu[3])
        phys_sig_lam = (
            20.0 * torch.sigmoid(mu[3] + sigma[3])
            - 20.0 * torch.sigmoid(mu[3] - sigma[3])
        ) / 2.0
        uncert_dict = {
            "V0": (float(phys_mu_V0), float(phys_sig_V0)),
            "r0": (float(phys_mu_r0), float(phys_sig_r0)),
            "a": (float(phys_mu_a), float(phys_sig_a)),
            "lam_so": (float(phys_mu_lam), float(phys_sig_lam)),
        }
    LE, LR, LTH, LPH, LBC = [torch.tensor(0.0, device=device) for _ in range(5)]
    rows = []
    for i, st in enumerate(states):
        E_target = E_vec[i]
        E_pred = energy_rayleigh_full3d(
            wave_net, E_vec, sample, st, params, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
        )
        LE = LE + (E_pred - E_target) ** 2
        LR = LR + loss_pde_radial_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            params,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            N=512,
        )
        LTH = LTH + loss_pde_theta_full3d(wave_net, E_vec, sample, st, emm=emm, N=512)
        LPH = LPH + loss_pde_phi_full3d(
            wave_net, E_vec, sample, st, emm=emm, N=512, Np_bc=64
        )
        LBC = LBC + loss_bc_full3d(
            wave_net,
            E_vec,
            sample,
            st,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
            Nb=128,
        )
        rows.append((st, float(E_pred.detach().cpu()), float(E_target.detach().cpu())))
    LE, LR, LTH, LPH, LBC = (
        LE / K_states,
        LR / K_states,
        LTH / K_states,
        LPH / K_states,
        LBC / K_states,
    )
    LORTH = loss_orthogonality_full3d(
        wave_net,
        sample,
        E_vec,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        N=4096,
    )
    # Weighted sum of spectral, PDE, boundary, orthogonality and KL terms.
    loss = (
        wE * LE
        + wR * LR
        + wTh * LTH
        + wPh * LPH
        + wBC * LBC
        + wORTH * LORTH
        + wKL * L_KL
    )
    return (loss, LE, LR, LTH, LPH, LBC, LORTH, L_KL, uncert_dict, rows)


# Training and posterior inference

def train_single_nucleus_full3d(
    dataset_path="ws_fd_dataset.npz",
    A = experiment["A"],
    Z = experiment["Z"],
    is_proton=experiment["is_proton"]
    max_states=experiment["max_states"],
    epochs=training["epochs"],
    lr_wave=training["lr_wave"],
    lr_param=training["lr_param"],
    n_r_points = experiment["n_r_points"],
    Nr_norm = experiment["Nr_norm"],
    Nth_norm = experiment["Nth_norm"],
    Nph_norm = experiment["Nph_norm"],
    hidden_wave = training["hidden_wave"],
    hidden_param = training["hidden_param"],
    emm = training["emm"],
    wE = training["wE"],
    wR = training["wR"],
    wTh = training["wTh"],
    wPh = training["wPh"],
    wBC = training["wBC"],
    wORTH = training["wORTH"],
    wKL = training["wKL"],
    use_scheduler_wave = training["use_scheduler_wave"],
    use_scheduler_param = training["use_scheduler_param"],
    gamma_wave = training["gamma_wave"]
    gamma_param = training["gamma_param"]

    step_size_wave = training["step_size_wave"]
    step_size_param = training["step_size_param"]

    print_every = training["print_every"]
):
    """Train WaveNet and ParamNet jointly for one nucleus and nucleon species."""
    dataset = load_fd_dataset(dataset_path)
    sample = get_sample_by_nucleus(
        dataset, A=A, Z=Z, is_proton=is_proton, max_states=max_states
    )
    K_states = len(sample["states"])
    param_input_dim = K_states * n_r_points + K_states + 3 + K_states * 3
    wave_net = WaveNet3D(n_states=K_states, hidden=hidden_wave, depth=5, beta=0.6).to(
        device
    )
    param_net = ProbabilisticParamNet(
        input_dim=param_input_dim, hidden=hidden_param, depth=3
    ).to(device)
    optimizer_wave = torch.optim.Adam(wave_net.parameters(), lr=lr_wave)
    optimizer_param = torch.optim.Adam(param_net.parameters(), lr=lr_param)
    scheduler_wave = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_wave, step_size=step_size_wave, gamma=gamma_wave
        )
        if use_scheduler_wave
        else None
    )
    scheduler_param = (
        torch.optim.lr_scheduler.StepLR(
            optimizer_param, step_size=step_size_param, gamma=gamma_param
        )
        if use_scheduler_param
        else None
    )
    history = []
    print(
        f"\nTraining Probabilistic Inverse PINN: A={A}, Z={Z}, species={('p' if is_proton else 'n')}"
    )
    print(f"States used: {K_states} | wKL: {wKL}")
    print(
        f"\nTraining inverse full-3D PINN: A={A}, Z={Z}, species={('p' if is_proton else 'n')}"
    )
    print(f"States used: {K_states}")
    print(f"ParamNet input dim: {param_input_dim}")
    print(f"m used in angular equations: {emm}")
    print(f"lr_wave={lr_wave}, lr_param={lr_param}")
    print(
        f"use_scheduler_wave={use_scheduler_wave}, gamma_wave={gamma_wave}, step_size_wave={step_size_wave}"
    )
    print(
        f"use_scheduler_param={use_scheduler_param}, gamma_param={gamma_param}, step_size_param={step_size_param}"
    )
    for st in sample["states"]:
        print(f"  nr={st['nr']} l={st['l']} j={st['j']:.1f} E={st['energy']:.6f}")
    for ep in range(1, epochs + 1):
        wave_net.train()
        param_net.train()
        optimizer_wave.zero_grad()
        optimizer_param.zero_grad()
        loss, LE, LR, LTH, LPH, LBC, LORTH, LKL, uncert_dict, rows = (
            compute_sample_loss_full3d(
                wave_net,
                param_net,
                sample,
                n_r_points=n_r_points,
                Nr_norm=Nr_norm,
                Nth_norm=Nth_norm,
                Nph_norm=Nph_norm,
                emm=emm,
                wE=wE,
                wR=wR,
                wTh=wTh,
                wPh=wPh,
                wBC=wBC,
                wORTH=wORTH,
                wKL=wKL,
                is_training=True,
            )
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(wave_net.parameters()) + list(param_net.parameters()), 5.0
        )
        optimizer_wave.step()
        optimizer_param.step()
        if scheduler_wave:
            scheduler_wave.step()
        if scheduler_param:
            scheduler_param.step()
        if ep % print_every == 0 or ep == 1:
            mu_V0, sig_V0 = uncert_dict["V0"]
            mu_r0, sig_r0 = uncert_dict["r0"]
            mu_a, sig_a = uncert_dict["a"]
            mu_lam, sig_lam = uncert_dict["lam_so"]
            print(
                f"[ep={ep:5d}] LOSS={loss.item():.3e} (KL={LKL.item():.2e}) LE={LE.item():.3e} LR={LR.item():.3e} "
            )
            print(
                f"    -> V0: {mu_V0:5.2f} ± {sig_V0:.2f} MeV | r0: {mu_r0:4.3f} ± {sig_r0:.3f} fm | a: {mu_a:4.3f} ± {sig_a:.3f} fm | lam_so: {mu_lam:5.2f} ± {sig_lam:.2f}"
            )
    return (wave_net, param_net, history, sample)


@torch.no_grad()
def infer_parameters_full3d(
    wave_net,
    param_net,
    sample,
    n_samples=1000,
    n_r_points = experiment["n_r_points"],
    Nr_norm = experiment["Nr_norm"],
    Nth_norm = experiment["Nth_norm"],
    Nph_norm = experiment["Nph_norm"],
):
    """Estimate posterior means and standard deviations in physical parameter space."""
    wave_net.eval()
    param_net.eval()
    E_vec = torch.tensor(
        [float(st["energy"]) for st in sample["states"]],
        dtype=torch.float32,
        device=device,
    )
    x_param = make_param_input_from_radial_psi(
        wave_net,
        sample,
        E_vec,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )
    mu, sigma, _ = param_net(x_param)
    eps = torch.randn(n_samples, 4, device=device)
    raw = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    V0 = 40.0 + 25.0 * torch.sigmoid(raw[:, 0])
    r0 = 1.15 + 0.2 * torch.sigmoid(raw[:, 1])
    a = 0.55 + 0.2 * torch.sigmoid(raw[:, 2])
    lam_so = 15.0 + 20.0 * torch.sigmoid(raw[:, 3])
    return {
        "V0": (float(V0.mean()), float(V0.std())),
        "r0": (float(r0.mean()), float(r0.std())),
        "a": (float(a.mean()), float(a.std())),
        "lam_so": (float(lam_so.mean()), float(lam_so.std())),
    }


# Example experiment

if __name__ == "__main__":
    wave_net, param_net, history, sample = train_single_nucleus_full3d(
        dataset_path="ws_fd_dataset.npz",
        A = experiment["A"],
        Z = experiment["Z"],
        is_proton=experiment["is_proton"]
        max_states=experiment["max_states"],
         epochs=training["epochs"],
        lr_wave=training["lr_wave"],
        lr_param=training["lr_param"],
        n_r_points = experiment["n_r_points"],
        Nr_norm = experiment["Nr_norm"],
        Nth_norm = experiment["Nth_norm"],
        Nph_norm = experiment["Nph_norm"],
        hidden_wave = training["hidden_wave"],
        hidden_param = training["hidden_param"],
        emm = training["emm"],
        wE = training["wE"],
        wR = training["wR"],
        wTh = training["wTh"],
        wPh = training["wPh"],
        wBC = training["wBC"],
        wORTH = training["wORTH"],
        wKL = training["wKL"],
        print_every = training["print_every"],
    )
    pred = infer_parameters_full3d(
        wave_net,
        param_net,
        sample,
        n_r_points = experiment["n_r_points"],
        Nr_norm = experiment["Nr_norm"],
        Nth_norm = experiment["Nth_norm"],
        Nph_norm = experiment["Nph_norm"],
    )
    print("\n==============================================")
    print("FINAL INFERRED PARAMETERS WITH UNCERTAINTIES")
    print("==============================================")
    print(f"V0     = {pred['V0'][0]:.4f} ± {pred['V0'][1]:.4f} MeV")
    print(f"r0     = {pred['r0'][0]:.4f} ± {pred['r0'][1]:.4f} fm")
    print(f"a      = {pred['a'][0]:.4f}  ± {pred['a'][1]:.4f} fm")
    print(f"lam_so = {pred['lam_so'][0]:.4f} ± {pred['lam_so'][1]:.4f}")
