import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

# =========================================================
# Device / seeds
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# =========================================================
# Constants
# =========================================================
hc = 197.3269804
u  = 931.49410242
e2 = 1.43996448

R_MAX = 25.0

kappa_fixed = 0.639
r0_so_fixed = 1.16
a_so_fixed  = 0.662

# =========================================================
# Dataset
# =========================================================
def load_fd_dataset(path="ws_fd_dataset.npz"):
    raw = np.load(path, allow_pickle=True)["data"]
    return list(raw)


def get_sample_by_nucleus(dataset, A, Z, is_proton, max_states=6):
    for item in dataset:
        if int(item["A"]) == A and int(item["Z"]) == Z and bool(item["is_proton"]) == bool(is_proton):
            states = sorted(item["states"], key=lambda x: float(x["energy"]))[:max_states]
            return {
                "A": int(A),
                "Z": int(Z),
                "is_proton": bool(is_proton),
                "states": [
                    {
                        "nr": int(st["nr"]),
                        "l": int(st["l"]),
                        "j": float(st["j"]),
                        "energy": float(st["energy"]),
                    }
                    for st in states
                ],
            }

    raise ValueError(f"No sample found for A={A}, Z={Z}, is_proton={is_proton}")


# =========================================================
# Physics helpers
# =========================================================
def reduced_mass(A: int, is_proton: bool):
    m_core = float(A - 1) * u
    m_nucl = (1.007276466621 * u) if is_proton else (1.00866491588 * u)
    return (m_core * m_nucl) / (m_core + m_nucl)


def K_value(mu):
    return hc**2 / (2.0 * mu)


def l_dot_s(l: int, j: float):
    return 0.5 * (j * (j + 1.0) - l * (l + 1.0) - 0.75)


def tdotT_expectation(A: int, Z: int, is_proton: bool):
    N = A - Z

    if N == Z:
        rhs = 3.0
    elif N > Z:
        rhs = ((N - Z + 1) if is_proton else -(N - Z + 1)) + 2.0
    else:
        rhs = ((N - Z - 1) if is_proton else -(N - Z - 1)) + 2.0

    return -0.25 * rhs


@dataclass
class WSParams:
    V0: torch.Tensor
    r0: torch.Tensor
    a: torch.Tensor
    lam_so: torch.Tensor


# =========================================================
# Scaling
# =========================================================
def scale_energy(E):
    return E / 100.0


def scale_r(r):
    return r / R_MAX


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


# =========================================================
# Woods–Saxon potential
# =========================================================
def R_central(A, params: WSParams):
    return params.r0 * (A ** (1.0 / 3.0))


def R_spin_orbit(A):
    return r0_so_fixed * (A ** (1.0 / 3.0))


def f_ws(r, R, a):
    return 1.0 / (1.0 + torch.exp((r - R) / a))


def df_ws_dr(r, R, a):
    x = (r - R) / a
    ex = torch.exp(x)
    return -(ex / (1.0 + ex) ** 2) * (1.0 / a)


def V_depth(A, Z, is_proton, params: WSParams):
    tdotT = tdotT_expectation(A, Z, is_proton)
    return params.V0 * (1.0 - (4.0 * kappa_fixed / float(A)) * tdotT)


def V_central(r, A, Z, is_proton, params: WSParams):
    V = V_depth(A, Z, is_proton, params)
    R = R_central(A, params)
    return -V * f_ws(r, R, params.a)


def V_coulomb(r, A, Z, is_proton, params: WSParams):
    if not is_proton:
        return torch.zeros_like(r)

    Z_core = max(int(Z) - 1, 0)
    R = R_central(A, params)
    rc = r.clamp_min(1e-10)

    inside = rc <= R
    outside = ~inside

    out = torch.empty_like(rc)
    out[inside] = (Z_core * e2 / (2.0 * R)) * (
        3.0 - (rc[inside] / R) ** 2
    )
    out[outside] = (Z_core * e2) / rc[outside]

    return out


def V_spin_orbit(r, A, Z, is_proton, l, j, params: WSParams):
    mu = reduced_mass(A, is_proton)

    rc = r.clamp_min(1e-10)
    Rso = R_spin_orbit(A)

    Vtilde = params.lam_so * params.V0
    dV = Vtilde * df_ws_dr(rc, Rso, a_so_fixed)

    pref = hc**2 / (2.0 * mu**2 * rc)
    return pref * dV * float(l_dot_s(l, j))


def V_eff(r, A, Z, is_proton, l, j, params: WSParams):
    return (
        V_central(r, A, Z, is_proton, params)
        + V_coulomb(r, A, Z, is_proton, params)
        + V_spin_orbit(r, A, Z, is_proton, l, j, params)
    )


# =========================================================
# WaveNet: r + energies + metadata + state -> R(r)
# =========================================================
class WaveNet(nn.Module):
    def __init__(self, n_states, hidden=256, depth=5):
        super().__init__()

        self.n_states = n_states
        input_dim = 1 + n_states + 3 + 3

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def forward(self, r, E_vec, sample, st):
        N = r.shape[0]

        r_scaled = scale_r(r)
        E_scaled = scale_energy(E_vec).unsqueeze(0).repeat(N, 1)

        nuc = scale_nucleus(
            sample["A"],
            sample["Z"],
            sample["is_proton"],
        ).unsqueeze(0).repeat(N, 1)

        q = scale_quantum(
            st["nr"],
            st["l"],
            st["j"],
        ).unsqueeze(0).repeat(N, 1)

        x = torch.cat([r_scaled, E_scaled, nuc, q], dim=1)
        raw = self.net(x)

        rc = r.clamp_min(1e-8)
        beta = 0.6

        return (rc ** int(st["l"])) * torch.exp(-beta * rc) * raw


# =========================================================
# ParamNet: sampled normalized wavefunctions -> parameters
# =========================================================
class ParamNet(nn.Module):
    def __init__(self, input_dim, hidden=256, depth=4):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 4)]

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(m.bias)

    def forward(self, x):
        raw = self.net(x)

        V0 = 30.0 + 50.0 * torch.sigmoid(raw[0])
        r0 = 0.8  + 0.8  * torch.sigmoid(raw[1])
        a  = 0.3  + 0.8  * torch.sigmoid(raw[2])
        lam_so = 5.0 + 45.0 * torch.sigmoid(raw[3])

        return WSParams(V0=V0, r0=r0, a=a, lam_so=lam_so)


# =========================================================
# Autograd
# =========================================================
def d(f, x):
    return torch.autograd.grad(
        f,
        x,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
    )[0]


# =========================================================
# Wavefunction helpers
# =========================================================
def eval_R(wave_net, r, E_vec, sample, st):
    return wave_net(r, E_vec, sample, st)


def norm_scale(wave_net, E_vec, sample, st, Nr=1024):
    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    R = eval_R(wave_net, r, E_vec, sample, st)

    r1 = r.squeeze()
    I = torch.trapz((R.squeeze() ** 2) * r1**2, r1).clamp_min(1e-12)

    return torch.rsqrt(I)


def eval_R_norm(wave_net, r, E_vec, sample, st):
    s = norm_scale(wave_net, E_vec, sample, st)
    return s * eval_R(wave_net, r, E_vec, sample, st)


# =========================================================
# ParamNet input from sampled Ψ
# =========================================================
def make_param_input_from_psi(
    wave_net,
    sample,
    E_vec,
    n_psi_points=96,
):
    """
    ParamNet input:
    - fixed samples of normalized R(r) for every state
    - energies
    - nucleus info
    - quantum numbers

    The r-grid is fixed, not random.
    """
    r = torch.linspace(0.0, R_MAX, n_psi_points, device=device).unsqueeze(1)

    psi_parts = []

    for st in sample["states"]:
        Rn = eval_R_norm(wave_net, r, E_vec, sample, st).squeeze()

        # sign convention: remove R -> -R ambiguity
        sign = torch.sign(Rn[1].detach())
        if sign.item() == 0.0:
            sign = torch.tensor(1.0, device=device)

        Rn = sign * Rn

        psi_parts.append(Rn)

    psi_vec = torch.cat(psi_parts)

    E_scaled = scale_energy(E_vec)

    nuc = scale_nucleus(
        sample["A"],
        sample["Z"],
        sample["is_proton"],
    )

    q_vec = torch.cat([
        scale_quantum(st["nr"], st["l"], st["j"])
        for st in sample["states"]
    ])

    return torch.cat([psi_vec, E_scaled, nuc, q_vec])


# =========================================================
# Rayleigh energy
# =========================================================
def energy_rayleigh(wave_net, E_vec, sample, st, params, Nr=1024):
    A = sample["A"]
    Z = sample["Z"]
    is_proton = sample["is_proton"]

    l = int(st["l"])
    j = float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    r.requires_grad_(True)

    Rn = eval_R_norm(wave_net, r, E_vec, sample, st)
    Rr = d(Rn, r)

    r1 = r.squeeze()
    V = V_eff(r, A, Z, is_proton, l, j, params).squeeze()

    kinetic = K * torch.trapz((Rr.squeeze() ** 2) * r1**2, r1)

    centrifugal = K * l * (l + 1.0) * torch.trapz(
        Rn.squeeze() ** 2,
        r1,
    )

    potential = torch.trapz(
        V * (Rn.squeeze() ** 2) * r1**2,
        r1,
    )

    return kinetic + centrifugal + potential


# =========================================================
# PDE residual
# =========================================================
def loss_pde(wave_net, E_vec, sample, st, params, N=512):
    A = sample["A"]
    Z = sample["Z"]
    is_proton = sample["is_proton"]

    l = int(st["l"])
    j = float(st["j"])

    mu = reduced_mass(A, is_proton)
    K = K_value(mu)

    Ns = N // 2

    r_small = torch.rand(Ns, 1, device=device) * (0.25 * R_MAX)
    r_rest = 0.25 * R_MAX + torch.rand(N - Ns, 1, device=device) * (0.75 * R_MAX)

    r = torch.cat([r_small, r_rest], dim=0)
    r.requires_grad_(True)

    Rn = eval_R_norm(wave_net, r, E_vec, sample, st)
    Rr = d(Rn, r)
    Rrr = d(Rr, r)

    E = energy_rayleigh(wave_net, E_vec, sample, st, params).detach()

    rc = r.clamp_min(1e-8)
    V = V_eff(rc, A, Z, is_proton, l, j, params)

    residual = (
        -K * (rc**2 * Rrr + 2.0 * rc * Rr - l * (l + 1.0) * Rn)
        + ((rc**2) * V - (rc**2) * E) * Rn
    )

    return torch.mean(residual**2)


# =========================================================
# Boundary loss
# =========================================================
def loss_bc(wave_net, E_vec, sample, st, Nb=128):
    l = int(st["l"])

    rR = torch.full((Nb, 1), R_MAX, device=device)
    rR.requires_grad_(True)

    RR = eval_R_norm(wave_net, rR, E_vec, sample, st)
    L_Rmax = torch.mean(RR**2)

    r0 = torch.zeros((Nb, 1), device=device)
    r0.requires_grad_(True)

    R0 = eval_R_norm(wave_net, r0, E_vec, sample, st)

    if l == 0:
        dR0 = d(R0, r0)
        L_0 = torch.mean(dR0**2)
    elif l == 1:
        L_0 = torch.mean(R0**2)
    else:
        dR0 = d(R0, r0)
        L_0 = torch.mean(R0**2) + torch.mean(dR0**2)

    return L_Rmax + L_0


# =========================================================
# Orthogonality loss
# =========================================================
def loss_orthogonality(wave_net, sample, E_vec, Nr=512):
    groups = {}

    for st in sample["states"]:
        key = (int(st["l"]), float(st["j"]))
        groups.setdefault(key, []).append(st)

    total = torch.tensor(0.0, device=device)

    r = torch.linspace(0.0, R_MAX, Nr, device=device).unsqueeze(1)
    r1 = r.squeeze()

    for _, group in groups.items():
        if len(group) < 2:
            continue

        group = sorted(group, key=lambda x: int(x["nr"]))

        Rs = []

        for st in group:
            Rn = eval_R_norm(wave_net, r, E_vec, sample, st).squeeze()
            Rs.append(Rn)

        for i in range(len(Rs)):
            for k in range(i):
                overlap = torch.trapz(Rs[i] * Rs[k] * r1**2, r1)
                total = total + overlap**2

    return total


# =========================================================
# One sample loss
# =========================================================
def compute_sample_loss(
    wave_net,
    param_net,
    sample,
    n_psi_points=96,
    wE=50.0,
    wPDE=5.0,
    wBC=5.0,
    wORTH=10.0,
):
    states = sample["states"]
    K = len(states)

    E_vec = torch.tensor(
        [float(st["energy"]) for st in states],
        dtype=torch.float32,
        device=device,
    )

    x_param = make_param_input_from_psi(
        wave_net,
        sample,
        E_vec,
        n_psi_points=n_psi_points,
    )

    params = param_net(x_param)

    LE = torch.tensor(0.0, device=device)
    LPDE = torch.tensor(0.0, device=device)
    LBC = torch.tensor(0.0, device=device)

    rows = []

    for i, st in enumerate(states):
        E_target = E_vec[i]

        E_pred = energy_rayleigh(
            wave_net,
            E_vec,
            sample,
            st,
            params,
        )

        LE = LE + (E_pred - E_target) ** 2

        LPDE = LPDE + loss_pde(
            wave_net,
            E_vec,
            sample,
            st,
            params,
        )

        LBC = LBC + loss_bc(
            wave_net,
            E_vec,
            sample,
            st,
        )

        rows.append((st, float(E_pred.detach().cpu()), float(E_target.detach().cpu())))

    LE = LE / K
    LPDE = LPDE / K
    LBC = LBC / K

    LORTH = loss_orthogonality(wave_net, sample, E_vec)

    loss = (
        wE * LE
        + wPDE * LPDE
        + wBC * LBC
        + wORTH * LORTH
    )

    return loss, LE, LPDE, LBC, LORTH, params, rows


# =========================================================
# Single nucleus training
# =========================================================
def train_single_nucleus(
    dataset_path="ws_fd_dataset.npz",
    A=56,
    Z=28,
    is_proton=False,
    max_states=6,
    epochs=3000,
    lr=1e-3,
    n_psi_points=96,
    hidden_wave=256,
    hidden_param=256,
    wE=50.0,
    wPDE=5.0,
    wBC=5.0,
    wORTH=10.0,
    print_every=100,
):
    dataset = load_fd_dataset(dataset_path)

    sample = get_sample_by_nucleus(
        dataset,
        A=A,
        Z=Z,
        is_proton=is_proton,
        max_states=max_states,
    )

    K = len(sample["states"])

    param_input_dim = K * n_psi_points + K + 3 + K * 3

    wave_net = WaveNet(
        n_states=K,
        hidden=hidden_wave,
        depth=5,
    ).to(device)

    param_net = ParamNet(
        input_dim=param_input_dim,
        hidden=hidden_param,
        depth=4,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(wave_net.parameters()) + list(param_net.parameters()),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1000,
        gamma=0.5,
    )

    history = []

    print(f"\nTraining single nucleus: A={A}, Z={Z}, species={'p' if is_proton else 'n'}")
    print(f"States used: {K}")
    for st in sample["states"]:
        print(
            f"  nr={st['nr']} l={st['l']} j={st['j']:.1f} "
            f"E={st['energy']:.6f}"
        )

    for ep in range(1, epochs + 1):
        optimizer.zero_grad()

        loss, LE, LPDE, LBC, LORTH, params, rows = compute_sample_loss(
            wave_net,
            param_net,
            sample,
            n_psi_points=n_psi_points,
            wE=wE,
            wPDE=wPDE,
            wBC=wBC,
            wORTH=wORTH,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(wave_net.parameters()) + list(param_net.parameters()),
            5.0,
        )

        optimizer.step()
        scheduler.step()

        rec = {
            "epoch": ep,
            "loss": float(loss.item()),
            "LE": float(LE.item()),
            "LPDE": float(LPDE.item()),
            "LBC": float(LBC.item()),
            "LORTH": float(LORTH.item()),
            "V0": float(params.V0.detach().cpu()),
            "r0": float(params.r0.detach().cpu()),
            "a": float(params.a.detach().cpu()),
            "lam_so": float(params.lam_so.detach().cpu()),
        }

        history.append(rec)

        if ep % print_every == 0 or ep == 1:
            print(
                f"[ep={ep:5d}] "
                f"LOSS={rec['loss']:.3e} "
                f"LE={rec['LE']:.3e} "
                f"LPDE={rec['LPDE']:.3e} "
                f"LBC={rec['LBC']:.3e} "
                f"LORTH={rec['LORTH']:.3e} | "
                f"V0={rec['V0']:.4f} "
                f"r0={rec['r0']:.4f} "
                f"a={rec['a']:.4f} "
                f"lam_so={rec['lam_so']:.4f}"
            )

            for st, epred, etar in rows:
                print(
                    f"    nr={st['nr']} l={st['l']} j={st['j']:.1f} "
                    f"E_pred={epred:.6f} E_target={etar:.6f}"
                )

    return wave_net, param_net, history, sample


# =========================================================
# Inference
# =========================================================
@torch.no_grad()
def infer_parameters(
    wave_net,
    param_net,
    sample,
    n_psi_points=96,
):
    E_vec = torch.tensor(
        [float(st["energy"]) for st in sample["states"]],
        dtype=torch.float32,
        device=device,
    )

    x_param = make_param_input_from_psi(
        wave_net,
        sample,
        E_vec,
        n_psi_points=n_psi_points,
    )

    params = param_net(x_param)

    return {
        "V0": float(params.V0.detach().cpu()),
        "r0": float(params.r0.detach().cpu()),
        "a": float(params.a.detach().cpu()),
        "lam_so": float(params.lam_so.detach().cpu()),
    }


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    wave_net, param_net, history, sample = train_single_nucleus(
        dataset_path="ws_fd_dataset.npz",
        A=56,
        Z=28,
        is_proton=False,
        max_states=6,
        epochs=7000,
        lr=5e-4,
        n_psi_points=96,
        hidden_wave=128,
        hidden_param=128,
        wE=50.0,
        wPDE=5.0,
        wBC=5.0,
        wORTH=10.0,
        print_every=100,
    )

    pred = infer_parameters(
        wave_net,
        param_net,
        sample,
        n_psi_points=96,
    )

    print("\nFinal inferred parameters:")
    print(f"V0     = {pred['V0']:.6f} MeV")
    print(f"r0     = {pred['r0']:.6f} fm")
    print(f"a      = {pred['a']:.6f} fm")
    print(f"lam_so = {pred['lam_so']:.6f}")