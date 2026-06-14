"""
Training, inference and diagnostic utilities for multi-nucleus inverse
Woods-Saxon PINN experiments with Weights & Biases logging.
"""

import os
import json
import time
import math
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False


# Import the base PINN implementation when this module is used standalone
if "WaveNet3D" not in globals():
    module_name = os.environ.get("PINN_BASE_MODULE", None)
    if module_name:
        base = importlib.import_module(module_name)
        for name in [
            "device", "R_MAX", "PI", "TWOPI", "WSParams",
            "load_fd_dataset", "get_sample_by_nucleus",
            "WaveNet3D", "ProbabilisticParamNet",
            "compute_sample_loss_full3d",
            "make_param_input_from_radial_psi",
            "eval_R", "eval_Theta", "eval_Phi",
            "psi_scale_only", "eval_psi_norm",
            "energy_rayleigh_full3d",
        ]:
            globals()[name] = getattr(base, name)
    else:
        print(
            "WARNING: Base PINN symbols are unavailable. "
            "Set PINN_BASE_MODULE or execute this code after the base model."
        )


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


def nucleus_tag(sample):
    species = "p" if bool(sample["is_proton"]) else "n"
    return f"A{int(sample['A'])}_Z{int(sample['Z'])}_{species}"


def state_label(st):
    return f"nr={int(st['nr'])}, l={int(st['l'])}, j={float(st['j']):.1f}"


def safe_label(text):
    return (
        str(text)
        .replace("=", "")
        .replace(",", "")
        .replace(" ", "_")
        .replace(".", "p")
    )


def wb_log(payload, step=None):
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(payload, step=step)


# Parameter inference and diagnostics
@torch.no_grad()
def infer_param_stats_mc(
    wave_net,
    param_net,
    sample,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    n_mc_samples=2000,
):
    """Monte-Carlo uncertainty in physical parameter space."""
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
    eps = torch.randn(n_mc_samples, 4, device=device)
    raw = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

    V0 = 40.0 + 25.0 * torch.sigmoid(raw[:, 0])
    r0 = 1.15 + 0.20 * torch.sigmoid(raw[:, 1])
    a = 0.55 + 0.20 * torch.sigmoid(raw[:, 2])
    lam_so = 15.0 + 20.0 * torch.sigmoid(raw[:, 3])

    return {
        "V0_mu": float(V0.mean().cpu()),
        "V0_sigma": float(V0.std(unbiased=False).cpu()),
        "r0_mu": float(r0.mean().cpu()),
        "r0_sigma": float(r0.std(unbiased=False).cpu()),
        "a_mu": float(a.mean().cpu()),
        "a_sigma": float(a.std(unbiased=False).cpu()),
        "lam_so_mu": float(lam_so.mean().cpu()),
        "lam_so_sigma": float(lam_so.std(unbiased=False).cpu()),
    }


def wsparams_from_stats(stats):
    return WSParams(
        V0=torch.tensor(stats["V0_mu"], dtype=torch.float32, device=device),
        r0=torch.tensor(stats["r0_mu"], dtype=torch.float32, device=device),
        a=torch.tensor(stats["a_mu"], dtype=torch.float32, device=device),
        lam_so=torch.tensor(stats["lam_so_mu"], dtype=torch.float32, device=device),
    )


@torch.no_grad()
def make_param_input_from_random_radial_psi(
    wave_net,
    sample,
    E_vec,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    r_min=0.0,
    r_max=R_MAX,
):
    """
    Builds the ParamNet input using randomly sampled radial points.

    This is similar to make_param_input_from_radial_psi(...), but instead of
    using a fixed linspace grid, it samples a new random sorted radial grid.
    """

    wave_net.eval()

    # Sample an unseen radial grid for robustness testing
    r = r_min + (r_max - r_min) * torch.rand(n_r_points, 1, device=device)
    r, _ = torch.sort(r, dim=0)

    radial_parts = []

    for st in sample["states"]:
        _, _, _, _, _, s = psi_scale_only(
            wave_net,
            E_vec,
            sample,
            st,
            Nr=Nr_norm,
            Nth=Nth_norm,
            Nph=Nph_norm,
        )

        Rn = (s * eval_R(wave_net, r, E_vec, sample, st)).squeeze()

        # Use a consistent phase convention across probe sets
        idx = min(5, Rn.numel() - 1)
        sign = torch.sign(Rn[idx].detach())

        if sign.item() == 0.0:
            sign = torch.tensor(1.0, device=device)

        Rn = sign * Rn
        radial_parts.append(Rn)

    psi_vec = torch.cat(radial_parts)

    E_scaled = scale_energy(E_vec)

    nuc = scale_nucleus(
        sample["A"],
        sample["Z"],
        sample["is_proton"],
    )

    q_vec = torch.cat(
        [
            scale_quantum(st["nr"], st["l"], st["j"])
            for st in sample["states"]
        ]
    )

    x_param = torch.cat([psi_vec, E_scaled, nuc, q_vec])

    return x_param, r.squeeze().detach().cpu().numpy()

@torch.no_grad()
def infer_param_stats_random_radial_probes(
    wave_net,
    param_net,
    sample,
    n_probe_sets=20,
    n_r_points=96,
    Nr_norm=1024,
    Nth_norm=512,
    Nph_norm=512,
    n_mc_samples=1000,
):
    """
    Repeats parameter inference using different randomly sampled radial grids.

    This evaluates how sensitive the learned parameter inference is to the
    radial sampling points used to construct the ParamNet input.
    """

    wave_net.eval()
    param_net.eval()

    E_vec = torch.tensor(
        [float(st["energy"]) for st in sample["states"]],
        dtype=torch.float32,
        device=device,
    )

    rows = []

    for probe_id in range(n_probe_sets):
        x_param, r_probe = make_param_input_from_random_radial_psi(
            wave_net,
            sample,
            E_vec,
            n_r_points=n_r_points,
            Nr_norm=Nr_norm,
            Nth_norm=Nth_norm,
            Nph_norm=Nph_norm,
        )

        mu, sigma, _ = param_net(x_param)

        eps = torch.randn(n_mc_samples, 4, device=device)
        raw = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

        V0 = 40.0 + 25.0 * torch.sigmoid(raw[:, 0])
        r0 = 1.15 + 0.20 * torch.sigmoid(raw[:, 1])
        a = 0.55 + 0.20 * torch.sigmoid(raw[:, 2])
        lam_so = 15.0 + 20.0 * torch.sigmoid(raw[:, 3])

        rows.append(
            {
                "probe_id": probe_id,
                "V0_mu": float(V0.mean().cpu()),
                "V0_sigma": float(V0.std(unbiased=False).cpu()),
                "r0_mu": float(r0.mean().cpu()),
                "r0_sigma": float(r0.std(unbiased=False).cpu()),
                "a_mu": float(a.mean().cpu()),
                "a_sigma": float(a.std(unbiased=False).cpu()),
                "lam_so_mu": float(lam_so.mean().cpu()),
                "lam_so_sigma": float(lam_so.std(unbiased=False).cpu()),
                "r_min": float(np.min(r_probe)),
                "r_max": float(np.max(r_probe)),
            }
        )

    df = pd.DataFrame(rows)

    summary = {}

    for p in ["V0", "r0", "a", "lam_so"]:
        summary[f"{p}_probe_mean"] = float(df[f"{p}_mu"].mean())
        summary[f"{p}_probe_std"] = float(df[f"{p}_mu"].std(ddof=0))
        summary[f"{p}_mean_uncertainty"] = float(df[f"{p}_sigma"].mean())

    return df, summary

# Plot 1: parameter values vs epochs
def plot_parameter_history(history, out_dir, tag, log_wandb=True):
    df = pd.DataFrame(history)
    if df.empty:
        return []

    out_dir = ensure_dir(out_dir)
    paths = []

    for p, unit in [("V0", "MeV"), ("r0", "fm"), ("a", "fm"), ("lam_so", "")]:
        mu_col = f"{p}_mu"
        sig_col = f"{p}_sigma"
        if mu_col not in df:
            continue

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        x = df["epoch"].to_numpy()
        mu = df[mu_col].to_numpy()
        ax.plot(x, mu, linewidth=2.0, label=f"{p} mean")

        if sig_col in df:
            sig = df[sig_col].to_numpy()
            ax.fill_between(x, mu - sig, mu + sig, alpha=0.25, label=r"$\pm1\sigma$")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{p} ({unit})" if unit else p)
        ax.set_title(f"{tag}: {p} evolution")
        ax.grid(True, alpha=0.3)
        ax.legend()
        path = save_fig(fig, out_dir / f"{tag}_param_{p}_vs_epoch.png")
        paths.append(path)

        if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({f"plots/parameters/{p}_vs_epoch": wandb.Image(path)})

    return paths


def plot_loss_history(history, out_dir, tag, log_wandb=True):
    df = pd.DataFrame(history)
    if df.empty:
        return None

    out_dir = ensure_dir(out_dir)
    cols = [c for c in ["loss", "LE", "LR", "LTH", "LPH", "LBC", "LORTH", "LKL"] if c in df]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for c in cols:
        y = np.maximum(df[c].to_numpy(dtype=float), 1e-30)
        ax.plot(df["epoch"], y, linewidth=1.7, label=c)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")
    ax.set_title(f"{tag}: training losses")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    path = save_fig(fig, out_dir / f"{tag}_loss_components.png")

    if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"plots/loss_components": wandb.Image(path)})

    return path


# Plot 2: full Psi^2 vs r, theta, phi
@torch.no_grad()
def collect_full_psi2_data(
    wave_net,
    sample,
    n_r=600,
    n_th=360,
    n_ph=360,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    theta_ref=None,
    phi_ref=0.0,
):
    """
    Full |Psi|^2 one-dimensional diagnostics.

    Because |Psi(r,theta,phi)|^2 depends on all 3 variables, this produces:

    A. Slice curves:
       |Psi(r, theta0, phi0)|^2 vs r
       |Psi(r0, theta, phi0)|^2 vs theta
       |Psi(r0, theta0, phi)|^2 vs phi

    B. Marginal probability densities:
       P(r)     = integral |Psi|^2 r^2 sin(theta) dtheta dphi
       P(theta) = integral |Psi|^2 r^2 sin(theta) dr dphi
       P(phi)   = integral |Psi|^2 r^2 sin(theta) dr dtheta

    For each state, r0 is chosen as the peak of P(r).
    """
    if theta_ref is None:
        theta_ref = PI / 2.0

    wave_net.eval()

    states = sample["states"]
    E_vec = torch.tensor([float(st["energy"]) for st in states], dtype=torch.float32, device=device)

    r = torch.linspace(0.0, R_MAX, n_r, device=device).unsqueeze(1)
    th = torch.linspace(1e-5, PI - 1e-5, n_th, device=device).unsqueeze(1)
    ph = torch.linspace(0.0, TWOPI, n_ph + 1, device=device)[:-1].unsqueeze(1)

    r1 = r.squeeze()
    th1 = th.squeeze()
    ph1 = ph.squeeze()

    data = {
        "r": r1.cpu().numpy(),
        "theta": th1.cpu().numpy(),
        "phi": ph1.cpu().numpy(),
        "states": [],
    }

    for st in states:
        # Normalize the full separable wavefunction.
        _, _, _, _, _, s = psi_scale_only(
            wave_net, E_vec, sample, st, Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm
        )

        # Evaluate separable components used in the marginal densities.
        Rn = (s * eval_R(wave_net, r, E_vec, sample, st)).squeeze()
        Th = eval_Theta(wave_net, th, E_vec, sample, st).squeeze()
        Ph_re, Ph_im = eval_Phi(wave_net, ph, E_vec, sample, st)
        Ph_re = Ph_re.squeeze()
        Ph_im = Ph_im.squeeze()

        R2 = Rn**2
        Th2 = Th**2
        Ph2 = Ph_re**2 + Ph_im**2
        sinth = torch.sin(th1).clamp_min(1e-8)

        IR = torch.trapz(R2 * r1**2, r1)
        Ith = torch.trapz(Th2 * sinth, th1)
        Iphi = torch.trapz(Ph2, ph1)

        P_r = (R2 * r1**2 * Ith * Iphi).cpu().numpy()
        P_theta = (Th2 * sinth * IR * Iphi).cpu().numpy()
        P_phi = (Ph2 * IR * Ith).cpu().numpy()

        # Reference radius: peak of marginal radial distribution.
        r_np = r1.cpu().numpy()
        r_ref = float(r_np[int(np.argmax(P_r))])

        # Slice |Psi(r, theta0, phi0)|^2
        theta0 = torch.full_like(r, float(theta_ref))
        phi0 = torch.full_like(r, float(phi_ref))
        Re_r, Im_r, _ = eval_psi_norm(
            wave_net, r, theta0, phi0, E_vec, sample, st,
            Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm,
        )
        psi2_r = (Re_r.squeeze()**2 + Im_r.squeeze()**2).cpu().numpy()

        # Slice |Psi(r0, theta, phi0)|^2
        r0_th = torch.full_like(th, r_ref)
        phi0_th = torch.full_like(th, float(phi_ref))
        Re_th, Im_th, _ = eval_psi_norm(
            wave_net, r0_th, th, phi0_th, E_vec, sample, st,
            Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm,
        )
        psi2_theta = (Re_th.squeeze()**2 + Im_th.squeeze()**2).cpu().numpy()

        # Slice |Psi(r0, theta0, phi)|^2
        r0_ph = torch.full_like(ph, r_ref)
        theta0_ph = torch.full_like(ph, float(theta_ref))
        Re_ph, Im_ph, _ = eval_psi_norm(
            wave_net, r0_ph, theta0_ph, ph, E_vec, sample, st,
            Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm,
        )
        psi2_phi = (Re_ph.squeeze()**2 + Im_ph.squeeze()**2).cpu().numpy()

        data["states"].append({
            "label": state_label(st),
            "state": dict(st),
            "r_ref": r_ref,
            "theta_ref": float(theta_ref),
            "phi_ref": float(phi_ref),
            "psi2_vs_r_slice": psi2_r,
            "psi2_vs_theta_slice": psi2_theta,
            "psi2_vs_phi_slice": psi2_phi,
            "P_r": P_r,
            "P_theta": P_theta,
            "P_phi": P_phi,
        })

    return data


def plot_full_psi2_curves(
    wave_net,
    sample,
    out_dir,
    tag,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    theta_ref=None,
    phi_ref=0.0,
    log_wandb=True,
):
    """Creates |Psi|^2 vs r/theta/phi slices and marginal plots."""
    out_dir = ensure_dir(out_dir)
    data = collect_full_psi2_data(
        wave_net,
        sample,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        theta_ref=theta_ref,
        phi_ref=phi_ref,
    )

    specs = [
        ("psi2_vs_r_slice", "r", "r (fm)", r"$|\Psi(r,\theta_0,\phi_0)|^2$", "psi2_slice_vs_r", "full wavefunction slice vs r"),
        ("psi2_vs_theta_slice", "theta", r"$\theta$ (rad)", r"$|\Psi(r_0,\theta,\phi_0)|^2$", "psi2_slice_vs_theta", "full wavefunction slice vs theta"),
        ("psi2_vs_phi_slice", "phi", r"$\phi$ (rad)", r"$|\Psi(r_0,\theta_0,\phi)|^2$", "psi2_slice_vs_phi", "full wavefunction slice vs phi"),
        ("P_r", "r", "r (fm)", r"$P(r)$", "psi2_marginal_vs_r", "marginal probability density vs r"),
        ("P_theta", "theta", r"$\theta$ (rad)", r"$P(\theta)$", "psi2_marginal_vs_theta", "marginal probability density vs theta"),
        ("P_phi", "phi", r"$\phi$ (rad)", r"$P(\phi)$", "psi2_marginal_vs_phi", "marginal probability density vs phi"),
    ]

    paths = []
    for ykey, xkey, xlabel, ylabel, fname, title in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for item in data["states"]:
            ax.plot(data[xkey], item[ykey], linewidth=1.6, label=item["label"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{tag}: {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        path = save_fig(fig, out_dir / f"{tag}_{fname}.png")
        paths.append(path)
        if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({f"plots/full_psi/{fname}": wandb.Image(path)})

    return paths


# Plot 3: heatmaps of full Psi^2(theta, phi)
@torch.no_grad()
def plot_full_psi2_theta_phi_heatmaps(
    wave_net,
    sample,
    out_dir,
    tag,
    n_th=220,
    n_ph=300,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    log_wandb=True,
):
    """
    Heatmap of |Psi(r0, theta, phi)|^2 for each selected state.
    r0 is chosen as the peak of P(r) for that state.
    """
    out_dir = ensure_dir(out_dir)
    wave_net.eval()

    states = sample["states"]
    E_vec = torch.tensor([float(st["energy"]) for st in states], dtype=torch.float32, device=device)

    th = torch.linspace(1e-5, PI - 1e-5, n_th, device=device).unsqueeze(1)
    ph = torch.linspace(0.0, TWOPI, n_ph + 1, device=device)[:-1].unsqueeze(1)
    TH, PH = torch.meshgrid(th.squeeze(), ph.squeeze(), indexing="ij")
    th_flat = TH.reshape(-1, 1)
    ph_flat = PH.reshape(-1, 1)

    # Use the peak of the radial marginal as the reference radius.
    curve_data = collect_full_psi2_data(
        wave_net,
        sample,
        n_r=600,
        n_th=180,
        n_ph=180,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )

    paths = []
    for idx, st in enumerate(states):
        r_ref = float(curve_data["states"][idx]["r_ref"])
        r_flat = torch.full_like(th_flat, r_ref)

        Re, Im, _ = eval_psi_norm(
            wave_net, r_flat, th_flat, ph_flat, E_vec, sample, st,
            Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm,
        )
        psi2 = (Re.squeeze()**2 + Im.squeeze()**2).reshape(n_th, n_ph).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6.8, 5.0))
        im = ax.imshow(
            psi2,
            origin="lower",
            aspect="auto",
            extent=[0.0, 2.0 * math.pi, 0.0, math.pi],
        )
        ax.set_xlabel(r"$\phi$ (rad)")
        ax.set_ylabel(r"$\theta$ (rad)")
        ax.set_title(f"{tag}: $|\\Psi(r_0,\\theta,\\phi)|^2$, {state_label(st)}, r0={r_ref:.2f} fm")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"$|\Psi(r_0,\theta,\phi)|^2$")

        path = save_fig(fig, out_dir / f"{tag}_full_psi2_theta_phi_heatmap_state{idx}_{safe_label(state_label(st))}.png")
        paths.append(path)
        if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({f"plots/full_psi_heatmaps/state_{idx}": wandb.Image(path)})

    return paths


def plot_random_probe_parameter_inference(
    probe_df,
    out_dir,
    tag,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)

    paths = []

    for p, unit in [
        ("V0", "MeV"),
        ("r0", "fm"),
        ("a", "fm"),
        ("lam_so", ""),
    ]:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))

        x = probe_df["probe_id"].to_numpy()
        mu = probe_df[f"{p}_mu"].to_numpy()
        sig = probe_df[f"{p}_sigma"].to_numpy()

        ax.errorbar(
            x,
            mu,
            yerr=sig,
            fmt="o",
            capsize=4,
            linewidth=1.5,
        )

        ax.axhline(mu.mean(), linestyle="--", linewidth=1.2)

        ax.set_xlabel("Random radial probe set")
        ax.set_ylabel(f"{p} ({unit})" if unit else p)
        ax.set_title(f"{tag}: {p} inference under random radial probes")
        ax.grid(True, alpha=0.3)

        path = save_fig(
            fig,
            out_dir / f"{tag}_{p}_random_radial_probe_inference.png",
        )

        paths.append(path)

        if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(
                {
                    f"plots/random_probe_inference/{p}": wandb.Image(path)
                }
            )

    return paths

# Extra supporting diagnostics: energy and orthogonality
def compute_energy_table(
    wave_net,
    param_net,
    sample,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
):
    """Uses gradients because energy_rayleigh_full3d calls autograd."""
    stats = infer_param_stats_mc(
        wave_net,
        param_net,
        sample,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        n_mc_samples=1000,
    )
    params = wsparams_from_stats(stats)

    E_vec = torch.tensor([float(st["energy"]) for st in sample["states"]], dtype=torch.float32, device=device)
    rows = []
    wave_net.eval()
    param_net.eval()

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
        E_pred = float(E_pred.detach().cpu())
        rows.append({
            "nr": int(st["nr"]),
            "l": int(st["l"]),
            "j": float(st["j"]),
            "E_target": E_target,
            "E_pred": E_pred,
            "residual": E_pred - E_target,
        })

    return pd.DataFrame(rows)


def plot_energy_diagnostics(
    wave_net,
    param_net,
    sample,
    out_dir,
    tag,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    log_wandb=True,
):
    out_dir = ensure_dir(out_dir)
    df = compute_energy_table(
        wave_net,
        param_net,
        sample,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
    )
    df.to_csv(out_dir / f"{tag}_energy_table.csv", index=False)

    rmse = float(np.sqrt(np.mean(df["residual"].values**2)))
    mae = float(np.mean(np.abs(df["residual"].values)))

    fig, ax = plt.subplots(figsize=(5.4, 5.1))
    ax.scatter(df["E_target"], df["E_pred"], s=52)
    lo = min(float(df["E_target"].min()), float(df["E_pred"].min()))
    hi = max(float(df["E_target"].max()), float(df["E_pred"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4)
    ax.set_xlabel(r"Target energy $E_{target}$ (MeV)")
    ax.set_ylabel(r"PINN energy $E_{PINN}$ (MeV)")
    ax.set_title(f"{tag}: predicted vs target energies")
    ax.grid(True, alpha=0.3)
    p1 = save_fig(fig, out_dir / f"{tag}_energy_pred_vs_target.png")

    labels = [f"{int(row.nr)},{int(row.l)},{row.j:.1f}" for _, row in df.iterrows()]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(np.arange(len(df)), df["residual"].values)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel(r"state $(n_r,l,j)$")
    ax.set_ylabel(r"$E_{PINN} - E_{target}$ (MeV)")
    ax.set_title(f"{tag}: energy residuals")
    ax.grid(True, axis="y", alpha=0.3)
    p2 = save_fig(fig, out_dir / f"{tag}_energy_residuals.png")

    if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "final/energy_rmse": rmse,
            "final/energy_mae": mae,
            "plots/energy/pred_vs_target": wandb.Image(p1),
            "plots/energy/residuals": wandb.Image(p2),
            "tables/energy_table": wandb.Table(dataframe=df),
        })

    return {"rmse": rmse, "mae": mae, "table": df, "paths": [p1, p2]}


@torch.no_grad()
def compute_overlap_matrix(wave_net, sample, Nr_norm=512, Nth_norm=256, Nph_norm=256, N=12000):
    wave_net.eval()
    states = sample["states"]
    K = len(states)
    E_vec = torch.tensor([float(st["energy"]) for st in states], dtype=torch.float32, device=device)

    r = torch.rand(N, 1, device=device) * R_MAX
    th = torch.rand(N, 1, device=device) * PI
    ph = torch.rand(N, 1, device=device) * TWOPI
    w = r.squeeze()**2 * torch.sin(th.squeeze()).clamp_min(1e-12)
    vol = R_MAX * PI * TWOPI

    psis = []
    for st in states:
        Re, Im, _ = eval_psi_norm(
            wave_net, r, th, ph, E_vec, sample, st,
            Nr=Nr_norm, Nth=Nth_norm, Nph=Nph_norm,
        )
        psis.append((Re.squeeze(), Im.squeeze()))

    O_re = torch.zeros(K, K, device=device)
    O_im = torch.zeros(K, K, device=device)
    for i in range(K):
        Re_i, Im_i = psis[i]
        for j in range(K):
            Re_j, Im_j = psis[j]
            ip_re = Re_i * Re_j + Im_i * Im_j
            ip_im = Re_i * Im_j - Im_i * Re_j
            O_re[i, j] = vol * torch.mean(ip_re * w)
            O_im[i, j] = vol * torch.mean(ip_im * w)

    return O_re.cpu().numpy(), O_im.cpu().numpy()


def plot_overlap_matrix(wave_net, sample, out_dir, tag, Nr_norm=512, Nth_norm=256, Nph_norm=256, log_wandb=True):
    out_dir = ensure_dir(out_dir)
    O_re, O_im = compute_overlap_matrix(wave_net, sample, Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm)
    O_abs = np.sqrt(O_re**2 + O_im**2)
    labels = [f"{int(st['nr'])},{int(st['l'])},{float(st['j']):.1f}" for st in sample["states"]]

    fig, ax = plt.subplots(figsize=(5.9, 5.1))
    im = ax.imshow(O_abs, origin="lower", vmin=0.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"state $(n_r,l,j)$")
    ax.set_ylabel(r"state $(n_r,l,j)$")
    ax.set_title(f"{tag}: overlap matrix")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|\langle \psi_i | \psi_j \rangle|$")
    path = save_fig(fig, out_dir / f"{tag}_overlap_matrix.png")

    off = O_abs.copy()
    np.fill_diagonal(off, 0.0)
    max_off = float(off.max())

    if log_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "final/max_abs_offdiag_overlap": max_off,
            "plots/orthogonality/overlap_matrix": wandb.Image(path),
        })

    return {"path": path, "max_abs_offdiag_overlap": max_off, "O_abs": O_abs}


# Main training function with logging and final plots
def train_single_nucleus_instrumented(
    dataset_path="ws_fd_dataset.npz",
    A=56,
    Z=28,
    is_proton=False,
    max_states=6,
    epochs=3000,
    lr_wave=5e-4,
    lr_param=1e-3,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    hidden_wave=128,
    hidden_param=128,
    emm=0,
    wE=0.5,
    wR=10.0,
    wTh=5.0,
    wPh=5.0,
    wBC=5.0,
    wORTH=5.0,
    wKL=1e-3,
    use_scheduler_wave=True,
    use_scheduler_param=True,
    gamma_wave=0.6,
    gamma_param=0.5,
    step_size_wave=1000,
    step_size_param=1500,
    print_every=100,
    log_every=100,
    plot_every=None,
    out_root="paper_outputs",
    wandb_project="inverse-ws-pinn",
    wandb_group=None,
    use_wandb=True,
):
    dataset = load_fd_dataset(dataset_path)
    sample = get_sample_by_nucleus(dataset, A=A, Z=Z, is_proton=is_proton, max_states=max_states)
    tag = nucleus_tag(sample)
    out_dir = ensure_dir(Path(out_root) / tag)
    K_states = len(sample["states"])
    param_input_dim = K_states * n_r_points + K_states + 3 + K_states * 3

    wave_net = WaveNet3D(n_states=K_states, hidden=hidden_wave, depth=5, beta=0.6).to(device)
    param_net = ProbabilisticParamNet(input_dim=param_input_dim, hidden=hidden_param, depth=4).to(device)

    optimizer_wave = torch.optim.Adam(wave_net.parameters(), lr=lr_wave)
    optimizer_param = torch.optim.Adam(param_net.parameters(), lr=lr_param)

    scheduler_wave = torch.optim.lr_scheduler.StepLR(optimizer_wave, step_size=step_size_wave, gamma=gamma_wave) if use_scheduler_wave else None
    scheduler_param = torch.optim.lr_scheduler.StepLR(optimizer_param, step_size=step_size_param, gamma=gamma_param) if use_scheduler_param else None

    config = dict(
        A=A, Z=Z, is_proton=bool(is_proton), species="p" if is_proton else "n",
        max_states=max_states, K_states=K_states, epochs=epochs,
        lr_wave=lr_wave, lr_param=lr_param, n_r_points=n_r_points,
        Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm,
        hidden_wave=hidden_wave, hidden_param=hidden_param,
        emm=emm, wE=wE, wR=wR, wTh=wTh, wPh=wPh, wBC=wBC, wORTH=wORTH, wKL=wKL,
        param_input_dim=param_input_dim,
    )

    if use_wandb and WANDB_AVAILABLE:
        wandb.init(project=wandb_project, group=wandb_group, name=tag, config=config, reinit=True)
        wandb.log({"tables/states_used": wandb.Table(dataframe=pd.DataFrame(sample["states"]))})

    print("\n==============================================")
    print(f"Training {tag}")
    print("==============================================")
    print(json.dumps(config, indent=2))
    print("States:")
    for st in sample["states"]:
        print(f"  {state_label(st)} E={float(st['energy']):.6f}")

    history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        wave_net.train()
        param_net.train()
        optimizer_wave.zero_grad()
        optimizer_param.zero_grad()

        loss, LE, LR, LTH, LPH, LBC, LORTH, LKL, uncert_dict, rows = compute_sample_loss_full3d(
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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(wave_net.parameters()) + list(param_net.parameters()), 5.0)
        optimizer_wave.step()
        optimizer_param.step()

        if scheduler_wave:
            scheduler_wave.step()
        if scheduler_param:
            scheduler_param.step()

        if ep == 1 or ep % log_every == 0 or ep == epochs:
            row = {
                "epoch": ep,
                "time_sec": time.time() - t0,
                "loss": float(loss.detach().cpu()),
                "LE": float(LE.detach().cpu()),
                "LR": float(LR.detach().cpu()),
                "LTH": float(LTH.detach().cpu()),
                "LPH": float(LPH.detach().cpu()),
                "LBC": float(LBC.detach().cpu()),
                "LORTH": float(LORTH.detach().cpu()),
                "LKL": float(LKL.detach().cpu()),
                "V0_mu": uncert_dict["V0"][0],
                "V0_sigma": uncert_dict["V0"][1],
                "r0_mu": uncert_dict["r0"][0],
                "r0_sigma": uncert_dict["r0"][1],
                "a_mu": uncert_dict["a"][0],
                "a_sigma": uncert_dict["a"][1],
                "lam_so_mu": uncert_dict["lam_so"][0],
                "lam_so_sigma": uncert_dict["lam_so"][1],
                "lr_wave": optimizer_wave.param_groups[0]["lr"],
                "lr_param": optimizer_param.param_groups[0]["lr"],
            }
            history.append(row)

            wb_log({
                "epoch": ep,
                "loss/total": row["loss"],
                "loss/LE_energy": row["LE"],
                "loss/LR_radial": row["LR"],
                "loss/LTH_theta": row["LTH"],
                "loss/LPH_phi": row["LPH"],
                "loss/LBC_boundary": row["LBC"],
                "loss/LORTH_orthogonality": row["LORTH"],
                "loss/LKL": row["LKL"],
                "params/V0_mu": row["V0_mu"],
                "params/V0_sigma": row["V0_sigma"],
                "params/r0_mu": row["r0_mu"],
                "params/r0_sigma": row["r0_sigma"],
                "params/a_mu": row["a_mu"],
                "params/a_sigma": row["a_sigma"],
                "params/lam_so_mu": row["lam_so_mu"],
                "params/lam_so_sigma": row["lam_so_sigma"],
                "lr/wave": row["lr_wave"],
                "lr/param": row["lr_param"],
                "time/sec": row["time_sec"],
            }, step=ep)

        if ep == 1 or ep % print_every == 0 or ep == epochs:
            print(
                f"[{tag} ep={ep:5d}] LOSS={loss.item():.3e} "
                f"LE={LE.item():.3e} LR={LR.item():.3e} LTH={LTH.item():.3e} "
                f"LPH={LPH.item():.3e} LBC={LBC.item():.3e} LORTH={LORTH.item():.3e} LKL={LKL.item():.3e}"
            )
            print(
                f"    V0={uncert_dict['V0'][0]:.4f}±{uncert_dict['V0'][1]:.4f}, "
                f"r0={uncert_dict['r0'][0]:.4f}±{uncert_dict['r0'][1]:.4f}, "
                f"a={uncert_dict['a'][0]:.4f}±{uncert_dict['a'][1]:.4f}, "
                f"lam_so={uncert_dict['lam_so'][0]:.4f}±{uncert_dict['lam_so'][1]:.4f}"
            )

        if plot_every is not None and ep % plot_every == 0:
            plot_parameter_history(history, out_dir, tag, log_wandb=True)
            plot_loss_history(history, out_dir, tag, log_wandb=True)

    # Save training history and final diagnostics.
    hist_df = pd.DataFrame(history)
    hist_path = out_dir / f"{tag}_training_history.csv"
    hist_df.to_csv(hist_path, index=False)

    final_stats = infer_param_stats_mc(
        wave_net,
        param_net,
        sample,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        n_mc_samples=2000,
    )

    random_probe_df, random_probe_summary = infer_param_stats_random_radial_probes(
    wave_net,
    param_net,
    sample,
    n_probe_sets=5,
    n_r_points=n_r_points,
    Nr_norm=Nr_norm,
    Nth_norm=Nth_norm,
    Nph_norm=Nph_norm,
    n_mc_samples=1000,
    )
    
    random_probe_df.to_csv(
        out_dir / f"{tag}_random_radial_probe_parameter_inference.csv",
        index=False,
    )
    
    with open(
        out_dir / f"{tag}_random_radial_probe_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(random_probe_summary, f, indent=2)
    
    plot_random_probe_parameter_inference(
        random_probe_df,
        out_dir,
        tag,
        log_wandb=True,
    )
    
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log(
            {
                "tables/random_radial_probe_inference": wandb.Table(
                    dataframe=random_probe_df
                )
            }
        )
    
        for k, v in random_probe_summary.items():
            wandb.run.summary[k] = v
    with open(out_dir / f"{tag}_final_parameters.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    # Generate figures and tables used in the analysis.
    plot_parameter_history(history, out_dir, tag, log_wandb=True)
    plot_loss_history(history, out_dir, tag, log_wandb=True)
    plot_full_psi2_curves(wave_net, sample, out_dir, tag, Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm, log_wandb=True)
    plot_full_psi2_theta_phi_heatmaps(wave_net, sample, out_dir, tag, Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm, log_wandb=True)
    energy_diag = plot_energy_diagnostics(wave_net, param_net, sample, out_dir, tag, n_r_points=n_r_points, Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm, log_wandb=True)
    overlap_diag = plot_overlap_matrix(wave_net, sample, out_dir, tag, Nr_norm=Nr_norm, Nth_norm=Nth_norm, Nph_norm=Nph_norm, log_wandb=True)

    torch.save(wave_net.state_dict(), out_dir / f"{tag}_wave_net.pt")
    torch.save(param_net.state_dict(), out_dir / f"{tag}_param_net.pt")

    summary = {
        "tag": tag,
        "A": int(A),
        "Z": int(Z),
        "is_proton": bool(is_proton),
        **final_stats,
        **random_probe_summary,
        "energy_rmse": energy_diag["rmse"],
        "energy_mae": energy_diag["mae"],
        "max_abs_offdiag_overlap": overlap_diag["max_abs_offdiag_overlap"],
        "history_csv": str(hist_path),
        "out_dir": str(out_dir),
    }
    with open(out_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({"tables/training_history": wandb.Table(dataframe=hist_df)})
        for k, v in summary.items():
            if isinstance(v, (int, float, bool, str)):
                wandb.run.summary[k] = v
        wandb.finish()

    return wave_net, param_net, history, sample, summary




# Multi-nucleus experiment
def list_available_nuclei(dataset_path="ws_fd_dataset.npz"):
    dataset = load_fd_dataset(dataset_path)
    rows = []
    for item in dataset:
        rows.append({
            "A": int(item["A"]),
            "Z": int(item["Z"]),
            "is_proton": bool(item["is_proton"]),
            "species": "p" if bool(item["is_proton"]) else "n",
            "n_states_available": len(item["states"]),
        })
    return pd.DataFrame(rows)


def default_10_nucleus_cases(dataset_path="ws_fd_dataset.npz"):
    desired = [
        (60, 28, False),
        (60, 28, True),
        (72, 30, False),
        (72, 30, True),
        (90, 40, False),
        (90, 40, True),
    ]
    available_df = list_available_nuclei(dataset_path)
    available = set((int(r.A), int(r.Z), bool(r.is_proton)) for _, r in available_df.iterrows())
    cases = [c for c in desired if c in available]

    if len(cases) < 10:
        for _, r in available_df.iterrows():
            c = (int(r.A), int(r.Z), bool(r.is_proton))
            if c not in cases:
                cases.append(c)
            if len(cases) >= 10:
                break
    return cases[:10]


def plot_multi_nucleus_summaries(summary_df, out_root):
    out_root = ensure_dir(out_root)
    if summary_df.empty:
        return []

    paths = []
    labels = summary_df["tag"].tolist()
    x = np.arange(len(labels))

    for p, unit in [("V0", "MeV"), ("r0", "fm"), ("a", "fm"), ("lam_so", "")]:
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.errorbar(x, summary_df[f"{p}_mu"], yerr=summary_df[f"{p}_sigma"], fmt="o", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"{p} ({unit})" if unit else p)
        ax.set_title(f"Parameter summary across nuclei: {p}")
        ax.grid(True, alpha=0.3)
        paths.append(save_fig(fig, out_root / f"multi_nucleus_{p}_summary.png"))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x, summary_df["energy_rmse"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Energy RMSE (MeV)")
    ax.set_title("Energy reconstruction error across nuclei")
    ax.grid(True, axis="y", alpha=0.3)
    paths.append(save_fig(fig, out_root / "multi_nucleus_energy_rmse.png"))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x, summary_df["max_abs_offdiag_overlap"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"max off-diagonal $|\langle \psi_i|\psi_j\rangle|$")
    ax.set_title("Orthogonality error across nuclei")
    ax.grid(True, axis="y", alpha=0.3)
    paths.append(save_fig(fig, out_root / "multi_nucleus_max_offdiag_overlap.png"))

    return paths


def run_many_nuclei_experiment(
    dataset_path="ws_fd_dataset.npz",
    cases=None,
    max_states=6,
    epochs=3000,
    out_root="paper_outputs",
    wandb_project="inverse-ws-pinn",
    wandb_group="10_nuclei_benchmark",
    use_wandb=True,
    common_train_kwargs=None,
):
    if cases is None:
        cases = default_10_nucleus_cases(dataset_path)
    if common_train_kwargs is None:
        common_train_kwargs = {}

    summaries = []
    for A, Z, is_proton in cases:
        print("\n\n############################################################")
        print(f"Running A={A}, Z={Z}, species={'p' if is_proton else 'n'}")
        print("############################################################")
        try:
            _, _, _, _, summary = train_single_nucleus_instrumented(
                dataset_path=dataset_path,
                A=A,
                Z=Z,
                is_proton=is_proton,
                max_states=max_states,
                epochs=epochs,
                out_root=out_root,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                use_wandb=use_wandb,
                **common_train_kwargs,
            )
            summaries.append(summary)
        except RuntimeError as e:
            print(f"RuntimeError for A={A}, Z={Z}, is_proton={is_proton}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed case A={A}, Z={Z}, is_proton={is_proton}: {e}")

    summary_df = pd.DataFrame(summaries)
    out_root = ensure_dir(out_root)
    summary_df.to_csv(out_root / "multi_nucleus_summary.csv", index=False)
    plot_multi_nucleus_summaries(summary_df, out_root)
    return summary_df


# Experiment presets

def run_full_multi_nucleus_benchmark():
    return run_many_nuclei_experiment(
        dataset_path="ws_fd_dataset.npz",
        cases=None,
        max_states=7,
        epochs=15000,
        out_root="paper_outputs_final",
        wandb_project="inverse-ws-pinn_final2",
        wandb_group="10_nuclei_final",
        use_wandb=True,
        common_train_kwargs={
            "lr_wave": 5e-4,
            "lr_param": 1e-3,
            "n_r_points": 96,
            "Nr_norm": 1024,
            "Nth_norm": 512,
            "Nph_norm": 256,
            "hidden_wave": 128,
            "hidden_param": 128,
            "emm": 0,
            "wE": 0.5,
            "wR": 10.0,
            "wTh": 5.0,
            "wPh": 5.0,
            "wBC": 5.0,
            "wORTH": 1.0,
            "wKL": 1e-3,
            "print_every": 100,
            "log_every": 100,
            "plot_every": None,
        },
    )


if __name__ == "__main__":
    summary_df = run_full_multi_nucleus_benchmark()
    print(summary_df)