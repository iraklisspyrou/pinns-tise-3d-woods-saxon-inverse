"""Tables and plotting routines used for manuscript diagnostics."""

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    PLOTTING_AVAILABLE = True
except Exception:
    plt = None
    PLOTTING_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from .constants import R_MAX, PI, TWOPI
from .runtime import device
from .inference import infer_global_parameters, infer_physical_mean_parameters
from .parameters import DEFAULT_BOUNDS, ParameterBounds
from .losses import energy_rayleigh_full3d
from .wavefunctions import eval_Phi, eval_R, eval_Theta, eval_psi_norm, psi_scale_only

PARAM_NAMES = ['V0', 'kappa', 'r0', 'a', 'lam_so', 'r0_so']
PARAM_LABELS = {'V0': r'$V_0$', 'kappa': r'$\kappa$', 'r0': r'$r_0$', 'a': r'$a$', 'lam_so': r'$\lambda_{\mathrm{SO}}$', 'r0_so': r'$r_{0,\mathrm{SO}}$'}
PARAM_UNITS = {'V0': 'MeV', 'kappa': '', 'r0': 'fm', 'a': 'fm', 'lam_so': '', 'r0_so': 'fm'}

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig, path, dpi=300, save_pdf=True):
    """Save the requested raster image and a vector-PDF companion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf and path.suffix.lower() != ".pdf":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
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
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=None,
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
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=seed,
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
    n_mc_samples=10_000,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=0,
):
    """Return the physical-space distribution mean used in the manuscript."""
    return infer_physical_mean_parameters(
        wave_net=wave_net,
        param_net=param_net,
        samples=samples,
        n_samples=n_mc_samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=seed,
    )


def wsparams_to_dict(params):
    return {
        "V0": float(params.V0.detach().cpu()),
        "kappa": float(params.kappa.detach().cpu()),
        "r0": float(params.r0.detach().cpu()),
        "a": float(params.a.detach().cpu()),
        "lam_so": float(params.lam_so.detach().cpu()),
        "r0_so": float(params.r0_so.detach().cpu()),
    }


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
        ax.plot(x, mu, linewidth=1.9, label="Physical-space output mean")

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

    fig.suptitle("Final global parameter output distribution", y=1.01)
    fig.tight_layout()
    path = save_fig(fig, out_dir / "final_global_parameter_summary.png")

    if log_wandb:
        wb_log({
            "plots/parameters/final_summary": wandb.Image(path),
            "tables/final_global_parameters": wandb.Table(dataframe=df),
        })

    return {"path": path, "table": df}


def compute_all_energy_tables(
    wave_net,
    param_net,
    samples,
    n_r_points=96,
    Nr_norm=512,
    Nth_norm=256,
    Nph_norm=256,
    n_mc_samples=10_000,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=0,
):
    """
    Evaluate Rayleigh energies for every selected state using the single
    physical-space distribution-mean parameter set.
    """
    params = global_mean_params(
        wave_net,
        param_net,
        samples,
        n_r_points=n_r_points,
        Nr_norm=Nr_norm,
        Nth_norm=Nth_norm,
        Nph_norm=Nph_norm,
        n_mc_samples=n_mc_samples,
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=seed,
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
    n_mc_samples=10_000,
    parameter_bounds: ParameterBounds = DEFAULT_BOUNDS,
    radial_normalization="full_separable",
    sign_probe_index=5,
    seed=0,
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
        n_mc_samples=n_mc_samples,
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=seed,
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
