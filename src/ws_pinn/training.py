"""Checkpointed joint optimization of WaveNet and ParamNet.

The command-line entry point resolves every paper-relevant value from YAML
before calling this module.  This file deliberately contains no competing set
of experiment defaults.
"""

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    wandb = None
    WANDB_AVAILABLE = False

from .data_loader import build_multinucleus_samples
from .diagnostics import (
    PARAM_NAMES,
    build_samples_table,
    ensure_dir,
    infer_global_stats,
    plot_full_psi2_slice_vs_theta,
    plot_global_energy_diagnostics,
    plot_learning_rates,
    plot_loss_history,
    plot_overlap_matrices,
    plot_parameter_history,
    plot_parameter_summary,
    plot_radial_wavefunctions,
    plot_theta_phi_heatmaps,
    save_history_csv,
    species_tag,
    wb_log,
)
from .inference import infer_global_raw_latent
from .losses import compute_multinucleus_loss_full3d
from .models import GlobalContextParamNet, WaveNet3D
from .parameters import ParameterBounds
from .potentials import set_potential_expression
from .runtime import device, runtime_metadata

MODULE_NAME = "ws_pinn.training"


def _json_ready(value: Any) -> Any:
    """Convert resolved settings into JSON/checkpoint-safe values."""
    if isinstance(value, ParameterBounds):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _required_setting(settings: Mapping[str, Any], name: str) -> Any:
    if name not in settings:
        raise KeyError(f"Missing resolved training setting: {name}")
    return settings[name]


def train_global_multinucleus(settings: Mapping[str, Any]):
    """Train one fully resolved experiment.

    Parameters
    ----------
    settings:
        Mapping returned by :func:`ws_pinn.config.training_settings`.  Keeping
        one mapping as the API prevents direct calls from silently falling
        back to settings that differ from the manuscript YAML files.
    """
    potential = str(_required_setting(settings, "potential"))
    dataset_path = str(_required_setting(settings, "dataset_path"))
    cases = list(_required_setting(settings, "cases"))
    max_states = int(_required_setting(settings, "max_states"))
    epochs = int(_required_setting(settings, "epochs"))
    batch_size = _required_setting(settings, "batch_size")
    lr_wave = float(_required_setting(settings, "lr_wave"))
    lr_param = float(_required_setting(settings, "lr_param"))
    n_r_points = int(_required_setting(settings, "n_r_points"))
    Nr_norm = int(_required_setting(settings, "Nr_norm"))
    Nth_norm = int(_required_setting(settings, "Nth_norm"))
    Nph_norm = int(_required_setting(settings, "Nph_norm"))
    hidden_wave = int(_required_setting(settings, "hidden_wave"))
    hidden_param = int(_required_setting(settings, "hidden_param"))
    wave_depth = int(_required_setting(settings, "wave_depth"))
    param_depth = int(_required_setting(settings, "param_depth"))
    beta = float(_required_setting(settings, "beta"))
    emm = int(_required_setting(settings, "emm"))
    wE = float(_required_setting(settings, "wE"))
    wR = float(_required_setting(settings, "wR"))
    wTh = float(_required_setting(settings, "wTh"))
    wPh = float(_required_setting(settings, "wPh"))
    wBC = float(_required_setting(settings, "wBC"))
    wORTH = float(_required_setting(settings, "wORTH"))
    wSO = float(_required_setting(settings, "wSO"))
    wKL = float(_required_setting(settings, "wKL"))
    prior_std = float(_required_setting(settings, "prior_std"))
    parameter_bounds = _required_setting(settings, "parameter_bounds")
    if not isinstance(parameter_bounds, ParameterBounds):
        raise TypeError("parameter_bounds must be a ParameterBounds instance")
    radial_normalization = str(
        _required_setting(settings, "radial_normalization")
    )
    sign_probe_index = int(_required_setting(settings, "sign_probe_index"))
    orth_points = int(_required_setting(settings, "orth_points"))
    use_scheduler_wave = bool(_required_setting(settings, "use_scheduler_wave"))
    use_scheduler_param = bool(_required_setting(settings, "use_scheduler_param"))
    gamma_wave = float(_required_setting(settings, "gamma_wave"))
    gamma_param = float(_required_setting(settings, "gamma_param"))
    step_size_wave = int(_required_setting(settings, "step_size_wave"))
    step_size_param = int(_required_setting(settings, "step_size_param"))
    print_every = int(_required_setting(settings, "print_every"))
    log_every = int(_required_setting(settings, "log_every"))
    plot_every = _required_setting(settings, "plot_every")
    checkpoint_every = int(_required_setting(settings, "checkpoint_every"))
    resume = bool(_required_setting(settings, "resume"))
    gradient_clip = float(_required_setting(settings, "gradient_clip"))
    inference_samples = int(_required_setting(settings, "inference_samples"))
    inference_seed = int(_required_setting(settings, "inference_seed"))
    reference = _required_setting(settings, "reference")
    config_source = str(_required_setting(settings, "config_source"))
    out_dir = str(_required_setting(settings, "out_dir"))
    use_wandb = bool(_required_setting(settings, "use_wandb"))
    wandb_project = str(_required_setting(settings, "wandb_project"))
    wandb_run_name = str(_required_setting(settings, "wandb_run_name"))
    wandb_group = _required_setting(settings, "wandb_group")
    log_model_watch = bool(_required_setting(settings, "log_model_watch"))
    final_wavefunction_cases = int(
        _required_setting(settings, "final_wavefunction_cases")
    )
    final_heatmap_cases = int(
        _required_setting(settings, "final_heatmap_cases")
    )
    final_overlap_cases = int(
        _required_setting(settings, "final_overlap_cases")
    )
    diagnostic_radial_points = int(
        _required_setting(settings, "diagnostic_radial_points")
    )
    diagnostic_peak_radial_points = int(
        _required_setting(settings, "diagnostic_peak_radial_points")
    )
    diagnostic_heatmap_theta_points = int(
        _required_setting(settings, "diagnostic_heatmap_theta_points")
    )
    diagnostic_heatmap_phi_points = int(
        _required_setting(settings, "diagnostic_heatmap_phi_points")
    )
    diagnostic_theta_slice_points = int(
        _required_setting(settings, "diagnostic_theta_slice_points")
    )
    diagnostic_overlap_points = int(
        _required_setting(settings, "diagnostic_overlap_points")
    )

    set_potential_expression(potential)
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
        depth=wave_depth,
        beta=beta,
    ).to(device)

    param_net = GlobalContextParamNet(
        input_dim=param_input_dim,
        hidden=hidden_param,
        depth=param_depth,
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

    # Preserve both the selected YAML values and every flattened effective
    # setting.  Derived dimensions and runtime metadata are recorded separately
    # so a checkpoint is self-describing without confusing this manifest with
    # the input configuration object.
    run_manifest = {
        "module": MODULE_NAME,
        "source_config": _json_ready(settings.get("source_config", {})),
        "resolved_settings": _json_ready({
            key: value
            for key, value in settings.items()
            if key != "source_config"
        }),
        "derived": {
            "n_samples": len(samples),
            "skipped": _json_ready(skipped),
            "param_input_dim": param_input_dim,
            "parameter_mode": (
                "one_global_context_informed_output_distribution"
            ),
        },
        "runtime": runtime_metadata(),
    }

    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    # Copy the exact human-authored YAML used to start this run.  The manifest
    # above contains resolved paths and derived values; this file preserves the
    # original public experiment description.
    source_path = Path(config_source)
    if source_path.is_file():
        shutil.copy2(source_path, out_dir / "input_config.yaml")

    if use_wandb:
        if not WANDB_AVAILABLE:
            raise ImportError(
                "use_wandb=True, but wandb is not installed."
            )

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config=run_manifest,
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
    print("GLOBAL MULTI-NUCLEUS COUPLED FORWARD-INVERSE TRAINING")
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
            "run_manifest": run_manifest,
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

        loss, metrics, _per_sample_rows = compute_multinucleus_loss_full3d(
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
            parameter_bounds=parameter_bounds,
            radial_normalization=radial_normalization,
            sign_probe_index=sign_probe_index,
            **loss_kwargs,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at epoch {epoch}: {loss}"
            )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(wave_net.parameters()) + list(param_net.parameters()),
            max_norm=gradient_clip,
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
                    "run_manifest": run_manifest,
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
                n_mc_samples=min(1000, inference_samples),
                parameter_bounds=parameter_bounds,
                radial_normalization=radial_normalization,
                sign_probe_index=sign_probe_index,
                seed=inference_seed,
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

                if reference is not None and p in reference:
                    payload[f"parameters/{p}_error_to_reference"] = (
                        row[f"{p}_mu"] - float(reference[p])
                    )
                    payload[f"parameters/{p}_abs_error_to_reference"] = abs(
                        row[f"{p}_mu"] - float(reference[p])
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
                reference=reference,
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
        n_mc_samples=inference_samples,
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=inference_seed,
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
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
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
        reference=reference,
        log_wandb=use_wandb,
    )
    plot_parameter_summary(
        final_stats,
        final_plots_dir,
        reference=reference,
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
        n_mc_samples=inference_samples,
        parameter_bounds=parameter_bounds,
        radial_normalization=radial_normalization,
        sign_probe_index=sign_probe_index,
        seed=inference_seed,
        log_wandb=use_wandb,
    )

    plot_radial_wavefunctions(
        wave_net,
        samples,
        final_plots_dir / "radial_wavefunctions",
        n_r=diagnostic_radial_points,
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
        radial_peak_points=diagnostic_peak_radial_points,
        n_theta=diagnostic_heatmap_theta_points,
        n_phi=diagnostic_heatmap_phi_points,
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
        radial_peak_points=diagnostic_peak_radial_points,
        n_theta=diagnostic_theta_slice_points,
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
        n_points=diagnostic_overlap_points,
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


# Compatibility alias for notebooks written against the first modular release.
train_global_multinucleus_instrumented = train_global_multinucleus
