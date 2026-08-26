"""YAML configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .parameters import ParameterBounds


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration value: {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one experiment configuration."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")

    experiment = _required(config, "experiment", "root")
    potential = str(experiment.get("potential", "")).lower()
    if potential not in {"seminole", "wahlborn"}:
        raise ValueError("experiment.potential must be seminole or wahlborn")

    # Each parameterization has one configuration file.  Change only
    # data.mode to switch between its synthetic and experimental settings.
    data = _required(config, "data", "root")
    if "datasets" in data:
        mode = str(data.get("mode", "synthetic")).lower()
        if mode not in {"synthetic", "experimental"}:
            raise ValueError("data.mode must be synthetic or experimental")
        datasets = data["datasets"]
        if mode not in datasets:
            raise KeyError(f"No data.datasets.{mode} section was provided")
        selected = deepcopy(datasets[mode])
        selected["mode"] = mode
        config["data"] = selected

    normalization = config.get("paramnet_input", {}).get(
        "radial_normalization", "full_separable"
    )
    if normalization not in {"full_separable", "radial_l2"}:
        raise ValueError(
            "paramnet_input.radial_normalization must be full_separable or radial_l2."
        )

    ParameterBounds.from_mapping(config.get("parameter_bounds"))
    config["_config_path"] = str(config_path)
    config["_repository_root"] = str(config_path.parent.parent)
    return config


def _resolve_repo_path(config: dict[str, Any], value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_repository_root"]) / path
    return str(path.resolve())


def training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the nested public configuration into the training API."""
    data = _required(config, "data", "root")
    model = config.get("model", {})
    quadrature = config.get("quadrature", {})
    training = config.get("training", {})
    weights = config.get("loss_weights", {})
    scheduler = config.get("scheduler", {})
    inference = config.get("inference", {})
    output = config.get("output", {})
    logging = config.get("logging", {})
    probes = config.get("paramnet_input", {})

    cases = []
    for entry in _required(data, "cases", "data"):
        if len(entry) != 3:
            raise ValueError("Each data.cases entry must contain [A, Z, species].")
        A, Z, species = entry
        species_text = str(species).lower()
        if species_text not in {"proton", "neutron", "p", "n"}:
            raise ValueError(f"Unknown nucleon species: {species}")
        cases.append((int(A), int(Z), species_text in {"proton", "p"}))

    return {
        "potential": str(config["experiment"]["potential"]).lower(),
        "dataset_path": _resolve_repo_path(config, _required(data, "path", "data")),
        "cases": cases,
        "max_states": int(data.get("states_per_system", 6)),
        "epochs": int(data.get("epochs", training.get("epochs", 30_000))),
        "batch_size": training.get("system_batch_size", 2),
        "lr_wave": float(training.get("wave_learning_rate", 5e-4)),
        "lr_param": float(training.get("param_learning_rate", 1e-3)),
        "n_r_points": int(probes.get("radial_probe_points", 96)),
        "Nr_norm": int(quadrature.get("radial", 1024)),
        "Nth_norm": int(quadrature.get("polar", 512)),
        "Nph_norm": int(quadrature.get("azimuthal", 256)),
        "hidden_wave": int(model.get("wave_width", 256)),
        "hidden_param": int(model.get("param_width", 256)),
        "wave_depth": int(model.get("wave_depth", 5)),
        "param_depth": int(model.get("param_depth", 3)),
        "beta": float(model.get("radial_decay", 0.6)),
        "emm": int(model.get("magnetic_projection", 0)),
        "wE": float(weights.get("energy", 0.5)),
        "wR": float(weights.get("radial_residual", 10.0)),
        "wTh": float(weights.get("polar_residual", 5.0)),
        "wPh": float(weights.get("azimuthal_residual", 5.0)),
        "wBC": float(weights.get("boundary", 5.0)),
        "wORTH": float(weights.get("orthogonality", 5.0)),
        "wSO": float(weights.get("spin_orbit", 0.2)),
        "wKL": float(weights.get("kl", 1e-4)),
        "prior_std": float(training.get("latent_prior_std", 2.0)),
        "parameter_bounds": ParameterBounds.from_mapping(
            config.get("parameter_bounds")
        ),
        "radial_normalization": probes.get(
            "radial_normalization", "full_separable"
        ),
        "sign_probe_index": int(probes.get("sign_probe_index", 5)),
        "orth_points": int(quadrature.get("orthogonality_points", 512)),
        "use_scheduler_wave": bool(scheduler.get("wave_enabled", True)),
        "use_scheduler_param": bool(scheduler.get("param_enabled", True)),
        "gamma_wave": float(scheduler.get("wave_factor", 0.6)),
        "gamma_param": float(scheduler.get("param_factor", 0.5)),
        "step_size_wave": int(scheduler.get("wave_step", 1500)),
        "step_size_param": int(scheduler.get("param_step", 2000)),
        "print_every": int(logging.get("print_every", 100)),
        "log_every": int(logging.get("log_every", 100)),
        "plot_every": logging.get("plot_every", 1000),
        "checkpoint_every": int(output.get("checkpoint_every", 100)),
        "resume": bool(output.get("resume", True)),
        "gradient_clip": float(training.get("gradient_clip", 5.0)),
        "inference_samples": int(inference.get("monte_carlo_samples", 10_000)),
        "inference_seed": int(inference.get("seed", 0)),
        "reference": deepcopy(config.get("reference_parameters")),
        "config_source": config["_config_path"],
        "out_dir": _resolve_repo_path(
            config,
            data.get(
                "output_directory",
                output.get("directory", f"outputs/{config['experiment']['name']}")
            ),
        ),
        "use_wandb": bool(logging.get("wandb", False)),
        "wandb_project": logging.get("wandb_project", "inverse-ws-pinn"),
        "wandb_run_name": logging.get(
            "wandb_run_name", config["experiment"].get("name", "seminole")
        ),
        "wandb_group": logging.get("wandb_group"),
        "log_model_watch": bool(logging.get("wandb_watch", False)),
        "final_wavefunction_cases": int(output.get("wavefunction_cases", 4)),
        "final_heatmap_cases": int(output.get("heatmap_cases", 2)),
        "final_overlap_cases": int(output.get("overlap_cases", 3)),
    }
