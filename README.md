# Coupled forward--inverse Woods--Saxon PINN

Reproducible code for the paper **Probabilistic Physics-Informed Neural
Solvers for Woods--Saxon Parameter Identification: A Coupled Forward--Inverse
Approach**.

The implementation learns separated radial, polar and azimuthal spatial
wavefunction components with WaveNet and a dataset-conditioned output
distribution over six global Woods--Saxon parameters with ParamNet. The code
supports both the Seminole and Wahlborn potential expressions.

## Files

```text
configs/
  seminole.yaml                 Seminole synthetic/experimental settings
  wahlborn.yaml                 Wahlborn synthetic/experimental settings
src/ws_pinn/
  models.py                     MLP, WaveNet and ParamNet declarations
  data_loader.py                NPZ loader and state selection
  training.py                   joint forward--inverse training
  fd_solver.py                  independent radial finite-difference solver
  potentials.py                Seminole and Wahlborn Hamiltonians
  losses.py                     physics-informed objective
scripts/
  train.py                      training command
  generate_data.py              synthetic-data generation
  fd_validation.py              independent closure validation
  lsq_fit.py                    Levenberg--Marquardt least-squares baseline
visualization/
  training_visualization.py     loss and parameter histories
  results_visualization.py      energy and residual figures
  spectrum_plots.py             publication level-scheme figures
checkpoints/
  README.md                     names and contents of saved model files
```

The supplied datasets remain at repository root:

- `seminole_synthetic_dataset.npz`
- `wahlborn_synthetic_dataset.npz`
- `experimental_dataset.npz`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
```

Install optional Weights & Biases support with:

```bash
pip install -e ".[tracking]"
```

## Choose the experiment

There are exactly two configuration files, one per potential expression. In
either file, change only:

```yaml
data:
  mode: synthetic
```

to:

```yaml
data:
  mode: experimental
```

The corresponding dataset path, selected systems, number of states, epochs
and output directory are then selected automatically.

## Train

```bash
python scripts/train.py --config configs/seminole.yaml
python scripts/train.py --config configs/wahlborn.yaml
```

Training saves updated weights and resumable checkpoints in the configured output directory.

## Generate synthetic data

```bash
python scripts/generate_data.py --config configs/seminole.yaml \
  --output generated_seminole_dataset.npz
```

Use `configs/wahlborn.yaml` for the Wahlborn expression.

## Independent finite-difference validation

Validate the reference values from the configuration:

```bash
python scripts/fd_validation.py --config configs/seminole.yaml \
  --dataset seminole_synthetic_dataset.npz
```

Validate a trained distribution-mean estimate:

```bash
python scripts/fd_validation.py --config configs/seminole.yaml \
  --parameters outputs/seminole_synthetic/final_global_parameter_prediction.json \
  --dataset seminole_synthetic_dataset.npz
```

## Least-squares baseline

```bash
python scripts/lsq_fit.py --dataset experimental_dataset.npz
```

This reproduces the separate finite-difference Levenberg--Marquardt comparison
on the same 42 identification levels

## Figures

```bash
python visualization/training_visualization.py \
  outputs/seminole_synthetic/training_history.csv

python visualization/results_visualization.py \
  outputs/seminole_synthetic/plots/final/all_nuclei_energy_table.csv

python visualization/spectrum_plots.py \
  --experimental experimental_dataset.npz \
  --seminole seminole_reference_spectrum.npz \
  --pinn pinn_spectrum.npz
```

The visualization commands save PDF by default where applicable, so plots can
be regenerated without retraining when the corresponding CSV/NPZ result files
are available.

## Reproducibility notes

- NumPy and PyTorch seeds are set through the YAML files.
- The finite-difference solver is independent of the neural-network code.
- GPU results may differ slightly across CUDA and hardware versions.
- Full paper runs are computationally expensive; use fewer epochs and smaller
  quadrature grids only for a smoke test.
- ParamNet standard deviations are model-derived output spreads, not calibrated
  Bayesian credible intervals.

