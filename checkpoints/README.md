# Trained model checkpoints

Training stores checkpoints inside the output directory selected by the YAML,
for example `outputs/seminole_synthetic_phi_only/checkpoints/`. This repository-level
folder documents the format; it does not contain fabricated or unverified
weights.

## Files written during a run

| File | Purpose |
|---|---|
| `latest_checkpoint.pt` | Most recent resumable state, updated at `checkpoint_every` and at normal completion |
| `final_checkpoint.pt` | Resumable state written after the configured final epoch |
| `best_training_loss.pt` | State with the lowest observed training objective |
| `wave_net_global_multinucleus.pt` | Final WaveNet `state_dict` only |
| `param_net_global_multinucleus.pt` | Final ParamNet `state_dict` only |

The resumable checkpoints contain:

- WaveNet and ParamNet weights.
- Both optimizer states.
- Both scheduler states when enabled.
- Completed epoch and current/best loss.
- Training history.
- NumPy, PyTorch, and CUDA random-number states.
- The complete resolved run manifest.
- The fixed initial Phi reference norms, their identification contexts and norm
  grids, and the polar integration rule (`normalization_state`).

The initial reference norms are also exported to
`normalization_reference.json`. They are restored on resume and inference;
**never recompute them from trained weights**. Latest, final and best-loss
checkpoints all include them. For corrected models, use the complete checkpoint
with `ws_pinn.checkpoints.load_models_and_samples`, rather than loading only the
bare WaveNet/ParamNet `state_dict` files. The latter do not contain the reference
constants.

The loader can also read the working Colab `phi_only_normalization_v1`
checkpoint when its original `normalization_reference.json` remains beside the
run's `checkpoints/` folder. It requires the matching dataset and configuration.
Old repository checkpoints without a normalization state are evaluated using
their original uniform polar quadrature and uncorrected radial probes. Other
experimental normalization formats are rejected, rather than silently treating
them as the Phi-only correction.

Every output directory also contains `input_config.yaml` (an exact copy of the
submitted YAML) and `run_manifest.json` (resolved paths, effective settings,
derived dimensions, and runtime metadata).

## Resume behavior

With `output.resume: true`, `scripts/train.py` resumes automatically when
`latest_checkpoint.pt` exists in the selected output directory. To begin a
scientifically independent run, use a **new output directory** and set:

```yaml
output:
  resume: false
```

Do not resume a checkpoint after changing the model architecture, parameter
bounds, selected state count, ParamNet normalization convention, or potential
expression. Those changes alter the meaning or shape of the learned state.
The trainer checks the saved normalization settings, energy contexts, norm
point counts, potential, bounds, probe convention and KL schedule before
continuing. With a KL schedule, epochs retain their absolute numbering: a run
resuming at 7601 uses the final weight, not a restarted warmup.

The new Phi-corrected configurations cannot silently resume old uncorrected
checkpoints. For deliberate reproduction of the old objective, set
`normalization.phi_only: false`,
`normalization.polar_quadrature: uniform_trapezoid_v1` and omit the KL schedule;
use the original dataset, bounds and model settings as well.

## Published weights

The source repository does not currently include the large `.pt` artifacts.
For permanent publication, the exact checkpoints used for the manuscript
should be attached to the versioned Zenodo archive or GitHub Release and linked
from the main README. This lets reviewers regenerate figures without repeating
the complete training runs.
