# Trained model checkpoints

Training stores checkpoints inside the output directory selected by the YAML,
for example `outputs/seminole_synthetic/checkpoints/`. This repository-level
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

Every output directory also contains `input_config.yaml` (an exact copy of the
submitted YAML) and `run_manifest.json` (resolved paths, effective settings,
derived dimensions, and runtime metadata).

## Resume behavior

With `output.resume: true`, `scripts/train.py` resumes automatically when
`latest_checkpoint.pt` exists in the selected output directory. To begin a
scientifically independent run, use a new output directory or set:

```yaml
output:
  resume: false
```

Do not resume a checkpoint after changing the model architecture, parameter
bounds, selected state count, ParamNet normalization convention, or potential
expression. Those changes alter the meaning or shape of the learned state.

## Published weights

The source repository does not currently include the large `.pt` artifacts.
For permanent publication, the exact checkpoints used for the manuscript
should be attached to the versioned Zenodo archive or GitHub Release and linked
from the main README. This lets reviewers regenerate figures without repeating
the complete training runs.
