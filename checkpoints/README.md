# Trained model weights

Training writes the updated weights automatically to the output directory
selected in the YAML configuration:

- `wave_net_global_multinucleus.pt`
- `param_net_global_multinucleus.pt`
- `checkpoints/latest_checkpoint.pt`
- `checkpoints/final_checkpoint.pt`

The resumable checkpoints also contain optimizer, scheduler, random-number and
training-history state. No trained `.pt` file was supplied with the repository
update, so this directory does not contain fabricated or unverified weights.

