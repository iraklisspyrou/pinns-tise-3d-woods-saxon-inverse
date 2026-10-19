"""Per-run Phi reference norms for the jointly learned separated solution.

These constants affect radial residuals/boundaries and ParamNet probes only.
The physical wavefunction still uses alpha=(I_R I_theta I_phi)**(-1/2).
"""

from copy import deepcopy
import json
import math
from pathlib import Path

import torch

VERSION = "phi_only_normalization_v1"
POLAR_RULE = "log_gauss_legendre_v1"
LEGACY_RULE = "uniform_trapezoid_v1"


def phi_reference_key(sample, st):
    """Include the energy context so repeated noisy observations stay distinct."""
    return json.dumps([
        int(sample["A"]), int(sample["Z"]), bool(sample["is_proton"]),
        int(st["nr"]), int(st["l"]), float(st["j"]),
        [float(s["energy"]) for s in sample["states"]],
    ], separators=(",", ":"))


def configure_normalization(wave, samples, grids, *, phi_only=True,
                            polar_rule=POLAR_RULE, state=None):
    """Initialize once, or restore the exact constants saved at initialization.

    Does not draw random numbers, change network weights, or freeze networks.
    Caller restores model weights before passing a checkpoint's state.
    """
    from .wavefunctions import norm_parts

    if polar_rule not in {POLAR_RULE, LEGACY_RULE}:
        raise ValueError(f"Unknown polar quadrature: {polar_rule}")
    keys = {phi_reference_key(s, st) for s in samples for st in s["states"]}
    if state is not None:
        if state.get("version") != VERSION:
            raise ValueError("Unsupported normalization checkpoint version.")
        if (state["phi_only"] != phi_only or state["polar_quadrature"] != polar_rule
                or tuple(state["reference_grids"]) != tuple(grids)):
            raise ValueError("Normalization settings differ from this checkpoint; use a new run.")
        refs = deepcopy(state["phi_norms"])
        # Context validation is required even when the Phi correction is off.
        if state.get("identification_keys") != sorted(keys):
            raise ValueError("Identification contexts differ from this checkpoint; use a new run.")
    else:
        refs = {}
        wave.polar_quadrature = polar_rule
        if phi_only:
            p = next(wave.parameters())
            with torch.no_grad():
                for sample in samples:
                    E = p.new_tensor([st["energy"] for st in sample["states"]])
                    for st in sample["states"]:
                        refs[phi_reference_key(sample, st)] = float(
                            norm_parts(wave, E, sample, st, *grids)[2])
    if ((phi_only and set(refs) != keys) or (not phi_only and refs)
            or any(not math.isfinite(v) or v <= 1e-12 for v in refs.values())):
        raise ValueError("Invalid or mismatched Phi reference norms.")
    result = dict(version=VERSION, phi_only=bool(phi_only),
                  polar_quadrature=polar_rule, reference_grids=list(grids),
                  identification_keys=sorted(keys), phi_norms=refs)
    wave.polar_quadrature = polar_rule
    wave.phi_reference_norms = refs if phi_only else None
    wave.normalization_state = deepcopy(result)
    return result


def restore_checkpoint_normalization(wave, samples, settings, checkpoint,
                                     checkpoint_path):
    """Restore corrected models; preserve old repository checkpoints explicitly.

    Never estimate initialization constants from trained weights. The Colab
    Phi-only checkpoint format uses its adjacent saved JSON reference file.
    """
    state = checkpoint.get("normalization_state")
    if state is not None:
        return configure_normalization(
            wave, samples, state["reference_grids"],
            phi_only=state["phi_only"], polar_rule=state["polar_quadrature"], state=state)
    correction = checkpoint.get("run_manifest", {}).get("normalization_correction", {})
    if correction.get("version") == VERSION:
        path = Path(checkpoint_path).parent.parent / correction.get(
            "reference_file", "normalization_reference.json")
        record = json.loads(path.read_text())
        if record.get("version") != "phi_only_reference_v1":
            raise ValueError("Expected the original Colab Phi-only reference JSON.")
        refs = {}
        for sample in samples:
            for st in sample["states"]:
                old = ":".join(map(str, (sample["A"], sample["Z"], int(sample["is_proton"]),
                                        st["nr"], st["l"], st["j"])))
                refs[phi_reference_key(sample, st)] = record["phi_norms"][old]
        grids = [settings[k] for k in ("Nr_norm", "Nth_norm", "Nph_norm")]
        state = dict(version=VERSION, phi_only=True, polar_quadrature=POLAR_RULE,
                     reference_grids=grids, identification_keys=sorted(refs), phi_norms=refs)
        return configure_normalization(wave, samples, grids, state=state)
    if correction:
        raise ValueError("This checkpoint uses a different normalization experiment.")
    wave.phi_reference_norms = None
    wave.polar_quadrature = LEGACY_RULE
    wave.normalization_state = None
    return None
