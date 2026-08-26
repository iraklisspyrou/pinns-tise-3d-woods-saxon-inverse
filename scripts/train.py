#!/usr/bin/env python3
"""Train one configured Woods--Saxon inverse experiment."""

from __future__ import annotations

import argparse

from ws_pinn.config import load_config, training_kwargs
from ws_pinn.runtime import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    runtime = config.get("runtime", {})
    configure_runtime(
        requested_device=str(runtime.get("device", "auto")),
        seed=int(runtime.get("seed", 0)),
        deterministic=bool(runtime.get("deterministic", False)),
    )

    # Import after runtime configuration so every module uses the requested device.
    from ws_pinn.training import train_global_multinucleus_instrumented

    train_global_multinucleus_instrumented(**training_kwargs(config))


if __name__ == "__main__":
    main()

