#!/usr/bin/env python3
"""Convert an RSL-RL v1 actor checkpoint for inference with RSL-RL v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    legacy = source["model_state_dict"]
    actor: dict[str, torch.Tensor] = {}
    for key, value in legacy.items():
        if key == "std":
            actor["distribution.std_param"] = value
        elif key.startswith("actor."):
            actor["mlp." + key.removeprefix("actor.")] = value
        elif key.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + key.removeprefix("actor_obs_normalizer.")] = value

    required = {
        "distribution.std_param",
        "mlp.0.weight",
        "mlp.6.bias",
        "obs_normalizer._mean",
        "obs_normalizer._std",
        "obs_normalizer.count",
    }
    missing = sorted(required - actor.keys())
    if missing:
        raise ValueError(f"Legacy checkpoint is missing required actor entries: {missing}")

    converted = {
        "actor_state_dict": actor,
        "iter": source.get("iter", 0),
        "infos": {
            "converted_from": str(args.input.resolve()),
            "source_infos": source.get("infos"),
            "conversion": "legacy_rsl_model_state_dict_to_rsl_v3_actor_only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(f"converted={args.output.resolve()} actor_tensors={len(actor)} iter={converted['iter']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
