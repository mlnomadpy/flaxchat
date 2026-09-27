"""Short-lived TPU topology and checkpoint probe; exits before trainer starts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from flaxchat.checkpoint import load_checkpoint_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "tpu" or jax.device_count() != 8 or jax.process_count() != 1:
        raise RuntimeError("Pilot requires one physical eight-chip TPU host")
    expected = json.loads(args.identity.read_text())["input_identity"]
    metadata = load_checkpoint_metadata(args.checkpoint, step=args.step)
    if metadata["step"] != args.step:
        raise ValueError("Cloned checkpoint cursor differs")
    for key, value in (("resolved_config", expected["resolved_config"]),
                       ("tokenizer_identity", expected["tokenizer"]),
                       ("data_manifest_identity", expected["data_manifest"]),
                       ("source_python_sha256", expected["source_python_sha256"])):
        if metadata.get(key) != value:
            raise ValueError("Cloned checkpoint identity differs: " + key)
    print(json.dumps({"event": "scale_pilot_probe", "passed": True,
                      "backend": "tpu", "devices": 8, "processes": 1,
                      "checkpoint_step": args.step}))


if __name__ == "__main__":
    main()
