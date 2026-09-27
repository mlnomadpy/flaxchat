"""Bounded physical 8-chip exact-resume pilot from an isolated Orbax copy.

The launcher must supply an independent cleanup guard. This worker never
changes the production checkpoint prefix and rejects a changed training recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.admission) != args.admission_sha256:
        raise ValueError("Pilot admission changed")
    admission = json.loads(args.admission.read_text())
    draft = json.loads(args.draft.read_text())
    if (admission.get("status") != "ADMITTED_8_CHIP_PILOT"
            or admission.get("source_checkpoint_prefix") == admission.get("pilot_checkpoint_prefix")
            or admission.get("target_devices") != 8
            or admission.get("target_processes") != 1
            or time.time() >= admission.get("expires_unix", 0)):
        raise ValueError("Invalid or expired isolated pilot admission")
    if sha(args.draft) != admission["draft_sha256"] or sha(args.identity) != admission["identity_sha256"]:
        raise ValueError("Frozen pilot recipe or identity changed")
    if sha(Path(__file__)) != admission["worker_sha256"]:
        raise ValueError("Pilot worker changed after admission")
    if sha(Path("scripts/encoder_scale_pilot8_probe.py")) != admission["probe_sha256"]:
        raise ValueError("Pilot probe changed after admission")
    prior = json.loads(args.identity.read_text())["input_identity"]
    start = admission["start_step"]
    end = admission["end_step"]
    if not 0 < end - start <= 64:
        raise ValueError("Pilot must be bounded to at most 64 updates")
    argv = list(draft["argv_without_output"])
    for flag in ("--initialize-from-checkpoint", "--initialize-step"):
        index = argv.index(flag)
        del argv[index:index + 2]
    if "--resume" in argv or "--stop-after" in argv or "--output" in argv:
        raise ValueError("Draft unexpectedly contains resume/output flags")
    command = [sys.executable, *argv, "--resume", "--stop-after", str(end),
               "--output", admission["pilot_checkpoint_prefix"]]
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"complete": False, "start_step": start, "end_step": end,
              "source_checkpoint_prefix": admission["source_checkpoint_prefix"],
              "pilot_checkpoint_prefix": admission["pilot_checkpoint_prefix"]}
    log = args.output / "train.log"
    try:
        # TPU PJRT permits only one owning process. Perform topology and
        # checkpoint checks in a child that exits before the trainer starts.
        with (args.output / "probe.log").open("w") as stream:
            subprocess.run([sys.executable, "scripts/encoder_scale_pilot8_probe.py",
                            "--checkpoint", admission["pilot_checkpoint_prefix"],
                            "--step", str(start), "--identity", str(args.identity)], stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=180,
                           env=os.environ | {"JAX_PLATFORMS": "tpu"})
        with log.open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           timeout=min(1800, admission["workload_seconds"] - 60),
                           env=os.environ | {"JAX_PLATFORMS": "tpu"})
        events = []
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
        configs = [e for e in events if e.get("event") == "run_config"]
        steps = [e for e in events if e.get("event") == "train_step"]
        if (len(configs) != 1 or configs[0].get("backend") != "tpu"
                or configs[0].get("devices") != 8 or configs[0].get("processes") != 1
                or configs[0].get("fsdp") != 1
                or [e["step"] for e in steps] != list(range(start + 1, end + 1))
                or not all(e.get("updated") is True for e in steps)):
            raise ValueError("Pilot lacks ordered applied eight-chip updates")
        for key in ("resolved_config", "tokenizer", "data_manifest", "source_python_sha256"):
            if configs[0]["input_identity"].get(key) != prior[key]:
                raise ValueError("Pilot training identity differs from four-chip stage: " + key)
        report.update(complete=True, status="pilot_finished", device_count=8,
                      applied_updates=len(steps),
                      warm_step_seconds=[e["seconds"] for e in steps[1:]
                                         if not e.get("includes_profiling")],
                      checkpoint_step=end)
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        evidence = admission["evidence_prefix"].rstrip("/")
        for name in ("report.json", "probe.log", "train.log"):
            source = args.output / name
            if source.is_file():
                subprocess.run(["gcloud", "storage", "cp", "--if-generation-match=0",
                                str(source), evidence + "/" + name], check=True,
                               timeout=120)


if __name__ == "__main__":
    main()
