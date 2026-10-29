"""Actual-parent, real-data qualification before the first GOAT contrastive stage."""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import os
import math
import time
import xml.etree.ElementTree as ET
from flaxchat.goat_contrastive_campaign import stage_arguments
from flaxchat.checkpoint_metadata import read_committed_metadata
from scripts.release_contract import canonical_hash


def run(argv, log, seconds):
    with log.open("wb") as f:
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=seconds)
    if r.returncode:
        raise RuntimeError("Qualification child failed: " + log.name)


def steps(log):
    rows = []
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(r, dict) and r.get("event") == "embedding_step":
            rows.append(r)
    return rows


def compare(reference, first, resumed, left, right):
    for log in (reference, first, resumed):
        runs = []
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "embedding_run":
                runs.append(event)
        if len(runs) != 1 or any(
            runs[0].get(k) != v
            for k, v in dict(backend="tpu", devices=8, processes=1).items()
        ):
            raise ValueError(
                "Every real training trace requires physical eight-chip topology"
            )
    a, b = steps(reference), steps(first) + steps(resumed)
    if [x["step"] for x in a] != [1, 2, 3, 4] or [x["step"] for x in b] != [1, 2, 3, 4]:
        raise ValueError("Exact four-update traces required")
    fields = [
        "loss",
        "gradient_norm",
        "learning_rate",
        "batch_sha256",
        "candidate_masking",
        "global_pairs",
        "pair_only",
    ]
    for x, y in zip(a, b, strict=True):
        if any(x.get(k) != y.get(k) for k in fields):
            raise ValueError("Continuous/split optimizer input/result mismatch")
        if (
            any(
                not math.isfinite(x[k])
                for k in ["loss", "gradient_norm", "learning_rate"]
            )
            or x["global_pairs"] != 128
        ):
            raise ValueError("Nonfinite or wrong-global-batch real update")
    read_committed_metadata(left, 4, include_receipt=True)
    read_committed_metadata(right, 4, include_receipt=True)
    manifests = [
        json.loads((Path(p) / "4/manifest/metadata").read_text()) for p in [left, right]
    ]
    for k in ["model_state", "optimizer_state", "training_state"]:
        if not manifests[0].get(k) or manifests[0][k] != manifests[1].get(k):
            raise ValueError("Exact recovery failed: " + k)
    return dict(
        updates=4,
        global_pairs=128,
        exact_model_Adam_training_state=True,
        exact_batches_losses_gradients_LR=True,
        manifest_state_sha256={
            k: canonical_hash(manifests[0][k])
            for k in ["model_state", "optimizer_state", "training_state"]
        },
        steps=[{k: r[k] for k in ["step", *fields]} for r in a],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    Q = root / "qualification"
    Q.mkdir(exist_ok=False)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        JAX_PLATFORMS="tpu",
        FLAXCHAT_PHYSICAL_TPU="1",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    result = dict(
        schema_version=1,
        passed=False,
        backend="tpu",
        manifest_sha256=os.environ["FLAXCHAT_RUN_MANIFEST_SHA256"],
        runtime_lock_sha256=os.environ["FLAXCHAT_RUNTIME_LOCK_SHA256"],
        source_tree_sha256=os.environ["FLAXCHAT_SOURCE_TREE_SHA256"],
        tests=[],
        started_unix=time.time(),
    )

    def save():
        a.receipt.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        m = json.loads((root / "parent/checkpoint-manifest.json").read_text())
        parent_out = Q / "parent-import.json"
        run(
            [
                sys.executable,
                "-m",
                "scripts.validate_production_parent_tpu",
                "--parent",
                str(root / "parent"),
                "--output",
                str(parent_out),
                "--expected-family",
                "modernbert",
                "--expected-step",
                "379079",
                "--expected-metadata-sha256",
                m["metadata_sha256"],
                "--expected-manifest-sha256",
                canonical_hash(m),
                "--expected-leaves",
                "181",
            ],
            Q / "parent-import.log",
            600,
        )
        imported = json.loads(parent_out.read_text())
        devices = imported["runtime"]["devices"]
        assert imported["weight_import_qualified"] and len(devices) == 8
        result.update(devices=8, processes=1)
        result["tests"].append(dict(id="actual_parent_import", status="passed"))
        save()
        xml = Q / "hard-negative.xml"
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_embedding_hard_negative_physical_tpu.py",
                "--junitxml",
                str(xml),
            ],
            Q / "hard-negative.log",
            1200,
        )
        cases = list(ET.parse(xml).getroot().iter("testcase"))
        assert len(cases) == 4 and all(
            not any(x.find(tag) is not None for tag in ("failure", "error", "skipped"))
            for x in cases
        ), "Four unskipped physical loss/gradient cases required"
        result["tests"].append(
            dict(id="hard_negative_gradient_masking", status="passed")
        )
        save()
        reference = Q / "reference/checkpoints"
        split = Q / "split/checkpoints"
        for name, out, stop, resume in [
            ("continuous", reference, 4, False),
            ("split-first", split, 2, False),
            ("split-resumed", split, 4, True),
        ]:
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train_yat_embedding_finetune",
                    *stage_arguments(
                        root, out, scope="qualification", stop=stop, resume=resume
                    ),
                ],
                Q / (name + ".log"),
                1200,
            )
        result["recovery"] = compare(
            Q / "continuous.log",
            Q / "split-first.log",
            Q / "split-resumed.log",
            reference,
            split,
        )
        result["tests"] += [
            dict(id=x, status="passed")
            for x in [
                "finite_real_updates",
                "exact_model_Adam_sampler_recovery",
                "matched_rows_losses_gradients_LR",
            ]
        ]
        result["passed"] = True
    except BaseException as e:
        result["failure"] = dict(type=type(e).__name__, message=str(e))
        raise
    finally:
        result["finished_unix"] = time.time()
        save()
        manifest = json.loads((root / "run.json").read_text())
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                str(Q),
                manifest["output_prefix"] + "/qualification",
                "--recursive",
            ],
            check=True,
            timeout=600,
        )
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(a.receipt),
                manifest["output_prefix"] + "/qualification.json",
            ],
            check=True,
            timeout=120,
        )


if __name__ == "__main__":
    main()
