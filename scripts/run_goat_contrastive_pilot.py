"""One bounded new stage; continue only successful committed segments, never retry."""

from pathlib import Path
import argparse
import json
import math
import subprocess
import sys
import time
import threading

from flaxchat.goat_contrastive_campaign import stage_arguments


def next_stop(cursor, elapsed_segment, remaining, *, segment=250, margin=1800):
    """Reserve final evidence and use measured end-to-end duration conservatively."""
    if remaining <= margin:
        return None
    forecast = 5400 if elapsed_segment is None else max(600, elapsed_segment * 1.5)
    if not math.isfinite(forecast) or remaining - margin < forecast:
        return None
    return min(10000, cursor + segment) if cursor < 10000 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    root = a.root
    manifest = json.loads((root / "run.json").read_text())
    if a.output != manifest["output_prefix"] + "/checkpoints":
        raise ValueError("Frozen fresh stage output required")
    q = json.loads((root / "qualification.json").read_text())
    if q.get("passed") is not True:
        raise ValueError("Complete physical qualification required")
    work = root / "production"
    work.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline = started + 27000
    result = dict(
        status="running",
        parent_step=379079,
        optimizer_policy="fresh",
        segments=[],
        latest_committed_step=None,
        maximum_stage_steps=10000,
        work_seconds=27000,
        final_evidence_reserve_seconds=1800,
    )

    def save():
        (work / "status.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    cursor = 0
    elapsed = None
    save()
    try:
        while (
            stop := next_stop(cursor, elapsed, deadline - time.monotonic())
        ) is not None:
            log = work / f"segment-{cursor}-{stop}.log"
            begun = time.monotonic()
            with log.open("wb") as f:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scripts.train_yat_embedding_finetune",
                        *stage_arguments(root, a.output, stop=stop, resume=cursor > 0),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                def tee(process=process, log_file=f):
                    for line in process.stdout:
                        log_file.write(line)
                        log_file.flush()
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()

                reader = threading.Thread(target=tee, daemon=True)
                reader.start()
                try:
                    code = process.wait(
                        timeout=max(1, deadline - time.monotonic() - 1800)
                    )
                except subprocess.TimeoutExpired as error:
                    process.terminate()
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise RuntimeError(
                        "Segment exceeded fixed work/evidence boundary; no retry"
                    ) from error
                finally:
                    reader.join(timeout=30)
            elapsed = time.monotonic() - begun
            if code:
                raise RuntimeError(f"Segment failed ({code}); no retry")
            from flaxchat.checkpoint_metadata import read_committed_metadata

            metadata = read_committed_metadata(a.output, stop, include_receipt=True)
            if metadata["step"] != stop:
                raise ValueError("Missing exact committed recovery checkpoint")
            cursor = stop
            result["latest_committed_step"] = cursor
            result["segments"].append(
                dict(stop=stop, elapsed_seconds=elapsed, returncode=code)
            )
            save()
        result["status"] = "completed_bounded_pilot"
        if cursor == 0:
            raise RuntimeError("No production segment fit remaining bounded window")
    except BaseException as error:
        result.update(status="failed_no_retry", failure=repr(error))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        save()
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                str(work),
                manifest["output_prefix"] + "/production",
                "--recursive",
            ],
            check=True,
            timeout=900,
        )


if __name__ == "__main__":
    main()
