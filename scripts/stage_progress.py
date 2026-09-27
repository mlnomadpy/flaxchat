"""Run bounded stages while preserving diagnostic logs before stage completion."""

from collections.abc import Callable, Mapping, Sequence
import json
import math
from pathlib import Path
import subprocess
import time


def run_stage(
    argv: Sequence[str],
    *,
    log: Path,
    env: Mapping[str, str],
    timeout: float,
    publish: Callable[[Path, float], None],
    interval: float = 60,
) -> int:
    """Publish partial logs; only the child exit status determines stage success.

    Publisher failures are diagnostic and do not stop training. The publisher
    receives the remaining deadline and must bound its own I/O by that value.
    Status is written atomically beside the log, never to SSH stdout: a blocked
    transport must not prevent polling or deadline enforcement. Publishers may
    upload this sidecar with the log. Final authoritative evidence publication
    remains the caller's responsibility.
    """
    if not all(math.isfinite(v) and v > 0 for v in (timeout, interval)):
        raise ValueError("Require positive finite timeout and interval")
    end = time.monotonic() + timeout
    wall_end = time.time() + timeout
    status_path = log.with_suffix('.status.json')
    record = {}

    def remaining_seconds():
        return min(end - time.monotonic(), wall_end - time.time())

    def status(state, **details):
        record.update(state=state, observed_epoch=time.time(),
                      log_bytes=log.stat().st_size, deadline_epoch=wall_end, **details)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record) + '\n')
        temporary.replace(status_path)
    with log.open("w") as stream:
        with subprocess.Popen(
            argv,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=dict(env) | {"PYTHONUNBUFFERED": "1"},
        ) as process:
            try:
                status('running', pid=process.pid)
                while True:
                    remaining = remaining_seconds()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    try:
                        code = process.wait(timeout=min(interval, remaining))
                        status('exited', pid=process.pid, returncode=code)
                        return code
                    except subprocess.TimeoutExpired:
                        remaining = remaining_seconds()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(argv, timeout) from None
                        stream.flush()
                        status('publishing', pid=process.pid)
                        try:
                            publish(log, remaining)
                        except (OSError, subprocess.SubprocessError) as error:
                            status('running', pid=process.pid,
                                   publication_error=f'{type(error).__name__}: {error}')
                        else:
                            status('running', pid=process.pid, last_publication_epoch=time.time())
            except subprocess.TimeoutExpired:
                status('timed_out', pid=process.pid)
                raise
            except BaseException as error:
                status('monitor_failed', pid=process.pid, error=f'{type(error).__name__}: {error}')
                raise
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
