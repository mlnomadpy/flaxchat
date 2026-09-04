"""Opt-in, best-effort Python reliability guard for trusted evaluation code.

This module is deliberately not a security sandbox. User-facing deployments
must keep generated-code execution disabled unless they supply an independently
reviewed container/VM isolation backend.
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of an attempted guarded code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **kw): raise IOError
    def readline(self, *a, **kw): raise IOError
    def readlines(self, *a, **kw): raise IOError
    def readable(self, *a, **kw): return False


class BoundedStringIO(io.StringIO):
    """Capture at most ``limit`` characters and reject output floods."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def write(self, value: str) -> int:
        if self.tell() + len(value) > self.limit:
            raise RuntimeError("Output limit exceeded")
        return super().write(value)


class _RedirectStdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def time_limit(seconds: float):
    """Raise TimeoutException after `seconds` (no-op on Windows)."""
    if platform.system() == "Windows":
        yield
        return
    def handler(signum, frame):
        raise TimeoutException("Timed out!")
    previous_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextlib.contextmanager
def capture_io(maximum_output_chars: int = 1_000_000):
    """Capture stdout/stderr, block stdin."""
    f_out = BoundedStringIO(maximum_output_chars)
    f_err = BoundedStringIO(maximum_output_chars)
    f_in = WriteOnlyStringIO()
    with contextlib.redirect_stdout(f_out):
        with contextlib.redirect_stderr(f_err):
            with _RedirectStdin(f_in):
                yield f_out, f_err


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        cwd = os.getcwd()
        os.chdir(dirname)
        try:
            yield dirname
        finally:
            os.chdir(cwd)


def _linux_memory_baseline() -> tuple[int, int]:
    """Return the process's current virtual and data mappings in bytes.

    ``RLIMIT_AS`` and ``RLIMIT_DATA`` are absolute ceilings, not allocation
    budgets.  A spawned interpreter can already have more than 256 MiB mapped
    after importing JAX, so applying a raw 256 MiB ceiling can make the dynamic
    loader abort before user code starts.  Linux exposes the two baselines in
    ``/proc/self/statm`` without requiring psutil.
    """
    with open("/proc/self/statm", encoding="ascii") as statm:
        fields = statm.read().split()
    page_size = os.sysconf("SC_PAGE_SIZE")
    return int(fields[0]) * page_size, int(fields[5]) * page_size


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    Disable dangerous functions. NOT a security sandbox — best-effort
    guard against accidental destructive actions from generated code.
    """
    if maximum_memory_bytes is not None and platform.system() == "Linux":
        import resource
        virtual_bytes, data_bytes = _linux_memory_baseline()
        address_limit = virtual_bytes + maximum_memory_bytes
        data_limit = data_bytes + maximum_memory_bytes
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        resource.setrlimit(resource.RLIMIT_DATA, (data_limit, data_limit))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None

    __builtins__["help"] = None

    import sys
    sys.modules["ipdb"] = None  # type: ignore[assignment]
    sys.modules["joblib"] = None  # type: ignore[assignment]
    sys.modules["resource"] = None  # type: ignore[assignment]
    sys.modules["psutil"] = None  # type: ignore[assignment]
    sys.modules["tkinter"] = None  # type: ignore[assignment]


def _unsafe_execute(
    code, timeout, maximum_memory_bytes, maximum_output_chars, result_connection
):
    """Execute code in a subprocess with safety guards."""
    result = {
        "success": False, "stdout": "", "stderr": "",
        "timeout": False, "memory_exceeded": False, "error": None,
    }
    with create_tempdir():
        # Save functions needed for tempdir cleanup
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Never expose parent credentials to evaluated code. This process still
        # is not a sandbox; clearing the environment is defense in depth only.
        os.environ.clear()
        os.environ.update({"PATH": "/usr/bin:/bin", "JAX_PLATFORMS": "cpu"})
        reliability_guard(maximum_memory_bytes)

        try:
            exec_globals = {}
            with capture_io(maximum_output_chars) as (stdout_f, stderr_f):
                with time_limit(timeout):
                    exec(code, exec_globals)
            result.update({
                "success": True,
                "stdout": stdout_f.getvalue(),
                "stderr": stderr_f.getvalue(),
            })
        except TimeoutException:
            result.update({"timeout": True, "error": "Execution timed out"})
        except MemoryError as e:
            result.update({"memory_exceeded": True, "error": f"Memory limit exceeded: {e}"})
        except BaseException as e:
            result.update({"error": f"{type(e).__name__}: {e}"})

        # Restore for cleanup
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink
    result_connection.send(result)
    result_connection.close()


def execute_code(
    code: str,
    timeout: float = 5.0,
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024,
    maximum_output_chars: int = 1_000_000,
    *,
    trusted: bool = False,
) -> ExecutionResult:
    """
    Execute explicitly trusted Python code behind a reliability guard.

    Args:
        code: Python source to execute.
        timeout: Max seconds before kill.
        maximum_memory_bytes: Additional memory budget (None to disable).
        maximum_output_chars: Combined per-stream output bound.
        trusted: Explicit acknowledgement that code is trusted. Untrusted code
            is disabled by default because a subprocess and denylist are not a
            security boundary.

    Returns:
        ExecutionResult with success, stdout, stderr, error, timeout, memory_exceeded.
    """
    if not trusted:
        return ExecutionResult(
            error=(
                "Untrusted Python execution is disabled. Configure an isolated "
                "container/VM backend or pass trusted=True only for reviewed code."
            )
        )
    if timeout <= 0 or maximum_output_chars <= 0:
        raise ValueError("timeout and maximum_output_chars must be positive")

    # JAX owns background threads, so forking the parent process can deadlock.
    # A fresh interpreter is slower to start but safe on every supported OS.
    # On an accelerator host the isolated child must not reacquire the parent's
    # exclusive TPU/GPU runtime; generated Python is intentionally CPU-only.
    context = multiprocessing.get_context("spawn")
    previous_platforms = os.environ.get("JAX_PLATFORMS")
    os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_unsafe_execute,
            args=(code, timeout, maximum_memory_bytes, maximum_output_chars, sender),
        )
        process.start()
        sender.close()
        process.join(timeout=timeout + 1)

        if process.is_alive():
            process.kill()
            process.join()
            receiver.close()
            return ExecutionResult(timeout=True, error="Execution timed out (process killed)")

        try:
            result = receiver.recv() if receiver.poll() else None
        except EOFError:
            result = None
        finally:
            receiver.close()
        if result is None:
            return ExecutionResult(error="Execution failed (no result)")

        return ExecutionResult(
            success=result.get("success", False),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            error=result.get("error"),
            timeout=result.get("timeout", False),
            memory_exceeded=result.get("memory_exceeded", False),
        )
    finally:
        if previous_platforms is None:
            os.environ.pop("JAX_PLATFORMS", None)
        else:
            os.environ["JAX_PLATFORMS"] = previous_platforms
