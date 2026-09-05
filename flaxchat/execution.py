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
import subprocess
import tempfile
import threading
import queue
import re
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ExecutionResult:
    """Result of an attempted guarded code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False


_IMAGE_DIGEST = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class IsolatedExecutionConfig:
    """External container boundary for untrusted generated Python."""

    image: str
    runtime: Literal["docker", "podman"] = "docker"
    cpus: float = 1.0
    memory_bytes: int = 256 * 1024 * 1024
    pids_limit: int = 16
    tmpfs_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if not _IMAGE_DIGEST.fullmatch(self.image):
            raise ValueError("container image must be pinned by sha256 digest")
        if self.cpus <= 0 or self.memory_bytes <= 0:
            raise ValueError("container CPU and memory limits must be positive")
        if self.pids_limit < 1 or self.tmpfs_bytes <= 0:
            raise ValueError("container process and tmpfs limits must be positive")

    def command(self, *, container_name: str | None = None) -> tuple[str, ...]:
        """Return a shell-free, deny-by-default container invocation."""
        command = (
            self.runtime,
            "run",
            "--rm",
            "--interactive",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory_bytes}",
            f"--cpus={self.cpus}",
            "--user=65534:65534",
            f"--tmpfs=/tmp:rw,noexec,nosuid,size={self.tmpfs_bytes}",
            "--env=PYTHONHASHSEED=0",
        )
        if container_name is not None:
            command += (f"--name={container_name}",)
        return command + (
            self.image,
            "python",
            "-I",
            "-S",
            "-",
        )


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
    # These APIs are platform-dependent, so avoid static attribute assumptions.
    setattr(os, "lchflags", None)
    setattr(os, "lchmod", None)
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


def _run_bounded_process(
    command: tuple[str, ...], code: str, timeout: float, maximum_output_bytes: int
) -> ExecutionResult:
    """Run an isolated process without allowing pipe output to exhaust the parent."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        start_new_session=os.name != "nt",
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdin = process.stdin
    chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=8)

    def read_stream(name: str, stream) -> None:
        while block := stream.read(8192):
            chunks.put((name, block))
        chunks.put((name, None))

    def write_stdin() -> None:
        try:
            stdin.write(code.encode("utf-8"))
            stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def terminate_process_tree() -> None:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    readers = [
        threading.Thread(target=read_stream, args=("stdout", process.stdout), daemon=True),
        threading.Thread(target=read_stream, args=("stderr", process.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + timeout
    writer = threading.Thread(target=write_stdin, daemon=True)
    writer.start()
    try:
        outputs = {"stdout": bytearray(), "stderr": bytearray()}
        finished_streams = 0
        while finished_streams < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_process_tree()
                return ExecutionResult(timeout=True, error="Execution timed out")
            try:
                name, block = chunks.get(timeout=remaining)
            except queue.Empty:
                terminate_process_tree()
                return ExecutionResult(timeout=True, error="Execution timed out")
            if block is None:
                finished_streams += 1
                continue
            outputs[name].extend(block)
            if sum(map(len, outputs.values())) > maximum_output_bytes:
                terminate_process_tree()
                return ExecutionResult(error="Output limit exceeded")
        return_code = process.wait()
        stdout = outputs["stdout"].decode("utf-8", errors="replace")
        stderr = outputs["stderr"].decode("utf-8", errors="replace")
        return ExecutionResult(
            success=return_code == 0,
            stdout=stdout,
            stderr=stderr,
            error=None if return_code == 0 else f"Isolated process exited {return_code}",
            memory_exceeded=return_code in {137, -9},
        )
    finally:
        if process.poll() is None:
            terminate_process_tree()


def execute_code_isolated(
    code: str,
    config: IsolatedExecutionConfig,
    *,
    timeout: float = 5.0,
    maximum_output_bytes: int = 1_000_000,
) -> ExecutionResult:
    """Execute untrusted Python only inside a hardened external container."""
    if timeout <= 0 or maximum_output_bytes <= 0:
        raise ValueError("timeout and maximum_output_bytes must be positive")
    container_name = f"flaxchat-exec-{uuid.uuid4().hex}"
    try:
        try:
            return _run_bounded_process(
                config.command(container_name=container_name),
                code,
                timeout,
                maximum_output_bytes,
            )
        except FileNotFoundError:
            return ExecutionResult(error=f"isolation runtime {config.runtime!r} not found")
        except OSError as error:
            return ExecutionResult(error=f"isolation runtime failed: {error}")
    finally:
        try:
            subprocess.run(
                (config.runtime, "rm", "--force", container_name),
                check=False,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass


def execute_generated_code(
    code: str,
    *,
    timeout: float = 5.0,
    maximum_output_bytes: int = 1_000_000,
) -> ExecutionResult:
    """Fail closed or use the operator-configured external isolation backend."""
    image = os.environ.get("FLAXCHAT_EXECUTION_IMAGE")
    if not image:
        return ExecutionResult(error="Untrusted Python execution is disabled")
    runtime_name = os.environ.get("FLAXCHAT_EXECUTION_RUNTIME", "docker")
    if runtime_name not in {"docker", "podman"}:
        return ExecutionResult(error="isolation runtime must be docker or podman")
    runtime: Literal["docker", "podman"] = (
        "podman" if runtime_name == "podman" else "docker"
    )
    try:
        config = IsolatedExecutionConfig(image=image, runtime=runtime)
    except ValueError as error:
        return ExecutionResult(error=str(error))
    return execute_code_isolated(
        code,
        config,
        timeout=timeout,
        maximum_output_bytes=maximum_output_bytes,
    )


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
