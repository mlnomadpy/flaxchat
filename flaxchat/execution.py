"""
Sandboxed Python code execution for flaxchat evaluation.

Ported from nanochat's execution.py. Uses multiprocessing for isolation
with signal-based timeouts, IO capture, and a reliability guard that
disables dangerous builtins and modules.
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
    """Result of a sandboxed code execution."""
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
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def capture_io():
    """Capture stdout/stderr, block stdin."""
    f_out = io.StringIO()
    f_err = io.StringIO()
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


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    Disable dangerous functions. NOT a security sandbox — best-effort
    guard against accidental destructive actions from generated code.
    """
    if maximum_memory_bytes is not None and platform.uname().system != "Darwin":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))

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
    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _unsafe_execute(code, timeout, maximum_memory_bytes, result_dict):
    """Execute code in a subprocess with safety guards."""
    with create_tempdir():
        # Save functions needed for tempdir cleanup
        import os
        import shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        reliability_guard(maximum_memory_bytes)

        result_dict.update({
            "success": False, "stdout": "", "stderr": "",
            "timeout": False, "memory_exceeded": False, "error": None,
        })

        try:
            exec_globals = {}
            with capture_io() as (stdout_f, stderr_f):
                with time_limit(timeout):
                    exec(code, exec_globals)
            result_dict.update({
                "success": True,
                "stdout": stdout_f.getvalue(),
                "stderr": stderr_f.getvalue(),
            })
        except TimeoutException:
            result_dict.update({"timeout": True, "error": "Execution timed out"})
        except MemoryError as e:
            result_dict.update({"memory_exceeded": True, "error": f"Memory limit exceeded: {e}"})
        except BaseException as e:
            result_dict.update({"error": f"{type(e).__name__}: {e}"})

        # Restore for cleanup
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0,
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024,
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed subprocess.

    Args:
        code: Python source to execute.
        timeout: Max seconds before kill.
        maximum_memory_bytes: Memory limit (None to disable).

    Returns:
        ExecutionResult with success, stdout, stderr, error, timeout, memory_exceeded.
    """
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict),
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return ExecutionResult(timeout=True, error="Execution timed out (process killed)")

    if not result_dict:
        return ExecutionResult(error="Execution failed (no result)")

    return ExecutionResult(
        success=result_dict.get("success", False),
        stdout=result_dict.get("stdout", ""),
        stderr=result_dict.get("stderr", ""),
        error=result_dict.get("error"),
        timeout=result_dict.get("timeout", False),
        memory_exceeded=result_dict.get("memory_exceeded", False),
    )
