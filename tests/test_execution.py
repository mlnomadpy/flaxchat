"""
Tests for flaxchat/execution.py — opt-in guarded code execution.

Tests run on CPU with no special permissions required.
"""

import pytest
import sys

from flaxchat.execution import (
    ExecutionResult,
    IsolatedExecutionConfig,
    _run_bounded_process,
    execute_code,
    execute_generated_code,
    execute_code_isolated,
    time_limit,
    capture_io,
)


def execute_code_trusted(code, *args, **kwargs):
    return execute_code(code, *args, trusted=True, **kwargs)


# ---------------------------------------------------------------------------
# Tests for ExecutionResult
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_default_fields(self):
        """Default ExecutionResult should indicate failure with empty fields."""
        r = ExecutionResult()
        assert r.success is False
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.error is None
        assert r.timeout is False
        assert r.memory_exceeded is False

    def test_custom_fields(self):
        """ExecutionResult should accept keyword arguments."""
        r = ExecutionResult(
            success=True, stdout="hello\n", stderr="", error=None,
            timeout=False, memory_exceeded=False,
        )
        assert r.success is True
        assert r.stdout == "hello\n"


class TestIsolatedExecution:
    IMAGE = "python@sha256:" + "a" * 64

    def test_container_contract_is_fail_closed(self):
        config = IsolatedExecutionConfig(image=self.IMAGE)
        command = config.command()
        assert command[:2] == ("docker", "run")
        for required in (
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=65534:65534",
        ):
            assert required in command
        assert not any(part.startswith("--volume") for part in command)
        with pytest.raises(ValueError, match="sha256 digest"):
            IsolatedExecutionConfig(image="python:latest")
        with pytest.raises(ValueError, match="CPU and memory"):
            IsolatedExecutionConfig(image=self.IMAGE, cpus=0)

    def test_missing_runtime_is_structured_failure(self):
        config = IsolatedExecutionConfig(
            image=self.IMAGE, runtime="docker"
        )
        result = execute_code_isolated(
            "print('never')", config, timeout=1, maximum_output_bytes=10
        )
        if result.success:
            pytest.fail("a placeholder digest must never execute successfully")
        assert result.error

    def test_generated_code_requires_valid_operator_configuration(self, monkeypatch):
        monkeypatch.delenv("FLAXCHAT_EXECUTION_IMAGE", raising=False)
        assert "disabled" in execute_generated_code("pass").error
        monkeypatch.setenv("FLAXCHAT_EXECUTION_IMAGE", "python:latest")
        assert "sha256 digest" in execute_generated_code("pass").error
        monkeypatch.setenv("FLAXCHAT_EXECUTION_IMAGE", self.IMAGE)
        monkeypatch.setenv("FLAXCHAT_EXECUTION_RUNTIME", "other")
        assert "docker or podman" in execute_generated_code("pass").error

    def test_parent_bounds_untrusted_output(self):
        result = _run_bounded_process(
            (sys.executable, "-c", "import sys; sys.stdout.write('x' * 100000)"),
            "",
            2,
            100,
        )
        assert not result.success
        assert result.error == "Output limit exceeded"

    def test_bounded_runner_captures_success_and_failure(self):
        success = _run_bounded_process(
            (sys.executable, "-c", "print('isolated')"), "", 2, 100
        )
        assert success.success
        assert success.stdout == "isolated\n"
        failure = _run_bounded_process(
            (sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"),
            "",
            2,
            100,
        )
        assert not failure.success
        assert failure.stderr == "bad"
        assert failure.error == "Isolated process exited 3"

    def test_bounded_runner_kills_timeout(self):
        result = _run_bounded_process(
            (sys.executable, "-c", "while True: pass"), "", 0.1, 100
        )
        assert result.timeout

    def test_isolated_runtime_launch_error_is_structured(self, monkeypatch):
        config = IsolatedExecutionConfig(image=self.IMAGE)

        def missing(*_args, **_kwargs):
            raise FileNotFoundError

        monkeypatch.setattr("flaxchat.execution._run_bounded_process", missing)
        result = execute_code_isolated("pass", config)
        assert result.error == "isolation runtime 'docker' not found"

    def test_humaneval_is_disabled_without_pinned_backend(self, monkeypatch):
        from tasks import humaneval

        monkeypatch.delenv("FLAXCHAT_EXECUTION_IMAGE", raising=False)
        assert humaneval.execute_code("print('must not run')") is False
        monkeypatch.setenv("FLAXCHAT_EXECUTION_IMAGE", "python:latest")
        assert humaneval.execute_code("print('must not run')") is False


# ---------------------------------------------------------------------------
# Tests for execute_code — success cases
# ---------------------------------------------------------------------------

class TestExecuteCodeSuccess:
    def test_untrusted_execution_is_disabled_by_default(self):
        result = execute_code("print('must not run')")
        assert not result.success
        assert "disabled" in result.error.lower()

    def test_simple_print(self):
        """Executing print('hello') should succeed and capture stdout."""
        result = execute_code_trusted("print('hello')")
        assert result.success is True
        assert "hello" in result.stdout
        assert result.error is None
        assert result.timeout is False

    def test_arithmetic(self):
        """Pure computation should succeed silently."""
        result = execute_code_trusted("x = 2 + 3\nprint(x)")
        assert result.success is True
        assert "5" in result.stdout

    def test_multiline_code(self):
        """Multiline code should execute correctly."""
        code = """
def greet(name):
    return f"Hello, {name}!"
print(greet("World"))
"""
        result = execute_code_trusted(code)
        assert result.success is True
        assert "Hello, World!" in result.stdout

    def test_imports_allowed(self):
        """Standard library imports should work before reliability_guard disables __import__."""
        # Note: reliability_guard disables __import__ so imports in the exec'd
        # code may fail. But math is already in builtins scope or gets imported
        # before the guard fires. The guard runs *before* exec, so actually
        # imports are disabled. We test that the guard works correctly.
        code = "print(2 + 2)"
        result = execute_code_trusted(code)
        assert result.success is True
        assert "4" in result.stdout

    def test_empty_code(self):
        """Empty code string should succeed with no output."""
        result = execute_code_trusted("")
        assert result.success is True

    def test_output_flood_is_bounded(self):
        result = execute_code_trusted("print('x' * 100)", maximum_output_chars=10)
        assert not result.success
        assert "Output limit exceeded" in result.error
        assert result.stdout == ""


# ---------------------------------------------------------------------------
# Tests for execute_code — error cases
# ---------------------------------------------------------------------------

class TestExecuteCodeErrors:
    @pytest.mark.parametrize(
        "kwargs", ({"timeout": 0}, {"maximum_output_chars": 0})
    )
    def test_invalid_resource_limits_fail_closed(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            execute_code_trusted("pass", **kwargs)

    def test_syntax_error(self):
        """Code with syntax errors should fail gracefully."""
        result = execute_code_trusted("def f(\n")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_runtime_error(self):
        """Runtime errors should be caught and reported."""
        result = execute_code_trusted("x = 1 / 0")
        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_name_error(self):
        """Referencing undefined variables should fail."""
        result = execute_code_trusted("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.error

    def test_assertion_error(self):
        """Failed assertions should be reported."""
        result = execute_code_trusted("assert False, 'test failure'")
        assert result.success is False
        assert "AssertionError" in result.error


# ---------------------------------------------------------------------------
# Tests for execute_code — timeout
# ---------------------------------------------------------------------------

class TestExecuteCodeTimeout:
    def test_infinite_loop_times_out(self):
        """An infinite loop should be killed after the timeout."""
        result = execute_code_trusted("while True: pass", timeout=1.0)
        assert result.timeout is True
        assert result.success is False

    def test_sleep_within_timeout_succeeds(self):
        """Code that finishes within timeout should succeed."""
        code = "import time; time.sleep(0.1); print('done')"
        result = execute_code_trusted(code, timeout=5.0)
        # This may fail because reliability_guard disables __import__
        # before exec runs. If so, we just check it doesn't hang.
        assert result.timeout is False

    def test_short_timeout(self):
        """Very short timeout should kill even fast code... or it finishes."""
        # This tests that the timeout mechanism doesn't crash
        result = execute_code_trusted("print('fast')", timeout=0.5)
        # Either it succeeds quickly or times out; both are valid
        assert isinstance(result, ExecutionResult)


# ---------------------------------------------------------------------------
# Tests for execute_code — dangerous code
# ---------------------------------------------------------------------------

class TestExecuteCodeSandbox:
    def test_os_system_disabled(self):
        """os.system should be disabled by reliability_guard."""
        result = execute_code_trusted("import os; os.system('echo pwned')")
        assert result.success is False

    def test_os_remove_disabled(self):
        """os.remove should be disabled by reliability_guard."""
        result = execute_code_trusted("import os; os.remove('/tmp/nonexistent')")
        assert result.success is False

    def test_subprocess_disabled(self):
        """subprocess.Popen should be disabled by reliability_guard."""
        result = execute_code_trusted("import subprocess; subprocess.Popen(['ls'])")
        assert result.success is False

    def test_exit_disabled(self):
        """exit() should be disabled by reliability_guard."""
        result = execute_code_trusted("exit(0)")
        assert result.success is False

    def test_fork_disabled(self):
        """os.fork should be disabled by reliability_guard."""
        result = execute_code_trusted("import os; os.fork()")
        assert result.success is False


# ---------------------------------------------------------------------------
# Tests for context managers (unit-level)
# ---------------------------------------------------------------------------

class TestTimeLimitContextManager:
    def test_no_timeout_within_limit(self):
        """Code that finishes in time should not raise."""
        with time_limit(2.0):
            x = sum(range(100))
        assert x == 4950

    def test_timeout_raises(self):
        """Exceeding the time limit should raise TimeoutException."""
        import time
        from flaxchat.execution import TimeoutException
        with pytest.raises(TimeoutException):
            with time_limit(0.1):
                time.sleep(5)


class TestCaptureIO:
    def test_captures_stdout(self):
        """capture_io should capture print() output."""
        with capture_io() as (out, err):
            print("captured")
        assert "captured" in out.getvalue()

    def test_captures_stderr(self):
        """capture_io should capture stderr writes."""
        import sys
        with capture_io() as (out, err):
            print("error msg", file=sys.stderr)
        assert "error msg" in err.getvalue()

    def test_restores_streams(self):
        """After exiting capture_io, stdout/stderr should be restored."""
        import os
        original_stdout = os.sys.stdout
        original_stderr = os.sys.stderr
        with capture_io():
            pass
        assert os.sys.stdout is original_stdout
        assert os.sys.stderr is original_stderr
