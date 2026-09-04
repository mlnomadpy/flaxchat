from scripts.ci_scope import select_scope


def test_core_change_keeps_full_validation_and_relevant_expensive_checks():
    scope = select_scope(["flaxchat/checkpoint.py", "flaxchat/engine.py"])
    assert scope["mode"] == "full"
    assert scope["run_multidevice"] is True
    assert scope["run_e2e"] is True
    assert scope["run_audit"] is False


def test_benchmark_change_only_selects_benchmark_tests():
    scope = select_scope(["benchmarks/compare.py"])
    assert scope == {
        "mode": "targeted",
        "tests": [
            "tests/test_benchmark_compare.py",
            "tests/test_benchmark_protocol.py",
            "tests/test_matched_benchmark.py",
            "tests/test_training_scaling.py",
        ],
        "run_audit": False,
        "run_build": False,
        "run_multidevice": False,
        "run_e2e": False,
    }


def test_changed_test_runs_without_global_coverage_job():
    scope = select_scope(["tests/test_chat.py"])
    assert scope["mode"] == "targeted"
    assert scope["tests"] == ["tests/test_chat.py"]


def test_dependency_change_runs_full_audit():
    scope = select_scope(["pixi.lock"])
    assert scope["mode"] == "full"
    assert scope["run_audit"] is True


def test_manual_dispatch_forces_full_validation():
    assert select_scope([], force_full=True)["mode"] == "full"


def test_kaggle_monitor_change_only_runs_its_contract_tests():
    scope = select_scope(["scripts/kaggle_tpu_tests.py", "tests/test_kaggle_launcher.py"])
    assert scope["mode"] == "targeted"
    assert scope["tests"] == ["tests/test_kaggle_launcher.py"]


def test_accelerator_template_change_runs_launcher_contract_only():
    scope = select_scope(["accelerators/kaggle/matched.py"])
    assert scope["mode"] == "targeted"
    assert scope["tests"] == ["tests/test_kaggle_launcher.py"]
