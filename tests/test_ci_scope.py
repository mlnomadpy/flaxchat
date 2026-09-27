from scripts.ci_scope import select_scope


def test_prepared_trainer_change_requires_distributed_validation():
    scope = select_scope(['scripts/train_gpt2.py'])
    assert scope['run_multidevice']
    assert 'tests/test_token_pool.py' in scope['tests']


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


def test_release_workflow_change_runs_policy_tests_without_full_suite():
    scope = select_scope([".github/workflows/release.yml"])
    assert scope == {
        "mode": "targeted",
        "tests": ["tests/test_quality_policy.py"],
        "run_audit": False,
        "run_build": False,
        "run_multidevice": False,
        "run_e2e": False,
    }


def test_non_cpu_workflow_changes_run_only_policy_tests():
    for path in (
        ".github/workflows/deploy.yaml",
        ".github/workflows/kaggle-tpu.yml",
        ".github/workflows/macos-compatibility.yml",
    ):
        scope = select_scope([path])
        assert scope["mode"] == "targeted"
        assert scope["tests"] == ["tests/test_quality_policy.py"]


def test_ci_selector_and_artifact_verifier_have_precise_test_routes():
    assert select_scope(["scripts/ci_scope.py"])["tests"] == ["tests/test_ci_scope.py"]
    assert select_scope(["scripts/verify_artifact.py"])["tests"] == [
        "tests/test_published_artifact.py"
    ]
    assert select_scope(["infra/tpu/flexstart.sh"])["tests"] == [
        "tests/test_quality_policy.py"
    ]


def test_encoder_preflight_and_evidence_regressions_are_selected():
    scope = select_scope(['scripts/train_encoder.py'])
    assert 'tests/test_encoder_snapshot.py' in scope['tests']
    assert scope['run_multidevice']
    assert 'tests/test_encoder_projection_benchmark.py' in select_scope(
        ['scripts/benchmark_encoder_projection.py'])['tests']
    assert 'tests/test_encoder_qualification.py' in select_scope(
        ['scripts/validate_encoder_tpu.py'])['tests']


def test_classifier_training_selects_task_and_multidevice_checks():
    scope = select_scope(['scripts/finetune_encoder_classifier.py'])
    assert scope['tests'] == ['tests/test_encoder_classification.py', 'tests/test_encoder_ner_finetuning.py']
    assert scope['run_multidevice']
    assert select_scope(['scripts/prepare_encoder_classification.py'])['tests'] == [
        'tests/test_encoder_classification.py'
    ]


def test_ner_evaluator_selects_training_metrics_and_multidevice():
    scope = select_scope(['scripts/evaluate_encoder_ner.py'])
    assert scope['tests'] == ['tests/test_encoder_ner_finetuning.py', 'tests/test_ner.py']
    assert scope['run_multidevice']


def test_accumulation_reference_selects_physical_mesh_simulation():
    scope = select_scope(['scripts/replay_mlm_accumulation.py'])
    assert scope['run_multidevice']
    assert 'tests/test_local_mlm_accumulation.py' in scope['tests']
    assert 'tests/test_mlm_accumulation_replay.py' in scope['tests']


def test_tied_embedding_diagnostic_selects_multidevice_validation():
    scope = select_scope(['scripts/diagnose_tied_embedding.py'])
    assert scope['run_multidevice']
    assert 'tests/test_tied_embedding_diagnostic.py' in scope['tests']
