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
        "manual_physical_tests": [],
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
        "manual_physical_tests": [],
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


def test_independent_development_routes_admission_and_manual_physical_acceptance():
    assert 'tests/test_gcs_parent_exposure_metadata.py' in select_scope(
        ['scripts/scan_gcs_parent_exposure.py'])['tests']
    assert 'tests/test_parent_exposure_inventory_metadata.py' in select_scope(
        ['scripts/build_parent_exposure_inventory.py'])['tests']
    scope = select_scope(['scripts/prepare_embedding_retrieval_dev.py'])
    assert 'tests/test_independent_retrieval_dev_metadata.py' in scope['tests']
    assert 'tests/test_development_quarantine_metadata.py' in scope['tests']
    assert scope['manual_physical_tests'] == ['tests/test_embedding_trainer_physical_tpu.py']
    assert not scope['run_multidevice'] and not scope['run_e2e']
    scope = select_scope(['scripts/prepare_yat_embedding_finetune.py'])
    assert 'tests/test_independent_retrieval_dev_metadata.py' in scope['tests']
    assert 'tests/test_code_provenance_metadata.py' in scope['tests']


def test_raw_publication_evidence_selects_contract_and_manual_parity():
    scope = select_scope(['scripts/parity_evidence.py'])
    assert scope['tests'] == ['tests/test_evaluation_integration_metadata.py', 'tests/test_release_contract.py']
    assert scope['manual_physical_tests'] == ['tests/test_torch_parity_tpu.py']
    assert not scope['run_multidevice'] and not scope['run_e2e']


def test_parity_campaign_routes_resource_admission_checks():
    scope = select_scope(['scripts/run_yat_parity_campaign.py'])
    assert 'tests/test_parity_resource_admission_metadata.py' in scope['tests']
    assert scope['manual_physical_tests'] == ['tests/test_torch_parity_tpu.py']


def test_new_data_parent_tools_route_metadata_and_keep_tpu_work_manual():
    scope = select_scope(['scripts/filter_representation_development.py',
                          'scripts/prepare_enriched_parent_export.py',
                          'tests/test_embedding_gradient_cache_physical_tpu.py'])
    assert scope['mode'] == 'targeted'
    assert 'tests/test_candidate_collision_filter_metadata.py' in scope['tests']
    assert 'tests/test_enriched_parent_export_metadata.py' in scope['tests']
    assert 'tests/test_cache_diagnostic_metadata.py' in scope['tests']
    assert 'tests/test_embedding_gradient_cache_physical_tpu.py' not in scope['tests']
    assert scope['manual_physical_tests'] == ['tests/test_embedding_gradient_cache_physical_tpu.py']


def test_data_controller_and_reporting_only_helpers_have_no_cpu_model_routes():
    mapping = {
        'scripts/bounded_data_job.py': 'tests/test_bounded_data_job_metadata.py',
        'scripts/diagnose_artifact_transfer.py': 'tests/test_artifact_transfer_metadata.py',
        'scripts/report_full_corpus_retrieval.py': 'tests/test_full_corpus_retrieval_metadata.py',
        'flaxchat/full_corpus_retrieval.py': 'tests/test_full_corpus_retrieval_metadata.py',
        'flaxchat/embedding_telemetry.py': 'tests/test_embedding_telemetry_metadata.py',
        'flaxchat/embedding_uncertainty.py': 'tests/test_embedding_uncertainty_metadata.py',
    }
    for source, test in mapping.items():
        for change in ([source], [test], [source, test]):
            scope = select_scope(change)
            assert scope['mode'] == 'targeted'
            assert scope['tests'] == [test]
            assert scope['manual_physical_tests'] == []
            assert not scope['run_multidevice']
            assert not scope['run_e2e']


def test_metadata_exception_does_not_hide_core_training_changes():
    scope = select_scope(['flaxchat/embedding_telemetry.py', 'flaxchat/embedding.py'])
    assert scope['mode'] == 'full'
    assert 'tests/test_embedding_trainer_physical_tpu.py' in scope['manual_physical_tests']
    assert select_scope(['flaxchat/embedding_telemetry_extra.py'])['mode'] == 'full'


def test_full_corpus_encoder_and_scorer_keep_model_execution_physical():
    for source in ('scripts/evaluate_yat_full_corpus_tpu.py', 'flaxchat/full_corpus_tpu.py'):
        scope = select_scope([source])
        if scope['mode'] == 'targeted':
            assert 'tests/test_yat_full_corpus_tpu_metadata.py' in scope['tests']
        else:
            assert source == 'flaxchat/full_corpus_tpu.py'
        assert 'tests/test_full_corpus_retrieval_physical_tpu.py' in scope['manual_physical_tests']
        assert 'tests/test_full_corpus_retrieval_physical_tpu.py' not in scope['tests']
    scope = select_scope(['tests/test_full_corpus_retrieval_physical_tpu.py'])
    assert scope['tests'] == []
    assert scope['manual_physical_tests'] == ['tests/test_full_corpus_retrieval_physical_tpu.py']
