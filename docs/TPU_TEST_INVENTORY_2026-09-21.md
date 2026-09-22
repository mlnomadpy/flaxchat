# Physical TPU test inventory — 2026-09-21

406 passed; 2 skipped; 0 failed. Host: v5e, eight devices, one process.

This is the complete pytest inventory. Tests that explicitly spawn CPU/virtual-device subprocesses are CPU evidence, even though the parent suite ran on a TPU VM. The run also includes hardware TPU attention parity and real-data training acceptance.

[Full validation report](TPU_VALIDATION_2026-09-21.md) · [Machine-readable evidence](../benchmarks/results/gcp-spot-v5e-8-20260921.json)

| Module | Passed | Skipped |
|---|---:|---:|
| test_attention_accelerator | 4 | 0 |
| test_benchmark_compare | 9 | 0 |
| test_benchmark_protocol | 1 | 0 |
| test_chat | 7 | 0 |
| test_chat_web | 6 | 0 |
| test_checkpoint | 16 | 0 |
| test_checkpoint_topology | 1 | 0 |
| test_ci_scope | 10 | 0 |
| test_common | 24 | 0 |
| test_config | 35 | 0 |
| test_dataloader | 16 | 0 |
| test_dataset | 7 | 0 |
| test_docs | 3 | 0 |
| test_engine | 23 | 0 |
| test_eval | 13 | 0 |
| test_execution | 45 | 0 |
| test_kaggle_launcher | 18 | 0 |
| test_matched_benchmark | 7 | 0 |
| test_model | 34 | 1 |
| test_multihost_acceptance | 9 | 0 |
| test_optim | 14 | 0 |
| test_pipeline | 4 | 0 |
| test_prefetch | 9 | 0 |
| test_published_artifact | 3 | 0 |
| test_quality_policy | 13 | 0 |
| test_report | 4 | 0 |
| test_result_provenance | 5 | 0 |
| test_sharding | 7 | 1 |
| test_stage_functions | 13 | 0 |
| test_token_pool | 5 | 0 |
| test_tokenizer | 30 | 0 |
| test_tpu_validation | 2 | 0 |
| test_training_safety | 4 | 0 |
| test_training_scaling | 5 | 0 |

## test_attention_accelerator

- **PASS** `tests.test_attention_accelerator::test_splash_matches_xla_forward_and_gradient[False-32]`
- **PASS** `tests.test_attention_accelerator::test_splash_matches_xla_forward_and_gradient[False-128]`
- **PASS** `tests.test_attention_accelerator::test_splash_matches_xla_forward_and_gradient[True-32]`
- **PASS** `tests.test_attention_accelerator::test_splash_matches_xla_forward_and_gradient[True-128]`

## test_benchmark_compare

- **PASS** `tests.test_benchmark_compare::test_compare_requires_identical_controls`
- **PASS** `tests.test_benchmark_compare::test_compare_reports_normalized_throughput`
- **PASS** `tests.test_benchmark_compare::test_record_schema_fails_closed`
- **PASS** `tests.test_benchmark_compare::test_record_rejects_placeholder_revision_and_nonfinite_values`
- **PASS** `tests.test_benchmark_compare::test_record_rejects_boolean_and_fractional_integer_fields`
- **PASS** `tests.test_benchmark_compare::test_comparison_rejects_duplicate_frameworks`
- **PASS** `tests.test_benchmark_compare::test_superlinear_efficiency_is_valid`
- **PASS** `tests.test_benchmark_compare::test_parameter_counts_may_differ_inside_declared_tolerance`
- **PASS** `tests.test_benchmark_compare::test_protocol_hash_uses_exact_file_bytes`

## test_benchmark_protocol

- **PASS** `tests.test_benchmark_protocol::test_pinned_baseline_plans_match_canonical_protocol`

## test_chat

- **PASS** `tests.test_chat::test_generation_is_seed_deterministic`
- **PASS** `tests.test_chat::test_context_limit_fails_before_generation`
- **PASS** `tests.test_chat::test_empty_prompt_is_rejected`
- **PASS** `tests.test_chat::test_generate_text_stops_at_assistant_end`
- **PASS** `tests.test_chat::test_stream_honors_cancellation_before_decode`
- **PASS** `tests.test_chat::test_stream_preserves_byte_tokenizer_unicode`
- **PASS** `tests.test_chat::test_loader_rejects_unknown_checkpoint_type_before_io`

## test_chat_web

- **PASS** `tests.test_chat_web::test_import_has_no_model_loading`
- **PASS** `tests.test_chat_web::test_application_factory_health_and_websocket_protocol`
- **PASS** `tests.test_chat_web::test_websocket_rejects_oversized_input`
- **PASS** `tests.test_chat_web::test_websocket_disconnect_cancels_silent_generation`
- **PASS** `tests.test_chat_web::test_websocket_returns_sanitized_model_error`
- **PASS** `tests.test_chat_web::test_web_settings_reject_nonpositive_bounds`

## test_checkpoint

- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_async_policy_is_honored[True]`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_async_policy_is_honored[False]`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_creates_directory`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_returns_manager`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_max_to_keep`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_existing_directory_ok`
- **PASS** `tests.test_checkpoint.TestCreateCheckpointManager::test_relative_directory_is_normalized_for_tensorstore`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_save_and_load_latest`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_save_and_load_specific_step`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_load_no_checkpoints_raises`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_param_values_preserved`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_manifest_detects_modified_state_before_mutation`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_partial_checkpoint_fails_before_mutating_model`
- **PASS** `tests.test_checkpoint.TestSaveLoadRoundTrip::test_interrupted_training_matches_uninterrupted_training`
- **PASS** `tests.test_checkpoint.TestRestoreModelFromCheckpoint::test_restores_in_place`
- **PASS** `tests.test_checkpoint.TestRestoreModelFromCheckpoint::test_returns_metadata`

## test_checkpoint_topology

- **PASS** `tests.test_checkpoint_topology::test_checkpoint_restores_from_eight_devices_to_one`

## test_ci_scope

- **PASS** `tests.test_ci_scope::test_core_change_keeps_full_validation_and_relevant_expensive_checks`
- **PASS** `tests.test_ci_scope::test_benchmark_change_only_selects_benchmark_tests`
- **PASS** `tests.test_ci_scope::test_changed_test_runs_without_global_coverage_job`
- **PASS** `tests.test_ci_scope::test_dependency_change_runs_full_audit`
- **PASS** `tests.test_ci_scope::test_manual_dispatch_forces_full_validation`
- **PASS** `tests.test_ci_scope::test_kaggle_monitor_change_only_runs_its_contract_tests`
- **PASS** `tests.test_ci_scope::test_accelerator_template_change_runs_launcher_contract_only`
- **PASS** `tests.test_ci_scope::test_release_workflow_change_runs_policy_tests_without_full_suite`
- **PASS** `tests.test_ci_scope::test_non_cpu_workflow_changes_run_only_policy_tests`
- **PASS** `tests.test_ci_scope::test_ci_selector_and_artifact_verifier_have_precise_test_routes`

## test_common

- **PASS** `tests.test_common.TestComputeDtype::test_dtype_is_valid`
- **PASS** `tests.test_common.TestComputeDtype::test_reason_is_string`
- **PASS** `tests.test_common.TestGetBaseDir::test_returns_string`
- **PASS** `tests.test_common.TestGetPeakFlops::test_known_tpu`
- **PASS** `tests.test_common.TestGetPeakFlops::test_unknown_device`
- **PASS** `tests.test_common.TestGetPeakFlops::test_case_insensitive`
- **PASS** `tests.test_common.TestDummyWandb::test_log_noop`
- **PASS** `tests.test_common.TestComputeInit::test_setup_mesh_registers_global_mesh`
- **PASS** `tests.test_common.TestComputeInit::test_runtime_initializes_before_backend_discovery`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment0]`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment1]`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment2]`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment3]`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment4]`
- **PASS** `tests.test_common.TestComputeInit::test_multi_process_launchers_are_detected[environment5]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment0]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment1]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment2]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment3]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment4]`
- **PASS** `tests.test_common.TestComputeInit::test_single_process_markers_are_not_distributed[environment5]`
- **PASS** `tests.test_common.TestComputeInit::test_distributed_initialization_precedes_topology_queries`
- **PASS** `tests.test_common.TestComputeInit::test_initialized_launcher_is_not_initialized_twice`
- **PASS** `tests.test_common.TestComputeInit::test_kaggle_single_host_does_not_initialize_distributed`

## test_config

- **PASS** `tests.test_config.TestGPTConfig::test_defaults`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_defaults`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_depth_12`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_depth_24`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_depth_with_overrides`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_dict`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_dict_with_depth`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_yaml`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_from_json`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_to_dict`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_roundtrip`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[model]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[training]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[tpu]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[checkpoint]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[logging]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[tokenizer]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[data]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[evaluation]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_nested_field_is_rejected[generation]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_unknown_depth_override_is_rejected`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_depth_can_be_combined_with_training_section`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_model_contract_is_rejected[kwargs0-divisible by n_head]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_model_contract_is_rejected[kwargs1-divisible by n_kv_head]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_model_contract_is_rejected[kwargs2-window_pattern]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_model_contract_is_rejected[kwargs3-attention_backend]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_tpu_contract_is_rejected`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section0-device_batch_size]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section1-warmdown_ratio]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section2-max_to_keep]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section3-fsdp]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section4-log_interval]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section5-vocab_size]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section6-max_per_task]`
- **PASS** `tests.test_config.TestFlaxChatConfig::test_invalid_mutable_section_contract_is_rejected[section7-temperature]`

## test_dataloader

- **PASS** `tests.test_dataloader.TestDocumentBatches::test_two_hosts_are_disjoint_and_resume_global_order`
- **PASS** `tests.test_dataloader.TestDocumentBatches::test_yields_text_batches`
- **PASS** `tests.test_dataloader.TestDocumentBatches::test_resume_state`
- **PASS** `tests.test_dataloader.TestDocumentBatches::test_multi_host_sharding`
- **PASS** `tests.test_dataloader.TestDocumentBatches::test_val_split_uses_last_shard`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_corrupt_document_is_not_skipped`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_resume_reproduces_exact_next_batches[1]`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_resume_reproduces_exact_next_batches[2]`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_resume_reproduces_exact_next_batches[5]`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_resume_reproduces_exact_next_batches[9]`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_resume_rejects_changed_topology_or_packing`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_output_shapes`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_targets_are_shifted_inputs`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_state_dict_returned`
- **PASS** `tests.test_dataloader.TestDataLoaderBOSBestFit::test_invalid_split_raises`
- **PASS** `tests.test_dataloader.TestDataLoaderNoState::test_omits_state_dict`

## test_dataset

- **PASS** `tests.test_dataset::test_dataset_url_is_pinned_to_immutable_revision`
- **PASS** `tests.test_dataset::test_download_shard_uses_canonical_name`
- **PASS** `tests.test_dataset::test_download_shard_rejects_out_of_range_ids[-1]`
- **PASS** `tests.test_dataset::test_download_shard_rejects_out_of_range_ids[6542]`
- **PASS** `tests.test_dataset::test_download_range_is_validated_before_network_access[-1-1]`
- **PASS** `tests.test_dataset::test_download_range_is_validated_before_network_access[2-1]`
- **PASS** `tests.test_dataset::test_download_range_is_validated_before_network_access[0-6543]`

## test_docs

- **PASS** `tests.test_docs::test_validator_reports_missing_targets_and_unclosed_fences`
- **PASS** `tests.test_docs::test_validator_extracts_mermaid_blocks_for_the_real_parser`
- **PASS** `tests.test_docs::test_validator_rejects_empty_mermaid_blocks`

## test_engine

- **PASS** `tests.test_engine.TestGenerate::test_basic_generation`
- **PASS** `tests.test_engine.TestGenerate::test_greedy_deterministic`
- **PASS** `tests.test_engine.TestGenerate::test_top_k`
- **PASS** `tests.test_engine.TestGenerate::test_single_token_prompt`
- **PASS** `tests.test_engine.TestGenerate::test_shakespeare_generation`
- **PASS** `tests.test_engine.TestGenerateWithCache::test_cache_uses_model_placement`
- **PASS** `tests.test_engine.TestGenerateWithCache::test_basic_cached_generation`
- **PASS** `tests.test_engine.TestGenerateWithCache::test_greedy_deterministic`
- **PASS** `tests.test_engine.TestGenerateWithCache::test_matches_simple_generate`
- **PASS** `tests.test_engine.TestGenerateWithCache::test_top_k_cached`
- **PASS** `tests.test_engine.TestGenerateFast::test_fast_generation_basic`
- **PASS** `tests.test_engine.TestGenerateFast::test_fast_greedy_deterministic`
- **PASS** `tests.test_engine.TestGenerateFast::test_fast_matches_cached`
- **PASS** `tests.test_engine.TestSpeculativeDecoding::test_speculative_basic`
- **PASS** `tests.test_engine.TestSpeculativeDecoding::test_speculative_greedy_matches`
- **PASS** `tests.test_engine.TestSpeculativeDecoding::test_speculative_greedy_matches_after_rejection`
- **PASS** `tests.test_engine.TestSpeculativeDecoding::test_speculative_validates_decode_window`
- **PASS** `tests.test_engine.TestCalculator::test_basic_math`
- **PASS** `tests.test_engine.TestCalculator::test_string_count`
- **PASS** `tests.test_engine.TestCalculator::test_rejects_dangerous`
- **PASS** `tests.test_engine.TestCalculator::test_rejects_power`
- **PASS** `tests.test_engine.TestCalculator::test_comma_removal`
- **PASS** `tests.test_engine.TestExecuteCode::test_execute_code_tool`

## test_eval

- **PASS** `tests.test_eval::test_multiple_choice_prompt_matches_golden_reference`
- **PASS** `tests.test_eval::test_language_model_prompt_matches_golden_reference`
- **PASS** `tests.test_eval::test_multiple_choice_label_and_scores_match_golden_reference`
- **PASS** `tests.test_eval.TestForwardModel::test_output_shapes`
- **PASS** `tests.test_eval.TestForwardModel::test_last_position_nan`
- **PASS** `tests.test_eval.TestForwardModel::test_losses_positive`
- **PASS** `tests.test_eval.TestFindCommonLength::test_common_prefix`
- **PASS** `tests.test_eval.TestFindCommonLength::test_no_common_prefix`
- **PASS** `tests.test_eval.TestFindCommonLength::test_common_suffix`
- **PASS** `tests.test_eval.TestFindCommonLength::test_identical_sequences`
- **PASS** `tests.test_eval.TestCoreProtocol::test_declared_fewshot_count_and_manifest`
- **PASS** `tests.test_eval.TestCoreProtocol::test_task_failure_invalidates_aggregate`
- **PASS** `tests.test_eval.TestRenderMC::test_basic`

## test_execution

- **PASS** `tests.test_execution.TestExecutionResult::test_default_fields`
- **PASS** `tests.test_execution.TestExecutionResult::test_custom_fields`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_container_contract_is_fail_closed`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_missing_runtime_is_structured_failure`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_generated_code_requires_valid_operator_configuration`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_parent_bounds_untrusted_output`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_bounded_runner_captures_success_and_failure`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_bounded_runner_kills_timeout`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_bounded_runner_times_out_while_child_ignores_large_stdin`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_bounded_runner_kills_descendants_holding_output_descriptors`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_isolated_runner_always_attempts_named_container_cleanup`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[open('/tmp/escape', 'w').write('x')]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[from pathlib import Path; Path('/tmp/escape').write_text('x')]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[import socket; socket.create_connection(('example.com', 80))]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[import ctypes; ctypes.CDLL(None)]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[import importlib; importlib.import_module('subprocess')]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_adversarial_payloads_fail_closed_without_backend[import os; os.fork()]`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_isolated_runtime_launch_error_is_structured`
- **PASS** `tests.test_execution.TestIsolatedExecution::test_humaneval_is_disabled_without_pinned_backend`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_untrusted_execution_is_disabled_by_default`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_simple_print`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_arithmetic`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_multiline_code`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_imports_allowed`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_empty_code`
- **PASS** `tests.test_execution.TestExecuteCodeSuccess::test_output_flood_is_bounded`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_invalid_resource_limits_fail_closed[kwargs0]`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_invalid_resource_limits_fail_closed[kwargs1]`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_syntax_error`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_runtime_error`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_name_error`
- **PASS** `tests.test_execution.TestExecuteCodeErrors::test_assertion_error`
- **PASS** `tests.test_execution.TestExecuteCodeTimeout::test_infinite_loop_times_out`
- **PASS** `tests.test_execution.TestExecuteCodeTimeout::test_sleep_within_timeout_succeeds`
- **PASS** `tests.test_execution.TestExecuteCodeTimeout::test_short_timeout`
- **PASS** `tests.test_execution.TestExecuteCodeSandbox::test_os_system_disabled`
- **PASS** `tests.test_execution.TestExecuteCodeSandbox::test_os_remove_disabled`
- **PASS** `tests.test_execution.TestExecuteCodeSandbox::test_subprocess_disabled`
- **PASS** `tests.test_execution.TestExecuteCodeSandbox::test_exit_disabled`
- **PASS** `tests.test_execution.TestExecuteCodeSandbox::test_fork_disabled`
- **PASS** `tests.test_execution.TestTimeLimitContextManager::test_no_timeout_within_limit`
- **PASS** `tests.test_execution.TestTimeLimitContextManager::test_timeout_raises`
- **PASS** `tests.test_execution.TestCaptureIO::test_captures_stdout`
- **PASS** `tests.test_execution.TestCaptureIO::test_captures_stderr`
- **PASS** `tests.test_execution.TestCaptureIO::test_restores_streams`

## test_kaggle_launcher

- **PASS** `tests.test_kaggle_launcher::test_acceptance_bundle_installs_all_test_feature_extras`
- **PASS** `tests.test_kaggle_launcher::test_kaggle_launcher_requires_immutable_full_revision`
- **PASS** `tests.test_kaggle_launcher::test_launch_spec_round_trip_and_secret_values_are_rejected`
- **PASS** `tests.test_kaggle_launcher::test_launch_spec_executes_argv_without_shell`
- **PASS** `tests.test_kaggle_launcher::test_gcp_lifecycle_tears_down_after_failure[up]`
- **PASS** `tests.test_kaggle_launcher::test_gcp_lifecycle_tears_down_after_failure[run]`
- **PASS** `tests.test_kaggle_launcher::test_gcp_lifecycle_resume_collect_and_teardown`
- **PASS** `tests.test_kaggle_launcher::test_platform_dry_runs_share_one_manifest_contract`
- **PASS** `tests.test_kaggle_launcher::test_training_bundle_runs_real_pipeline_at_exact_revision`
- **PASS** `tests.test_kaggle_launcher::test_fineweb_gpt2_bundle_uses_gpt2_tokenizer_below_vocab_ceiling`
- **PASS** `tests.test_kaggle_launcher::test_monitor_recovers_after_transport_reset`
- **PASS** `tests.test_kaggle_launcher::test_resume_monitor_never_submits_a_new_version`
- **PASS** `tests.test_kaggle_launcher::test_partial_artifact_download_keeps_previous_complete_output`
- **PASS** `tests.test_kaggle_launcher::test_monitor_exhausts_transport_outage_budget`
- **PASS** `tests.test_kaggle_launcher::test_kernel_error_is_not_treated_as_transport_failure`
- **PASS** `tests.test_kaggle_launcher::test_kaggle_command_bounds_a_hung_cli`
- **PASS** `tests.test_kaggle_launcher::test_matched_gpu_bundle_pins_all_three_repositories`
- **PASS** `tests.test_kaggle_launcher::test_matched_preflight_uses_cpu_before_spending_gpu_quota`

## test_matched_benchmark

- **PASS** `tests.test_matched_benchmark::test_framework_neutral_byte_encoding_is_deterministic`
- **PASS** `tests.test_matched_benchmark::test_sequences_use_next_token_targets`
- **PASS** `tests.test_matched_benchmark::test_batch_loader_fails_closed_on_shape_drift`
- **PASS** `tests.test_matched_benchmark::test_record_carries_shared_data_identity`
- **PASS** `tests.test_matched_benchmark::test_hardware_guard_rejects_unmatched_accelerators`
- **PASS** `tests.test_matched_benchmark::test_parameter_budget_is_inclusive_at_five_percent`
- **PASS** `tests.test_matched_benchmark::test_nanochat_token_batches_are_promoted_to_int64`

## test_model

- **PASS** `tests.test_model.TestRMSNorm::test_output_shape`
- **PASS** `tests.test_model.TestRMSNorm::test_normalization`
- **PASS** `tests.test_model.TestRMSNorm::test_zero_input`
- **PASS** `tests.test_model.TestRotaryEmbeddings::test_shape`
- **PASS** `tests.test_model.TestRotaryEmbeddings::test_apply_shape`
- **PASS** `tests.test_model.TestRotaryEmbeddings::test_deterministic`
- **PASS** `tests.test_model.TestHasVE::test_alternating`
- **PASS** `tests.test_model.TestHasVE::test_last_layer_included`
- **PASS** `tests.test_model.TestMLP::test_forward_shape`
- **PASS** `tests.test_model.TestMLP::test_relu_squared`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_auto_backend_metadata_includes_sequence_compatibility`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_xla_attention_matches_dense_reference[3]`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_xla_attention_matches_dense_reference[8]`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_xla_attention_gradient_matches_dense_reference`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_xla_attention_supports_padding_and_grouped_query_heads`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_attention_rejects_invalid_padding_shape`
- **SKIP** `tests.test_model.TestCausalSelfAttention::test_splash_fails_clearly_without_tpu` — CPU/GPU fallback test
- **PASS** `tests.test_model.TestCausalSelfAttention::test_forward_shape`
- **PASS** `tests.test_model.TestCausalSelfAttention::test_causal_masking`
- **PASS** `tests.test_model.TestBlock::test_forward_shape`
- **PASS** `tests.test_model.TestGPT::test_declared_rng_controls_initialization`
- **PASS** `tests.test_model.TestGPT::test_construction`
- **PASS** `tests.test_model.TestGPT::test_forward_logits`
- **PASS** `tests.test_model.TestGPT::test_forward_loss`
- **PASS** `tests.test_model.TestGPT::test_forward_shakespeare`
- **PASS** `tests.test_model.TestGPT::test_num_params`
- **PASS** `tests.test_model.TestGPT::test_estimate_flops`
- **PASS** `tests.test_model.TestGPT::test_window_sizes`
- **PASS** `tests.test_model.TestGPT::test_value_embeddings_alternating`
- **PASS** `tests.test_model.TestGPT::test_ignore_index`
- **PASS** `tests.test_model.TestGPT::test_softcap`
- **PASS** `tests.test_model.TestGPT::test_jit_compatible`
- **PASS** `tests.test_model.TestGPT::test_grad_computable`
- **PASS** `tests.test_model.TestGPTSmall::test_forward`
- **PASS** `tests.test_model.TestGPTSmall::test_loss`

## test_multihost_acceptance

- **PASS** `tests.test_multihost_acceptance::test_summary_requires_disjoint_complete_matching_workers`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[overlap]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[duplicate]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[loss]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[nonfinite]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[revision]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[digest]`
- **PASS** `tests.test_multihost_acceptance::test_summary_fails_closed_on_inconsistent_evidence[failed]`
- **PASS** `tests.test_multihost_acceptance::test_summary_rejects_single_host_and_invalid_cost`

## test_optim

- **PASS** `tests.test_optim.TestMuonOptimizer::test_init_and_step`
- **PASS** `tests.test_optim.TestMuonOptimizer::test_tall_and_wide_matrices`
- **PASS** `tests.test_optim.TestMuonOptimizer::test_weight_decay`
- **PASS** `tests.test_optim.TestMuonOptimizer::test_polar_express_coefficients`
- **PASS** `tests.test_optim.TestSetupOptimizer::test_creates_optimizer`
- **PASS** `tests.test_optim.TestSetupOptimizer::test_optimizer_step`
- **PASS** `tests.test_optim.TestSetupOptimizer::test_loss_decreases`
- **PASS** `tests.test_optim.TestLRSchedule::test_warmup`
- **PASS** `tests.test_optim.TestLRSchedule::test_constant_phase`
- **PASS** `tests.test_optim.TestLRSchedule::test_warmdown`
- **PASS** `tests.test_optim.TestLRSchedule::test_final_lr_frac`
- **PASS** `tests.test_optim.TestWDSchedule::test_cosine_decay`
- **PASS** `tests.test_optim.TestMuonMomentumSchedule::test_warmup`
- **PASS** `tests.test_optim.TestMuonMomentumSchedule::test_stable_phase`

## test_pipeline

- **PASS** `tests.test_pipeline::test_pipeline_config_rejects_invalid_shapes`
- **PASS** `tests.test_pipeline::test_distributed_tpu_attention_fallback_is_explicit`
- **PASS** `tests.test_pipeline::test_fixture_split_is_deterministic`
- **PASS** `tests.test_pipeline::test_complete_pipeline_emits_restorable_artifacts`

## test_prefetch

- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_basic_iteration`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_next_protocol`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_prefetch_ahead`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_stop_cleanup`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_empty_data`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_arrays_are_jax`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_worker_failure_is_not_silently_reported_as_eof`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_queue_applies_bounded_backpressure`
- **PASS** `tests.test_prefetch.TestBackgroundPrefetcher::test_invalid_prefetch_count_fails_early`

## test_published_artifact

- **PASS** `tests.test_published_artifact::test_published_artifact_is_intact_and_generates_deterministically`
- **PASS** `tests.test_published_artifact::test_artifact_path_cannot_escape_root`
- **PASS** `tests.test_published_artifact::test_artifact_checksum_verification_rejects_tampering`

## test_quality_policy

- **PASS** `tests.test_quality_policy::test_third_party_actions_are_immutable`
- **PASS** `tests.test_quality_policy::test_workflow_permissions_remain_least_privilege`
- **PASS** `tests.test_quality_policy::test_dependabot_covers_actions_and_python_dependency_metadata`
- **PASS** `tests.test_quality_policy::test_release_publishes_and_smokes_the_checkpoint_only_on_tags`
- **PASS** `tests.test_quality_policy::test_expensive_workflows_are_opt_in_and_routine_ci_is_linux_only`
- **PASS** `tests.test_quality_policy::test_release_reuses_default_branch_validation_instead_of_retesting`
- **PASS** `tests.test_quality_policy::test_python_and_optional_dependency_contracts_are_truthful_and_disjoint`
- **PASS** `tests.test_quality_policy::test_pages_uses_default_branch_and_pr_builds_without_deploying`
- **PASS** `tests.test_quality_policy::test_module_coverage_floor_reports_missing_and_low_files`
- **PASS** `tests.test_quality_policy::test_module_coverage_deltas_are_signed_and_risk_scoped`
- **PASS** `tests.test_quality_policy::test_module_coverage_cli_reports_delta_and_fails_low_module`
- **PASS** `tests.test_quality_policy::test_current_tpu_results_share_one_immutable_revision_and_are_linked`
- **PASS** `tests.test_quality_policy::test_paid_multihost_launcher_fails_closed_and_uses_bounded_defaults`

## test_report

- **PASS** `tests.test_report::test_unknown_device_cost_is_not_invented`
- **PASS** `tests.test_report::test_tpu_cost_scales_by_time_and_device_count`
- **PASS** `tests.test_report::test_git_info_degrades_cleanly_outside_checkout`
- **PASS** `tests.test_report::test_report_writes_matching_markdown_and_json`

## test_result_provenance

- **PASS** `tests.test_result_provenance::test_published_result_provenance_is_valid`
- **PASS** `tests.test_result_provenance::test_validator_rejects_stale_revision`
- **PASS** `tests.test_result_provenance::test_validator_rejects_missing_evidence_link`
- **PASS** `tests.test_result_provenance::test_validator_rejects_divergent_claim_value`
- **PASS** `tests.test_result_provenance::test_validator_rejects_missing_identity`

## test_sharding

- **PASS** `tests.test_sharding::test_default_mesh_consumes_every_device`
- **PASS** `tests.test_sharding::test_batch_is_sharded_and_state_is_replicated`
- **SKIP** `tests.test_sharding::test_virtual_multidevice_job_really_has_eight_devices` — dedicated virtual multi-device job only
- **PASS** `tests.test_sharding::test_optimizer_state_inherits_replicated_parameter_sharding`
- **PASS** `tests.test_sharding::test_inference_uses_model_devices_not_global_mesh[generate]`
- **PASS** `tests.test_sharding::test_inference_uses_model_devices_not_global_mesh[generate_with_cache]`
- **PASS** `tests.test_sharding::test_inference_uses_model_devices_not_global_mesh[generate_fast]`
- **PASS** `tests.test_sharding::test_inference_uses_model_devices_not_global_mesh[generate_speculative]`

## test_stage_functions

- **PASS** `tests.test_stage_functions::test_sft_batch_supervises_only_assistant_tokens`
- **PASS** `tests.test_stage_functions::test_sft_batch_fails_without_examples`
- **PASS** `tests.test_stage_functions::test_sft_batch_rejects_examples_without_supervised_tokens`
- **PASS** `tests.test_stage_functions::test_load_conversations_from_jsonl_obeys_limit`
- **PASS** `tests.test_stage_functions::test_preference_objective_is_finite_and_centered`
- **PASS** `tests.test_stage_functions::test_cli_modules_are_import_safe_and_expose_main`
- **PASS** `tests.test_stage_functions::test_stage_cli_options_resolve_to_typed_requests[PretrainRequest-build_parser-argv0-depth-3]`
- **PASS** `tests.test_stage_functions::test_stage_cli_options_resolve_to_typed_requests[SFTRequest-build_parser-argv1-batch_size-2]`
- **PASS** `tests.test_stage_functions::test_stage_cli_options_resolve_to_typed_requests[RLRequest-build_parser-argv2-num_samples-4]`
- **PASS** `tests.test_stage_functions::test_stage_cli_options_resolve_to_typed_requests[EvalRequest-build_parser-argv3-tasks-core,mmlu]`
- **PASS** `tests.test_stage_functions::test_stage_result_is_machine_readable`
- **PASS** `tests.test_stage_functions::test_all_stage_requests_accept_one_validated_resolved_config`
- **PASS** `tests.test_stage_functions::test_pretrain_smoke_resolves_frozen_model_configuration`

## test_token_pool

- **PASS** `tests.test_token_pool::test_packing_has_no_padding_and_resume_is_exact`
- **PASS** `tests.test_token_pool::test_corruption_and_path_escape_rejected`
- **PASS** `tests.test_token_pool::test_gpt_resume_matches_uninterrupted`
- **PASS** `tests.test_token_pool::test_invalid_ids_rejected`
- **PASS** `tests.test_token_pool::test_manifest_changes_identity`

## test_tokenizer

- **PASS** `tests.test_tokenizer.TestTrainFromIterator::test_vocab_size`
- **PASS** `tests.test_tokenizer.TestTrainFromIterator::test_special_tokens_present`
- **PASS** `tests.test_tokenizer.TestTrainFromIterator::test_special_tokens_have_ids`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_returns_list_of_ints`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_decode_roundtrip`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_batch`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_with_prepend`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_with_append`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_empty_string`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_encode_invalid_type_raises`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_callable`
- **PASS** `tests.test_tokenizer.TestEncodeDecode::test_unicode_roundtrip`
- **PASS** `tests.test_tokenizer.TestSpecialTokens::test_bos_token_id`
- **PASS** `tests.test_tokenizer.TestSpecialTokens::test_encode_special_returns_int`
- **PASS** `tests.test_tokenizer.TestSpecialTokens::test_id_to_token`
- **PASS** `tests.test_tokenizer.TestConversationRendering::test_huggingface_backend_marks_only_assistant_targets`
- **PASS** `tests.test_tokenizer.TestConversationRendering::test_completion_removes_reference_answer`
- **PASS** `tests.test_tokenizer.TestSaveLoad::test_save_and_reload`
- **PASS** `tests.test_tokenizer.TestSaveLoad::test_save_creates_file`
- **PASS** `tests.test_tokenizer.TestSaveLoad::test_loaded_vocab_size_matches`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_uses_byt5_byte_id_mapping`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_unicode_roundtrip`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_batch_prepend_and_append`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_vocab_contains_bytes_reserved_and_chat_tokens`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_special_tokens_are_explicit_not_parsed_from_text`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_conversation_rendering`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_save_load_and_factory_roundtrip`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_stream_decode_preserves_multibyte_characters`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_rejects_unknown_special_and_invalid_id`
- **PASS** `tests.test_tokenizer.TestByteTokenizer::test_training_cli_needs_no_dataset`

## test_tpu_validation

- **PASS** `tests.test_tpu_validation::test_junit_preserves_failures_errors_and_skip_reasons`
- **PASS** `tests.test_tpu_validation::test_source_hash_detects_uncommitted_changes`

## test_training_safety

- **PASS** `tests.test_training_safety::test_nonfinite_step_preserves_model_and_optimizer`
- **PASS** `tests.test_training_safety::test_fp32_microbatch_accumulation_matches_large_batch`
- **PASS** `tests.test_training_safety::test_single_microbatch_path_is_supported`
- **PASS** `tests.test_training_safety::test_schedule_is_defined_before_optimizer_construction`

## test_training_scaling

- **PASS** `tests.test_training_scaling::test_device_counts_are_sorted_unique_and_bounded`
- **PASS** `tests.test_training_scaling::test_strong_scaling_efficiency_uses_one_device_baseline`
- **PASS** `tests.test_training_scaling::test_efficiency_rejects_nonbaseline_first_record`
- **PASS** `tests.test_training_scaling::test_efficiency_rejects_invalid_threshold`
- **PASS** `tests.test_training_scaling::test_software_metadata_is_explicit_and_versioned`
