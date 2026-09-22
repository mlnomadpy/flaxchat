# Final-source TPU test inventory — 2026-09-21

Source: `14c688f0d5077d960d7c50620eab05bd08dc264660e0cd78f7cd5f728f4a4148`. Two-host Spot v5p-16 allocation; unit suite isolated to one host with four physical chips.

**503 passed, 4 skipped, 0 failed**, 241.11 seconds. Full case names, durations and skip reasons: `artifacts/harness-fixes/tpu-unit-inventory-accepted.json`; raw JUnit: `artifacts/harness-fixes/tpu-accepted.xml`. Counts below are unique within this suite, not summed with overlapping local runs.

| Test module | Passed | Skipped |
|---|---:|---:|
| `tests.test_attention_accelerator` | 4 | 0 |
| `tests.test_benchmark_compare` | 9 | 0 |
| `tests.test_benchmark_protocol` | 1 | 0 |
| `tests.test_chat` | 7 | 0 |
| `tests.test_chat_web` | 6 | 0 |
| `tests.test_checkpoint` | 18 | 0 |
| `tests.test_checkpoint_topology` | 1 | 0 |
| `tests.test_ci_scope` | 10 | 0 |
| `tests.test_common` | 25 | 0 |
| `tests.test_config` | 35 | 0 |
| `tests.test_dataloader` | 16 | 0 |
| `tests.test_dataset` | 7 | 0 |
| `tests.test_distributed_cpu` | 0 | 2 |
| `tests.test_docs` | 3 | 0 |
| `tests.test_engine` | 23 | 0 |
| `tests.test_eval` | 13 | 0 |
| `tests.test_execution` | 45 | 0 |
| `tests.test_finetuning_resume` | 2 | 0 |
| `tests.test_gcp_cleanup_guard` | 9 | 0 |
| `tests.test_gcp_tpu_preflight` | 2 | 0 |
| `tests.test_gcp_tpu_run` | 5 | 0 |
| `tests.test_harness_regressions` | 29 | 0 |
| `tests.test_kaggle_launcher` | 18 | 0 |
| `tests.test_matched_benchmark` | 7 | 0 |
| `tests.test_model` | 34 | 1 |
| `tests.test_multihost_acceptance` | 15 | 0 |
| `tests.test_optim` | 14 | 0 |
| `tests.test_pipeline` | 4 | 0 |
| `tests.test_prefetch` | 9 | 0 |
| `tests.test_published_artifact` | 3 | 0 |
| `tests.test_quality_policy` | 13 | 0 |
| `tests.test_report` | 4 | 0 |
| `tests.test_result_provenance` | 5 | 0 |
| `tests.test_sharding` | 7 | 1 |
| `tests.test_spot_watchdog` | 2 | 0 |
| `tests.test_stage_functions` | 13 | 0 |
| `tests.test_token_pool` | 23 | 0 |
| `tests.test_tokenizer` | 30 | 0 |
| `tests.test_tpu_scale_plan` | 18 | 0 |
| `tests.test_tpu_validation` | 2 | 0 |
| `tests.test_training_safety` | 7 | 0 |
| `tests.test_training_scaling` | 5 | 0 |

## Complementary checks

- 498 CPU tests passed; 9 skips covered by accelerator or explicit multi-device/process jobs.
- 36 BF16 CPU regressions passed.
- 25 checkpoint/accumulation tests passed on eight virtual devices.
- 8 sharding tests passed on eight virtual devices.
- 2 localhost two-process collective/checkpoint tests passed.

The physical suite skips the two opt-in localhost process tests, a CPU/GPU-only unsupported-Splash check, and the explicit eight-virtual-device assertion. Those are covered by the corresponding local runs.

## Failure history

The initial physical run found a topology-order mismatch in checkpoint hashing and four numerical assertions. The checkpoint implementation now preserves the input mesh. The final suite includes a reversed-mesh/partial-chunk regression, independent analytic FP32 accumulation checks, and explicit BF16 error bounds. The initial failures and intermediate diagnostic timeout remain in `artifacts/harness-fixes/`; they are not overwritten or counted as passes.

BF16 accumulated-versus-full gradient relative L2 error measured 0.0055576474 (0.556%), below the 2% limit. Scan checks retain strict FP32 comparisons and per-leaf BF16 norm bounds with the original absolute allowance for near-zero leaves.

Multi-host recovery results are recorded separately in `artifacts/harness-fixes/multihost-accepted/summary.json`. A passing unit suite alone is not full recovery or all-scale validation.

## Physical multi-host acceptance

All nine stages passed on two physical hosts/eight chips. Interrupted data-parallel recovery, FSDP=4 recovery and the CPU/BF16 single-host bridge back to TPU produced exactly matching model, optimizer and training-state manifests. Recipe: Muon, accumulation=2, BF16, loss chunks=16 and rematerialization.

A separate 12-layer GPT-2 FSDP=8 check passed four updates and checkpoint commit at global batch 16/sequence length 128 (8,192 tokens). This is cross-host parameter sharding, not proof of convergence or larger-pod scaling. Summaries: `artifacts/harness-fixes/multihost-accepted/summary.json` and `artifacts/harness-fixes/scale-training-summary/training_summary.json`.
