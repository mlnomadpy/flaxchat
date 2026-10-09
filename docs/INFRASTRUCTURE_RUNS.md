# Guarded representation deployment and accounting

Use `python -m scripts.representation_run launch --manifest run.json --manifest-uri gs://bucket/manifests/run.json --output artifacts/run-receipts` for a new representation continuation or qualification. The manifest in GCS must be byte-identical to the local manifest. This routes through `gcp_spot_supervisor`: server-verified cleanup before creation, fresh queue/node identity, bounded setup/workload, and verified absence of both objects after completion. The default two-hour maximum remains; explicitly admitted long leases require `deployment.allow_long_lease: true` and retain the same global deadline and independent cleanup checks. The legacy `train_tpu` adapter supports dry-run rendering and `--guarded-manifest`; its local-watchdog paid path is disabled.

A run manifest has schema version 1 and these fields:

| Field | Required meaning |
| --- | --- |
| `run_id`, `stage_id` | Distinct lowercase run/stage identities; output prefix must end in `/run_id/stage_id` |
| `parent_identity` | Immutable last-good checkpoint identity; trainer semantic preflight still validates it |
| `cleanup_owner` | Operational owner of the independent cloud lease |
| `expected_topology` | Requested accelerator type, matching deployment |
| `expected_device_count`, `expected_process_count` | Required explicit JAX logical topology assertions |
| `distributed` | Whether worker physical checks initialize JAX distributed before device discovery |
| `source` | `{uri, sha256, target: "source", archive: true}`; contained source archive, including the current worker code |
| `runtime_lock` | `{path, sha256, wheelhouse_target}` of complete Linux exact pins and the hashed wheelhouse artifact target; optional `python_executable` defaults to `python3.12` |
| `artifacts` | Verified `{uri, sha256, target, archive}` input/parent packages; target relative to run root |
| `overlays` | Same identity contract for deliberate evaluator/source overlays; applied after source |
| `setup_checks` | Substantive validation argv; trivial/help probes are rejected; successful execution alone is not physical acceptance |
| `workload` | Nonempty argv; `{python}`, `{root}`, `{output_prefix}` and `{checkpoint_output}` placeholders are substituted without shell expansion |
| `checkpoint_output` | Training output, default `<output_prefix>/checkpoints`; exactly one actual `--output VALUE` or `--output=VALUE` must match it |
| `qualification` | Physical receipt contract `{receipt, required_tests, format}`; mandatory for known training workloads; optional for qualification-only workloads |
| `output_prefix` | New stage GCS namespace; pass the matching checkpoint path to the trainer argv |
| `deployment` | Project, zone, accelerator/runtime, whole-slice hourly rate, budget, lease, optional positive `capacity_wait_seconds` within the lease and shared campaign ledger |

`deployment.pricing` requires `sku`, `region`, `provisioning_model`, `source`, timezone-aware `observed_at`, `accelerator_type`, `currency: "USD"`, `unit: "whole_slice_hour"`, and `hourly_usd`. The accelerator and rate must match deployment. Do not derive this number from JAX device count. The `--campaign-ledger` path must be shared by all controllers on the same filesystem, with the same budget cap; independent laptops need a shared controller rather than separate local ledgers. Reservations are conservative and survive process loss; posted charges remain separate. `RunLedger.reconcile_completed_reservation` can release unused reservations only for consistent terminal campaigns with verified cleanup and a later independent queue/node absence observation. A finalized interrupted code75 campaign additionally requires matching attempt/event codes and ordered work-finished, resource-absent, retry-eligible history. Original events and the budget cap are preserved. Retained elapsed-time bounds are not provider charges.

An actual training workload must have exactly one output argument bound to the declared namespace, for example `--output {checkpoint_output}`. The manifest adapter rejects conflicting, duplicate or missing training output arguments before provisioning. `workload_kind: "training"` declares custom training wrappers; known `scripts.train_*` modules and `train_*.py` entrypoints are detected automatically. Qualification-only/test workloads can run without prior numerical acceptance.

Successful setup records `schema_version: 2`, `setup_checks_executed: true`, and `physical_acceptance: false` unless substantive tests produced a verified qualification receipt. Backend discovery alone does not qualify checkpoint recovery, gradients or quality. A known training workload requires a qualification contract, so copying `true`, `echo`, `--help` or a print-only probe cannot admit real training.

For compatibility with the existing physical suite, declare:

```json
{
  "qualification": {
    "receipt": "qualification/summary.json",
    "format": "test_suite",
    "required_tests": ["EXACT_JUNIT_TEST_ID_FOR_OBJECTIVE", "EXACT_JUNIT_TEST_ID_FOR_RECOVERY"]
  }
}
```

Run `scripts.validate_test_suite` as the substantive setup check and direct its output into that root-relative directory. Required IDs must be the actual JUnit node names for the declared scope. The adapter verifies passing module/test inventory without skips, current Python source digest, sibling `hardware.json` TPU backend/topology, JAX/Flax/Optax/Orbax/libtpu package versions against the exact lock, and selected interpreter version. This is selected-test acceptance, not production quality or universal scale validation. The existing suite is single-host; physical multi-host acceptance needs its appropriate scope and bound receipt.

The default `format: "bound"` accepts a custom physical receipt with `schema_version: 1`, `passed: true`, `backend: "tpu"`, active `manifest_sha256`, `runtime_lock_sha256`, `source_tree_sha256`, `devices`, `processes`, and a `tests` array of unique `{id, status: "passed"}` entries covering every required ID. Setup supplies manifest/lock/source hashes in environment variables `FLAXCHAT_RUN_MANIFEST_SHA256`, `FLAXCHAT_RUNTIME_LOCK_SHA256` and `FLAXCHAT_SOURCE_TREE_SHA256`. Place generated receipts outside immutable source. Workers bind the exact receipt/hardware hashes and revalidate them before admission. These checks authenticate declared test evidence; they do not turn fabricated fixture files into physical evidence.

Set real parent/data/stage arguments from the [representation runbook](REPRESENTATION_TRAINING_RUNBOOK.md). Old setup receipts are deliberately invalid under the stronger schema and must be rebuilt with the qualified current runtime.

Prepare immutable deployment inputs on the existing selected Linux runtime without importing a model:

```bash
python -m scripts.build_runtime_bundle --python /path/to/qualified/venv/bin/python \
  --output /tmp/representation-runtime
```

This captures exact complete pins, downloads compatible wheels with no dependency resolution, hashes every wheel, and packages the wheelhouse. For an environment containing CPU torch wheels, explicitly provide the matching official CPU wheel index; for torch_xla, provide its official wheel index. Runtime capture refuses macOS; download-only cross preparation accepts an exact lock and explicit target platform/Python/ABI flags. For lock-based download preparation on the selected Linux interpreter, derive **all** supported host wheel platforms instead of guessing a few glibc tags:

```bash
python -m scripts.build_runtime_bundle --python /path/to/linux/python3.12 \
  --lock infra/tpu/validation-lock.txt --auto-host-platforms \
  --python-version 3.12 --abi cp312 --extra-index-url https://download.pytorch.org/whl/cpu \
  --output /tmp/representation-runtime
```

This reads the complete tag set from the selected interpreter's `pip._vendor.packaging.tags.sys_tags`, deduplicates every supported platform, and records the original tags in the bundle receipt. It rejects a non-Linux interpreter, a target Python/ABI mismatch, or mixing automatic and manually restricted platforms. A short guessed list can exclude compatible intermediate glibc wheels such as `manylinux_2_27_x86_64`; do not reuse the old three-platform example. Download-only cross preparation from macOS remains available with explicit platform flags, but must include the complete target image's compatible platforms and is not automatically inferred from the Mac host.

This command downloads wheels only and records `cross_download_only: true`; it does not execute a CPU model or qualify an environment. The old validation lock includes CPU torch for tooling; choose a separately declared torch_xla lock for TPU PyTorch execution.

The generated receipt identifies Python/architecture and package indexes, but does not claim physical numerical acceptance. Include `runtime-lock.txt` in your source archive; include `wheelhouse.tar.gz` in manifest artifacts with `target: "wheels", archive: true`, and set `runtime_lock.wheelhouse_target: "wheels"`.

If the selected TPU image lacks Python 3.12, stage a fixed official standalone interpreter instead of downloading a floating installer:

```bash
python -m scripts.prepare_python_artifact --release RELEASE_DATE \
  --python-version EXACT_VERSION --url PINNED_OFFICIAL_ASSET_URL \
  --sha256 VERIFIED_OFFICIAL_SHA256 --output /tmp/cpython-linux.tar.gz
```

Select the exact release/version/asset/hash from the [official python-build-standalone releases](https://github.com/astral-sh/python-build-standalone/releases). The helper accepts only that pinned project's Linux x86_64 install-only URL, verifies its bytes and executable, and never executes it locally. Its sidecar gives the manifest artifact (`target: "cpython", archive: true, allow_internal_links: true`) and `runtime_lock.python_executable: "{root}/cpython/python/bin/python3.12"`. Keep source archives strict; internal symlinks are opt-in for interpreter artifacts and cannot escape the extraction destination. The bootstrap source extraction uses explicit containment/type checks compatible with system Python 3.10, without relying on newer tar `filter` support.

A manifest template can replace `sha256` fields with `local_path` for source/input/wheelhouse/overlay entries and the runtime lock. Seal and optionally upload it to fresh generation-zero GCS names:

```bash
python -m scripts.representation_run seal --manifest run-template.json \
  --output run.json
python -m scripts.representation_run seal --manifest run-template.json \
  --output run.json --upload --manifest-uri gs://bucket/manifests/run.json
```

The first command only hashes local artifacts and validates metadata. The second explicitly uploads those exact files and the sealed manifest, refusing overwrite of existing objects. Source and parent/data archives must be prepared from the intended current revision and last-good checkpoint. Preserve their receipts; do not copy a historical manifest hash onto changed files.

Run `python -m scripts.representation_run check --manifest run.json` before uploading the manifest or provisioning. It verifies metadata and identities. Setup verifies each downloaded input before extraction, rejects traversal/symlinks, installs exact frozen dependencies offline with `--no-deps --no-index --find-links` from the verified wheelhouse, checks dependency completeness and installed versions, and records the complete package/interpreter inventory. Re-setup invalidates old passing receipts before mutations and forces installation from verified wheel bytes. Workers recheck the deployed source tree/lock, live normalized installed package inventory, interpreter version/binary SHA256, and qualification receipts before model execution. Package version equality alone does not detect same-version manual edits to site-package files; keep the run environment immutable after setup. It deliberately does not install floating editable extras. Source is loaded through `PYTHONPATH`; no unresolved dependency expansion is allowed. The existing [accepted validation freeze](../infra/tpu/validation-lock.txt) is a starting point, not evidence that the newer MTEB/torch_xla environment is qualified. Capture the complete selected Linux runtime, then physically qualify it before claiming it tested. Setup uses `python3.12` by default; set `runtime_lock.python_executable` explicitly for the selected VM. A hashed standalone interpreter can be an artifact and use `{root}/python/bin/python3.12`. A different default VM Python does not silently select another runtime.

Historical embedding full/pilot/eval and torch setup wrappers use `FLAXCHAT_RUN_MANIFEST` and `FLAXCHAT_RUN_ROOT`; a missing manifest fails before downloads/model work. Periodic/final uploads use bounded subprocess deadlines. Training exit codes survive telemetry failures; local status receipts record upload failures. The independent cleanup lease still bounds the allocation if the worker process or controller disappears.

## Posted billing

Generate a parameterized read-only query, without executing it:

```bash
python -m scripts.cloud_accounting billing \
  --table billingproject.dataset.gcp_billing_export_v1_ACCOUNT \
  --project tpubuilders --start 2026-09-01T00:00:00Z --end 2026-10-01T00:00:00Z \
  --output artifacts/billing-query.json
```

Add `--execute --maximum-bytes-billed 100000000` to query an existing authorized BigQuery export. The output separates posted gross costs, signed credits, promotional credits and net costs, with service/SKU/location/run-label breakdowns (add `--detailed` for resource names in detailed-export tables) and export/usage freshness. Missing labels remain unattributed rather than disappearing from totals. Empty exports mean unknown charges. Delayed billing and adjustments prevent a claim that the snapshot is final. Remaining promotional balance cannot be inferred from this usage export. No export or IAM is created automatically. See the [official billing export schema](https://docs.cloud.google.com/billing/docs/how-to/export-data-bigquery-tables/standard-usage).

## Retained storage

```bash
python -m scripts.cloud_accounting storage --prefix gs://bucket/run \
  --output artifacts/storage-inventory.json
python -m scripts.cloud_accounting plan --inventory artifacts/storage-inventory.json \
  --disposable-prefix cache/ --protected-prefix checkpoints/ --protected-prefix release/ \
  --output artifacts/storage-plan.json
```

Inventory includes live, noncurrent and soft-deleted generations/bytes and raw metadata. Names in cleanup rules are object names relative to the bucket, exactly as the inventory records them. It uses three sequential reads and reports that consistency limit, preserving every observed live generation even if it disappears before the later version listing. It refuses missing/unrecognized object metadata rather than declaring an empty bucket. This CLI only proposes cleanup; it never deletes. Protect the last-good checkpoint, public-release lineage and evidence explicitly. Soft-deleted bytes remain retained until expiry. Storage posted charges appear separately by service in the billing report; bytes alone are not a pricing estimate. Flags follow the [official storage listing reference](https://docs.cloud.google.com/sdk/gcloud/reference/storage/ls).

## Verification scope

`tests/test_cloud_accounting_metadata.py` verifies price units/nonfinite rejection, signed billing/unknown balances, real gcloud metadata envelope normalization, versioned storage retention, exact locks/archive safety, stage identity, guarded launch generation/dispatch, concurrent campaign admission and stalled telemetry preserving workload failures. These are model-free checks. Physical controller-loss recovery, fresh VM dependency installation, target-topology restore and posted export permissions remain acceptance evidence to collect; no test fixture substitutes for them.

## Historical MIRACL and publication commands

The historical MIRACL and GCE publication shell entry points now delegate to
`python3 -m scripts.representation_workflow` using `FLAXCHAT_RUN_MANIFEST` and
`FLAXCHAT_RUN_ROOT`. MIRACL still accepts its positional shard index 2–7; its
manifest must contain `workflow_contract: {kind: miracl, shard_index: 3,
subsets_target: shard.json}`, a hashed artifact targeting `shard.json`, and the
`evaluate_yat_mteb_subsets` workload with `--task MIRACLRetrievalHardNegatives
--subsets-file {root}/shard.json`. Include the exact published model and subset
inputs as hashed artifacts. `setup_and_run_yat_miracl_single_tpu.sh 3` performs
setup then evaluation; `run_yat_miracl_single_tpu.sh 3` reuses verified setup.

Publication manifests select `workload_kind: publication` and
`workflow_contract: {kind: publish-flax, repo: owner/new-model,
release_target: release}` (or `publish-torch`). The workload executes the
corresponding `scripts.publish_yat_embedding_from_gcp` or
`scripts.publish_yat_torch_from_gcp` module with `{root}/release`, the explicitly
delivered token file and matching `--repo`. The release is a hashed archive
artifact targeting `release`. These non-model publication workers install the
same frozen offline runtime and revalidate it; they do not initialize JAX or
claim physical model acceptance. Tokens must never be manifest artifacts. The
GCE wrappers retain restrictive token permissions and exit cleanup. Publication
identity/parity requirements remain enforced by the publisher. Use the exact
same caller-authorized repo and release identity; this wrapper does not authorize
a new upload. No wrapper installs floating dependencies or downloads mutable code.

Guarded TPU launches attach `flaxchat-run` and `flaxchat-stage` provider labels.
Valid names remain unchanged; invalid or oversized values receive a deterministic
hash suffix, with original identities retained in the local resource receipt.
After readiness, the controller verifies actual TPU VM labels and saves
`billing-labels.json`. Provider Billing export propagation/attribution still
requires a real posted query; labels alone are not dollar evidence. Campaign
coordination remains one controller/shared filesystem. Worker result receipts
are written before final telemetry, and signals targeting an exited child group
preserve the recorded workload return code.

## Durable parity publication follow-up

The v4 publication path requires `--parity-evidence` pointing to private retained
raw outputs, inputs, sidecars and case reports outside the public release tree.
It rehashes and recomputes numerical gates before Hub mutation. A Torch workflow
must provide a separately hashed `parity_evidence_target` artifact and explicit
`--host-ram-budget-bytes` and `--evidence-budget-bytes`. Compressed and NPY-header
sizes are checked before allocation. Historical v3 summaries alone do not qualify
publication under this strengthened contract.

The physical campaign also requires `--scratch-budget-bytes`; use its
`--preflight-only` mode to check the complete advertised matrix without model
execution before choosing a worker. Estimates cover host collection/comparison
and evidence storage; TPU HBM and actual peak performance still need physical
measurement. No automatic paid workflow is enabled by these changes.

The supervisor separates capacity from startup: observed `PROVISIONING`/`ACTIVE` or node startup switches irreversibly to startup waiting. Declare `deployment.startup_wait_seconds` (positive integer within `attempt_seconds`) in new frozen manifests, alongside the capacity budget. Both phases remain bounded by the original global lease; the adapter passes both budgets and receipts retain queue/node transitions. This orchestration change has model-free checks, not physical startup acceptance.
