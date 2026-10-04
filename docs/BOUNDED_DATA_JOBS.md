# Bounded data jobs on an existing Linux cloud host

`scripts/bounded_data_job.py` is a provider-neutral controller for source-authenticated data jobs on an already provisioned Linux host. It does not allocate TPUs or VMs, admit a campaign budget, publish receipts, or qualify model behavior. Use the existing campaign admission and generation-pinned upload tools for those steps. Existing running temporary launchers must be observed through their original handles; adopting this tool is not permission to replace or restart them.

The temporary parent-materialization and MIRACL-filter jobs motivated this controller: an SSH connection can fail before or after a worker starts, and a successful SSH return code can coexist with a failed durable worker receipt. The controller makes those states explicit. `launch` starts a detached supervisor and returns admission information; it never reports application success. `observe` is read-only. A missing ownership handle, expired deadline, transport failure, or missing terminal receipt never permits a duplicate launch. A retry is a separately admitted job with a fresh job ID and root after authoritative termination and exact cleanup evidence for the original job.

## Immutable admission

The JSON specification uses schema version 1:

```json
{
  "schema_version": 1,
  "job_id": "parent-materialize-unique-02",
  "root": "/tmp/flaxchat-data-parent-unique-02",
  "controller_sha256": "<SHA256 of the deployed bounded_data_job.py>",
  "timeout_seconds": 1800,
  "kill_after_seconds": 30,
  "command": ["/usr/bin/python3", "{input:runner.py}", "--scratch-directory", "{scratch}"],
  "inputs": [
    {"name": "runner.py", "path": "/tmp/admitted-inputs/runner.py", "bytes": 1234, "sha256": "<SHA256>"}
  ]
}
```

The example values are placeholders, not a deployable admission. The caller must retain the canonical specification SHA256 outside the remote root and account for the full timeout, kill allowance, startup freeze (at most 120 seconds), and final cleanup in its campaign lease and cost reservation. The tool accepts at most 64 flat inputs, each at most 4 GiB; those bounds are not a disk-capacity guarantee. Independently validate free disk and total input bytes before admission. Paths must be absolute without symlink ancestors, and the root must be fresh with a `flaxchat-data-` basename. No shell interpolation is performed. Commands may refer to `{input:name}` and `{scratch}`. Source files are copied to the retained `inputs` directory and verified against the admitted hashes before execution; the controller itself is copied and verified too.

Provisioning, input downloads and generation pinning remain outside this controller. A network download inside a workload must have its own finite request timeout, immutable remote identity and byte/hash verification. Interpreter and package locks are also an independent admission requirement; a frozen source is not a frozen runtime. The worker must publish or move desired outputs to its retained result namespace before exiting: its `work` scratch is deleted after exact process cleanup, while logs, inputs and receipts remain under the job root.

## Launch and reconnect

```sh
python3 scripts/bounded_data_job.py launch --spec /tmp/admitted-job.json
python3 scripts/bounded_data_job.py observe --root /tmp/flaxchat-data-parent-unique-02 --expected-spec-sha256 <retained-canonical-spec-SHA256>
```

The first root `mkdir` is an admission lock, and `supervisor-claim.json` is a second once-only workload lock. Neither is removed to allow retry. The detached supervisor writes `ownership.json` after starting GNU `timeout`; that process independently enforces the workload deadline. Ownership contains the Linux boot ID and process start ticks, so a recycled PID cannot be treated as a live worker. A unique random owner nonce plus the exact scratch `TMPDIR` tags descendants; cleanup checks both, then rechecks PID identity immediately before signalling. It never emits complete process environments.

The independent expected specification SHA256 is mandatory for both the `observe`
CLI and Python API. Retain it outside the remote job root before launch; do not
derive it from that root's current files during observation. Missing or malformed
pins fail before reading the remote root. Rewriting local spec, admission and
terminal files to share a new self-consistent hash cannot replace the original
independent admission identity. The Python call is
`observe(root, expected_spec_sha256)` and never has an unpinned observation mode.

`terminal.json` is the durable authoritative execution receipt. `worker_returncode`, `status`, `cleanup_verified`, residual process identities and `scratch_removed` must all be read. `execution_succeeded` only says the executable returned zero: it does not authenticate its exported model/data, satisfy an application acceptance gate, or establish TPU qualification. Cleanup can fail even after a successful executable, in which case the supervisor returns failure and the receipt retains the unresolved state. Cloud publication should use an independently bounded, generation-zero upload and verify the uploaded bytes. Upload or attachment success never substitutes for this worker receipt.

## Limits and qualification

The independent timeout protects against a lost SSH attachment and a killed supervisor for children still in its process group. A supervisor SIGKILL can prevent residual nonce-tag cleanup and terminal publication; escaped process groups, host destruction, unavailable `/proc`, and changed ownership permissions therefore remain unresolved until independently observed. This is not a sustained fault-recovery qualification. Never infer clean resources or paid-resource deletion from scratch cleanup. The tool controls processes only, so cloud allocation cleanup must be verified separately.

The metadata suite executes real bounded model-free subprocesses covering zero exit, nonzero exit and timeout, plus specification/ownership/receipt mutation checks. A detached `/proc` integration test runs only on physical Linux; macOS explicitly skips that test and does not prove cloud liveness, cleanup or application behavior. TPU workloads and numerical/performance tests remain physical-TPU-only under the training skill.

## Actual Linux acceptance, October 1

The frozen then-current controller passed the standard-library-only [physical Linux lifecycle selftest](audit-2026-09-30/bounded-data-job-linux-selftest-01/selftest.json) in Cloud Shell: actual live observation, duplicate launch rejection, worker return17 despite successful attachment, timeout124 and escaped-session child termination. All four terminal receipts reported verified owned cleanup. This qualifies this process-lifecycle source on the observed Linux executor, not cloud provisioning, whole-controller SIGKILL, TPU models or production training. The macOS pytest Linux integration remains a declared skip. Owned receipt/source directories are retained; consult independent cleanup evidence for scoped process/scratch absence.

Subsequent required external-pin enforcement changes the controller source from
that historical Linux freeze. Omission and self-consistent rewrite rejection have
fresh model-free coverage; historical execution does not qualify this new source
on Linux. Existing live jobs retain their original frozen controller and handles.

## Changed-source Linux acceptance

The current controller SHA14c42ccb9433df61dfc81bff459d5f979cf809b7328d433ea32bade3592cdb49
passed the [actual changed-source Linux selftest](audit-2026-09-30/bounded-data-job-linux-selftest-02/selftest.json).
Six checks passed: mandatory API/CLI external pin, self-consistent replacement rejection,
live success/duplicate observation, worker17 failure, timeout124 and escaped child cleanup.
The separate [wrapper observation](audit-2026-09-30/bounded-data-job-linux-selftest-02/wrapper.json)
confirmed all four owned process groups/handles and scratch absent. Whole-controller,
provider and model numerical scopes remain unqualified. Source/receipt directories
remain retained; this is not project-wide cleanup.
