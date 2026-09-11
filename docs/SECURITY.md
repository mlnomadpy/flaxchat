# Security and generated-code execution

`flaxchat.execution.execute_code` is a best-effort reliability guard for code
that an operator has already reviewed and trusts. It is **not a security
sandbox**. The default is fail-closed: calls must pass `trusted=True`.

The guarded child process has a wall-clock timeout, bounded captured output,
best-effort Linux memory/process/file limits, an empty environment, and a
temporary working directory. These controls protect the parent from common
accidents; they do not defend against a malicious Python program, native code,
kernel exploits, filesystem traversal, or all network/import bypasses.

Production systems must run untrusted generated code in a separately reviewed
container, microVM, or remote worker. Flaxchat ships an opt-in Docker/Podman
adapter, `execute_code_isolated`, that requires a digest-pinned image and uses
no network, no inherited credentials, no host mounts, a read-only root, an
unprivileged user, dropped capabilities, no-new-privileges, isolated Python,
CPU/memory/process/output/time quotas, and bounded disposable tmpfs storage.
The container runtime and kernel remain part of the trusted computing base.

HumanEval is fail-closed unless `FLAXCHAT_EXECUTION_IMAGE` names a reviewed
image using `repository@sha256:<64 lowercase hex characters>`. Set
`FLAXCHAT_EXECUTION_RUNTIME=podman` to use Podman; Docker is the default. Never
replace the digest with a mutable tag. Deployments needing a stronger boundary
should substitute a reviewed microVM or remote-worker backend.
