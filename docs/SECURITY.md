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
container, microVM, or remote worker with all of the following enforced outside
Python: no network, no inherited credentials, a read-only/minimal filesystem,
an allowlisted runtime, CPU/memory/process/output/time quotas, and disposable
storage. Flaxchat does not currently ship that backend, so untrusted execution
must remain disabled.
