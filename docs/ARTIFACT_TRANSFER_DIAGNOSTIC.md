# Diagnose parent transport before retrying materialization

The actual parent06 worker timed out while copying the retained 1.23GB weight
object with `gcloud storage cp` after 180 seconds. It did not record byte progress,
so the receipt cannot distinguish transfer throughput, startup, checksum work or
client stalls. This does not justify repeating the same attempt with a guessed
timeout.

The provider describes the retained weight as a composite object with 25 components,
CRC32C and no MD5. Client CRC overhead is a plausible explanation, not an observed
cause. The locally installed `gcloud storage cat --help` explicitly documents
inclusive `--range=0-67108863` and states that cat does not compute a checksum.
Check the executor's CLI version/help too; local SDK evidence does not establish
the cloud executor runtime or checksum implementation.

`scripts/diagnose_artifact_transfer.py` observes one stream, with no retries. Its
read-only range mode receives at most 64MiB, hashes the streamed bytes, reports
received bytes and elapsed time every five seconds, and removes the range payload.
The partial SHA identifies observed range bytes; it cannot authenticate the whole
weight artifact. Exact requested size, client return code and finite deadline are
required for a completed diagnostic. Stderr volume/hash and a truncation indicator
are recorded without exposing stderr payload or credential-bearing URLs.

A frozen source/manifest proposal is at
`/private/tmp/flaxchat-fixes-20260930/transfer-diagnostic-01`. Its manifest preserves
the actual original generation, full-artifact size/SHA and independent public
identity. The draft declares a 180-second stream budget inside 600 seconds with
240 seconds reserved for finalization. It does not allocate compute or dispatch
anything. Root must verify parent06 cleanup, executor identity, fresh owned paths,
budget reservation and immutable source before execution.

From the verified unpacked source, the read-only measurement command is:

```sh
python3 diagnose_artifact_transfer.py --manifest artifact.json \
  --output /tmp/flaxchat-transfer-diagnostic-1001a/range-payload \
  --receipt /tmp/flaxchat-transfer-diagnostic-1001a/receipt.json \
  --mode range --range-bytes 67108864 --phase-seconds 180 \
  --total-seconds 600 --finalization-margin-seconds 240
```

The caller must retain an independent outer timeout and exact owned-process/root
cleanup. The utility owns its stream's process group and kills it on timeout or
handled TERM/INT; whole executor loss remains outside this qualification. Receipts
and progress should be collected before deleting the caller's owned root. A failed
observation or SSH loss requires inspecting the same execution, not starting another.

The proposed full mode streams the same pinned generation into a separate partial
file, checks exact full byte count and the independently pinned SHA, then publishes
the separate output through an exclusive hard link. A mismatch, timeout, client
failure or existing output never commits weights. Full mode is implemented and
literal-byte tested, but not integrated into parent materialization or physically
validated. Choose a full-transfer budget only after actual range/client evidence;
leave enough time for enrichment, archive, upload, receipts and cleanup in the
original parent lease. Warm range throughput alone is not a guaranteed full-run ETA.

Fifteen model-free subprocess tests cover range semantics, no false whole-auth
claim, full-byte/SHA commit, short/oversized/wrong bytes, client failure, stderr
bounds, timeout cleanup, exact generation/range command, invalid pins, fresh output
and deadline margin. No model tensors, cloud reads or TPU allocation occur in them.
