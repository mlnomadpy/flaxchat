# TPU testing upgrade — 2026-09-23

## Results

The combined inventory contains **764 distinct passing test identities**. This
combines different runs and backends; it is not one completely passing TPU run.
All 55 test modules were attempted on a single four-chip Spot v5p host.

| Run | Passed | Failed | Skipped | Scope |
| --- | ---: | ---: | ---: | --- |
| Full v5p suite | 750 | 4 | 8 | All 55 modules; no unattempted modules |
| Targeted v5e retest | 23 | 0 | 0 | Splash, distributed CPU, encoder training, runner contracts |
| Local eight-device CPU checks | 2 | 0 | 0 | CPU-only fallback and virtual topology cases |
| Final local regression checks | 46 | 0 | 0 | Runner, supervisor, cleanup, CI routing, documentation |

Repeated tests are counted once in the combined inventory. Original failures and
skips remain in its per-test history. Six opt-in localhost CPU distributed tests
skipped in the initial run were enabled in the retest. The two inherently CPU-only
cases were run locally. Host orchestration and CPU child-process tests are included
in these totals; they are not all numerical TPU tests.

Both physical runs used JAX/jaxlib 0.11.1, Flax 0.12.9, Optax 0.2.8,
Orbax 0.12.4, libtpu 0.0.46.1 and Python 3.12.14.

## Defects found and corrected

1. The suite forced highest matmul precision on BF16 Splash attention. The pinned
   compiler rejected that combination with `Bad lhs type`. The runner now uses
   native/default precision for this module and records it. All four forward and
   gradient comparisons passed on v5e. Tolerances and training code were unchanged.
   The corrected setting has not been newly rerun on v5p.
2. Blocking queued-resource creation could consume the lease before the separate
   capacity deadline started. Submission now uses `--async` with a maximum
   60-second call timeout, followed by bounded readiness polling. A regression
   assertion covers this, and the final v5e launch exercised the corrected path.
3. Opt-in distributed tests were initially skipped. The runner now provides an
   explicit flag for them and distinguishes selected retests from full campaigns.

The new runner isolates modules, kills timed-out process groups, uploads individual
JUnit/log evidence after each module, and rejects empty, missing or all-skipped
reports as successful qualification. CI routes runner changes to its regression
tests. Lint, type checking and final local regressions passed.

## Provenance and retained evidence

Base commit: `011125f7a0dda8e294052666c425aa14c33c49cb`, with uncommitted changes.

- Initial source archive SHA256:
  `a9b2484ff7c9f74b61517330aef1529c0ce6d47950a255cd1f30e51f912d97e9`.
- Retest archive SHA256:
  `ccd47e2355d93beae3047894952047c677f7eb44a5f988c2208f8a94e75563ea`.
- Archive comparison found changes only in `scripts/validate_test_suite.py` and
  `tests/test_tpu_validation.py`; training implementation was identical.
- The subsequent local supervisor fix has separate local regression evidence and
  a launcher hash; it was not part of the remote source archive.

Detailed local evidence is retained under `artifacts/tpu-suite-0923/`:
`combined-inventory.json`, original and retry `summary.json`/JUnit/logs, hardware
records, source manifests, launch arguments, ledgers and cleanup receipts.
`verified-downloads.json` records SHA256 hashes for all 127 cloud objects after
checking their downloaded sizes and MD5 values against GCS metadata (3,144,001
bytes total). These artifacts are ignored by Git; this document does not make
them publicly downloadable.

## Limits and next qualification work

The v6e request did not obtain capacity. A second v5p request did not become ready
within the intended window and was cleaned up. The final bounded v5e retry passed.
This campaign does not qualify v6e, all scales, physical multi-host recovery,
released-model parity at production size, sustained throughput, or multilingual
model quality. Existing multi-host evidence remains a separate campaign.

Next priorities are a sustained physical multi-host run, independent verification
of raw recovery evidence, and multilingual quality acceptance. The broader gaps
in `HARNESS_AUDIT_2026_09_23.md` are not closed merely by passing this test suite.

## Cost and cleanup

Every request had a durable cleanup guard, bounded lease and conservative ledger
reservation. Reservations are planning bounds, not actual charges or credit usage.
Posted campaign cost is not yet established. The pre-run billing snapshot showed
$861.86 in marketing credits and a separate $1,000 TPU program credit with
different eligibility; these are not a combined unrestricted cash balance.

All four attempt ledgers ended in `resource_absent`. Fresh API listings confirmed
zero TPU nodes and queued resources in all three campaign zones: `us-central1-a`,
`us-east5-a`, and `us-west4-a`.

Local copies preserve the reports before deletion of the campaign's temporary
GCS prefix. A post-cleanup listing confirmed no live objects in that prefix.
Existing bucket soft-delete retention is preserved, so deleted objects
can remain billable during retention. This does not claim all project storage is
empty or that unrelated project services have zero cost.
