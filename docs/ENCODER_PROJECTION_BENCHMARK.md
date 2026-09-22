# MLM projection optimization and matched benchmark

The baseline projects every encoded position into all 256,000 vocabulary entries.
Only selected MLM labels contribute to the loss. `--mlm-projection masked`
compacts selected positions into static per-row buffers; overflow runs the dense
path without discarding labels. The default remains dense; the optional backends have performance evidence below,
but downstream model quality and multi-host encoder execution remain unqualified.

## Fused TPU backend

**Qualification status (2026-09-22): experimental; default remains `xla`.**
After the shared-head refactor, both optimized backends pass the released-model
gradient regression. Earlier failures are retained below to explain the fix.
Downstream quality and physical multi-host encoder execution remain unqualified.

`--mlm-loss-backend pallas --mlm-vocab-tile 1024` selects the experimental fused
projection/cross-entropy implementation. It composes with masked selection.

The forward kernel computes vocabulary tiles in TPU VMEM, returning only their
log-sum-exp and target scores. A stable reduction combines normalizers across the
entire vocabulary. The custom backward recomputes each tile, forms all softmax
gradients, and computes hidden/decoder/bias gradients. Hidden gradients accumulate in FP32 in VMEM across vocabulary tiles.
For more than 512 projected rows, a second token axis keeps decoder tiles resident
and accumulates decoder gradients in FP32 before casting them once. There is no vocabulary sampling or gradient pruning.
BF16 dot outputs are rounded before the FP32 bias addition, matching the existing
head's precision contract. Tile/reduction ordering can cause numerical differences.

The backend uses explicit data sharding around device-local Pallas calls. Both
loss sum and selected-token count are reduced globally; decoder gradients include
all shards. Token tiles are 128 or 512 rows; hidden size and vocabulary tile size must
be multiples of 128. Tail vocabulary padding is excluded from normalization.
Full encoder outputs remain available; only training/evaluation with labels uses
the configured loss backend. Parameters, residuals, and Adam state retain the
existing FP32 policy. Projection/backend/tile are checkpoint identity fields,
and the fused implementation is included in the source hash.

The implementation avoids the full token-by-vocabulary logit tensor, but still
allocates decoder gradients and per-tile normalization statistics. It is not a claim of
zero temporary memory or guaranteed speedup.

## Measurement protocol

`python -m scripts.benchmark_encoder_projection --output LOCAL_DIR --prefix FRESH_GCS_PREFIX`
must run inside the external Spot budget/deadline supervisor. Each case gets a
fresh process. Results and failure logs are uploaded after each case.

1. Compare fused versus dense forward/backward on odd and full vocabulary sizes.
   Kernel qualification checks loss and all hidden/weight/bias gradients.
2. Sweep configurable vocabulary tiles and projected-row counts; a tile must pass
   every requested row count. Use 5 warmups and 50 synchronized forward/backward
   measurements. The first campaign tested tiles through 4096 at 128 rows; the
   revised campaign tests 512/1024 tiles at 128/512/1024 rows.
3. Gate new backends on released-weight loss and all-parameter gradients against
   masked XLA: absolute loss error below 1e-3 and global relative gradient L2 below
   0.03. These are numerical acceptance limits, not downstream quality claims.
4. Compare dense XLA, masked XLA, explicitly sharded unchunked masked XLA
   (`xla_full`), and masked Pallas on released mmBERT weights. Batch/context
   combinations are configurable. Each case runs 55 AdamW updates, discards
   5 warmups, and commits one checkpoint.
5. Report steady input tokens/s, median/p90 update time, compilation plus first
   update, checkpoint time, invocation time, per-device memory, mask overflow
   count, and estimated compute cost per billion input tokens.

All cases use the same source bundle, initial weights, data, seed, precision,
learning rate, and optimizer. The prepared fixture has no padding but is
synthetic/repetitive: this is a performance and execution campaign, not quality
qualification. Different batch sizes process different numbers of tokens;
compare projection modes at the same batch/context before attributing speedup.

Compiled temporary-memory accounting is recorded for the isolated head. Its
process peak also includes the dense numerical oracle and is labeled accordingly.
Full-training process peaks are isolated per case. Compilation measurements can
benefit from the configured compilation cache. Spot cost estimates use the
recorded quote, not posted billing, and steady rates exclude setup/compile/save.

## Local checks

- FP32/BF16 loss and custom-gradient comparison, vocabulary tails and ignored labels.
- All-ignored labels yield zero loss and gradients.
- Four-device CPU interpreter comparison with unequal per-shard mask counts.
- Encoder loss/gradient parity, overflow fallback, training and exact resume.
- Benchmark aggregation tests ensure warmups/checkpoints do not inflate steady rates.
- CI selects the fused tests and runs the four-device check when kernels change.

## References

- [JAX TPU Pallas details](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
- [JAX TPU matrix multiplication](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html)
- [Cut Your Losses](https://arxiv.org/abs/2411.09009)

## Measured v5p results — first kernel revision, 2026-09-22

Four chips (`v5p-8`), 307.8M parameters, vocabulary 256,000, BF16 compute and
FP32 residual/master/Adam state. Fifty timed updates per successful case.

| Batch × sequence | Dense XLA tokens/s | Masked XLA tokens/s | Masked fused tokens/s |
|---|---:|---:|---:|
| 4 × 512 | 28,042 | 36,735 | 36,931 |
| 16 × 512 | 50,162 | 97,339 | 110,866 |
| 32 × 512 | 55,241 | 127,122 | 170,225 |
| 4 × 2048 | 46,661 | 84,409 | Not run: campaign time bound |

At batch 32, fusion is 3.08× the dense baseline and 1.34× masked XLA.
Estimated steady compute cost is $6.55/billion input tokens versus $20.18 dense,
using the recorded $4.013452/hour whole-slice Spot quote. Short invocations cost
much more per token: compilation, setup and saving dominate a 55-step run.
Peak device memory is approximately 7.1–7.3 GiB across these cases; there is no
measured full-training peak-memory improvement to claim. Checkpoints take about
40–42 seconds. No measured masked-buffer overflows occurred.

The isolated 128-row fused head was slower than XLA (1.20 ms at tile 2048 versus
0.90 ms), despite reducing compiled temporary storage from 197.2 MB to 49.3 MB.
Tile 4096 failed with a VMEM allocation error and was excluded. End-to-end gains
include explicit sharding and larger projected batches, not fusion alone.

This campaign used the earlier kernel that stores hidden-gradient partials in
HBM. The current source accumulates them in VMEM and tiles both axes. Do not
attribute these measurements to that newer implementation. A frozen paired
v5p/v6e campaign tests the newer implementation and adds the `xla_full` control.

Evidence: `artifacts/encoder-projection-0922/fused1-summary.json` and
`fused1-comparison.json`; remote reports/logs live under
`gs://tpubuilders-flaxchat-validation-0921/encoder-projection-0922/fused1/`.
The first TPU and queued resource were confirmed deleted.

## v6e comparison

The queued v6e experiment uses four Spot chips (`v6e-4`) in `us-central1-a`,
with the same frozen source and workload as its paired v5p experiment.
The request stayed in `WAITING_FOR_RESOURCES` for approximately 24 minutes and
was cancelled. **No v6e throughput or numerical result was obtained.** Queue and
VM absence in both zones were independently confirmed after the runs. The
Cloud Billing catalog quote retrieved on 2026-09-22 is $0.680951/chip-hour,
or $2.723804/hour for the slice (SKU `9A7C-D787-1EFD`). This is a price estimate,
not posted spend or a throughput result. Evidence is saved in
`artifacts/encoder-projection-0922/v6-prices.json`.

Each allocation has an independent Cloud Workflows 40-minute cleanup deadline.
The supervisor reserves conservatively at on-demand rates plus cleanup time and
an ancillary reserve; its budget reservation is not an actual charge.


## Revised-kernel qualification — v5p

All requested kernel shapes (128/512/1024 rows; vocabulary tiles 512/1024) passed
loss and hidden/weight/bias gradient checks, including an irregular vocabulary.
At 128 rows, compiled temporary storage is 0.49 MB for tile 1024 (0.67 MB for
tile 512), versus 197.2 MB for dense XLA. The fused isolated head is still slower:
1.28 versus 0.90 ms at 128 rows, 3.98 versus 2.15 ms at 512 rows, and 9.91 versus
4.06 ms at 1024 rows. These are one-chip head measurements, not training rates.

The stronger released-mmBERT check found:

| Backend | Absolute loss difference | Relative L2 of gradient difference | Gate |
|---|---:|---:|---|
| `xla_full` | 0.00003909 | 6.474% | Fail |
| `pallas`, tile 1024 | 0.00000719 | 7.928% | Fail |

The limit was 3% before measurement and was not relaxed. Neither backend was
admitted to the revised training benchmark. A small loss difference and passing
isolated kernels do not establish acceptable full-model gradients. The earlier
170K tokens/s result is performance evidence only, not a qualified training
recommendation. The exact source of the gradient discrepancy remains unresolved;
sharding, BF16 rounding, and gradient accumulation order need diagnosis against
an FP32 reference on realistic held-out data. The gate now records per-parameter
absolute/relative error, norms, and cosine similarity for future runs. That richer
diagnostic was added after this frozen campaign and is not present in its results.

Local integration: 51 passed, one skipped, one accelerator-only test deselected
with four simulated CPU devices. Additional focused diagnostic/aggregation tests:
4 passed. Supervisor tests: 15 passed, including terminal capacity failures and
capacity wait expiry across host sleep. Lint and type checks passed.

The revised v5p campaign measured 50,271 tokens/s for dense XLA and 93,420 for
masked XLA at batch 16 × 512. Both batch-64 cases failed before training because
the fixture contains only 32 rows; this is not a TPU memory or scaling result.
The report preserves those failures. Before the next allocation, supply enough
prepared rows and run the new CPU-only fixture preflight:

```bash
python -m scripts.benchmark_encoder_projection --preflight-only \
  --output artifacts/projection-preflight --cases 16:512 64:512
```

The command correctly rejects the existing 32-row fixture for batch 64. The
campaign also checks fixture sizes at startup, and now returns nonzero for
incomplete coverage or failed qualification instead of reporting process success.
These reporting/preflight fixes were added after the frozen campaign. The
successful supervisor workload exit in the old ledger does not mean every case
passed. The two new backends remain unqualified; batch 64 and v6e remain unmeasured.

Estimated compute for the two completed v5p allocations is approximately
$2.58–$3.16 using observed READY-to-deletion and full provisioning-to-deletion
windows at the recorded Spot rate. This is not posted billing; storage/network
are additional. The v6e request never obtained hardware. The outcome and ledger
references are recorded in `artifacts/encoder-projection-0922/cloud-outcome.json`.

## Follow-up diagnosis and regression fix

The original failing fixture is retained. On v5p, freezing the encoder outputs
made Pallas and unchunked XLA agree with each other while both differed by about
9.11% from the chunked prediction head. The largest difference was in the head's
dense kernel (about 9.48%); its input gradient differed by about 10.47%. This
localizes the primary discrepancy to shared head computation/chunking rather than
only the custom Pallas backward implementation.

`prediction_features` now runs the dense/GELU/normalization transformation once
before decoder-loss chunking. All backends consume this common transformation.
Only the vocabulary projection/CE remains inside the rematerialized scan. The
objective and parameter shapes are unchanged, but BF16 rounding changes; source
identity therefore prevents silently resuming an older implementation.

`diagnose_encoder_projection` records per-parameter diagnostics and compares
against a separately constructed FP32 model. The first diagnostic did **not**
request highest TPU matmul precision, so its nominal FP32 comparisons must not be
used as a strict FP32 oracle. Subsequent diagnostics explicitly set and record
`matmul_precision=highest`.

`validate_projection_campaign` checks both backends against the original failing
fixture before allowing larger-batch throughput measurements. The 3% gradient
and 1e-3 loss thresholds are unchanged. `prepare_projection_fixture` creates a
separate deterministic 128-row multilingual synthetic fixture for batch-64
throughput testing, with tokenizer/checksum validation. Neither fixture qualifies
language-model quality.

The local refactor checks passed: 53 tests, one skip, one accelerator deselection
on four CPU devices. The launcher also now includes host sleep in its local SSH
wall deadline and reaps SSH process groups on timeout/cancellation. Independent
cloud cleanup remains required; the first diagnostic's cloud resources were
confirmed absent even when its old local SSH process remained stale.

Neither the four-chip v6e retry nor the single-chip fallback produced a workload
result or an observed READY event. Both requests were cleaned up. The controller
was interrupted during the single-chip wait, so absence of a READY event is not
proof of zero billable allocation; posted billing was not checked. The refactored v5p regression campaign is recorded
under `artifacts/encoder-validation-0922/headfix4/`.

### Shared-head regression result

The refactored v5p run passed the original failing fixture without changing the
thresholds. Full-model gradient relative L2 is **0.6875% for Pallas** (previously
7.9276%) and **1.9545% for unchunked XLA** (previously 6.4743%). Absolute loss
differences are 0.00004630 and 0.000000028 respectively, below the 1e-3 limit.
The baseline now computes its head once, as the other backends do. These are
comparisons of the corrected implementations; they do not assert bitwise identity
to the old chunked implementation.

Evidence: `artifacts/encoder-projection-0922/headfix4-diagnosis.json`, frozen source
provenance in `artifacts/encoder-validation-0922/headfix4/source.json`.

With explicitly highest TPU matmul precision, the corrected FP32 chunked versus
unchunked comparison has relative gradient L2 **5.59e-6** and absolute loss
difference **5.59e-9**. BF16 versus that FP32 reference is about 4.96–5.59% in
relative gradient L2 on this very low-loss fixture. This distinguishes backend
consistency from the broader BF16 precision policy; the latter still needs
real held-out/downstream quality evidence.

## Completed corrected batch-64 benchmark

Spot v5p-8 (four chips), batch 64 × sequence 512, released 307.8M mmBERT,
128-row synthetic fixture, BF16 compute and FP32 residual/master/Adam state.
Each case completed 55 accepted optimizer updates, with five warmups excluded
from the 50 timed updates. All numerical gates passed; no cases failed or were
omitted, and no masked-buffer overflows occurred.

| Projection/backend | Input tokens/s | Speedup over dense | Steady compute $/billion input tokens | Peak GiB/chip |
|---|---:|---:|---:|---:|
| Dense XLA | 53,458 | 1.00× | 20.85 | 7.24 |
| Masked chunked XLA | 152,678 | 2.86× | 7.30 | 7.25 |
| Masked unchunked XLA | **252,081** | **4.72×** | **4.42** | 7.24 |
| Masked fused Pallas | 233,122 | 4.36× | 4.78 | 7.25 |

**Recommendation for this measured shape: masked `xla_full`.** Pallas is 8.1%
slower in step time and provides no measured full-training peak-memory benefit
here. Its isolated temporary-memory savings do not imply a lower whole-training
peak. Keep Pallas available for further shape/hardware sweeps, without assuming
fusion is always faster. Defaults remain portable dense/XLA; the optimized
setting is explicit and checkpoint identity prevents silent backend switches.

For a new run on this tested hardware/shape, add:

```bash
--batch-size 64 --mlm-projection masked --mlm-loss-backend xla_full \
--dtype bfloat16 --residual-dtype float32
```

Use a 512-token prepared dataset with at least 64 rows and run input preflight
before allocation. The fixture-generation recipe is:

```bash
python -m scripts.prepare_projection_fixture --output artifacts/fresh-fixture \
  --tokenizer artifacts/mmbert-base/tokenizer.json --rows 128 --length 512
python -m scripts.benchmark_encoder_projection --preflight-only \
  --data-root artifacts/fresh-fixture --cases 64:512 --output artifacts/preflight
```

First compile/update took 49–53 seconds; checkpointing took 39–44 seconds. The
short-run invocation cost for the winning backend was $70.41/billion input tokens,
versus $4.42 steady compute. Both are estimates at $4.013452/hour; neither is
posted billing or a production training quote. Real data loading, interruptions,
checkpoint cadence, and quality targets can change costs.

The larger fixture's released-weight gradient gates also passed: 0.872% relative
L2 for unchunked XLA and 0.719% for Pallas. The strict FP32 diagnostic and original
failing-fixture gates precede these results. Complete reports:
`artifacts/encoder-projection-0922/headfix4-summary.json`,
`headfix4-comparison.json`, and `headfix4-diagnosis.json`. Remote logs/results:
`gs://tpubuilders-flaxchat-validation-0921/encoder-validation-0922/headfix4/results/`.

Remaining qualification: v6e execution/throughput, physical multi-host encoder
training, long-context/other-batch sweeps of this revision, physical fused
checkpoint interruption/recovery, and realistic multilingual held-out/downstream
quality. CPU exact resume passes; a successful TPU checkpoint write is not a
physical interruption/recovery test.

The final corrected verification allocation was cleaned up and its supervisor
exited successfully. Estimated compute for that allocation is **$1.23–$1.52**,
using observed READY-to-absence and full request-to-absence windows respectively.
Earlier diagnostic runs and storage/network are additional; no posted-billing
or remaining-credit claim is made. Evidence: `headfix4-cost.json` and the run ledger.
