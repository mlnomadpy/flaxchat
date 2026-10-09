# Authorized base MLM continuation — October 6

**Current status:** corpus preparation completed and is retained in GCS; TPU
setup failed before any MLM update because `{checkpoint_output}` was passed
literally to qualification. Owned controller, node and queue were deleted.
The placeholder-expansion fix and regression tests are local; no corrected
deployment has been started. The execution plan below records the failed attempt.

User explicitly ordered the wrong contrastive run stopped and requested the
correct base MLM continuation. Controller `yat-mlm-1006` in project `azettaai`,
zone `us-west4-a`, runs one claimed systemd execution `flaxchat-mlm-1006`.
No duplicate or automatic contrastive successor is authorized by this execution.

Parent: `mlnomad/yat-mmbert-base-mlm-48236` revision
`1788a8f4235ff05f167158ae38aef8fe7292a706`, model SHA256
`8afca9ad82a7f6e17d8bd6f1cb933c5002bb8aeb2cabc98baadc2ac2d1bba7c6`.
The original GCS bucket failed with billing-disabled403. Cloud downloads the
preserved public weights directly and authenticates original checkpoint leaves.
Original optimizer state is not available through the public release: this is a
new MLM stage with fresh AdamW, schedule and data cursor from those MLM weights.

Data target: at least1B prepared nonpadding tokens, source shares45% FineWeb2-HQ,
20% English FineWeb-Edu,15% FineWeb2 covering Hindi/Urdu/Swahili/Thai/Korean/Bengali,
and20% raw GitHub code. All27 natural languages are retained. Pinned revisions,
prior recipe coordinate exclusions, exact dedup, complete document windows,
source/code language/license provenance and disjoint heldout documents are saved.
Benchmark/semantic decontamination is not established by these checks.

Pipeline first prepares data on the CPU controller (no model execution), then
runs metadata admission, then requests one8-chip Spot v5e TPU. Allocation uses
an independent cleanup workflow with12-hour maximum lease. The inherited
`yat-embed-torch-` resource prefix is required by the scoped cleanup IAM condition;
it does not identify the objective. The stage is `base-mlm-continuation` and the
actual trainer is `scripts.train_encoder`, exclusively masked language modeling.

On TPU: import authenticated original MLM weights;4updates/save; identical-row
parent and child heldout MLM evaluation; resume to8updates/save/evaluate. All
must be finite and applied; heldout loss must remain within15% of the parent.
Only then does the bound qualification receipt admit sustained continuation.
This is import/update/resume qualification, not an uninterrupted numerical parity
comparison. Current physical acceptance must come from live receipts.

Planned horizon100,000updates, globalbatch64, accumulation4, sequence512,
masking15%, peak LR1e-5, warmup500, cosine decay, checkpoint every250updates,
retention3 and heldout checks every2,000updates. Actual executed steps are bounded
by the allocation deadline and quality checks;100,000 is not a completion claim.
Preserve parent architecture: YAT bias1/epsilon.01/trainablealpha, BF16 feature
compute, FP32 residuals and centered FP32 attention scores. No QAT/ternary change.

Cloud prefix: `gs://azettaai-yat-eval-0929/mlm-continuation-1006/`.
Pipeline status: `pipeline-status.json`; controller log: `controller/pipeline.log`.
The finalized manifest appears as `run.json` only after data hashes/admission.
Checkpoint prefix ends in
`yat-embed-torch-mlm-1006/base-mlm-continuation/checkpoints`; logs and MLM gate
receipts are in the sibling `checkpoints-evidence` prefix.

Cost: new allocation reserves$130 including conservative on-demand TPU ceiling,
cleanup and ancillary allowance, with prior$593.60 reservations retained under a
$750 campaign cap. These are reservations, not actual charges or credit balance.
Controller auto-deletes at16hours; systemd execution is capped at55,000seconds.
No laptop connection is required after dispatch.
