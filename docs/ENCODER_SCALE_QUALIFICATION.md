# Encoder scale and quality qualification

This protocol separates hardware correctness, bounded endurance, exact recovery,
scaling performance, and downstream quality. None automatically establishes the
others. Use a frozen source/data/weight bundle and the Spot supervisor with an
independent cloud deletion guard. Increase physical size only after the smaller
size passes. Record unsuccessful capacity requests as unvalidated.

If external SSH is unreachable but the existing IAP route is authorized, pass
`--tunnel-through-iap` to the supervisor. It selects the supported alpha TPU SSH
command without modifying IAM or firewalls. Individual SSH connection attempts
are bounded to 20 seconds. Setup has its own 300-second default workload timeout
(plus the transport's bounded grace), rather than consuming the training lease.

## Endurance and recovery

`scripts.validate_encoder_scale` runs on every physical worker. It checks the
requested topology, trains the declared horizon, repeats training with a committed
midpoint SIGKILL, resumes, and compares raw final model/optimizer/cursor manifests.
The default recipe uses BF16 compute, FP32 residuals/master weights/Adam state,
masked projection with an explicitly selected `xla`, `xla_full`, `xla_local` or `pallas` backend, and
a fixed global batch. It requires finite
accepted updates, no projection fallback, a final checkpoint and a declared
minimum measured training duration excluding compilation. Stage logs, hardware,
raw manifests and summaries are uploaded before the allocation is deleted.

`scripts.summarize_encoder_scale` independently checks all workers' raw evidence
and the controller completion report. It rejects missing fault markers, missing
workers, mismatched manifests and incomplete update horizons. A successful bounded
run is evidence for its measured duration, not a days-long production campaign.

Current encoder parameters are replicated; encoder FSDP is not implemented.
Mask-count-weighted gradient accumulation is available in the trainer. Each
microbatch must divide across the devices. Fixed-global-batch comparisons require the batch to divide
by the physical device count. Changing batch size changes the experiment and
must be reported, rather than presented as matched strong scaling.

## Multilingual quality probe

Before interpreting downstream scores, run `scripts.validate_encoder_reference`
in `export` then `verify` mode on the same local released snapshot. This compares
valid-token hidden states, mask-aware pooled embeddings and full-vocabulary logits
at selected positions against a frozen Transformers eager/FP32 oracle. Eight
languages and lengths 32/128/256 exercise padding and local-attention boundaries.
Weights, config, tokenizer and oracle bytes are checksum-bound. This establishes
implementation fidelity only; it cannot certify task quality.

`scripts.evaluate_encoder_xnli` evaluates a pinned `facebook/xnli` export through
our encoder implementation. The September 23 fixture uses dataset revision
`b8dd5d7af51114dbda02c0e3f6133f332186418e`, 1,024 English training pairs, and the first
512 test pairs in each of 15 languages. Training text and test text come from
separate official splits. NPZ input bytes are checksum-verified.

The probe separately mean-pools premise and hypothesis, concatenates both vectors,
their absolute difference and elementwise product, then fits a fixed ridge head
on standardized English training features. All languages use that same head and
training-only normalization statistics. Premise and hypothesis are separately
truncated/padded to 128 tokens. It reports per-language accuracy and
Wilson intervals, plus macro accuracy. Repeated premises and correlated examples
limit the interpretation of these binomial intervals.

`scripts.compare_encoder_xnli` rejects different datasets, missing languages,
changed sample counts and inconsistent macro scores before reporting candidate
deltas. Optional `--max-macro-drop-pp` and `--max-language-drop-pp` limits
(returning failure when exceeded) must both be declared before evaluating test
results. Without them the tool is descriptive only. Even a regression-gate pass
does not establish production acceptance.
Retain the evaluator source bundle identity for both runs.

This is a bounded frozen-encoder probe, **not** the published end-to-end fine-tuned
XNLI protocol or a production acceptance gate. Do not compare its score directly
with published mmBERT XNLI scores. It explicitly leaves production qualification
false. The public pretrained model's original data contamination is unknown.

Production qualification additionally needs a locked task/language/domain matrix,
full held-out datasets, a validated reference implementation, predeclared minimum
scores and regression margins, multiple seeds, retrieval and token-level tasks,
long-context evaluation, and evaluation of the actual resulting trained checkpoint.
An infrastructure pass or successful evaluation execution cannot substitute for
meeting those criteria.

## Campaign records

Local `artifacts/encoder-scale-0923/` contains the frozen source and input manifests,
quota snapshots, conservative campaign ledger, launch arguments, per-attempt
cleanup receipts, scale matrix and retained results. The campaign reserves at
most $150 without treating promotional balances or estimated costs as posted
charges. Deferred, failed, capacity-blocked and budget-blocked sizes must remain
explicitly unqualified. Temporary cloud artifacts are removed only after local
copies are verified; normal soft-delete retention remains in effect.
