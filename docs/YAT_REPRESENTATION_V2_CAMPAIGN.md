# YAT representation improvement and quantized releases

User-authorized October 5, 2026: continue the trained model to improve multilingual
representations, then quantization-aware training, then a separate ternary candidate.
Publish each completed, validated stage as its own public Hugging Face model.
This document records intended stages; it is not a training or publication receipt.

## Immutable starting point

Start from `mlnomad/yat-mmbert-base-embedding-v1` revision
`571d0ef3d8e1381055845fce334a48be253c3fea`, weight SHA256
`6ee56349adbf10a3c3d5805d94539f987250aea2c98611b4d89c6fd9b7142aff`.
This is the retrieval fine-tune at stage step 14,000, not the earlier MLM model.
Each changed objective/data recipe starts a new stage with an explicit fresh
optimizer/schedule/cursor, preserving the original checkpoint. Fixed YAT bias 1,
epsilon 0.01 and trainable alpha remain mandatory.

## Stage and release sequence

| Stage | Intended public repository | Promotion condition |
| --- | --- | --- |
| Representation continuation | `mlnomad/yat-mmbert-base-embedding-v2` | Independent development improvement with declared language/task regression bounds; retained exact checkpoint and final evaluation |
| INT8 QAT | `mlnomad/yat-mmbert-base-embedding-v2-int8` | Exact exported quantizer matches training; TPU gradient/recovery and original-versus-quantized quality checks; actual compressed loader validation |
| INT4 QAT candidate | `mlnomad/yat-mmbert-base-embedding-v2-int4` | Separately measured quality/storage/serving tradeoff passes its declared gates |
| Ternary candidate | `mlnomad/yat-mmbert-base-embedding-v2-ternary` | Separate trained student, explicit quantized tensor coverage, quality and real storage/serving measurements |

Repository names are planned, not created. Do not publish empty repositories or
copy unchanged weights into a new release and call that training. A failed
candidate stays retained privately with its failure evidence; it does not become
the parent of the next stage. Native Flax releases and PyTorch conversions have
separate implementation/parity scopes and must be labelled accordingly.

## Finite checklist

- [x] Authenticate portable training parent against all 181 committed tensor leaves.
- [ ] Restore that parent on physical TPU through the current trainer import path.
- [x] Screen retrieval, bitext, code and STS candidates against all six declared
  historical training sources; retain 10,772 rows with zero exact/aligned overlap.
- [x] Tokenize and verify the complete portable development bundle.
- [ ] Quarantine all development rows/groups from every new training source.
- [ ] Freeze training mixture, language sampling, hard-negative policy, steps,
  schedule, checkpoint retention, development thresholds and financial cap.
- [ ] Run one bounded direct-training qualification on TPU, including gradients,
  finite updates and interrupted optimizer/sampler recovery. Cache remains off
  until its separate unresolved optimizer-state acceptance passes.
- [ ] Record parent development scores; continue training and select by development
  quality, preserving best and latest recovery checkpoints separately.
- [ ] Evaluate the selected checkpoint on a pinned final suite; record matched
  EmbeddingGemma results separately from published paper context.
- [ ] Export, publish and verify the new immutable Hub revision and full file hashes.
- [ ] Add exact INT8 QAT and qualify its forward/backward/export behavior on TPU.
- [ ] Train/evaluate/publish INT8; only then promote INT4 and ternary candidates.

## Training hypothesis

The first continuation should strengthen multilingual retention while preserving
retrieval and code gains. Refresh useful hard negatives and source diversity;
translation pairs are explicit bitext supervision, not assumed retrieval labels.
Distillation is a candidate upgrade, dependent on an authenticated usable teacher
and measured benefit. Teacher identity, prompts, pooling and any cached outputs
must be bound to exact examples. Final benchmark tests do not select mixtures.
No claim is made that matching EmbeddingGemma is guaranteed by a fixed step count.

## Quantization contract

QAT retains floating-point master weights and optimizer state. INT8 initially
models the existing symmetric per-output-channel linear / per-vocabulary-row
embedding export, with FP32 scales and reconstruction followed by the encoder's
existing compute policy. It does not imply INT8 matrix multiplication or reduced
training-state memory. YAT FFN projection and prototype distance must consume the
same dequantized kernel. Keep alpha, norms, residual and attention-score policies
explicit; do not silently ternarize all parameters.

Ternary projection weights with higher-precision embeddings are a partially
ternary model. Report actual tensor coverage and byte size: the vocabulary table
is a substantial share of this encoder. Quantizer scale changes YAT geometry and
cannot be discarded as though YAT were an ordinary linear layer. Hardware latency
claims require warmed end-to-end measurements on the intended serving backend.

## Current execution

A standard-library-only bounded Cloud Shell job `parent-v2-1005` prepared a separate
enriched parent under `gs://azettaai-yat-eval-0929/representation-v2-1005/parent`.
Its deadline is 900 seconds plus a 15-second termination allowance. No TPU is
allocated by that job. Terminal status, full byte authentication and cleanup must
be observed before marking the first item complete. The source receipt is retained
outside the repository under `/private/tmp/flaxchat-improve-1005`.

Parent preparation authenticated all 181 stored tensor leaves. The first bulk
upload timed out after preparation; its failure receipt is preserved. Nine small
metadata files were subsequently uploaded and independently hash-verified under
`representation-v2-1005/parent-metadata`; existing immutable release weights and
tokenizer are reused, avoiding a duplicate large transfer. This is byte-level
authentication, not physical restoration or new training. The 18-configuration
development candidate was exported successfully; full historical filtering is
a separate running gate.


The first physical request was rejected before allocation because its name was
outside the cleanup principal's existing scope. The corrected request used that
scope without changing IAM, but its 180-second Spot capacity window expired.
No model execution began. Both the corrected request's queue and node were
independently verified absent. Original receipts are retained in
[physical01](representation-v2-2026-10-05/physical01-campaign-receipt.json) and
[physical02](representation-v2-2026-10-05/physical02-campaign-receipt.json).
These attempts do not qualify the current QAT code or parent restoration.

The INT8 STE implementation is present and preserves floating-point masters,
trainable alpha, bias 1 and epsilon 0.01. The frozen physical suite requires 15
explicit cases covering quantizer/export agreement, gradients, direct training,
checkpoint recovery and fault behavior. Those cases remain unrun on this freeze.
No new model weights or Hugging Face repositories have been published in this
campaign yet.


The final bounded Spot attempt used a 600-second capacity window within a
2,400-second global lease. It also expired without allocation or model execution;
[its terminal receipt](representation-v2-2026-10-05/physical03-campaign-receipt.json)
confirms queue and node absence. No further capacity requests are part of this
attempt. A reservation is not a posted charge.

Cloud Shell changed boot identity while the public-data repair was running and
lost its `/tmp` workspace. The last observed transfer progress is retained; no
terminal worker or process-cleanup receipt was recoverable. The authenticated
compressed source archive and exported development candidate remain in GCS.
Data preparation is moving to a bounded GCE executor with provider-enforced
termination, while model computation remains TPU-only.

Development preparation now supports a portable bundle root. References to the
complete historical raw files and original stage metadata remain inside that
bundle, and their full exposure proofs are recomputed on admission. Moving the
bundle and rejecting altered raw bytes passed model-free tests. The physical
qualification archive predates this data-only addition; a real training launch
must freeze the final source and prepared artifacts together.


The GCE executor completed the full six-source history scan. The original
512-row-per-configuration candidate failed the unchanged 64-row minimum after
filtering: Arabic retained 54 and Dutch 45. The
[diagnostic counts](representation-v2-2026-10-05/original-candidate-rejection-counts.json)
record every slice. Its original full-history scan found 23,213 overlapping
historical rows; that count is not the number of selected development examples.

A separate [expanded recipe](../configs/data/representation-development-next-stage-v2.json)
keeps identical pinned sources, seed and 10,000-row scan limit while selecting up
to 2,048 rows per configuration. Filtering passed with Arabic 199 and Dutch 169;
all eight bitext languages remain. Retained totals are 4,077 bitext, 3,776 native
retrieval, 1,419 code and 1,500 STS rows. The final clean-history recheck and portable
tokenization have separate acceptance receipts; filtering alone is not model or
production-training acceptance.


The expanded candidate's independent recheck completed successfully with zero
exact/aligned overlap across all six declared histories. Its controller also
verified process cleanup and removed scratch. The
[expanded development handoff](representation-v2-2026-10-05/expanded-development-handoff.json)
retains source hashes, execution receipts and generation-pinned GCS outputs, including the filtered
candidate archive and clean scan. This does not establish absence of undisclosed
MLM exposure or semantic/translation overlap. Portable tokenization is a separate
bounded job on the same VM; no new TPU request was issued.


Portable tokenization and finalization completed successfully. All three full
parent-exposure proofs were recomputed exactly; the complete generation-pinned
1,265,421,119-byte GCS archive was read back and matched SHA256
`4447de1376a17cd0c5c98cf082d16b37391469d34fa5a6225e317ebff8df2559`.
[Final portable receipts](representation-v2-2026-10-05/portable-development-handoff.json)
retain controller cleanup, source, proof replay and full readback evidence. The
initial 180-second aggregate replay timeout remains recorded; successful
finalization used the same verification with a 600-second limit and reused the
prepared bytes. The bundle contains shared historical evidence and all four
tokenized development tasks; parent model weights remain separately pinned.

**Next gate:** prepare and quarantine the new training mixture against this exact
bundle, freeze the stage, and obtain TPU capacity for current-source qualification
and a bounded real-data continuation. The current model is still embedding-v1;
no new model checkpoint or public Hugging Face release was produced in this
attempt. INT4 and ternary remain later stages, not implemented or trained releases.


The temporary GCE data VM was deleted after artifact verification. Independent
provider listings confirmed that both it and its boot disk are absent:
[cleanup receipt](representation-v2-2026-10-05/gce-cleanup.json). Both attempted
Spot queues/nodes were also verified absent. Authenticated GCS data and protected
model weights are retained.


## October 5 unattended continuation launch

A single cloud-owned continuation campaign has now been dispatched. Its exact
source/specification, provider lifecycle, additional100USD reservation cap and
first-bootstrap failure are recorded in [the overnight execution plan](representation-v2-2026-10-05/overnight-training.md)
and [launch observations](representation-v2-2026-10-05/overnight-launch-intent.json).
The corrected controller authenticated and materialized pinned cloud artifacts
and reached locked-runtime preparation. Data preparation, physical qualification
and actual model steps remain distinct later phases; do not report them as passed
from controller creation. The remote campaign owns one eventual8hSpotTPU
attempt and survives laptop shutdown. No Hugging Face publication has occurred.
