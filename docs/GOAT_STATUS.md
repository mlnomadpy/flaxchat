# GOAT release and training status

Updated October28,2026. This status supersedes historical campaign instructions in the training runbook.

## Published MLM base

The [native GOAT release](https://huggingface.co/mlnomad/goat-mmbert-yat-base-mlm-10b) contains checkpoint379079, initialized from scratch and pretrained with MLM. Its final corpus stage covered10,038,104,948 prepared nonpadding tokens after an earlier MLM stage. Complete optimizer/counter/sampler recovery and protected parents are retained separately from the public weight-only release. The public dataset catalog contains source provenance and code references; it is not a promise that every original code payload is publicly redistributed.

[Full benchmarks](GOAT_BENCHMARKS.md) cover41 English MTEB tasks,131 multilingual MTEB tasks,10 CoIR tasks and18 MIRACL hard-negative language subsets. Historical embedding-v1 and paper comparisons used fine-tuned models. They were not rerun alongside this base evaluation. A separate PyTorch release remains gated on complete physical parity; the existence of conversion tooling is not a parity pass.

## First contrastive pilot

A separately identified mixture has563,962 training rows across MS MARCO, code, multilingual bitext and eight native MIRACL sources. Its quarantined development candidate has8,082 retrieval/bitext/code/STS rows. The original full representation-v2 training archive was removed with older storage history; this prepared pilot is a rebuild from pinned sources and retained bitext, not an identical replacement archive. Undisclosed MLM and semantic/translation exposure remain unresolved.

The first bounded attempt imported all181 native parent leaves on an eight-chip TPU and passed four physical hard-negative loss/gradient cases. Qualification stopped in actual-data admission:32 development rows could not cover every language with at least two rows. No real optimizer updates or production contrastive checkpoints were created. The corrected128-row qualification sample passes model-free actual-data admission and a regression test. Full model/Adam/sampler recovery qualification and production remain physically unrun. There was no automatic retry; exact queue/node/controller/disk cleanup was independently verified.

The published MLM weights, full-state recovery parents and raw failure receipts are preserved. Deployment requires fresh account/credit/quota admission, a frozen source/data/runtime manifest, an explicit finite budget and an independently armed cleanup guard. No README or example command authorizes allocation.

## Accumulated tooling

- MLM corpus preparation: provenance-preserving tokenization, authenticated history exclusions, bounded worker memory and resumable parallel preparation.
- Recovery: full-manifest pinning, remaining-corpus sampling with padded tails, and explicit full-state sampler migration without resetting Adam or the learning-rate horizon.
- Native Hub loading: `flaxchat.hub.from_pretrained`, with immutable revision support.
- Contrastive admission: authenticated MLM-only lineage for the first contrastive stage, real-data qualification and bounded successful-segment continuation.
- [Looped encoder / LoopCD](LOOPED_ENCODER.md): opt-in implementation with separate physical acceptance requirements; not silently enabled in the published model.

Historical dated operational receipts are retained outside the Git source payload. They do not establish that old resources are running or that an unqualified feature passed. [Coverage transition design](coverage-transition-design.md), [training tools](TRAINING_TOOLS.md) and [runbook](REPRESENTATION_TRAINING_RUNBOOK.md) describe the implementation and its boundaries.
