# Active training objective — 2026-10-06

The user explicitly reaffirmed that the next stage is **continued masked language
model (MLM) pretraining of the YAT-mmBERT base**, not contrastive fine-tuning.
The current contrastive continuation was the wrong execution of that request.
User ordered it stopped on October 6; shutdown verification is recorded separately.

1. Identify and authenticate the last verified **MLM** checkpoint, tokenizer,
   architecture, optimizer/schedule and data cursor. Do not silently substitute
   the latest contrastive checkpoint or random initialization.
2. Continue MLM with the agreed multilingual natural-language and code recipe,
   using physical TPU validation, durable recovery checkpoints, a finite approved
   budget/lease, and cleanup. Preserve fixed YAT bias 1, epsilon 0.01, trainable alpha.
3. Evaluate and publish the improved base model as its own artifact.
4. Only then perform a separately identified contrastive embedding stage and
   compare its embedding quality with EmbeddingGemma. MLM alone does not establish
   a competitive embedding model.

The subsequent user request, "run the correct shit then", authorizes a new MLM
continuation. See [the active MLM execution](mlm-continuation-2026-10-06/README.md). Do not auto-restart
run yat-embed-torch-v2-overnight-1006. Keep its GCS checkpoints as separate
contrastive artifacts; never label them continued MLM pretraining.

Next action: follow the one active MLM execution through data preparation and physical
MLM qualification. Do not launch a duplicate. The preserved public MLM step48236
weights are the parent; this is a new optimizer/schedule stage, not exact optimizer
resume of the completed historical stage.
