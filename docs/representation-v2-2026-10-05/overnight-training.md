# October 5 single cloud continuation

The user requested continued multilingual/code representation training that survives laptop shutdown. This is a new contrastive stage from the authenticated embedding-v1 step14000 public weights, with a fresh optimizer for the changed data. It is not a new random initialization or a return to MLM.

Cloud campaign prefix: `gs://azettaai-yat-eval-0929/representation-v2-1005/overnight-2200`. Source and specification hashes, provider identity and current observations are in `overnight-launch-intent.json`. Read remote `campaign.json` for current phase; neither this plan nor controller creation proves training began.

- One owned GCE controller, e2-standard-8, 100GB auto-deleting boot disk, attached service account, provider deletion after12h. No user OAuth or Hugging Face token copied.
- Pinned full2,181,332 replay candidates plus eight native MIRACL sources. Six historical heldouts and the independent retrieval/bitext/code/STS bundle are excluded. Data split seed41 avoids the empty-development replay defect; source shuffle and trainer seed29.
- Mixture:45%MARCO,20%bitext,20%native multilingual retrieval,15%code. Code programming-language labels are unknown. These counts are before filtering, not measured trained exposure.
- Data preprocessing has7200s deadline, shared production admission1800s. Failure stops the campaign before allocation.
- One v5e8-chip Spot request, capacity wait1800s, startup600s, full lease28800s including waits/setup. Current-source parent import and ten direct loss/gradient/recovery/fault tests must pass before training. No cache or QAT in this continuation.
- Training horizon20000steps, batch128, query128/document256, learning rate1e-5, warmup500, recovery every100steps, quality every500steps. Latest3 recovery checkpoints and independent best retained in GCS. Runtime lease can stop before horizon; resume requires committed optimizer/cursor identity.
- Independent per-language/task regression tolerance0.02 remains unchanged. A completed run is not proof of matching EmbeddingGemma.
- Maximum additional reservation100USD, planned91.6USD (TPU81.6 including30mcleanup, ancillary2, controller8). Original147.4USD reservations are carried into247.4USD aggregate cap. Reservations are not posted charges. TPU ceiling uses8×1.20USD/hour on-demand price as conservative Spot allowance; actual Spot price is not asserted.

The reviewed opt-in long-lease workflow was deployed as revision000002-9d2. Default2h safeguards remain; explicit opt-in permits up to12h. Live workflow arguments/source/permissions still gate each actual allocation.

First controller boot authenticated source/spec but failed before data work because its bootstrap Python lacked tarfile filter support. It uploaded its failure log and self-deleted; instance and disk absence were verified. Corrected bootstrap uses explicit archive type/path containment checks. Source/spec and original absolute campaign deadline were unchanged. No TPU was requested by that failed bootstrap.

Validation: orchestration boundary/timeout tests99passed, cloud-wrapper metadata tests37passed, lint/syntax passed; actual replay-split metadata fixture passed with9796train/104dev. These are model-free checks, not physical TPU acceptance.
