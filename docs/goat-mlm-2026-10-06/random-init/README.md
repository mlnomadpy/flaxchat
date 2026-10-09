# Random GOAT base MLM stage

Explicit user instruction: start from random initialization. Controller `yat-goat-1006e`, project `azettaai`, zone `us-west4-a`, Spot `v5litepod-8`. Output: `gs://azettaai-yat-eval-0929/goat-mlm-1006e/yat-embed-torch-goat-1006e/goat-scratch-base-mlm/checkpoints`.

GOAT uses only V for attention scores, excludes diagonal, and retains the YAT FFN. All model weights are newly initialized with seed1006; no public weights are deployed or loaded. Reference config/tokenizer/metadata are pinned only to define architecture and data vocabulary. Fresh optimizer and cursor. Fixed YAT bias1, epsilon0.01, trainable alpha. BF16 compute with existing FP32 residual/score policy.

Prepared corpus: 1B non-padding tokens; 45% FineWeb2-HQ, 20% FineWeb-Edu English, 15% additional multilingual FineWeb2, 20% code, with 27 human languages and programming-language provenance. This bounded run may repeat rows. No production model quality claim follows from the initial numerical checks.

Schedule: ceiling100,000steps, sequence512, globalbatch64, accumulation4, AdamW peak3e-4, warmup500, cosine finalratio0.05, gradientclip1. Checkpoints250steps, latest3 retained; heldout evaluation every2,000steps. Training limit10hours after qualification; independent TPU lease12hours, controller providerDELETE15hours.

Qualification: all14 physical GOAT tests; full-model4updates/checkpoint; paired512-row evaluation against the identical random initializer; restore/update through8; second evaluation. Scratch sanity gate: finite loss within5% of random baseline on identical masked tokens. This replaces the trained-parent preservation gate for this explicitly random stage. Source/runtime/data identity binds subsequent continuation.

Available credit observed in Chrome: $858.59 (expires July22,2027). Prior retained conservative reservations $768.85; this attempt reserves $130, aggregate $898.85 under unchanged $900 cap. Actual Spot rate and per-run posted charges are not verified. Reservation uses $9.60/slicehour conservative on-demand ceiling plus ancillary/cleanup allowance.

Source, runtime, reference, corpus, and controller identities are in run.json and freeze.json. Read durable receipts before claiming numerical acceptance or active training. The controller operates independently of the laptop, admits only one attempt, and cleans the TPU after failure or deadline.

## Confirmed execution

All14 physical TPU cases passed (24.44seconds). Full random model updated/saved at4, resumed through8 and evaluated on identical512heldout rows/30,168masked tokens. Loss: initializer12.5444245308; step4 12.5083982645; step8 12.3784664232. Bound source/runtime/config evidence in mlm-qualification.json admitted the long workload.

Observed live step130, updated=true, training-batch loss8.8029203415, warmed useful rate55,677nonpaddingtokens/s. This training batch differs from heldout inputs, so do not present8.803 as heldout loss. Latest committed checkpoint at observation:8; automatic saves every250steps. Independent guard deadline October7 at10:50:20UTC (03:50PDT); workload has its own10hourlimit. Job runs in GCP independently of this laptop.
