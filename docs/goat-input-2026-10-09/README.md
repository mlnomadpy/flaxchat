# October 9 GOAT-input MLM continuation

Corrected exact continuation from committed step **35,500**. Attempt1009a failed
closed before model tests/updates: the35750 directory lacked its GCS commit marker.
Its cleanup is independently verified. Attempt1009b authenticated commitment but its assigned TPU image lacked gcloud
and failed before model execution; its cleanup is verified. Attempt1009c includes
a generation-pinned CloudCLI588 bootstrap for setup and workload, repeats an
SDK/GCS smoke check on the controller before allocation, and authenticates commitment and
metadata/manifest hashes **before TPU allocation**, then retains physical restore
and numerical admission. The earlier35750 recovery claim was incorrect. This preserves the
original GOAT-input scratch stage, frozen source/runtime, optimizer, schedule,
seed, sampler cursor, and retained multilingual/code corpus. Objective: MLM.
Scores use attention inputs; V remains projected; diagonal scores are excluded.

Controller: `yat-goat-input-1009c`, project `azettaai`, zone `us-west4-a`.
Evidence: `gs://azettaai-yat-eval-0929/goat-input-1009c`.
Checkpoints remain in the original `goat-input-1007b` stage namespace.
The request uses eight v5e Spot chips, one attempt, a one-hour capacity wait,
10-hour global TPU lease and up to 9-hour workload; capacity and setup reduce
available training time. Independent workflow cleanup precedes allocation;
controller provider deletion is bounded at 14 hours. Saves every 250 updates,
keeps three rolling recovery checkpoints; this does not retain every step.

Fresh physical admission requires the selected 30-test GOAT suite and full-model
held-out evaluation at batch16 on512 fixed rows, then exact optimizer/cursor
restore. First segment ends36000; later boundaries every2000 steps. Evaluation
must remain finite and within5% of the fresh35500 recovery evaluation. Full-model
batch8 evaluation remains known nonfinite and unqualified.

Local acceptance:37 ledger metadata checks and16 isolated external recovery
metadata checks passed. The no-JAX-import assertion requires isolated execution:
combining the accounting suite imports the flaxchat package. No CPU model tests
were used. Independent read-only review passed deployment/hash/budget checks.

Conservative reservations after evidence-backed terminal cleanup reconciliation:
$786.95 prior +$110.80 new = **$897.75**, within the unchanged$900 campaign cap.
Reconciled1007d,1008b,1009a and1009b retain their original events and interruption status;
only unused elapsed-time reservation was released;1009a retained$11.60 and1009b$11.61 after final cleanup. These are **not posted costs**.
Chrome billing UI observation: active credit$780.90; this is not per-model spend.
The9.60/hour whole-slice admission rate uses Google's1.20/chip-hour v5e us-west4
on-demand list ceiling, checked October9; actual Spot rate remains unverified.
Source: https://cloud.google.com/tpu/pricing.

See [deployment freeze](recovery-c/freeze.json), [manifest](recovery-c/run.json),
and the subsequent status receipt for actual physical progress. Controller creation
or a queued request alone must not be described as model training.

The CloudCLI bootstrap was tested on the existing Linux controller before a new
TPU request: version588.0.0 and authenticated35500 metadata read passed. Google's
[versioned archive guidance](https://docs.cloud.google.com/sdk/docs/downloads-versioned-archives)
links the public release bucket; pinned object generation1791292489113329 MD5
matched the downloaded bytes, whose full SHA is recorded in the manifest.
The latest-alias checksum belongs to a different object and was not used.

## Verified active continuation

1009c passed all30 physical GOAT tests (44.3seconds, no skips/failures). Restored
35500 evaluation: masked-token loss **1.9821179338773598**, batch16,512rows.
Independent worker read verified applied finite updates through **35544**.
Latest independently verified durable checkpoint remains35500 at this observation;
35750 is the next scheduled save, not yet claimed committed. The worker continues
cloud-owned training. See [active observation](recovery-c/training-observation.json).
