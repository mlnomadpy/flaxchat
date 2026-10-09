# Input-scored GOAT TPU validation and MLM training — October 7

Cloud controller `yat-goat-input-1007b` in `azettaai/us-west4-a` runs one bounded Spot v5e eight-chip attempt. Frozen source and inputs are recorded in `freeze.json` and `run.json`. Attention uses input geometry for YAT scores, independently projected values, diagonal exclusion, fixed bias1/epsilon0.01, trainable alpha. Random seed1006, no parent tensors, fresh optimizer.

Admission:30 physical attention cases including value-independent score geometry, full-model MLM updates, heldout comparison against the identical random initializer, checkpoint/exact resume. Only successful qualification permits the longer workload. No CPU numerical tests.

Workload deadline2h after qualification; TPU global lease2½h plus30-minute budget cleanup reserve with independent cleanup workflow; controller provider DELETE deadline6h. Checkpoints every250steps, evaluation every2000, latest3 retained. Training horizon100k is a schedule, not a claim this bounded attempt reaches100k.

Conservative reservation36.80USD includes2½h lease plus½h cleanup at9.60USD/slice-hour plus8USD ancillary. Retained prior reservations862.49USD; total899.29 within existing900 cap. These are bounds, not posted charges. Credit observation858.59USD is dated October6 22:43UTC.

Physical outcome:all30 TPU cases passed with0 skips/failures/errors in44.519seconds. Full-model initialization/update, heldout sanity and exact resume gates passed. Real input-scored MLM training reached checkpoint3000; last512-row heldout evaluation at2000 is3.869533 vs12.533409 at random initialization. Spot interruption ended this attempt; training is currently stopped. Controller/queue/node are independently absent. Durable receipts are in `evidence/`, and `live-status.json` overrides the worker’s pre-interruption running snapshot. Durable pipeline status and controller receipts: `gs://azettaai-yat-eval-0929/goat-input-1007b/`. Checkpoint prefix is recorded in `run.json`.

AttemptA passed input preflight but failed budget admission before any TPU request because the initial calculation omitted the supervisor’s1800second cleanup reserve. Its controller deleted itself; independent zone lists are empty. AttemptB fixes that calculation and adds admission before input downloads. The8USD ancillary bound covers controller setup, including the short failed attempt. Original failure receipts are retained in `attempt-a/`.

RecoveryC found checkpoint3000 heldout loss nonfinite after fresh30-case TPU attention acceptance. No resumed update was launched. The current recovery blocker and retained rawtrace are in `recovery-c/evidence/`; read-only physical TPU numerical diagnosis is pending. Prior finite step2000 heldout loss does not validate checkpoint3000.
