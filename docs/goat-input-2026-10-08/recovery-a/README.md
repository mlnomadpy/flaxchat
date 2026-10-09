# Exact GOAT-input MLM continuation — October 8

User authorized continuation from latest retained checkpoint27750. Frozen B model/trainer/runtime/corpus and original worker root are unchanged. Exact resume restores model, optimizer, schedule and sampler cursor; original random-initialization flag identifies the existing stage, not a fresh initialization.

Controller yat-goat-input-1008a, azettaai/us-west4-a. One Spot v5e8 attempt, 10-hour guarded TPU lease and 9-hour workload, independent workflow cleanup plus14-hour controller DELETE. Conservative new reservation110.80USD, prior786.04 retained, total896.84 below unchanged900cap. Fresh Chrome active credit observation790.91USD; this is not per-run posted billing.

Fresh physical qualification:30-case attention suite, full512-row/30168-mask heldout evaluation of27750 at batch16, source/runtime/arguments identity binding. Known batch8 full-model NaN remains unresolved. First resumed segment27751–28000 must contain finite applied updates; subsequent heldout regression gates compare against the newly qualified27750 checkpoint. Saves250steps, evaluations2000steps, latest3 retained. Status and physical results must be verified from current receipts, not inferred from dispatch.

Durable evidence: gs://azettaai-yat-eval-0929/goat-input-1008a/training-evidence/.
