# October 6 overnight continuation

User authorized a long unattended continuation from the existing checkpoint.
The frozen run resumes step500, preserving optimizer state, sampler and the
original20,000-step cosine schedule (warmup500). The step ceiling is60,000 and
physical TPU lease12hours. This is the existing contrastive representation
objective, not a new MLM run. Batch128; save100; evaluate500; keep3recent plus
separate best; protected starting/best copies are outside rotation.

The explicit migration makes finite development regressions reporting-only.
Nonfinite updates, invalid identity and checkpoint failures still fail closed.
Quality improvement is not guaranteed; final production selection remains gated.

Fixed an actual GCS retention defect: Orbax0.12.4 generic temporary cleanup
classified nested `best/` as incomplete. Readers are now read-only and all
managers disable this recursive sweep. The original best0 was recovered via GCS
softdelete, authenticated, and copied alongside step500 to the new namespace.
Object names, sizes and CRC32C match the originals (83objects).

The cloud controller installs a pinned offline runtime, executes11physicalTPU
checks including migration recovery, then resumes production training. It has
provider-enforced16hour deletion; an independent workflow guards the12hourTPU
lease. It does not depend on the laptop. See status.json for observed progress.

Budget: current credit observation$877.34; incremental conservative reservation
$127, comprising$122TPU/cleanup/ancillary and$5controller. These are ceilings,
not measured charges. Prior reservations retained: aggregate$461.60/cap$462.
