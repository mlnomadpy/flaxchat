# System audit: continued YAT representation training

Audit date: September 30, 2026, America/Los_Angeles. Goal: continue the trained
YAT embedding model to improve multilingual, semantic and code representations.

## Verdict

The system has useful physical-TPU training, checkpoint integrity, atomic
finite-update rollback, replayable sampling, distributed global batches and
independently guarded cloud execution. The newer embedding-stage path does not
inherit every assurance or capability from the older encoder/decoder trainers.
Resolve its data/loss/identity defects, add representation selection and replay,
and qualify the changed path on physical TPU before a substantial continuation.

Three agents reviewed training, infrastructure, and evaluation/skills in parallel.
The integrated audit has **22 findings: 13 P1 and 9 P2**, with no P0 corruption
established. All 22 are open in GitHub, with evidence and acceptance criteria,
under [continuation tracker #61](https://github.com/mlnomadpy/flaxchat/issues/61).
The six earlier open issues remain distinct; no old closed issue was reopened
merely because a newer trainer needs equivalent controls.

The reviewed base commit is `d64a0d18cac0343a936c13ebea6ce02be5adf879`, including
existing dirty and untracked embedding-stage files. The
[source inventory](audit-2026-09-30/source-inventory.json) hashes 30 cited files;
its inventory digest is
`21278663a68fcf2937638eaddd7ef5fefb8521dba4611106f394f139ffad906e`.
That is a working-tree evidence identity, not a claim that all cited code was
committed or released at the base revision.

## Scope and evidence limits

Reviewed training/data preparation, encoder/decoder shared integration boundaries,
losses, optimizer update safety, sharding, checkpoint/resume, runtime setup,
GCP launch/supervision/cleanup, budget and storage accounting, evaluation receipts
and merges, TPU test selection, PyTorch conversion/publication, package/runtime
reproducibility and operator documentation. Kernel mathematics and optimizer
algorithms were reviewed at their integration boundaries, not re-proven.

This was a source and retained-evidence audit with metadata-only reproductions.
No model forward/backward, CPU model benchmark, physical TPU allocation, CI
workflow dispatch, storage deletion, live cloud inventory or billing query ran.
No new all-scale, multi-host, production-quality or current-resource-cleanliness
claim follows from this audit. Reachable defects are distinguished from missing
capabilities and unmeasured historical incidence.

## Open findings

| ID | Priority | Finding | GitHub |
|---|---|---|---|
| TRAIN-01 | P1 | Known relevant passages can enter mined-negative columns | [#39](https://github.com/mlnomadpy/flaxchat/issues/39) |
| TRAIN-02 | P1 | Fresh-stage parent semantics and provenance not authenticated | [#40](https://github.com/mlnomadpy/flaxchat/issues/40) |
| TRAIN-03 | P1 | Exact-resume identity omits sampler/kernel/runtime dependencies | [#41](https://github.com/mlnomadpy/flaxchat/issues/41) |
| TRAIN-04 | P1 | Prepared token padding/schema/vocabulary admission incomplete | [#42](https://github.com/mlnomadpy/flaxchat/issues/42) |
| TRAIN-05 | P2 | Resume can silently do no work when stop precedes cursor | [#43](https://github.com/mlnomadpy/flaxchat/issues/43) |
| TRAIN-06 | P1 | Cross-source duplicate identities and global dev quarantine missing | [#44](https://github.com/mlnomadpy/flaxchat/issues/44) |
| TRAIN-07 | P2 | Case/whitespace folding conflates meaningful code | [#45](https://github.com/mlnomadpy/flaxchat/issues/45) |
| TRAIN-08 | P2 | Inherited tokenizer truncation can precede declared preparation policy | [#46](https://github.com/mlnomadpy/flaxchat/issues/46) |
| TRAIN-09 | P1 | Training selects by fixed dev loss without representation regression gates | [#47](https://github.com/mlnomadpy/flaxchat/issues/47) |
| TRAIN-10 | P1 | New trainer lacks configurable multilingual replay and exposure accounting | [#48](https://github.com/mlnomadpy/flaxchat/issues/48) |
| TRAIN-11 | P2 | Embedding profiling/scaling and avoidable encoding/input overhead | [#49](https://github.com/mlnomadpy/flaxchat/issues/49) |
| TRAIN-12 | P1 | Dedicated hard-negative physical-TPU semantics/recovery coverage missing | [#50](https://github.com/mlnomadpy/flaxchat/issues/50) |
| INFRA-01 | P1 | Legacy paid launcher relies on laptop cleanup timer | [#51](https://github.com/mlnomadpy/flaxchat/issues/51) |
| INFRA-02 | P1 | Current embedding setups do not consistently lock runtime/source overlays | [#52](https://github.com/mlnomadpy/flaxchat/issues/52) |
| INFRA-03 | P1 | Publication parity not bound to full artifacts; some NaN gates fail open | [#53](https://github.com/mlnomadpy/flaxchat/issues/53) |
| INFRA-04 | P2 | Static device-based cost guesses and per-run accounting insufficient | [#54](https://github.com/mlnomadpy/flaxchat/issues/54) |
| INFRA-05 | P2 | Historical fixed launchers and unbounded telemetry uploads | [#55](https://github.com/mlnomadpy/flaxchat/issues/55) |
| INFRA-06 | P2 | Persistent artifact/soft-delete storage accounting remains manual | [#56](https://github.com/mlnomadpy/flaxchat/issues/56) |
| EVAL-001 | P1 | Completeness can accept empty/nonfinite scores or missing subsets | [#57](https://github.com/mlnomadpy/flaxchat/issues/57) |
| EVAL-002 | P2 | Evaluation identity omits source/runtime/full protocol | [#58](https://github.com/mlnomadpy/flaxchat/issues/58) |
| EVAL-003 | P1 | New evaluator/port tests and port-only CI routing incomplete | [#59](https://github.com/mlnomadpy/flaxchat/issues/59) |
| EVAL-004 | P2 | Physical conversion parity limited to eight 128-token examples | [#60](https://github.com/mlnomadpy/flaxchat/issues/60) |

Detailed source evidence, impact, proposed changes, classification and acceptance
criteria are in the [training findings](audit-2026-09-30/training-findings.json),
[infrastructure findings](audit-2026-09-30/infra-findings.json), and
[evaluation findings](audit-2026-09-30/evaluation-findings.json).
The [issue index](audit-2026-09-30/github-issues.json) maps every finding to GitHub.
TRAIN-12 owns trainer semantics/recovery; EVAL-003 owns evaluator/port test routing
and complements it rather than prescribing duplicate test modules.

## Confirmed metadata reproductions

The evaluation agent extracted `_aggregate` with Python AST and standard-library
helpers, without importing JAX or loading a model. Empty scores, a NaN score and
a missing subset each returned `complete=true`. This proves the acceptance
contract is weak; it does not prove existing published receipts are malformed.

The CI selector given `torch_port/yat_encoder.py` selected no tests and disabled
every execution branch. The Linux workflow path filter also omits `torch_port`.
No workflow was enabled or dispatched to demonstrate this metadata defect.

## What is already working at the code/evidence level

- Physical TPU guards prevent the newer model trainer/evaluator from using CPU.
- Prepared source revisions, tokenizer/weight hashes and per-source dev quarantine
  exist; the missing assurance is semantic and mixture-wide validation.
- Global arrays and expected-global-batch checks preserve intended distributed
  contrastive semantics. No remote-negative stop-gradient was found.
- Deterministic epoch permutations support exact sample replay within a fixed
  stage. The dependency identity must cover their implementation.
- Checkpoint restore verifies stored byte manifests before live mutation; finite
  candidate model/optimizer checks support atomic update rejection.
- The preferred cloud supervisor has unique allocation identities, server-verified
  cleanup leases, bounded remote work, transport reattachment and independent
  queue/node absence checks. The legacy launcher is a separate weaker path.
- Existing cards disclose bitext regression, unmatched paper comparisons, code
  overlap uncertainty and narrow PyTorch parity scope. These disclosures should
  remain visible when representations improve.

## Implementation order and continuation gate

1. **Correct the embedding contract:** #39–#46. Authenticate parent semantics,
   validate tokens, complete identities and masks, and own code/truncation policy.
2. **Make improved representation measurable:** #47–#48 and #57–#59. Add held-out
   development metrics, regression thresholds, best checkpoint retention and
   replayable broader data. Keep final tests isolated from checkpoint selection.
3. **Qualify the actual path:** #50 and #52. Freeze runtimes and run a bounded
   physical-TPU loss/gradient/interruption/recovery suite on the selected topology.
   Use the existing guarded supervisor; #51 blocks use of the legacy launcher.
4. **Measure scaling economics:** #49 and #54–#56. Profile useful work and overhead,
   then choose batch/layout/caching using measured quality and throughput. Extra
   chips alone do not improve representations; accumulation alone does not expand
   the simultaneous InfoNCE negative pool.
5. **Release a selected new stage:** #53 and #60. Authenticate full artifact parity,
   evaluate final pinned benchmarks and preserve immutable model lineage.

Continue from existing trained weights. A new data/objective/schedule stage gets
its own output and explicit optimizer policy; an interruption within that stage
restores exact state. The next decision is whether a bounded continuation with
bitext replay improves declared development measures while retaining retrieval
and code capability. Existing test benchmarks influenced earlier decisions, so
use fresh held-out selection data where needed.

## Tools and documentation updated

The [tool inventory](TRAINING_TOOLS.md) identifies existing JAX/NNX/Optax, Orbax,
GCP guard, prepared-data and MTEB components to integrate. Main additions are
semantic preflight, shared relevance/data identity, quality checkpoint selection,
replay/exposure tracking, strict receipts, profiling and campaign accounting.
XProf and available billing exports are useful supporting tools; a new paid
experiment platform is not required.

The new [training skill](../skills/flaxchat-training/SKILL.md) and
[canonical runbook](REPRESENTATION_TRAINING_RUNBOOK.md) distinguish implemented
tools from proposals, existing authorization from new scope, exact resume from
new stages, physical TPU evidence from metadata checks, and estimates from
posted costs. An independent realistic
[skill forward review](audit-2026-09-30/skill-forward-review.md) passed.
README, documentation index and agent guidance now point here. Old MLM launch
records and session handles remain as explicitly marked historical evidence.

GitHub issue creation succeeded through signed-in Chrome. The connected
integration permitted reads but denied issue creation; local `gh` authentication
was invalid. No token was exposed or credential permission expanded.

## Existing open issues retained

- [#1](https://github.com/mlnomadpy/flaxchat/issues/1): repository-wide umbrella.
- [#11](https://github.com/mlnomadpy/flaxchat/issues/11): release-readiness contract.
- [#12](https://github.com/mlnomadpy/flaxchat/issues/12): generic physical multi-host
  acceptance; new embedding-specific coverage is #50.
- [#19](https://github.com/mlnomadpy/flaxchat/issues/19): stable public artifact/demo.
- [#23](https://github.com/mlnomadpy/flaxchat/issues/23): matched decoder cross-harness
  measurements, not a matched EmbeddingGemma evaluation.
- [#31](https://github.com/mlnomadpy/flaxchat/issues/31): Kaggle prepared input path,
  outside the immediate GCP continuation priority.
