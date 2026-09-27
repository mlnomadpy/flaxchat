# YAT kernel optimization

## Current status — September 25, 2026

The physically validated four-host v5e-16 implementation uses uniform block32
adaptive forward repair, the all-zero-sensitive-cotangent shortcut, factored
softmax backward, query-tiled attention, local data sharding and state donation.
It measured63,313 useful tokens/s over30 warm unprofiled steps and passed exact
checkpoint interruption/resume. This excludes compilation, profiling and
checkpoint overhead. Its256-row held-out MLM loss is17.30570, so it does not
qualify model quality or justify production training.

The profile identifies distance-repair loops as46–64% of the device step.
Individual zero-cotangent pruning, tile compaction and mask-aware sparse repair
are not adopted: their CPU results did not predict TPU performance. Mask-aware
repair also failed a TPU gradient-comparison gate. The always-dense attention
candidate is CPU-tested and awaiting physical qualification. The fused Pallas
attention prototype remains experimental because actual-activation gradients
are not adequately qualified. Q/K normalization and centered FFN experiments
also failed to establish better held-out model quality.

Bias1,epsilon.01 and trainable alpha remain required. BF16 geometry does not
imply BF16 optimizer/master parameters or hardware matrix accumulators. The
general library precision default and explicitly configured YAT training runs
must be distinguished. Exact evidence and later findings appear below; the
September23 sections describe historical experiments, not the current result.

## Initial adaptive repair experiment — September 23

The native adaptive BF16 implementation reuses the projection dot product:

`distance = ||x||² + ||w||² - 2 * dot(x, w)`

It computes direct squared differences only for cancellation-sensitive pairs.
This avoids the feature-wise broadcast over every token/prototype pair. The
relative-distance threshold is a heuristic, not a proven error bound.

The latest change keeps all tile slicing and updates inside the repair branch.
Previously, even an inactive tile sliced and updated the output matrix, adding
work to reverse-mode differentiation. Sparse repairs now also reduce 32 features
per iteration instead of 8. This changes BF16 summation order, so numerical and
gradient checks accompany the optimization. The direct reference mode and dense
attention fallback retain their original reduction order.

The formula remains `alpha * (dot + 1)² / (distance + 0.01)`, multiplied by the
linear gate in the FFN. Bias 1 and epsilon 0.01 are fixed; alpha remains trainable.
BF16 rounds epsilon to its representable value. There is no NMN runtime dependency.
Distance arrays and reductions remain BF16; this does not assert that hardware
matrix multipliers accumulate internally in BF16.

## Measured result

CPU, JAX 0.11.1; 21 synchronized repetitions after compilation and three warmups.
The benchmark includes forward, backward, and alpha gradients. FFN input is
128×768, with 1,152 intermediate features; attention is B1/L128/H4/D64.

| Case | Previous adaptive BF16 | Updated adaptive BF16 |
| --- | ---: | ---: |
| FFN, one near/equal prototype | 35.23 ms | 24.04 ms |
| Attention, independent Q/K | 1.86 ms | 1.85 ms |
| Attention, equal Q/K | 6.30 ms | 6.28 ms |

The repaired FFN case improved 1.47×. No attention speedup is established.
The random FFN path measured 4.43→3.55 ms, but its executed arithmetic was not
changed; do not attribute that difference to this optimization without repeated
hardware profiling. CPU results do not establish TPU speed or training quality.
Mixed precision remains the default and remains faster in these CPU measurements.

Evidence: `artifacts/yat-tile-optimization-0923/baseline-final.json` and `final.json`.
The baseline source and comparison runner are preserved alongside those reports.
Independent-output tile mapping and compacted tile indexing were explored but
were not retained because they did not improve the measured tradeoff.

## Reproduce and next steps

Run `python -m scripts.benchmark_yat --output report.json --repeats 21` on the
target machine. The report records device/backend, JAX version, source hashes,
compile time, synchronized timings, temporary memory, finite outputs/gradients,
forward drift, and presence of FP32 in forward StableHLO. It provisions nothing.
Benchmark methodology follows [JAX benchmarking guidance](https://docs.jax.dev/en/latest/benchmarking.html).

For a larger TPU improvement, profile a fused tiled implementation that keeps
the projection dot, squared norms, denominator, and gate within the same kernel,
with a dedicated backward pass. That is future work, not a measured speedup.
Fused attention would also need online softmax and mask-correct backward handling.
Keep near-collision tests: blindly using the BF16 norm identity can round a real
distance of 0.25 to zero. Do not replace the squared input norm with 1 merely
because LayerNorm or RMSNorm precedes the projection; neither generally produces
unit L2 vectors, and learned affine parameters further change their lengths.

## Dedicated BF16 distance backward

The direct-distance primitive now uses a custom reverse-mode derivative. For an
upstream distance gradient G, each coordinate block computes
`weighted[i,j,k] = G[i,j] * 2 * (x[i,k] - y[j,k])`, then reduces over j for dx
and over i (with a minus sign) for dy. It retains only x and y across the
forward/backward boundary, avoiding transposition of the forward accumulation
loop. Broadcasted batch gradients are reduced back to the operand shapes.

Subtract before multiplying: replacing this derivative with
`2 * (x * sum(G) - G @ y)` is algebraically equivalent over real numbers, but
reintroduces cancellation for nearby BF16 vectors. All distance forward and
backward arrays/reductions remain BF16. Hardware-internal accumulation precision
is a separate property.

The custom backward applies to direct mode and the dense attention repair path.
Sparse FFN repairs keep their prior autodiff implementation because the custom
rule did not improve that case. The reduction block remains 8 for direct mode;
a block of 32 increased memory significantly and was rejected. Forward values
and their summation order are unchanged. Bias=1, epsilon=0.01 and trainable alpha
are unchanged. Mixed precision remains the model default.

Like other [JAX custom VJPs](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html),
the optimized primitive supports reverse mode, but not a direct forward-mode JVP.
Call `squared_distance_bf16(..., custom_backward=False)` for the autodiff reference
when forward-mode differentiation is needed. Full BF16 encoder JVPs are not
qualified by this optimization.

Reproduction and source snapshots: `artifacts/yat-vjp-0923/compare.py`,
`baseline_yat.py`, `baseline-final.json`, and `final.json`. Measurements include
forward, backward and alpha gradients, synchronized after compilation and three
warmups. They measure isolated CPU operators, not TPU or end-to-end throughput.

Retained candidate, 21 repetitions on CPU with JAX 0.11.1:

| Case | Previous | Custom backward | Temporary memory, previous → updated |
| --- | ---: | ---: | ---: |
| Direct BF16 FFN, random | 146.29 ms | 112.51 ms | 24.58 → 19.56 MB |
| Direct BF16 FFN, close pair | 145.55 ms | 117.17 ms | 24.58 → 19.56 MB |
| Adaptive BF16 attention, equal Q/K | 6.24 ms | 5.81 ms | 7.74 → 4.72 MB |
| Adaptive BF16 FFN, close pair | 24.09 ms | 24.31 ms | unchanged |

Direct FFN improves 1.24–1.30×; attention temporary memory falls 39% for adaptive
mode (47% for direct mode). Sparse adaptive FFN has no established speedup.
Mixed FFN remains much faster, around 2 ms on this CPU. These timings do not
establish TPU performance; the already frozen TPU validation payload predates
this custom backward.

## Adaptive FFN backward: sparse and dense repairs

The wide-input (`features > 128`) adaptive path now has its own custom VJP.
It preserves the forward calculation and its 1/8 sensitivity heuristic. For
unrepaired pairs, it differentiates the squared norms and supplies `-2 * G` to
the existing projection dot product. For repaired pairs, it computes
`2 * (x - w) * G` directly in BF16, then accumulates input/prototype gradients.
This avoids differentiating repeated full distance-matrix updates. The dot
cotangent remains separate so the caller's existing combined projection/gate
GEMM receives the correct gradient. Alpha remains trainable.

Sparse repairs retain 16×64 tiles. When at least one quarter of tiles need repair,
the backward uses direct coordinate differences over strips of at most 128 rows;
its coordinate scratch does not grow with the full token count. Attention with
head dimensions up to 128 keeps the previous implementation. The model's mixed
precision default is unchanged. `custom_backward=False` on the adaptive distance
primitive selects the original autodiff implementation; wide adaptive forward-mode
JVPs require this reference option.

Gradient summation order can differ in BF16. Tests cover broadcast batches,
independent dot cotangents, odd dimensions, sparse and dense collisions, zero
vectors, and dense strip boundaries. A 131-row case exposed about 2.6% difference
from the old sequential BF16 gradient sum; the new result is checked against
analytic FP32 derivatives instead of treating the old sum as ground truth.
Forward values are unchanged. StableHLO checks reject FP32 operations in the
BF16 distance backward.

The benchmark now accepts `--collision-stress`, adding scattered and dense FFN
collisions to the existing random, single-collision, and attention cases.
Smaller repair tiles and a globally larger repair tile were explored and rejected:
smaller tiles increased dispatch overhead, while larger tiles regressed scattered
collisions. Intermediate reports are exploratory, not release measurements.

Final isolated CPU comparison, JAX 0.11.1, 21 synchronized repetitions after
three warmups; FFN 128×768 with 1,152 prototypes, forward plus all gradients:

| Pattern | Previous adaptive | Updated adaptive | Speedup |
| --- | ---: | ---: | ---: |
| Random | 3.92 ms | 3.67 ms | No robust fast-path gain claimed |
| One collision | 24.09 ms | 9.54 ms | 2.52× |
| Six scattered collisions | 44.35 ms | 37.13 ms | 1.19× |
| Dense collisions | 295.13 ms | 198.95 ms | 1.48× |

Forward values and x/weight/alpha gradients were identical on these four fixtures.
This does not imply bitwise identity for arbitrary inputs; the independent
numerical tests above cover changed reduction order. Compiler-reported temporary
memory increased from 16.13 MB to 19.42 MB (~20%, +3.29 MB). Mixed precision remains
faster on this CPU; this optimization improves BF16 repair cost, not evidence
that BF16 is faster overall. No new TPU allocation was made for these measurements.

Final evidence and reproducible snapshots:
`artifacts/yat-tiles-0923/final_patterns.py`, `baseline_yat.py`, `final_source.py`,
`final-patterns.json`, and `final-metadata.json`. The metadata records source
hashes and verifies the candidate snapshot matches the production source.
Validation: 31 YAT tests pass, plus targeted Ruff and Pyright checks. TPU timing,
end-to-end throughput, and training-quality parity remain unvalidated.

Final repository verification: 1,107 CPU tests passed, 21 skipped, 5 deselected;
81.46% coverage with all module floors met. All 31 focused YAT tests passed again
after adding explicit array return types to the custom-VJP wrappers. Repository
Pyright and Ruff pass. These annotations/casts do not change arithmetic; the
benchmark snapshots above retain their original hashes. Verification evidence:
`artifacts/encoder-regression-current-0923/verification.json` and adjacent logs.

## Compact selected-pair repairs

Wide adaptive FFN repairs now gather up to 32 selected pairs per tile. The old
path computed all 1,024 distances in a 16×64 tile even if only one pair needed
repair. The compact path subtracts only gathered coordinates and scatters the
result and gradients back. Forward accumulation retains the block-32 BF16 order.
Overflow selects the complete tile calculation; no selected pairs are dropped.
Narrow attention and the no-collision arithmetic are unchanged.

The static capacity follows [JAX nonzero requirements](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.nonzero.html):
`size` is fixed under JIT, and callers check the count before selecting the compact
branch. Padding is masked in backward and dropped in forward. Fully populated
small tiles retain the dense backward reduction: an intermediate dispatch rule
failed the existing 131-row gradient-accuracy test and was corrected rather than
loosening its tolerance. Pair-scatter accumulation can change BF16 rounding;
bitwise equivalence is not guaranteed for arbitrary inputs or hardware.

Bias 1, epsilon 0.01, trainable alpha, and BF16 distance array arithmetic are
preserved. Mixed precision remains the default. No cloud resources were allocated.
Validation: 35 focused YAT tests pass, including capacity boundaries, overflow,
repeated indices, odd widths, broadcast gradients, zero vectors, and no-FP32
StableHLO checks. Targeted Ruff and Pyright pass.

Reproduction: `PYTHONPATH=. .venv/bin/python artifacts/yat-pair-repair-0923/final_compare.py`.
The directory preserves both sources and synchronized forward-plus-backward
measurements, including alpha gradients. CPU timings cannot establish TPU
speedup, end-to-end training throughput, or quality parity.

Final CPU measurements, 21 repetitions after three warmups, FFN 128×768 with
1,152 prototypes (forward plus gradients):

| Pattern | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| random | 3.67 ms | 3.65 ms | 1.01× |
| one | 9.63 ms | 4.37 ms | 2.21× |
| scattered | 37.17 ms | 5.85 ms | 6.35× |
| dense | 199.12 ms | 26.89 ms | 7.40× |
| near_scattered | 37.37 ms | 6.10 ms | 6.13× |
| near_repeated | 199.67 ms | 27.01 ms | 7.39× |

No robust speed change is claimed for random inputs. Compiler-reported temporary
memory remains approximately 19.42 MB (448 bytes more). Forward values are
identical on all six fixtures. All gradients are identical except input gradients
in `near_repeated`, which differ by 0.4061% relative L2 due to BF16 accumulation
order; weight and alpha gradients match on that fixture. That is a measured
rounding difference, not proof of training-quality equivalence. Fully dense pair
sets continue to use fallback; the repeated-prototype fixture has many collisions
but sparse selected pairs within each tile.

Evidence: `artifacts/yat-pair-repair-0923/final.json` and `metadata.json`.

### Opt-in max-centered BF16 softmax backward

The core APIs now expose the probability-selected centering method:

```python
softmax_bf16(logits, backward_mode="max_centered")
windowed_yat_attention(
    q, k, v, segments, radius=64,
    softmax_backward_mode="max_centered",
)
```

The default remains `factored`. Max-centering changes only the reverse-mode rule;
forward probabilities and outputs are identical. It subtracts the cotangent at
the largest-probability key, then a probability-weighted centered mean, accounting
for the rounded BF16 probability mass. All this arithmetic stays BF16. The
automatic reference (`custom_backward=False`) retains forward-mode AD; combining
it with `max_centered` is rejected rather than silently ignoring the option.

The method reduced the worst Q/K gradient errors in a 32-fixture physical v5e
sweep from 7.65%/6.64% to 2.22%/2.27%, at about 6–11% additional isolated
attention value-and-gradient time. These are bounded synthetic results. A CPU
probe of real student activations exposed separate forward-score rounding
sensitivity that this backward rule does not resolve. It is available through encoder `yat_softmax_backward` and the training flag
`--yat-softmax-backward max_centered`. The selection is recorded in checkpoint
identity, so exact resume cannot silently switch backward methods. Full-model
physical and quality qualification remain outstanding. The measured frozen prototype and traces are preserved in
`artifacts/yat-maxcenter-v5e1-0925`.

Regression tests cover unchanged forward values, constant-cotangent null
directions, masking, offset stress, the small-scale near-collision case,
argument validation, and retained reference forward-mode differentiation.

### Exact-zero backward repair pruning (September25)

The adaptive distance VJP now intersects the sensitive-pair mask with `cotangent != 0` before selecting repair tiles and compacting pairs. This removes direct-difference work that contributes exactly zero derivative, including masked/saturated attention paths. It uses no tolerance, changes no forward arithmetic, and retains BF16 geometry, bias1, epsilon.01 and trainable alpha. Small nonzero cotangents remain eligible for repair.

The existing BF16/oracle/windowed suite passes58 tests. Three added fixtures check zero,2^-16 and.25 cotangents on a sensitive pair in partial edge tiles against exact expected gradients. Eight frozen-before/after near-collision fixtures cover0,1,31,32,33,65,300 and1105 nonzero cotangents; forward results match and observed gradient errors against FP64 are unchanged (maximum0.8354%). Ruff and Pyright pass. Source and reports: `artifacts/yat-repair-pruning-0925`.

A CPU initialized-model probe across22 layers uses one128-token row and all12 heads. Sensitive pairs are sparse in this sample: one in layer7 and12 in layer18. The saved layer7 attention probe gives exactly zero distance cotangent for its sensitive pair, so the new backward avoids that repair. The previously saved distilled sample (two heads) has no sensitive pairs and establishes no speed benefit. These tiny samples do not represent the four-host workload distribution. No new TPU was allocated, and physical TPU performance/recovery validation of this source change remains outstanding. Further optimization should compact active work across tiles while preserving complete overflow handling and cancellation-safe gradients.

### Sparse tile-compaction prototype

Two artifact-only backward implementations were compared against the zero-cotangent-pruned baseline. Global pair compaction scans the full pair array and was slower on CPU (one-pair full-shape case approximately5.64ms vs3.29ms), so it was not integrated. Compacting the smaller affected-tile array instead reduces visits to at most32 selected tiles, with an unchanged full traversal when capacity is exceeded. The compact loop uses a static bound and preserves ascending tile order. All geometry remains BF16.

At shape48×64×512×64, the second CPU sweep measured3.20→2.41ms with one nonzero pair,5.46→4.78ms with31,5.55→4.82ms with32, and12.97→12.74ms with129. These are isolated CPU backward timings with five repetitions, not TPU speedups or model throughput. Twelve numerical fixtures across two shapes stay within0.28% relative gradient error against the FP64 direct-difference oracle. Six additional exact affected-tile boundary fixtures cover0/1/31/32/33/64 tiles, including partial edges and capacity overflow; candidate gradients match baseline exactly and lowered distance code contains no FP32 arithmetic.

Neither prototype is wired into production. Source hashes, timing data and tests are preserved under `artifacts/yat-global-pairs-0925`; physical TPU execution and end-to-end impact are outstanding. No new cloud resource was allocated for this experiment.

### Physical TPU results revise the sparse-repair decision

The one-chip v5e benchmark completed and all six capacity/overflow boundary tests pass physically. At48×64×512×64 with32 nonzero cotangents, original/individually-pruned/tile-compacted backward medians are0.741/2.366/0.908ms. With129 they are0.779/4.011/4.012ms. JAX device module traces confirm the32-pair comparison at0.540/2.184/0.730ms; these device timings exclude host overhead and are not end-to-end model speeds. With one nonzero pair tile compaction improves0.751→0.312ms, but it is not a general replacement.

Pruning individual cotangents changes dispatch from efficient dense strips to slower sparse traversal. Consequently the production code now retains the original geometry-based dispatch and skips only when *all* sensitive pairs have exactly zero cotangent. The tile prototype is not integrated. The benchmark's all-zero pruned path took0.232ms versus0.702ms original, but the precise revised source still needs physical qualification.

A separate audit found that forward dense repair used coordinate blocks8 whereas sparse repair used32. Changing unrelated pairs could switch dispatch and alter an unchanged pair's distance. Both now use32. An adversarial16-seed probe improves dispatch invariance from5/16 to16/16; three regression tests cover widths33/64/128. This changes BF16 results: in this small probe maximum error relative to FP64 rises from0.665% to1.280%, so it establishes consistency, not universal accuracy improvement. All68 related numerical, attention and training/resume tests pass after both edits; Ruff/Pyright checks pass. The new forward reduction was not part of the frozen TPU benchmark.

Evidence: `artifacts/yat-tilecomp-v5e1-0925` and `artifacts/yat-distance-reduction-0925`. The admitted envelope is$886.56 of$900, not actual spend. Cleanup verification is pending at this writing.

### Packed-attention regression and cleanup confirmation

The dispatch inconsistency also reproduced in full windowed attention, not just the distance primitive. With256 tokens, a target document of8 tokens and near-collision Q/K, perturbing only the other document changed the target output by up to0.03125 before the fix. Both local(radius64) and global attention reproduced it; Q/K/V and trainable-alpha gradients also changed. With uniform forward reduction, the target output and all four gradients are exactly invariant in the same probe. Four permanent tests now cover both radii and both softmax backward modes. The full windowed suite passes16 tests. Evidence: `artifacts/yat-distance-reduction-0925/packed-isolation.json`.

The one-chip benchmark controller exited0, and both its TPU VM and queued resource were independently confirmed absent. Receipt: `artifacts/yat-tilecomp-v5e1-0925/cleanup-verification.json`. Storage evidence is retained.

The separate forced-four-device CPU suite also passes18 local-sharding and training/recovery cases with the current source. This does not replace physical multi-host qualification.

### Q/K normalization diagnostic (architecture experiment)

[Query-Key Normalization for Transformers](https://aclanthology.org/2020.findings-emnlp.379/) motivates explicit head-dimension L2 normalization and learned scaling; it does not validate YAT or this experiment. In exact arithmetic, applying the fixed-bias1/epsilon.01 kernel to unit vectors gives `(1+c)^2/(2-2c+.01)`, where c is their cosine similarity. This is monotone on[-1,1] for positive alpha, unlike unrestricted squared alignment. The diagnostic computes actual BF16 normalized-vector norms rather than substituting1 after rounding.

On one saved initial-layer0 example (128 tokens,12 heads), alpha-grid fitting uses the first64 queries and measures the remaining64, which share the same document and keys. Raw scalar-alpha YAT yields probability KL0.8454 and attention-output relative L2 error0.8222 against a FP64 dot-attention reference. Normalization at norm scale.25 plus separate alpha per head lowers KL to0.3994; norm scale.5 gives the lowest observed output error0.6659. Both remain substantial mismatches. This suggests a calibration experiment, not a model-quality result. It changes architecture, is not integrated, and has no TPU performance evidence. Files: `artifacts/yat-qknorm-0925/probe.py` and `attention-fit.json`.

### Uniform-reduction four-host calibration (September 25)

The frozen uniform-reduction/all-zero-gradient-shortcut implementation passes all ten numerical regressions on each of four physical v5e workers (16 chips total). Its 32-step calibration measures 63,312.78 useful tokens/s over 30 warm, unprofiled steps, with median step 0.7872 s. This excludes separate checkpoint saves, compilation, and profiler export; it is not end-to-end job throughput. The first step after profiler export takes 0.8076 s, consistent with the export synchronization removing the prior post-profile stall. Eleven local trace/summary regression tests pass. Recovery and held-out quality evaluation remain pending at this observation. Evidence: `artifacts/yat-uniform-v5e16-0925/throughput.json` and `physical-numerical-regressions.json`.

### Q/K normalization fails the separate-document MLM diagnostic

The artifact-only follow-up fits per-head alpha grids on32 training documents and evaluates only a layer0 attention replacement on32 different validation documents,128 positions and631 masked targets. Original mmBERT scores1.62449 MLM loss; raw YAT alpha1 scores2.31843, raw fitted scalar2.35515, raw fitted per-head2.48011. Q/K normalization with scales1/.5/.25 scores2.69159/2.69564/2.73513. Thus lower fitted attention KL does not establish improved MLM behavior. No normalization or per-head-alpha architecture change is adopted. This small CPU diagnostic does not establish multilingual model quality or TPU performance. Bias1,epsilon.01 and native BF16 geometry are preserved. Evidence: `artifacts/yat-qknorm-0925/heldout_probe.py` and `heldout-mlm.json`.

### Completed uniform-reduction TPU validation and reproducible profile groups

The four-host v5e-16 run completed calibration, deliberate interruption after checkpoint16, resume through32, and256-row held-out evaluation. Independently downloaded uninterrupted/resumed manifests are byte-identical; model,optimizer,and training-state hashes match. Held-out MLM loss remains17.30570 over14,904 targets, so this is infrastructure/numerical validation, not production quality. The controller exited0 and independent VM/queued-resource listings are empty. See `artifacts/yat-uniform-v5e16-0925/recovery-verification.json` and `cleanup-verification.json`.

All16 device traces show training modules of783.14–784.89ms. Shape-selected direct-distance repair loop interval unions are362.75–498.73ms (46.3–63.7% of the corresponding step). These are inclusive loop intervals, not exclusive arithmetic time. Selected collective interval unions are83.08–221.68ms and may include waiting. The trace continues to prioritize adaptive repair cost and imbalance; it does not establish that replacing collectives or increasing batch size is sufficient. The fused YAT prototype still requires actual-activation gradient qualification before integration.

The harness trace summarizer now accepts repeatable `--op-group NAME=REGEX` selectors and stores each selector with its interval-union results. Custom categories may overlap and must not be added together. Empty, invalid, reserved, and duplicate group definitions fail early. Example for this specific 48-head/batch,64-query-tile trace:

```sh
.venv/bin/python -m scripts.summarize_jax_trace TRACE.xplane.pb --output summary.json \
  --op-group 'attention_repair=%while[^=]*= \(s32[^,]*, bf16\[48,64,' \
  --op-group 'vocabulary_all_reduce=bf16\[256000,768\].*all-reduce\('
```

Selectors based on shapes must be rechecked when the model/batch/tile changes. The profiling and summary suites pass18 tests; Ruff and Pyright pass for the changed summarizer.

### Mask-aware repair prototype (September25)

The current production primitive repairs near-collision pairs before attention applies its eligibility mask. A separate prototype accepts an eligibility mask and removes discarded pairs from both forward repair and the custom distance VJP. It still computes actual BF16 norms and direct differences; fixed bias1,epsilon.01 and trainable attention alpha are unchanged. It does not threshold small gradients. The prototype is not integrated into production.

Twelve CPU checks pass: all/none/sparse masks, local/global attention with both softmax backward modes, packed-document adversarial isolation, and a lowered geometry value-and-gradient check containing no FP32 arithmetic. Synthetic512-token,12-head global-attention timings use alternating execution order, three warmups and12 measurements per variant with other test processes finished. For document lengths16 and64, baseline/candidate medians are232.42/176.13ms and235.32/174.92ms (1.320× and1.345×). A single512-token document has no speed benefit (0.984×). Outputs match exactly; Q/K/V/alpha gradients match except key gradients in the64-token-document fixture, with0.183% relativeL2 difference due to changed repair/reduction dispatch. This is not an accuracy improvement claim. Earlier unstructured-mask and256-token tests showed little benefit, emphasizing the dependence on eligible tile structure.

No TPU speedup or model-quality claim follows from these CPU fixtures. Before integration, run a bounded physical comparison covering short packed documents, full documents, local/global windows, captured activations and all four gradients. Source, tests, controlled samples and hashes: `artifacts/yat-mask-aware-0925/manifest.json` and `attention-benchmark-512-alternating.json`.

### Fused backward on captured activations: row-statistic attribution

The fused custom-VJP prototype was checked in CPU Pallas interpretation against the independent NumPy FP64 oracle on saved post-distillation Q/K/V from depths0,7,14,21 (one128-token row,two heads,random cotangent), with global and radius64 attention and query tiles32/128. It is not qualified: forward agreement alone hides substantial gradient error. For example, depth21 fused128 alpha relative error is45.4% global and53.7% local, versus6.1%/8.4% for the current reference.

An artifact-only diagnostic recomputes the BF16 softmax row statistic as sum(p*gp), using the same saved maximum and normalizer, instead of dot(output,go). It adds a complete query-tiled XLA pass; it is not a speed optimization. At depth21 this reduces alpha error to6.5%/8.9% and moves Q/K errors closer to the reference. Depth0 local alpha error falls54.0%→21.4%. However depth14 global alpha error worsens25.1%→37.3%, so the substitution is not uniformly more accurate and is not adopted. Depth0 global Q/K errors remain dominated by score quantization. Any proposed fused implementation must qualify all derivatives, longer multi-key-tile cases, cancellation-sensitive alpha, and physical TPU execution before integration.

Evidence: `artifacts/yat-streaming-0925/real-fused-probe.json`, `real-fused-row-delta-probe.json`, and `real-fused-delta-comparison.json`. The full objective remains unresolved; neither these diagnostics nor passing infrastructure tests establish model quality.

### Physical TPU rejects mask-aware repair

The bounded v5e-1 run passed12 numerical regressions, then rejected the candidate in the broader benchmark. For512-token global attention with16/64-token packed documents, candidate wall time regressed about7.79×. The captured64-token-document profile confirms device module time1.92→17.25ms; nested conditional/loop intervals identify costly sparse traversal. Fewer repaired pairs did not translate to faster TPU execution. Full-document global and local cases had roughly unchanged timing.

The local radius64,512-token single-document fixture had identical forward outputs but Q/K/V gradient differences19.20%/19.13%/4.43% and alpha153.74% relative to the reference. These are differences from the reference, not FP64 error. The benchmark correctly stopped at its3% gate after six synthetic cases; none of the16 captured-activation TPU cases ran. The cause of the backward discrepancy is not established. The prototype is rejected and production source is unchanged. Controller exited1 for validation failure; VM and queued resource are independently absent.

The failed benchmark retained seeds, metrics and profiles, but not its actual device-generated failing arrays. CPU and TPU seed-based generation is insufficient evidence of identical inputs. Added `scripts.numerical_evidence` to preserve device values, original/stored dtypes and NPZ checksums; BF16 values are stored exactly in FP32. The regular YAT benchmark now invokes it before raising on nonfinite output/gradient. A future version of this comparison also preserves the actual Q/K/V,segments,alpha,cotangent and both outputs/gradient sets on mismatch; it is saved separately as `benchmark_with_evidence.py`, without altering the already-run source. Twenty-one evidence/CI tests pass; Ruff/Pyright pass. Exact arrays from this failed TPU case cannot be recovered from the seed alone.

Evidence: `artifacts/yat-maskaware-v5e1-0925/results`, `profile-summary.json`, `profiles-tree`, and `cleanup-verification.json`.

A subsequent CPU-only fused-backward experiment centers the kernel term in the alpha-gradient sum using a global row-constant anchor. Although equivalent in real arithmetic, it does not consistently improve the captured activation cases: depth21 alpha errors worsen from6.5%/8.9% with row-statistic recomputation to13.2%/15.8%. It is not adopted. See `artifacts/yat-streaming-0925/real-fused-alpha-center-probe.json`.

### Dense attention repair candidate

The TPU trace rejects the assumption that fewer repaired pairs necessarily means faster execution. A separate native candidate uses dense repair whenever attention geometry(width≤128) has any sensitive pair, retaining the existing all-zero-cotangent shortcut, direct-difference gradients, uniform block32 forward arithmetic, and bounded128-row backward strips. Wide FFN geometry is unchanged. This eliminates the attention sparse traversal responsible for the measured regression; it does not change the YAT formula.

Twenty-nine CPU regression cases pass, including baseline comparisons, independent gradient checks and packed-document isolation. The22-case synthetic/captured-activation operator matrix has identical forward outputs and Q/K/V/alpha gradients. Its compiled CPU temporary-memory estimates do not increase. Offline TPU lowering passes batch4,12-head,width64 attention at lengths512/2048 with global and radius64 windows. None of these checks establishes physical TPU speed, TPU memory, or model quality. The candidate remains outside production pending hardware comparison; subsequent failure gates will preserve exact device arrays. Evidence: `artifacts/yat-dense-attention-0925/manifest.json`, `cpu/benchmark.json`, and `offline-lowering.json`.

### Sparse geometry coverage and a bounded independent oracle

The dense-attention candidate now has12 additional controlled sparse-geometry cases: zero,1,31,32,33,129 selected pairs at shapes2×17×65×33 and48×64×512×64. Gradients are restricted to the selected pairs and checked against direct NumPy FP64 coordinate differences, so unrelated fast-path gradients cannot hide repair errors. Outputs match baseline; worst relative gradient error against the independent oracle is0.306%. Exact inputs are archived with checksums for cross-platform replay. Dense repair is substantially slower in these CPU cases; it remains a TPU-specific hypothesis, not a general default. Evidence: `artifacts/yat-dense-attention-0925/sparse-cpu`.

Added `attention_and_gradients_tiled` to the independent test oracle. It retains the full eligible-key softmax per query while processing heads and query tiles separately, bounding coordinate scratch by query_tile×length×width. It matches the full oracle in36 cases covering padding,packed documents,negative/zero/positive alpha and local/global windows; a separate finite-difference check covers Q/K/V/alpha. A512-token,12-head,width64 case executes in0.336s with87.9MB process peak RSS on this Mac. This enables full saved-failure analysis without the former full batch/head/sequence coordinate tensor. These FP64 operations are test references only; the production BF16 requirement is unchanged.

### Profile overlap and repair imbalance

The trace summarizer now records pairwise category interval intersections without
double-counting nested events. Across all16 devices in the captured uniform run,
selected attention-repair loops and named collectives have zero temporal overlap.
Repair durations range362.75–498.73ms; collectives89.995–228.605ms. Their sum is
nearly constant at588.72–591.98ms, with cross-device correlation−0.999986. This
single-step observation is consistent with devices doing less repair waiting
longer at collectives. It does not prove causal attribution or bandwidth
saturation. Reducing uneven repair work therefore remains the first hypothesis
to test before changing communication strategy. Evidence:
`artifacts/yat-uniform-v5e16-0925/profile-overlap.json` and
`profile-overlap-findings.json`. Profiling suites pass22 tests; Ruff/Pyright pass.

The first dense-candidate cloud attempt failed before setup: a copied launch
prefix reused a resource identity still targeted by an earlier cleanup timer.
The failed attempt's queue and VM are independently absent. The supervisor now
adds a fresh random identifier to every run and saves the actual identity in
each attempt directory. Suspended or suspending queues fail readiness promptly.
Operations/guard suites pass39 tests. The unchanged candidate is being replayed
under `artifacts/yat-dense-retry-v5e1-0925`, with a separate$3 reservation and
the original frozen archive. Production qualification remains false.

The replay subsequently passed29 physical numerical tests,12 exact-input sparse
geometry cases, and all22 synthetic/captured attention cases. Every attention
output and Q/K/V/alpha gradient matched the reference exactly. Synthetic512-token
attention improved1.107–1.150×; captured128-token cases ranged0.973–1.652×.
Compiled temporary-memory estimates were50.6–56.6% of the reference. These are
compiler estimates for isolated operators, not measured whole-model peak HBM.
The packed64-token-document JAX trace confirms device module time1.921698→
1.642035ms. Controlled sparse geometry improved1.014–12.433× with worst
FP64-oracle gradient relative error0.2294%. Evidence:
`artifacts/yat-dense-retry-v5e1-0925/findings.json`, `profile-summary.json`, and
`results`. This validates the dense repair arithmetic on one v5e chip, not model
throughput, multi-host scaling, other TPU generations, or production quality.

The native primitive now chooses this dense path for width≤128 on TPU using
`jax.lax.platform_dependent`, which selects the target platform during lowering
rather than inspecting the host's default backend. CPU/GPU and wide-FFN
heuristics remain unchanged. See the
[JAX platform-dependent lowering API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.platform_dependent.html).
The integrated source passes85 CPU numerical/windowed/dispatch tests and offline
TPU export for512/2048-token local/global forward and backward. The dispatch
wrapper itself still needs physical integration and full-model measurement;
the hardware results above belong to the frozen candidate. Both cloud attempts
are independently verified deleted, and the successful controller exited0.

An additional18 simulated four-device CPU tests force the dense branch and pass
local-shard numerical comparisons, packing/padding isolation, full encoder
training, and exact checkpoint resume. These validate SPMD integration on CPU;
they do not qualify physical multi-host TPU scaling. The current cumulative
conservative planning exposure is$864.87 of the user's$900 cap, including both
new$3 reservations. This is not actual billed spend or remaining credit.

### Three-prototype FFN initialization diagnostic

A CPU experiment expands each original FFN channel into prototypes `+beta*w`,
`-beta*w`, and zero, with the original linear gate repeated three times. Fitted
coefficients are absorbed into the output projection, so every feature still
uses the native YAT formula with bias 1 and epsilon 0.01. Alpha is initialized
to 1 and can remain trainable. Expanding all FFNs would add about 116.8M
parameters, placing this variant near 424M; it is not adopted by the harness.

On layer 0, coefficients fitted to 32 training documents give validation branch
relative squared error 0.0232 on 16 separate documents, compared with 0.7822 for
a scalar-calibrated original-width YAT replacement on the same inputs. That
local result does not survive progressive replacement of all 22 FFNs. A new
32-document validation sample, excluded from the earlier beta probe, gives MLM
loss 8.0555 versus the original teacher's 1.4869 over 323 masked targets. Attention
remains original mmBERT in this diagnostic. The result rejects this initializer
as production-ready despite its good first-layer fit. Evidence:
`artifacts/yat-basis-initialization-0925/{controls,report,progressive-report,progressive-mlm}.json`.

### Matched model comparison admission

The frozen paired workload runs baseline calibration followed by dense-repair
calibration, checkpoint interruption/resume and evaluation on the same TPU
allocation. Model sources differ only in `flaxchat/yat.py`. Its first launch used
a prefix outside the existing cleanup IAM condition; the workflow failed with
403 before provisioning, and both resource identities were independently absent.
Guard failures now retain their cloud error and an explicit pre-provisioning
ledger event; operations/guard tests pass 39 cases. Permissions were unchanged.

The corrected-prefix request exceeded its five-minute readiness allowance while
VMs were creating and entered controller cleanup. It did not start the workload.
Do not interpret that timeout as proof that TPU capacity is unavailable. A
future attempt needs a longer readiness allowance within an independently
bounded lease. A separate immutable reconciliation retains a $2 ancillary
allowance for the failed guard and removes only its unused compute reservation;
all other reservations remain intact. Its conservative exposure is $878.87,
not actual billed spend: `artifacts/yat-budget-guard-reconcile-0925/reconciliation.json`.

The longer replay uses a 15-minute readiness allowance, a 35-minute independent
cleanup lease and a $13 reservation. Its conservative cumulative exposure is
$891.87. The allocation reached READY and setup passed on all four hosts.
Track `artifacts/yat-dense-model-long-v5e16-0925`; the immutable workload and
result prefix remain under `yat-dense-model-v5e16-0925`.

Added `scripts.compare_yat_training` to reject mismatched configuration,
initialization, data/tokenizer identity, topology, per-step token exposure or
profiling windows. It reports useful-token throughput, loss trajectories, and
estimated warm compute cost at an explicit whole-slice hourly rate. It excludes
compilation, profiling and checkpoint time and does not establish actual billed
spend, same physical allocation, or production quality. Launcher receipts and
source manifests remain required. Comparison/CI suites pass 25 tests; Ruff and
Pyright pass.

The three-prototype initializer was also instantiated in the actual encoder:
424,571,414 parameters. Native MLM loss is 8.0393 versus teacher 1.4854 on the
same 323 targets. A training-only ridge correction of the full block residual
scores 9.7685 and is also rejected. Native hidden states differ from separately
evaluated layer buffers by 19.4% and 30.2% relative L2 respectively, versus 1.46%
for the teacher, on four checked documents. The cause is not established; staged
buffers must not substitute for native graph validation. These are tiny CPU
diagnostics with original attention, not full YAT model quality evidence.
See `artifacts/yat-basis-initialization-0925/native-verification.json` and
`artifacts/yat-basis-ridge-initialization-0925`.

Another fused-backward CPU diagnostic computes alpha as a probability-weighted
row covariance, centering both the kernel and output cotangent with the stored
probability mass. It preserves BF16 array arithmetic but adds an XLA traversal;
it is not a speed optimization. Against the independent FP64 oracle, captured
layer14 global alpha relative error falls37.3%→12.9%, while layer21 global/local
errors rise6.48%/8.88%→12.56%/14.40%. It is not adopted. Evidence:
`artifacts/yat-streaming-0925/alpha-covariance-comparison.json`. This reinforces
the need to check every derivative and depth rather than select one favorable
fixture.


### Paired physical v5e-16 result (2026-09-25)

The dense TPU distance-repair implementation completed a matched full-model
comparison on one 16-chip, four-host allocation. Both variants used the frozen
307M model, identical initial weights, seed, batches, multilingual data,
optimizer schedule and sharding. Every host passed the 30 dispatch tests.
All four wrapper receipts passed; the controller exited 0. Both the queue and
VM were independently confirmed absent at 06:51:37 UTC.

Across 30 warm, unprofiled steps of a 32-step calibration, useful throughput
rose from 62,123 to 150,033 tokens/s: **2.415×**. Padded throughput rose from
79,973 to 193,142 positions/s. This is a short-run result for this configuration,
not an all-scale or sustained-production claim. Compiled temporary memory fell
only 0.443%, from 5,943,797,760 to 5,917,476,864 bytes.

JAX XPlane analysis on the writer host's four devices showed the profiled
module falling from about 785 to 314 ms. The shape-selected attention repair
loops fell from 363–499 to 34–43 ms; named collective intervals fell from
90–229 to 68–82 ms. Those categories are interval unions, not exclusive kernel
time, and collective durations include waiting. Pallas-named operations stayed
near 32.9 ms. This supports the repair traversal optimization as the main gain;
it does not establish network saturation. Remaining collective and projection
costs deserve further measurement, but the priority is quality.

The first two training losses matched exactly; later trajectories differed by
up to 0.103617 (mean signed candidate-minus-baseline difference −0.002822).
Baseline/candidate numerical equivalence is therefore **not** claimed. Within
the candidate, interruption after committed checkpoint 16 and resume to 32
reproduced exact model, optimizer and training-state hashes. Held-out loss was
17.284266 on 256 rows / 14,904 masked targets: production quality remains
unqualified, and no production training was launched. Fused-attention backward
prototypes remain experimental.

At the explicitly supplied planning rate of $9.60 per slice-hour, warm compute
would be $42.93 versus $17.77 per billion useful tokens. This excludes startup,
compilation, profiles, checkpoints, evaluation and storage; it is neither a
billing measurement nor a current credit balance. The conservative budget
ledger still retains the full $13 reservation and $891.87 planning exposure.

Profile summaries now bound displayed HLO operation names to 500 characters,
retaining full-name SHA256 and length for truncated names. Grouping, regex
classification and ranking still use full names. This avoids multi-megabyte
summaries of model parameter signatures. The comparison/profiling suite passes
34 tests; Ruff and Pyright pass.

Evidence: `artifacts/yat-dense-model-long-v5e16-0925/` contains
`training-comparison.json`, `matched-evidence-audit.json`, `profile-findings.json`,
`writer-profile-compact.json`, per-host receipts, exact recovery manifests,
raw XPlane traces and `cleanup-verification.json`.


### Gradient conditioning follow-up (2026-09-25)

A CPU probe of the native 22-layer raw-pretrained YAT encoder used one
64-token training row and two masked targets. The heuristic and dense
attention-repair variants produced exactly equal loss and all 175 layer-gradient
leaves. This does not reproduce or explain the observed physical TPU drift.

With attention alpha1, the layer-gradient norm was 487,969,622. Early QKV
projections dominated. This layer-only norm is a lower bound for the full model
norm on the same inputs. Under unit global clipping, every nonzero last-layer
component in this probe was below Adam's 1e-8 epsilon. Adam's normalization means
that uniform gradient scaling is not itself proof of vanishing updates; the
comparison to its epsilon identifies a concrete conditioning concern.

Attention alpha0.1 reduced the layer-gradient norm to984 and training-probe loss
from16.666 to12.346. Alpha0.01 gave norm787 and loss12.874. Bias stayed1,
epsilon stayed0.01, FFN alpha stayed1, and all alpha parameters stayed trainable.
These initialization probes performed no optimization updates.

Alpha0.1 was selected on the training row before checking32 held-out rows of64
tokens. Native held-out MLM losses were teacher1.37325, all-YAT alpha1
17.62491, and all-YAT attention alpha0.1 17.47487. This is a useful stability
lead but fails quality qualification. Keep defaults unchanged; test bounded
adaptation with this initialization and compare gradient-health trajectories.

The training harness now reports `gradient_norm_before_clip` and
`gradient_clip_scale`. Sixteen relevant CPU tests, Ruff, Pyright and diff checks
passed. The instrumentation has not yet been re-profiled on TPU, so no claim of
zero performance overhead is made. No new TPU allocation was created.

Exact inputs, source hashes, per-parameter gradients statistics and matched
held-out results are stored beside this report. Full objective remains active:
production quality, TPU trajectory drift and fused-attention backward accuracy
remain unresolved.


### Matched alpha adaptation (2026-09-25)

All three variants used32 CPU MLM updates, the same32 training rows truncated
to64 tokens, two masked targets per update, AdamW3e-5 and unit global clipping.
Layer parameters were trainable. Shared vocabulary, outside-block norms and
prediction head stayed frozen, verified by hashes. The reused32-row diagnostic
evaluation contains282 masked targets; this is not untouched final qualification.

| Initialization | Initial MLM loss | After32 updates | Median layer-gradient norm |
| --- | ---: | ---: | ---: |
| Attention alpha1, FFN alpha1 |17.6249|17.3436|671,760,928|
| Attention alpha0.1, FFN alpha1 |17.4749|14.0914|168.05|
| Attention alpha0.1, training-calibrated FFN alpha |16.7202|15.6304|see findings.json|

The teacher scores1.37325 on these same evaluation inputs. Smaller attention
alpha improves stability and short-run adaptation, but quality remains poor.
Reject the FFN-calibrated combination for the next trial. These CPU results do
not qualify TPU numerical behavior, sustained training or production quality.

The normal trainer, YAT recipe and physical validation wrapper now accept
`--yat-attention-alpha 0.1`, separately from `--yat-alpha 1`. Both parameters
remain trainable. Fixed kernel bias1 and epsilon0.01 are unchanged. Omitting
the new option preserves inheritance from the existing alpha. The override is
stored in resolved configuration and checkpoint identity; changing it is not
an execution-only override. Old frozen validation bundles remain unchanged;
strict source/config checks must not be bypassed to resume an old run.

Separate-alpha initialization, invalid inputs, recipe propagation, stage
compatibility, full toy training and exact interruption/resume are tested.
The new setting has not been physically tested on TPU. No cloud resources were
created for these experiments. Next step: qualify the opt-in initialization in
a bounded training run, with gradient telemetry and held-out quality checks.


### Alpha TPU admission and wrapper failure (2026-09-25)

The initial request was rejected locally because its$7 reservation was below
the supervisor's$7.20 requirement. No guard, ledger, queue or VM was created for
that request; resource absence was independently verified. A new immutable
reconciliation retains an additional$0.20 upload allowance.

The corrected admission reserved$7.25, bringing conservative planning exposure
to$899.32/$900. This is not billed usage or a remaining-credit balance. An
independent cleanup Workflow was verified before allocation.

The v5litepod-8 allocation provided one host with eight chips. Setup passed,
but the wrapper failed before tests or training with missingJAX_PROCESS_INDEX.
The plan also incorrectly expected two processes. No runtime configuration
amendment was applied. This attempt provides no training or model-validation
result. The controller began cleanup immediately; at07:26 UTC the VM remained
DELETING and controller session86695 was still live. Check authoritative cleanup
state before another request.

The corrected wrapper is prepared at
`artifacts/yat-alpha-v5e8-fixed-0925/` and has not been submitted. It uses the
new launcher-topology helper: missing distributed environment means rank0 of1;
partial/invalid environment fails closed; a multi-host requirement is never
silently downgraded. A dry run of the complete wrapper with mocked subprocesses
verified its single-host command wiring without training or cloud calls.

The supervisor now rejects underfunded envelopes before creating attempt files
or invoking cloud tools. The new gradient-health report rejects missing,
nonfinite, unordered or incomplete observations and does not infer Adam update
sizes from clipping scale.76 relevant tests, Ruff, Pyright and diff checks pass.

Additional CPU fused-gradient diagnostics on native attention-alpha0.1
activations are in`artifacts/yat-alpha-fused-gradient-0925/`. They still do not
qualify the fused backward. A late-layer global alpha gradient has the wrong
sign in the native BF16 reference and both fused prototypes versus FP64. Its
cancellation-normalized error is small, so relative error alone exaggerates
magnitude, but the direction discrepancy remains meaningful. No fused path was
adopted. Production quality remains unqualified.


### Follow-up: cleanup, corrected validation and lower learning rate

The failed alpha8 retry was independently verified absent at 07:28:04 UTC; see `artifacts/yat-alpha-retry-v5e8-0925/cleanup-final-verification.json`. A corrected one-host/eight-chip request is now running, with 45 physical tests passed and training compiling. The immutable budget reconciliation and current $7.25 reservation bring conservative planning exposure to $896.32 of $900, not billed spend.

A 424.6M-parameter three-prototype YAT model improved diagnostic held-out MLM loss from 10.817396 to 8.178701 after 32 CPU updates at learning rate 3e-6. The otherwise matched 3e-5 run worsened to 13.399938; teacher-KL at 3e-5 worsened to 14.870907. These very small runs reuse diagnostic held-out rows; mmBERT scores about 1.37325. The lower-rate result warrants fresh evaluation, not production training or a default change. Details: `artifacts/yat-basis-alpha-low-lr-0925/`.

Fresh follow-up evaluation on 128 rows (1,191 targets), excluding complete documents touched by the preceding diagnostic, also improved: 11.627256 → 8.429386; mmBERT scored 2.027036. The model-only parameter snapshot and identities are retained at `artifacts/yat-basis-alpha-fresh-0925/`. This is still development validation, not final qualification.


### Eight-chip alpha0.1 JAX profile (2026-09-25)

The calibration completed 32 accepted updates. Thirty warm unprofiled steps delivered 78,209 useful tokens/s (98,642 padded positions/s). Pre-clipping gradient norm ranged from 12.07 to 144.39, median22.73. Recovery and held-out qualification were still pending when these calibration findings were recorded.

The captured step spans 284.10–284.31ms across eight devices. Collective interval unions span 57.81–83.54ms; vocabulary-shaped collectives account for47.63–69.91ms. The BF16[256000,768] embedding-gradient all-reduce runs twice at accumulation2. Shape-selected attention repair spans14.76–30.13ms and Pallas-named operations32.90ms. Intervals include nesting/waiting; they are not additive exclusive costs.

The next communication experiment should accumulate gradients locally across microsteps and reduce once per optimizer update, preserving global masked-token weighting, empty-mask behavior, finite-update rejection and exact restart. Compare against the same frozen model/data on the same topology; a sharded vocabulary is another candidate. Neither change is implemented or performance-qualified by this profile. The current run differs in topology, batch and alpha from the previous16-chip run, so their raw timings do not establish an alpha speedup. Source trace, parser output and hashes: `artifacts/yat-alpha-v5e8-fixed-0925/`.


Final eight-chip alpha0.1 validation passed: 32 updates, 78,209 useful tokens/s over30 warm unprofiled steps, exact model/optimizer/training-state recovery after committed-step16 SIGKILL, and held-out MLM loss9.194311 on256 rows/14,904 targets. Both TPU VM and queue were independently verified absent at07:49:02 UTC. Profiles and evidence remain in storage. This qualifies the bounded single-host implementation/recovery check, not model quality, every scale, or the altered setting on physical multi-host. See `artifacts/yat-alpha-v5e8-fixed-0925/RESULTS.md`.

The communication hypothesis now has an opt-in implementation: `--local-gradient-accumulation`. Seven four-device CPU tests pass, including BF16 YAT and exact resume; CPU HLO moves gradient reductions outside the microstep loop. Physical TPU benchmarking remains pending. FP32 accumulated-gradient communication can equal two BF16 reductions in byte volume, so no speedup is claimed. Evidence: `artifacts/yat-local-accumulation-0925/`.


### Full-model accumulation diagnostic and projection nesting

A two-device CPU comparison on the full307M native YAT model found identical forward loss17.5355129242 but2.1696% global gradient relative L2 drift over181 leaves. Some small alpha gradients have larger relative differences; see absolute errors in `artifacts/yat-local-full-gradient-0925/report.json`. This six-target/four-row probe does not establish physical TPU behavior or equivalent training. The mode remains opt-in.

Inspection also found that fused MLM projection created a nested global shard_map inside the new local accumulation map. The projection now supports an internal local-only path, selected only on the local model copy; global weighting/reduction stays with the enclosing accumulation helper. Native XLA, chunked XLA and Pallas-interpreted recovery/weighted-loss tests exercise this integration. `validate_yat_mmbert` now forwards `--local-gradient-accumulation` to all training stages.

A direct summed-loss diagnostic did not improve the accumulation drift: the full-model CPU result remains2.1696% global gradient relative L2 with identical reported per-leaf errors. The explicit sum path avoids mean reconstruction but is not a measured numerical or throughput improvement. One attention-alpha scalar changes sign versus the previous BF16 path; neither is established as an oracle. Frozen source and evidence: `artifacts/yat-local-direct-sum-0925/`.

An independent per-example CPU alpha-gradient reference favors the local accumulation path:34/44 exact matches versus2/44 for the original path; summed absolute error0.139242 versus2.145055. The disputed layer17 attention-alpha sign matches the candidate. This reference uses BF16 derivatives accumulated in FP32, not an FP64 oracle, and scalar-only autodiff can compile differently from full-gradient autodiff. It supports further physical benchmarking, not default adoption. See `artifacts/yat-alpha-reduction-reference-0925/`. The training comparator now has an explicit `--compare-local-accumulation` mode that permits only the disabled-to-enabled flag change while continuing to reject unrelated configuration/data/topology differences.

Physical accumulation qualification stopped at14 passes/1 failure: one embedding-gradient element exceeded the unchanged tolerance. No matched full-model throughput was measured. Both TPU resources were subsequently verified absent. Set `FLAXCHAT_NUMERICAL_EVIDENCE_DIR` for the accumulation tests to retain actual initialized parameters, inputs and gradients before assertions. Replay with `python -m scripts.replay_mlm_accumulation CAPTURE --output REPORT` using the captured device count, or `--per-example-only` for a reduction diagnostic. Eight-device CPU captures replay exactly for both paths; this does not resolve the TPU failure. Evidence: `artifacts/yat-embedding-reduction-reference-0925/replay-verified.json`.

The replay tool also accepts `--profile-directory DIR`. It synchronizes warmup
and captures three warmed executions of each distributed accumulation path,
with separate trace directories. These small diagnostic traces identify
execution structure; they are not a full-model throughput benchmark. Replay
records effective matmul precision separately from environment variables.
The four-chip capture diagnostic at
`artifacts/yat-accumulation-capture-v5e4-0925/` compares default and highest
matmul precision without weakening the existing numerical assertion. Its
wrapper distinguishes a completed diagnostic from a passing numerical test.

A separate padding regression was reproduced and fixed after the four-chip
archive was frozen. Both accumulation modes now discard targets on pad tokens
before calculating microstep weights and the global denominator, matching the
encoder loss mask. The new test requires loss and every gradient leaf to be
unchanged when labels are added only at padding positions. The four-device CPU
accumulation/replay/profiling suite passed 31 tests. This post-freeze fix is not
covered by that TPU archive; its diagnostic inputs already ignore padding.

Physical four-chip capture/replay completed for default and highest matmul
precision. Both old-path comparison assertions still fail, but the independent
per-example BF16/FP32-sum reference favors local accumulation on all 21 gradient
leaves: 19/21 exact, maximum absolute error 1.62455e-6, versus 0/21 exact and
0.0038220221 for the existing path. The entire candidate embedding gradient
matches the reference. Both paths replay their own captured gradients exactly;
all 67 saved arrays are bit-identical across the two precision settings.

JAX traces contain one combined FP32 gradient reduction and one integer count
reduction per candidate update. Across three tiny-fixture executions, default
precision module interval unions fall from 0.957–0.981 ms to 0.512–0.581 ms and
collective unions from 0.416–0.441 ms to 0.029–0.066 ms. This supports the
communication hypothesis but is not a full-model speedup measurement. The
per-example reference is not an FP64 oracle, and the unchanged old-path gate
still fails. Full-model numerical/training comparison remains necessary.
See `artifacts/yat-accumulation-capture-v5e4-0925/RESULTS.md`.

The reference helper is now reused by 12 independent accumulation checks,
covering FP32/BF16, three masking cases and one/two examples per device.
Multi-row CPU BF16 derivatives do not satisfy the tight single-example bound:
maximum errors are 2.65747e-4 (empty-first) and 1.99318e-4 (uneven). They satisfy
the existing BF16 numerical budget; the tighter single-example check and the
original old-path agreement gate remain separate. On both multi-row cases,
local accumulation is closer on 11 leaves, the old path on 2, and 8 tie. Do not
generalize the earlier single-example TPU result into universal superiority.
CI now schedules multi-device tests for changes to the shared reference helper.
See `artifacts/yat-independent-accumulation-gate-0925/RESULTS.md`.

A 48-case CPU score-order ablation tested centering the eligible kernel row
before alpha multiplication, rather than centering only the backward
cotangent. It improves the late-layer alpha cases but introduces an early-layer
sign error and worsens alpha in six of eight standard-backward cases. Combining
it with the existing max-centered backward still yields two alpha sign errors.
Reassociation alone also has mixed Q/K/alpha results. None was adopted; bias1,
epsilon.01, trainable alpha and production arithmetic remain unchanged.
Evidence and frozen source: `artifacts/yat-prescale-center-0925/RESULTS.md`.


### Eight-chip independent accumulation gate (2026-09-25)

The matched full-model benchmark was blocked by the physical numerical gate:
21 tests passed and one failed (`2-uneven-True`). The BF16 two-example-per-device
case produced embedding gradient[7,5] = 0.017401427 versus independent per-example
reference 0.013730288 (absolute difference 0.003671139). The unchanged
`atol=.003, rtol=.04` contract rejects this element. All twelve independent
tensor captures were saved before assertions and checksum verified. FP32 cases
and single-example BF16 cases passed; the latter have maximum gradient errors
under 2.1e-6. This does not establish correctness for larger local batches.

The separately recorded legacy comparison also remains failed. No full-model
throughput or new full-model trace was produced, and local accumulation remains
opt-in. Do not widen tolerances or infer a production speedup from the previous
tiny-fixture trace. Investigate batch-dependent BF16 differentiation using the
actual saved tensors before another paid benchmark. Evidence:
`artifacts/yat-accumulation-reference-v5e8-0925/`.


### Gather-before-cast embedding gradient repair

The input embedding now gathers FP32 master rows before converting selected
activations to the residual dtype. This keeps repeated-token parameter-gradient
scatter accumulation in FP32; YAT geometry and BF16 forward values are unchanged.
A repeated-token fixture removes 0.0234375 rounding error exactly. CPU replay of
the captured failing TPU fixture reduces maximum batch/reference gradient error
6.15x, but physical TPU revalidation remains pending. Eight-device CPU encoder
and accumulation tests: 56 passed, 2 skipped. This changes training gradients,
so prior checkpoints remain structurally readable but their old trajectories
are not promised across the source change. No TPU speedup is claimed. Evidence:
`artifacts/yat-gather-first-0925/`.


Physical follow-up (`artifacts/yat-gather-tpu-v5e8-0925/`): the old path replayed
21/21 captured TPU gradients exactly. Gather-first improved the worst error only
2.8%, from .003671139 to .003567681; embedding[7,5] still violates the unchanged
contract. CPU improvement did not transfer. Both three-step tiny-fixture JAX
traces have overlapping op-time ranges, so no speedup is established. The gate
stopped all subsequent full-model training. Isolate input and tied-decoder
gradient contributions before another model benchmark.

The tied-gradient diagnostic is available as `python -m
scripts.diagnose_tied_embedding CAPTURE --output NEW_DIRECTORY`. It supports
only the native chunked XLA decoder used by the failing fixture, explicitly
rejecting paths that bypass its decoder instrumentation. It writes verified
gradient tensors and per-path loss/closure reports, including nonfinite flags
and source/runtime identity. Four-device CPU CI covers the decomposition.
Physical decomposition validation remains outstanding; the CPU cancellation
observation alone does not attribute the TPU failure.


Physical split-gradient attribution (`artifacts/yat-split-tpu-v5e8-0925/`)
reproduced both prior full embedding tensors exactly. At the disputed element,
76.6% of gather-first error comes through the input path and 23.4% through the
tied decoder. Decomposition and error-attribution residuals are <=2.98e-8. The
remaining failure therefore cannot be repaired solely by gather order. Six
tiny-fixture JAX profiles show overlapping full-path timing ranges, not a
qualified throughput improvement. Numerical and production gates remain open.

The forward batching command `python -m scripts.diagnose_encoder_batching
CAPTURE --output NEW_DIRECTORY` captures stage outputs at the captured local
batch size and per example. Independent uninstrumented output controls detect
compiler changes from returning intermediate tensors. CPU replay of the saved
TPU fixture is exact at every stage and both controls. Physical forward replay
is pending; this is not a distributed training or compacted-MLM diagnostic.
Evidence: `artifacts/yat-forward-batching-0925/`.


Single-chip physical forward replay (`artifacts/yat-forward-tpu-v5e1-0925/`)
found ordinary batched/per-example logit drift 0.0007565 and forward/autodiff
loss drift 0.0002527. Instrumented stage drift appears at layer 0, but exposing
intermediates itself changes logits (up to 0.0010988). Thus it cannot establish
a specific layer defect. Investigate fusion/rounding with controlled compiler
boundaries; do not infer a successful numerical repair or distributed speedup.
