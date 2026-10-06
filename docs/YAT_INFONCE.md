# Opt-in YAT InfoNCE

Issue #67 adds a contrastive objective for a new training stage. Existing runs
continue to use cosine similarity by default; this implementation does not change
or restart a running cloud job.

For unit-length embeddings, let `s = q.T @ d`. The kernel and loss are

```text
K(s) = alpha * (s + 1)^2 / (2 - 2*s + 0.01)
L_i = -K(s_i+)/temperature + logsumexp_j(K(s_ij)/temperature)
alpha = softplus(raw_alpha) + 1e-6
```

The bias is fixed at 1, epsilon at 0.01, and temperature is fixed. Only alpha is
learned as an additional scalar parameter with optimizer state. Exponential YAT
weights are implicit in stable log-softmax; do not exponentiate scores before
passing them to log-softmax.

Select `--contrastive-similarity yat`. Optionally set
`--yat-infonce-alpha-init 0.01`; the default 0.01 is an initial experimental value,
**not a calibrated or TPU-qualified recommendation**. Values must be finite and
greater than the alpha floor of `1e-6`. Supplying this flag with the default
cosine objective is rejected instead of silently ignored. At identical vectors,
`K(1) = 400*alpha`, so temperature and initialization materially affect sharpness.

Normalization uses FP32 max-absolute-value rescaling followed by L2 normalization.
The resulting dot product is clipped to `[-1, 1]` to handle rounding. Zero or tiny
vectors with norm at most `1e-12`, and nonfinite query/positive vectors, cause a
nonfinite loss so training fails its finite-loss guard. Hard-negative padding
marked `negative_valid=False` is exempt from validity checks and excluded from
the loss. LayerNorm alone does not establish unit L2 norm. The objective retains
the existing duplicate and false-negative masking policies.

The `contrastive_objective` recipe field binds the kernel, normalization, alpha
initialization and parameterization, and fixed scale policies to the checkpoint.
Cosine recipes receive no extra objective field, preserving legacy identities.
Changing a cosine stage to YAT requires a **new stage** initialized from its
parent's encoder weights, with fresh objective and optimizer state. It is not an
exact resume or a quality-policy migration. Exact resume of a YAT stage restores
its learned alpha together with encoder and optimizer state.

Positive alpha makes this kernel monotonic in cosine similarity, so fixed
embeddings retain their cosine retrieval ordering. A training benefit is not
established by that fact. Qualification must compare loss/gradients against a
reference, exercise duplicate and distributed negative masking, prove alpha and
optimizer restoration, and measure actual TPU latency/memory and representation
quality before promoting the objective to a production default.

## Physical qualification

On a separately admitted TPU checkout (do not replace an active training job):

```sh
FLAXCHAT_PHYSICAL_TPU=1 JAX_PLATFORMS=tpu python -m pytest -q \
  tests/test_yat_infonce_physical_tpu.py \
  tests/test_yat_export_physical_tpu.py \
  tests/test_embedding_gradient_cache_physical_tpu.py \
  tests/test_embedding_trainer_physical_tpu.py
```

The trainer tests include pair/triplet, cached/direct execution, exact interrupted
resume with alpha and Adam moments, incompatible-alpha resume rejection, and
fresh-stage transitions from a YAT parent. Export authenticates the complete
checkpoint before removing the training-only alpha from serving weights.

`embedding_step.contrastive_metrics` records alpha, allowed-logit extrema, and
logit finiteness; existing step telemetry records throughput and gradient norm.
Use the existing `--profile-dir`, `--profile-skip`, and `--profile-steps` options
for a bounded JAX trace. Compare warmed-up latency and peak memory against cosine
at the same shape/topology, including cached and direct paths. Initial alpha and
fixed temperature must be calibrated on development data before a long run.

Implementation and test collection are not physical qualification. No YAT TPU
results or matched quality/cost improvement are established by this change. Keep
#67 open until physical recovery/export checks and the matched experiment pass.
