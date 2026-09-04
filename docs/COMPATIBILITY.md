# Compatibility matrix

| Surface | Supported contract | Validation |
|---|---|---|
| Python | 3.11, 3.12, 3.13 | Wheel/sdist install smoke on release tags |
| CPU | Base `jax>=0.9`, macOS arm64 and Linux x86-64 | Linux on changes; macOS manual/release-candidate only |
| NVIDIA GPU | Install the JAX CUDA plugin appropriate to the host | Manual Kaggle GPU acceptance |
| TPU | `flaxchat[tpu]`, compatible `jax[tpu]>=0.9`, libtpu runtime | Manual Kaggle TPU acceptance |
| Kaggle CLI | `flaxchat[kaggle]` | Bundle rendering locally; remote run on demand |

Python 3.14 is intentionally excluded because the required `rustbpe` runtime
does not provide a compatible distribution in the tested environment. GPU and
TPU plugins must not be installed together blindly: choose the accelerator
extra/runtime for the target host and retain the resolved environment in the
run manifest.

Routine CI runs once on Linux and cancels superseded commits. macOS and Kaggle
workflows are `workflow_dispatch` only so compatibility evidence is collected
when a release or platform change needs it, without continuously consuming
paid runner minutes or accelerator quota.
