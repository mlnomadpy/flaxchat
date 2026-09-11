# Compatibility matrix

| Surface | Supported contract | Validation |
|---|---|---|
| Python | 3.11, 3.12, 3.13 | Wheel/sdist install smoke on release tags |
| CPU | Base `jax>=0.9`, macOS arm64 and Linux x86-64 | Linux on changes; macOS manual/release-candidate only |
| NVIDIA GPU | `flaxchat[cuda12]`, compatible `jax[cuda12]>=0.9` | Manual Kaggle GPU acceptance |
| TPU | `flaxchat[tpu]`, compatible `jax[tpu]>=0.9`, libtpu runtime | Manual Kaggle TPU acceptance |
| Kaggle CLI | `flaxchat[kaggle]` | Bundle rendering locally; remote run on demand |

Python 3.14 is intentionally excluded because the required `rustbpe` runtime
does not provide a compatible distribution in the tested environment. GPU and
TPU plugins must not be installed together blindly: choose the accelerator
extra/runtime for the target host and retain the resolved environment in the
run manifest.

The CUDA and TPU extras are resolver-tested on their corresponding manual
Kaggle target workflows. They stay optional so the CPU wheel, web UI, and
Kaggle CLI do not pull accelerator plugins into unrelated installations.

Routine CI runs once on Linux and cancels superseded commits. macOS and Kaggle
workflows are `workflow_dispatch` only so compatibility evidence is collected
when a release or platform change needs it, without continuously consuming
paid runner minutes or accelerator quota.
