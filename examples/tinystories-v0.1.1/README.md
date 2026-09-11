# Flaxchat TinyStories acceptance checkpoint

This small checkpoint was produced by the bundled Kaggle TPU v5e-8 run at
source revision `97ba1333bf46b51f85ca3d9099c8c2717438ce91`. It is intended for
deterministic loading/inference smoke tests, checkpoint portability, and API
examples—not language quality or production use.

- Data: `roneneldan/TinyStories` at revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (CDLA-Sharing-1.0).
- Stages: two pretraining updates, one SFT update, one preference update.
- Hardware: one Kaggle TPU v5e-8 host with eight devices.
- Model/tokenizer/checkpoint identities and measured losses are in
  `run_manifest.json`; the complete training configuration is also published
  separately as `resolved_config.json`.
- Artifact integrity is recorded in `SHA256SUMS.json`.

From a flaxchat v0.1.1 checkout:

```bash
python -m scripts.verify_artifact examples/tinystories-v0.1.1
python -m scripts.checkpoint_demo --artifact-dir examples/tinystories-v0.1.1
python -m scripts.chat_cli --artifact-dir examples/tinystories-v0.1.1 -p "Tell a story"
python -m scripts.chat_web --artifact-dir examples/tinystories-v0.1.1
```

All three inference entry points verify every file against `SHA256SUMS.json`
before loading. Checkpoint and tokenizer paths, model architecture, and source
revision are resolved from `run_manifest.json`; no model-name heuristic is used.

The repository code is MIT licensed. See `LICENSE` for the artifact-specific
notice and the TinyStories dataset terms that also apply. This branch copy is
compatible with flaxchat 0.1.1 but is not an immutable release: the manifest
intentionally records `published: false`. A future `v0.1.1` release must attach
the verified archive before that field and its checksums can be changed.
