# Releases and compatibility

flaxchat uses semantic version tags (`vMAJOR.MINOR.PATCH`). The tag must match
the package version or the release workflow fails. Every release reruns lint,
type checking, dependency audit, the complete CPU suite, the offline
end-to-end pipeline, and package construction. Wheels and source archives are
published with GitHub artifact attestations; the pipeline manifest is retained
as release evidence.

Checkpoint format compatibility is independent of the package version and is
defined in [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md). A format change must
increment `CHECKPOINT_FORMAT_VERSION`. Readers reject unknown formats before
mutating live state. Within a package major version, public configuration keys,
pipeline manifest fields, and checkpoint formats are backward compatible;
removals require a deprecation notice in at least one minor release.

Accelerator validation is intentionally separate because Kaggle TPU capacity
is queued. Before tagging, dispatch `.github/workflows/kaggle-tpu.yml` against
the exact candidate commit and link its downloaded `summary.json` in the
release notes. A failed or incomplete accelerator bundle blocks the tag.
