# Releases and compatibility

## v0.1.1 release candidate

This patch release carries the [TPU initialization fix from issue
#8](https://github.com/mlnomadpy/flaxchat/issues/8), first verified at
[commit `97ba1333bf46b51f85ca3d9099c8c2717438ce91`](https://github.com/mlnomadpy/flaxchat/commit/97ba1333bf46b51f85ca3d9099c8c2717438ce91).
The evidence consists of the repository's [GitHub Actions
runs](https://github.com/mlnomadpy/flaxchat/actions), [Kaggle TPU acceptance
kernel version 6](https://www.kaggle.com/code/skywolfmo/flaxchat-tpu-full-test-suite),
and the machine-readable records in [RESULTS.md](RESULTS.md). Checkpoint format
2 and the public configuration keys used by v0.1.0 remain compatible; the
cross-topology restore tests and tracked TinyStories checkpoint exercise that
contract before release.

flaxchat uses semantic version tags (`vMAJOR.MINOR.PATCH`). The tag must match
the package version or the release workflow fails. Every release reruns lint,
type checking, dependency audit, the complete CPU suite, the offline
end-to-end pipeline, and package construction. Wheels and source archives are
published with GitHub artifact attestations; the pipeline manifest is retained
as release evidence. Release assets also include SHA-256 checksums and a
CycloneDX SBOM. Supported Python versions are install-smoke-tested only on tag
builds, avoiding repeated matrix cost on ordinary pushes.

Checkpoint format compatibility is independent of the package version and is
defined in [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md). A format change must
increment `CHECKPOINT_FORMAT_VERSION`. Readers reject unknown formats before
mutating live state. Within a package major version, public configuration keys,
pipeline manifest fields, and checkpoint formats are backward compatible;
removals require a deprecation notice in at least one minor release.

Accelerator validation is intentionally separate because Kaggle TPU capacity
is queued. It has no schedule. Before tagging, manually dispatch
`.github/workflows/kaggle-tpu.yml` against
the exact candidate commit and link its downloaded `summary.json` in the
release notes. A failed or incomplete accelerator bundle blocks the tag.
