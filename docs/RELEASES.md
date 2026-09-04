# Releases and compatibility

## v0.1.1 release candidate

This patch release carries the [TPU initialization fix from issue
#8](https://github.com/mlnomadpy/flaxchat/issues/8), first verified at
[commit `97ba1333bf46b51f85ca3d9099c8c2717438ce91`](https://github.com/mlnomadpy/flaxchat/commit/97ba1333bf46b51f85ca3d9099c8c2717438ce91),
with the complete release-candidate suite subsequently verified at
[`12bfd8522f9a4dff46f05157108eb63159240882`](https://github.com/mlnomadpy/flaxchat/commit/12bfd8522f9a4dff46f05157108eb63159240882).
The evidence consists of the repository's [GitHub Actions
runs](https://github.com/mlnomadpy/flaxchat/actions), [Kaggle TPU acceptance
kernel version 12](https://www.kaggle.com/code/skywolfmo/flaxchat-tpu-full-test-suite),
and the machine-readable records in [RESULTS.md](RESULTS.md). Checkpoint format
2 and the public configuration keys used by v0.1.0 remain compatible; the
cross-topology restore tests and tracked TinyStories checkpoint exercise that
contract before release.

flaxchat uses semantic version tags (`vMAJOR.MINOR.PATCH`). The tag must match
the package version or the release workflow fails. A release accepts only a
tagged commit already present on `master` with a successful Linux-validation
run, avoiding a duplicate full suite. Tag builds retain the three supported
Python install-smokes on one Ubuntu runner and reuse one package build rather
than allocating a three-runner matrix. The extracted release checkpoint is
verified and exercised from an installed wheel in a clean temporary working
directory. Tag builds also retain deterministic checkpoint packaging, SHA-256
checksums, a CycloneDX SBOM, and GitHub artifact attestations.
The workflow uploads a draft first, downloads the published asset bytes into a
fresh directory, verifies their release-wide checksums, installs the downloaded
wheel in a clean Python 3.13 environment, and runs deterministic inference from
the downloaded checkpoint archive. The draft becomes public only after those
checks pass.

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
