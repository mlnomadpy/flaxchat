"""Measure checkpoint tensor drift on CPU; never replace exact-resume gates.

Restores one component pair at a time as NumPy arrays and verifies its manifest
before comparing. A finite numerical report is not an acceptance/quality claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def compare_arrays(reference, candidate):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Checkpoint leaf shape or dtype mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite checkpoint leaf')
    # Compute norms in bounded FP64 chunks; do not allocate FP64 model copies.
    a, b = a.reshape(-1), b.reshape(-1)
    error_squared = scale_squared = maximum = 0.0
    changed = zero_changes = sign_changes = 0
    exact_bytes = True
    for start in range(0, a.size, 262144):
        x, y = a[start:start+262144], b[start:start+262144]
        exact_bytes = exact_bytes and x.tobytes() == y.tobytes()
        delta = y.astype(np.float64) - x.astype(np.float64)
        error_squared += float(np.sum(delta * delta))
        scale_squared += float(np.sum(x.astype(np.float64) ** 2))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        changed += int(np.count_nonzero(x != y))
        zero_changes += int(np.count_nonzero((x == 0) != (y == 0)))
        sign_changes += int(np.count_nonzero(np.signbit(x) != np.signbit(y)))
    return dict(elements=int(a.size), changed_elements=changed,
                exact_bytes=exact_bytes, max_abs=maximum,
                error_l2=float(np.sqrt(error_squared)),
                reference_l2=float(np.sqrt(scale_squared)),
                relative_l2=float(np.sqrt(error_squared/scale_squared)) if scale_squared else None,
                zero_changes=zero_changes, sign_changes=sign_changes)


def restore_component(directory, item, expected):
    # Array-only comparisons run in TPU subprocess supervisors. Defer backend
    # imports so the supervisor does not acquire its child worker's TPU.
    import jax
    import orbax.checkpoint as ocp
    from flaxchat.checkpoint import _canonical_manifest_paths, _state_manifest

    with ocp.PyTreeCheckpointer() as checkpointer:
        path = str(directory / item)
        metadata = ocp.PyTreeCheckpointHandler().metadata(directory / item)
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata)
        tree = checkpointer.restore(path, args=ocp.args.PyTreeRestore(
            item=metadata, restore_args=restore_args))
    if _canonical_manifest_paths(_state_manifest(tree)) != _canonical_manifest_paths(expected):
        raise ValueError(f'{item}: restored bytes differ from checkpoint manifest')
    return {jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def compare_checkpoints(reference: Path, candidate: Path):
    manifests = [json.loads((p/'manifest/metadata').read_text()) for p in (reference, candidate)]
    if manifests[0]['step'] != manifests[1]['step']:
        raise ValueError('Checkpoint steps differ')
    comparisons = {}
    for item, field in [('model', 'model_state'), ('optimizer', 'optimizer_state'),
                        ('training_state', 'training_state')]:
        a = restore_component(reference, item, manifests[0][field])
        b = restore_component(candidate, item, manifests[1][field])
        if set(a) != set(b):
            raise ValueError(f'{item}: checkpoint leaf paths differ')
        comparisons[field] = {path: compare_arrays(a[path], b[path]) for path in a}
        del a, b
    return dict(complete=True, step=manifests[0]['step'], comparisons=comparisons,
                identity_equal=manifests[0]['identity'] == manifests[1]['identity'],
                scope='Diagnostic tensor drift only; no tolerance or production qualification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare_checkpoints(args.reference.resolve(), args.candidate.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
