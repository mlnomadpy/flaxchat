"""Execute receipt construction only; no model or numerical backend imports."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_explicit_executed_truncation_and_native_score_policy():
    source = Path('scripts/evaluate_yat_public_mteb.py')
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_identity')
    namespace = {'Path': Path, '__file__': str(source), 'file_hash': lambda path: str(path),
        'MTEB_VERSION': 'frozen', 'BENCHMARKS': {'english': 'suite'},
        'provenance': lambda *args, **kwargs: {'captured': True},
        'jax': SimpleNamespace(devices=lambda: [])}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'identity', 'exec'), namespace)
    identity = namespace['_identity'](Path('/fixture'), 'english', 512, 'model')
    assert identity['truncation_policy'] == 'right-truncate-token-ids-to-sequence-length-before-padding'
    assert identity['score_scale'] == 'native-mteb-main-score/no-rescaling'
    assert identity['sequence_length'] == 512
    assert identity['prompts'] == 'none'
    assert identity['normalization'] == 'L2 FP32'
    assert len(identity['files_sha256']) == 3
