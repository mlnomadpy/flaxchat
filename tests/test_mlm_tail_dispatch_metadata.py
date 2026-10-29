"""Static tail dispatch without importing a model or JAX backend."""

import ast
from pathlib import Path
import pytest


def dispatch():
    source = ast.parse(Path("flaxchat/training.py").read_text())
    function = next(
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "gradients_for_local_mlm_microbatches"
    )
    calls = []

    def record(name):
        def invoke(*args):
            calls.append((name, args))
            return name

        return invoke

    namespace = {
        "gradients_for_mlm_microbatches": record("global"),
        "_gradients_for_local_mlm_microbatches": record("local"),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<dispatch>", "exec"),
        namespace,
    )
    return namespace[function.name], calls


def test_full_batch_is_direct_local_and_tail_is_direct_reference():
    function, calls = dispatch()
    model, inputs, targets, mesh = (object() for _ in range(4))
    assert function(model, inputs, targets, mesh) == "local"
    assert calls == [("local", (model, inputs, targets, mesh))]
    calls.clear()
    assert function(model, inputs, targets, mesh, padded_rows=True) == "global"
    assert calls == [("global", (model, inputs, targets))]


@pytest.mark.parametrize("invalid", [0, 1, None, object()])
def test_nonstatic_padding_flag_is_rejected_without_model_execution(invalid):
    function, calls = dispatch()
    with pytest.raises(TypeError, match="static Python bool"):
        function(None, None, None, None, padded_rows=invalid)
    assert calls == []
