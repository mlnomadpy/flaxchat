"""Released ModernBERT parity: reference and JAX phases run in separate processes."""
import argparse
import json
import hashlib
from pathlib import Path

import numpy as np


def snapshot_identity(snapshot):
    root = Path(snapshot)
    files = [root / 'config.json', root / 'tokenizer.json', *sorted(root.glob('*.safetensors'))]
    if len(files) < 3:
        raise ValueError('A converted safetensors snapshot is required')
    result = {}
    for path in files:
        with path.open('rb') as stream:
            result[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result


def check_reference(data, *, dtype, residual_dtype, identity, diagnostic=False):
    if 'snapshot_identity' not in data:
        raise ValueError('Reference lacks snapshot provenance; regenerate it')
    if json.loads(str(data['snapshot_identity'])) != identity:
        raise ValueError('Reference snapshot identity mismatch')
    matches = str(data['dtype']) == dtype and str(data['residual_dtype']) == residual_dtype
    if not matches and not diagnostic:
        raise ValueError('Reference precision mismatch; use matching reference or --diagnostic')
    return matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['reference', 'compare'])
    p.add_argument('--snapshot', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--residual-dtype', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--diagnostic', action='store_true',
                   help='Report numerical drift without treating it as a qualification pass')
    a = p.parse_args()
    if a.dtype == 'float32' and a.residual_dtype != 'float32':
        p.error('BF16 residuals require BF16 compute')
    identity = snapshot_identity(a.snapshot)
    if a.phase == 'reference':
        import torch
        from transformers import AutoConfig, AutoModelForMaskedLM
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(Path(a.snapshot) / 'tokenizer.json'))
        tokenizer.enable_truncation(max_length=a.length)
        texts = ['The capital city of France is Paris.', 'عاصمة فرنسا هي باريس.',
                 'La capitale de la France est Paris.', '法国的首都是巴黎。']
        texts[2:] = [text * 20 for text in texts[2:]]
        x = np.zeros((4, a.length), np.int32)
        y = np.full_like(x, -1)
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text).ids
            x[i, :len(ids)] = ids
            position = max(1, len(ids) // 2)
            y[i, position] = x[i, position]
            x[i, position] = 4
        ref_config = AutoConfig.from_pretrained(a.snapshot, local_files_only=True)
        ref_config.reference_compile = False
        ref = AutoModelForMaskedLM.from_pretrained(a.snapshot, config=ref_config, local_files_only=True,
                    attn_implementation='eager').eval()
        ids = torch.tensor(x, dtype=torch.long)
        # Autocast preserves FP32 master parameters; explicit BF16 residual mode
        # also rounds embedding/norm outputs as the JAX implementation does.
        if a.dtype == 'bfloat16' and a.residual_dtype == 'bfloat16':
            def cast_output(module, inputs, output):
                return output.to(torch.bfloat16)
            for module in ref.modules():
                if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
                    module.register_forward_hook(cast_output)
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=a.dtype == 'bfloat16'):
            hidden = ref.model(ids, attention_mask=ids != 0).last_hidden_state
            selected = torch.tensor(y >= 0)
            logits = ref.decoder(ref.head(hidden[selected]))
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor(y[y >= 0], dtype=torch.long))
        loss.backward()
        np.savez(a.reference, snapshot_identity=json.dumps(identity, sort_keys=True), inputs=x, targets=y, dtype=a.dtype, residual_dtype=a.residual_dtype, hidden=hidden.detach().float().numpy(),
                 logits=logits.detach().float().numpy(), loss=loss.detach().numpy(),
                 qkv_gradient=ref.model.layers[0].attn.Wqkv.weight.grad.numpy().T)
        Path(a.output).write_text(json.dumps({'reference_loss': float(loss.detach()), 'length': a.length,
            'dtype': a.dtype, 'residual_dtype': a.residual_dtype, 'torch': torch.__version__, 'languages': ['English', 'Arabic', 'French', 'Chinese']}, indent=2))
        return
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flaxchat.encoder import EncoderConfig, ModernBert
    from scripts.train_encoder import load_pretrained, file_hash
    jax.config.update('jax_default_matmul_precision', 'highest')
    data = np.load(a.reference, allow_pickle=False)
    matched = check_reference(data, dtype=a.dtype, residual_dtype=a.residual_dtype,
                              identity=identity, diagnostic=a.diagnostic)
    config = EncoderConfig.from_hf(json.loads((Path(a.snapshot) / 'config.json').read_text()),
                                  compute_dtype=a.dtype, residual_dtype=a.residual_dtype, loss_chunk_size=16)
    model = ModernBert(config, rngs=nnx.Rngs(0))
    provenance = load_pretrained(model, a.snapshot)
    x, y = jnp.array(data['inputs']), jnp.array(data['targets'])
    valid = data['inputs'] != 0
    positions = np.where(data['targets'] >= 0)
    hidden = nnx.jit(lambda m, x: m.encode(x))(model, x)
    logits = model.project(hidden[positions])
    loss, grads = nnx.jit(nnx.value_and_grad(lambda m: m(x, y)))(model)
    actual = dict(hidden=np.asarray(hidden)[valid], logits=np.asarray(logits), loss=np.asarray(loss),
                  qkv_gradient=np.asarray(grads['layers'][0]['qkv']['kernel'][...]))
    expected = {k: data[k][valid] if k == 'hidden' else data[k] for k in actual}
    tolerance = {'float32': (2e-3, 2e-3), 'bfloat16': (.25, .05)}[a.dtype]
    errors = {k: float(np.max(np.abs(v - expected[k]))) for k, v in actual.items()}
    relative_l2 = {k: float(np.linalg.norm(v - expected[k]) / max(float(np.linalg.norm(expected[k])), 1e-12)) for k, v in actual.items()}
    passed = max(relative_l2.values()) < (.05 if a.dtype == 'bfloat16' else .002) and all(np.allclose(v, expected[k], atol=tolerance[0], rtol=tolerance[1]) for k, v in actual.items())
    report = dict(passed=passed if not a.diagnostic else None, numerical_gate_passed=passed,
                  qualification=not a.diagnostic, reference_precision_matches=matched, backend=jax.default_backend(), dtype=a.dtype, residual_dtype=a.residual_dtype,
                  reference_dtype=str(data['dtype']), reference_residual_dtype=str(data['residual_dtype']),
                  max_absolute_errors=errors, relative_l2_errors=relative_l2, absolute_tolerance=tolerance[0], relative_tolerance=tolerance[1],
                  loss=float(loss), parameters=sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param))),
                  weights_sha256=provenance, tokenizer_sha256=file_hash(Path(a.snapshot) / 'tokenizer.json'),
                  length=x.shape[1], masked_tokens=int((y >= 0).sum()), jax=jax.__version__)
    Path(a.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not passed and not a.diagnostic:
        raise SystemExit('Released checkpoint parity failed')


if __name__ == '__main__':
    main()
