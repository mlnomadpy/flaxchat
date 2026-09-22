"""Convert a local ModernBERT PyTorch state dictionary to safetensors safely."""
import argparse
import json
from pathlib import Path


def convert(directory):
    import torch
    import numpy as np
    from safetensors.numpy import save_file
    from flaxchat.encoder import EncoderConfig
    from scripts.train_encoder import file_hash
    root = Path(directory)
    EncoderConfig.from_hf(json.loads((root / 'config.json').read_text()))
    output = root / 'model.safetensors'
    if output.exists():
        raise ValueError('model.safetensors already exists; refusing to overwrite')
    source = root / 'pytorch_model.bin'
    state = torch.load(source, map_location='cpu', weights_only=True)
    arrays = {key: value.detach().float().numpy() for key, value in state.items()}
    embed = 'model.embeddings.tok_embeddings.weight'
    if 'decoder.weight' in arrays:
        if not np.array_equal(arrays['decoder.weight'], arrays[embed]):
            raise ValueError('Checkpoint does not have tied embeddings')
        del arrays['decoder.weight']
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite checkpoint tensor')
    save_file(arrays, str(output), metadata={'format': 'pt'})
    report = dict(source_sha256=file_hash(source), safetensors_sha256=file_hash(output),
                  tensors=len(arrays), loader='torch.load(weights_only=True)')
    (root / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', help='Local snapshot with config.json and pytorch_model.bin')
    print(json.dumps(convert(p.parse_args().snapshot), indent=2))
