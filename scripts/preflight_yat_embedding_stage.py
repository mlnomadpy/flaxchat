"""Authenticate an embedding stage's parent and prepared mixture without a model.

Run before allocating a TPU; imports do not initialize a device or execute any
forward/backward. Exact-resume state is verified again by the physical trainer.
"""
import argparse
import json
from pathlib import Path
from flaxchat.embedding_contract import source_identity, validate_parent_metadata
from flaxchat.embedding_contract import canonical_hash
from flaxchat.embedding_stage import add_stage_arguments, prepare_stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_stage_arguments(parser)
    parser.add_argument('--expected-devices', type=int, required=True)
    parser.add_argument('--expected-processes', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    admitted = prepare_stage(args, device_count=args.expected_devices, process_count=args.expected_processes)
    hashes, encoder = admitted['parent_hashes'], admitted['encoder_config']
    parent = {'public_files_sha256': hashes}
    if args.parent_checkpoint:
        from flaxchat.checkpoint_metadata import read_committed_metadata
        metadata = read_committed_metadata(args.parent_checkpoint, args.parent_step, include_receipt=True)
        parent.update(validate_parent_metadata(metadata, encoder, hashes['tokenizer.json']))
        parent['committed_artifact'] = metadata['committed_receipt']
    receipt = {'passed': True, 'scope': 'model_free_configuration_data_and_parent_admission', 'physical_tpu_qualified': False,
               'parent': parent, 'mixture': admitted['mixture_receipt'],
               'resolved_configuration': admitted['admission_receipt'],
               'data_manifests': {name: digest for name, _, _, digest in admitted['data_items']},
               'remaining_checks': ['actual physical topology', 'model/loss/gradient/recovery acceptance',
                                    'exact-resume optimizer and sampler cursor' if args.resume else 'initial model weights'],
               'source': source_identity(Path(__file__).resolve().parents[1])}
    receipt['identity_sha256'] = canonical_hash(receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()
