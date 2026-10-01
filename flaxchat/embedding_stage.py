"""Shared model-free configuration and prepared-data admission for embedding stages."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from flaxchat.encoder_data import file_hash
from flaxchat.embedding_contract import validate_arrays
from flaxchat.embedding_data import qualify_mixture
from flaxchat.embedding_quality import stratified_dev_indices, load_sts_dev, load_retrieval_dev, bind_production_sts

ARRAYS = ("query_tokens", "positive_tokens", "negative_tokens", "query_text_ids",
          "positive_text_ids", "negative_text_ids", "positive_group_ids", "negative_valid")
ID_ARRAYS = {"query_text_ids", "positive_text_ids", "negative_text_ids", "positive_group_ids"}
TRAIN_ARRAYS = ARRAYS + ("known_positive_text_ids",)

def _load_data(specification: str, tokenizer_sha: str, encoder: dict):
    name, separator, folder = specification.partition("=")
    if not separator or not name or not folder or not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--data must be a unique SOURCE=DIR")
    path = Path(folder)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") not in ("flaxchat-yat-embedding-triplets-v1", "flaxchat-yat-embedding-triplets-v2")
            or manifest.get("source") != name
            or manifest.get("tokenizer_sha256") != tokenizer_sha):
        raise ValueError(f"Prepared {name} identity mismatch")
    required_files = {f"{split}/{key}.npy" for split in ("train", "dev") for key in ARRAYS}
    if set(manifest["raw_files"]) != {"train.jsonl", "dev.jsonl"}:
        raise ValueError("Prepared raw checksum inventory is not canonical and complete")
    if set(manifest["tokenization"]["files"]) != required_files:
        raise ValueError("Prepared array checksum inventory is not canonical and complete")
    for filename, digest in manifest["raw_files"].items():
        if file_hash(path / filename) != digest:
            raise ValueError(f"Prepared {name} raw file changed: {filename}")
    for filename, entry in manifest["tokenization"]["files"].items():
        if file_hash(path / filename) != entry["sha256"]:
            raise ValueError(f"Prepared {name} token file changed: {filename}")
    arrays = {split: {key: np.load(path / split / f"{key}.npy", mmap_mode="r",
                                        allow_pickle=False)
                      for key in ARRAYS} for split in ("train", "dev")}
    for split in ("train", "dev"):
        if any(len(value) != manifest["rows"][split]
               for value in arrays[split].values()):
            raise ValueError("Prepared array row count mismatch")
    validate_arrays(arrays, manifest, encoder)
    return name, arrays, manifest, file_hash(manifest_path)

def _weights(value: str, names: set[str]):
    parts = {}
    for term in value.split(","):
        name, separator, number = term.partition("=")
        if not separator or name in parts:
            raise ValueError("Invalid or duplicate source weight")
        parts[name] = float(number)
    if (set(parts) != names or any(not math.isfinite(x) or x <= 0 for x in parts.values())
            or not math.isclose(sum(parts.values()), 1.0, abs_tol=1e-6)):
        raise ValueError("Source weights must be positive and sum to one")
    return parts

def _batch_counts(weights: dict[str, float], batch_size: int):
    names = sorted(weights)
    floors = {name: int(batch_size * weights[name]) for name in names}
    remaining = batch_size - sum(floors.values())
    order = sorted(names, key=lambda name: (batch_size * weights[name] - floors[name], name),
                   reverse=True)
    for name in order[:remaining]:
        floors[name] += 1
    if any(count < 2 for count in floors.values()):
        raise ValueError("Each selected source needs at least two rows per global batch")
    return floors

def add_stage_arguments(parser):
    parser.add_argument("--parent-public", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--parent-step", type=int)
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--source-weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-policy", choices=("mixed", "homogeneous"), default="mixed")
    parser.add_argument("--language-exponent", type=float, default=1.0)
    parser.add_argument("--dev-max-rows", type=int, default=1024)
    parser.add_argument("--encoder-chunk-size", type=int, default=0, help="Exact gradient cache encoder chunk rows; 0 disables")
    parser.add_argument("--sts-dev", action="append", default=[], help="Pinned held-out STS NAME=DIR")
    parser.add_argument('--retrieval-dev', action='append', default=[], help='Independent authenticated retrieval/bitext/code NAME=DIR')
    parser.add_argument('--training-scope', choices=('production', 'qualification'), default='production',
                        help='Production requires authenticated independent task coverage; qualification is an explicit narrower fixture scope')
    parser.add_argument('--quality-min-rows', type=int, default=64)
    parser.add_argument('--quality-min-rows-per-language', type=int, default=16)
    parser.add_argument('--quality-min-unique-documents', type=int, default=16)
    parser.add_argument("--max-dev-regression", type=float, default=0.02)
    parser.add_argument("--profile-dir")
    parser.add_argument("--profile-skip", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=2)
    return parser


def validate_stage_configuration(args, *, device_count, process_count):
    """Reject deterministic configuration errors without querying a backend."""
    for value in (device_count, process_count):
        if type(value) is not int or value < 1:
            raise ValueError('Explicit positive physical topology counts required')
    if device_count % process_count:
        raise ValueError('Physical device count must divide evenly across processes')
    if (args.batch_size < 2 or args.batch_size % device_count
            or not 1 <= args.steps <= 100_000 or not 0 <= args.warmup < args.steps
            or args.save_every < 1 or args.eval_every < 1 or args.keep_checkpoints < 1
            or not math.isfinite(args.temperature) or not 0 < args.temperature <= 1
            or not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3
            or not math.isfinite(args.weight_decay) or not 0 <= args.weight_decay <= 1
            or args.seed < 0 or (args.parent_step is not None and args.parent_step < 1)
            or args.stop_after is not None and not 1 <= args.stop_after <= args.steps):
        raise ValueError('Invalid bounded training configuration')
    if process_count > 1 and not args.output.startswith('gs://'):
        raise ValueError('Multi-host checkpoint output must be in GCS')
    if bool(args.parent_checkpoint) != (args.parent_step is not None):
        raise ValueError('Prior checkpoint and committed step must be supplied together')
    if args.parent_checkpoint and args.parent_checkpoint.rstrip('/') == args.output.rstrip('/'):
        raise ValueError('A new stage cannot overwrite its parent checkpoint')
    if (args.dev_max_rows < 2 or not math.isfinite(args.max_dev_regression)
            or not 0 <= args.max_dev_regression <= 1
            or not math.isfinite(args.language_exponent) or not 0 <= args.language_exponent <= 1
            or args.batch_policy not in ('mixed', 'homogeneous')):
        raise ValueError('Invalid development or language policy')
    if (args.encoder_chunk_size < 0 or args.encoder_chunk_size and
            (args.batch_size % args.encoder_chunk_size or args.encoder_chunk_size % device_count)):
        raise ValueError('Encoder chunk must divide global batch and align with data devices')
    if args.profile_skip < 0 or args.profile_steps < 1:
        raise ValueError('Invalid bounded profiling window')
    if (getattr(args, 'training_scope', 'production') not in ('production', 'qualification')
            or type(getattr(args, 'quality_min_rows', 64)) is not int or getattr(args, 'quality_min_rows', 64) < 16
            or type(getattr(args, 'quality_min_rows_per_language', 16)) is not int
            or getattr(args, 'quality_min_rows_per_language', 16) < 8
            or type(getattr(args, 'quality_min_unique_documents', 16)) is not int
            or getattr(args, 'quality_min_unique_documents', 16) < 8):
        raise ValueError('Invalid declared production quality coverage minima')


def prepare_stage(args, *, device_count, process_count):
    """Shared parent/data/policy admission used by preflight and physical trainer.

    No model, numerical backend, optimizer or checkpoint tensor is constructed.
    Parent checkpoint and exact-resume artifact metadata are separate contracts.
    """
    validate_stage_configuration(args, device_count=device_count, process_count=process_count)
    parent = Path(args.parent_public)
    parent_hashes = {name: file_hash(parent / name) for name in
                     ('config.json', 'tokenizer.json', 'model.safetensors')}
    if parent_hashes != json.loads(Path(args.parent_manifest).read_text()):
        raise ValueError('Parent model files differ from the published manifest')
    encoder = json.loads((parent / 'config.json').read_text())
    if encoder.get('yat_bias') != 1 or encoder.get('yat_epsilon') != .01 or encoder.get('yat_alpha_trainable') is not True:
        raise ValueError('YAT architecture identity mismatch')
    items = [_load_data(spec, parent_hashes['tokenizer.json'], encoder) for spec in args.data]
    if len({name for name, _, _, _ in items}) != len(items):
        raise ValueError('Duplicate source data')
    data = {name: arrays for name, arrays, _, _ in items}
    weights = _weights(args.source_weights, set(data))
    counts = (_batch_counts(weights, args.batch_size) if args.batch_policy == 'mixed'
              else {name: args.batch_size for name in data})
    if any(len(data[name]['train']['query_tokens']) < counts[name] for name in data):
        raise ValueError('Too few rows for a source batch')
    query_lengths = {item[2]['query_length'] for item in items}
    document_lengths = {item[2]['document_length'] for item in items}
    if len(query_lengths) != 1 or len(document_lengths) != 1:
        raise ValueError('All sources need the same query/document token lengths')
    mixture = qualify_mixture(data,
        {spec.partition('=')[0]: Path(spec.partition('=')[2]) for spec in args.data},
        {name: manifest for name, _, manifest, _ in items})
    dev_indices = {name: stratified_dev_indices(data[name]['dev']['language_ids'], args.dev_max_rows, args.seed)
                   for name in data}
    training_directories = [Path(spec.partition('=')[2]) for spec in args.data]
    sts_data, sts_receipts, sts_manifests, sts_directories = {}, {}, {}, {}
    for specification in args.sts_dev:
        name, separator, folder = specification.partition('=')
        if (not separator or not name or not folder or name in sts_data
                or not name.replace('_', '').replace('-', '').isalnum()):
            raise ValueError('--sts-dev must be a unique NAME=DIR')
        arrays, manifest, digest = load_sts_dev(folder, encoder, parent_hashes['tokenizer.json'],
            training_directories=training_directories,
            tokenizer_path=parent / 'tokenizer.json' if getattr(args, 'training_scope', 'production') == 'production' else None)
        indices = stratified_dev_indices(arrays['languages'], args.dev_max_rows, args.seed, minimum_rows=3)
        selected_scores = np.asarray(arrays['scores'][indices])
        selected_languages = arrays['languages'][indices]
        if np.ptp(selected_scores) == 0 or any(np.ptp(selected_scores[selected_languages == language]) == 0
                                               for language in set(selected_languages.tolist())):
            raise ValueError('STS development labels must vary in every selected language')
        sts_data[name] = (arrays, indices)
        sts_manifests[name], sts_directories[name] = manifest, folder
        sts_receipts[name] = {'manifest_sha256': digest, 'source_identity': manifest['source_identity'],
                             'selected_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()}
    retrieval_data, retrieval_receipts = {}, {}
    for specification in getattr(args, 'retrieval_dev', []):
        name, separator, folder = specification.partition('=')
        if (not separator or not name or not folder or name in retrieval_data
                or not name.replace('_', '').replace('-', '').isalnum()):
            raise ValueError('--retrieval-dev must be a unique NAME=DIR')
        arrays, manifest, digest = load_retrieval_dev(folder, encoder, parent_hashes['tokenizer.json'],
            training_directories=training_directories, parent_hashes=parent_hashes)
        indices = stratified_dev_indices(arrays['languages'], args.dev_max_rows, args.seed)
        retrieval_data[name] = (arrays, indices)
        retrieval_receipts[name] = {'manifest_sha256': digest, 'source_identity': manifest['source_identity'],
            'task': manifest['task'], 'candidate_identity': manifest['candidate_identity'],
            'parent_exposure': manifest['parent_exposure'],
            'selected_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()}
    if getattr(args, 'training_scope', 'production') == 'production' and sts_data:
        if not retrieval_receipts:
            raise ValueError('Production STS requires authenticated independent candidate/parent exposure')
        first = next(iter(retrieval_receipts))
        common = retrieval_receipts[first]
        if any(item['candidate_identity'] != common['candidate_identity'] or
               item['parent_exposure'] != common['parent_exposure'] for item in retrieval_receipts.values()):
            raise ValueError('Production tasks must share authenticated candidate and parent exposure identity')
        independent_directory = next(folder for specification in args.retrieval_dev
            for name, _, folder in [specification.partition('=')] if name == first)
        for name in sts_data:
            sts_receipts[name]['candidate_parent_binding'] = bind_production_sts(
                sts_directories[name], sts_manifests[name], independent_directory, common)
    quality_plan = production_quality_plan(args, sts_data, retrieval_data, retrieval_receipts)
    receipt = {'scope': 'resolved-embedding-configuration-and-data-v1',
        'production_quality_plan': quality_plan,
        'expected_device_count': device_count, 'expected_process_count': process_count,
        'source_weights': weights, 'batch_counts': counts, 'batch_policy': args.batch_policy,
        'batch_size': args.batch_size, 'encoder_chunk_size': args.encoder_chunk_size,
        'language_exponent': args.language_exponent, 'seed': args.seed,
        'dev_max_rows': args.dev_max_rows, 'max_dev_regression': args.max_dev_regression,
        'sts_development': sts_receipts,
        'independent_retrieval_development': retrieval_receipts,
        'selected_dev_indices_sha256': {name: hashlib.sha256(indices.tobytes()).hexdigest()
                                        for name, indices in dev_indices.items()},
        'steps': args.steps, 'warmup': args.warmup, 'learning_rate': args.learning_rate,
        'temperature': args.temperature, 'weight_decay': args.weight_decay,
        'eval_every': args.eval_every, 'save_every': args.save_every,
        'keep_checkpoints': args.keep_checkpoints, 'output': args.output,
        'physical_tpu_qualified': False}
    return {'parent_hashes': parent_hashes, 'encoder_config': encoder,
        'data_items': items, 'data': data, 'weights': weights, 'counts': counts,
        'query_lengths': query_lengths, 'document_lengths': document_lengths,
        'mixture_receipt': mixture, 'dev_indices': dev_indices,
        'sts_data': sts_data, 'sts_receipts': sts_receipts,
        'retrieval_data': retrieval_data, 'retrieval_receipts': retrieval_receipts,
        'production_quality_plan': quality_plan, 'admission_receipt': receipt}


def production_quality_plan(args, sts_data, retrieval_data, retrieval_receipts):
    """Coverage over already authenticated loaded rows, never receipt booleans."""
    scope = getattr(args, 'training_scope', 'production')
    minima = {'rows': getattr(args, 'quality_min_rows', 64),
              'rows_per_language': getattr(args, 'quality_min_rows_per_language', 16),
              'unique_documents': getattr(args, 'quality_min_unique_documents', 16)}
    coverage = {}
    tasks = set()
    for name, (arrays, indices) in retrieval_data.items():
        task = retrieval_receipts[name]['task']
        tasks.add(task)
        languages, counts = np.unique(arrays['languages'][indices], return_counts=True)
        documents = len(np.unique(arrays['positive_text_ids'][indices]))
        queries = len(np.unique(arrays['query_text_ids'][indices]))
        item = {'task': task, 'rows': len(indices), 'unique_queries': queries,
                'unique_documents': documents,
                'languages': dict(zip(map(str, languages), map(int, counts), strict=True))}
        labels = arrays.get('programming_languages', np.full(len(arrays['languages']), 'unknown'))[indices]
        programming, programming_counts = np.unique(labels, return_counts=True)
        item['programming_languages'] = dict(zip(map(str, programming), map(int, programming_counts), strict=True))
        if scope == 'production' and task == 'code':
            available = set(arrays.get('programming_languages', []).tolist()) if 'programming_languages' in arrays else set()
            real = available - {'unknown', 'not-applicable', 'und', ''}
            if any(item['programming_languages'].get(language, 0) < minima['rows_per_language'] for language in real):
                raise ValueError(f'Production programming-language coverage insufficient: {name}')
        coverage[f'independent/{name}'] = item
        if scope == 'production' and (len(indices) < minima['rows'] or
                min(counts) < minima['rows_per_language'] or
                min(queries, documents) < minima['unique_documents'] or
                task in ('retrieval', 'bitext') and
                (len(languages) < 2 or set(languages.tolist()) & {'und', 'unknown', ''})):
            raise ValueError(f'Production quality coverage insufficient: {name}')
    for name, (arrays, indices) in sts_data.items():
        tasks.add('sts')
        languages, counts = np.unique(arrays['languages'][indices], return_counts=True)
        coverage[f'sts/{name}'] = {'task': 'sts', 'rows': len(indices),
                'languages': dict(zip(map(str, languages), map(int, counts), strict=True))}
        if scope == 'production' and (len(indices) < minima['rows'] or
                min(counts) < minima['rows_per_language']):
            raise ValueError(f'Production quality coverage insufficient: {name}')
    required = {'retrieval', 'bitext', 'code', 'sts'}
    if scope == 'production' and not required <= tasks:
        raise ValueError('Production requires authenticated retrieval, bitext, code and STS development coverage')
    return {'policy': 'authenticated-production-task-coverage-v1', 'scope': scope,
            'declared_minima': minima, 'coverage': coverage,
            'required_tasks': sorted(required) if scope == 'production' else [],
            'physical_quality_qualified': False}
