"""Model-free integrity and finite numerical gates for YAT releases."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

ARTIFACTS = ("model.safetensors", "config.json", "tokenizer.json", "yat_encoder.py")
VERSION = "yat-release-parity-v4"
INTERMEDIATE_POLICY = "first-and-middle-layer-v1"
FIXTURE_POLICY = "multilingual/code/long/empty/all-padding-v2"


def sha256_identity(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def validate_case_scope(scope):
    if (not isinstance(scope, dict) or type(scope.get('rows')) is not int or scope['rows'] != 72
            or type(scope.get('sequence_length')) is not int or scope['sequence_length'] < 32
            or type(scope.get('batch_size')) is not int or scope['batch_size'] < 1
            or scope.get('retrieval_protocol') != '8-query/64-corpus-v1'
            or scope.get('fixture') != FIXTURE_POLICY
            or not sha256_identity(scope.get('inputs_sha256'))):
        raise ValueError('Complete frozen72-row parity scope required')
    metric_names(scope)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_hashes(root: Path, names=ARTIFACTS) -> dict:
    return {name: digest(root / name) for name in names}


def intermediate_scope(config):
    depth = config.get("num_hidden_layers")
    if type(depth) is not int or depth < 1:
        raise ValueError("Positive configured encoder depth required")
    return {"intermediate_policy": INTERMEDIATE_POLICY, "num_hidden_layers": depth,
            "intermediate_indices": sorted({0, (depth - 1) // 2})}


def metric_names(scope):
    expected = intermediate_scope({"num_hidden_layers": scope.get("num_hidden_layers")})
    if any(scope.get(key) != value for key, value in expected.items()):
        raise ValueError("Intermediate scope does not match configured depth/policy")
    return {"embedding", "hidden", "pool"} | {f"layer_{i}" for i in expected["intermediate_indices"]}


def validate_runtime_identity(runtime):
    required = {"runtime_packages", "interpreter", "numeric_environment", "effective_jax_config", "effective_torch_config", "deployment_receipts_sha256"}
    if not isinstance(runtime, dict) or not required <= set(runtime):
        raise ValueError("Full backend numerical runtime identity required")
    if not runtime["runtime_packages"] or not all(isinstance(runtime[key], dict) for key in required):
        raise ValueError("Invalid backend runtime identity")
    interpreter = runtime["interpreter"]
    if not all(interpreter.get(key) for key in ("version", "implementation", "build", "executable_sha256")):
        raise ValueError("Interpreter identity incomplete")
    deployment = runtime["deployment_receipts_sha256"]
    if set(deployment) != {"manifest-identity.txt", "runtime-receipt.json", "hardware-receipt.json"} or not all(deployment.values()):
        raise ValueError("Guarded deployment identity required for parity")


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_metrics(report: dict) -> None:
    if report.get("format") != VERSION:
        raise ValueError("Legacy/unbound parity receipt requires fresh TPU qualification")
    validate_case_scope(report.get('scope'))
    if report["jax_backend"] != "physical TPU" or report["torch_backend"] != "torch_xla TPU":
        raise ValueError("Physical TPU parity required")
    if set(report["metrics"]) != metric_names(report["scope"]):
        raise ValueError("Missing intermediate parity metrics")
    for name, metrics in report["metrics"].items():
        shape = metrics.get('shape')
        prefix = [72] if name == 'pool' else [72, report['scope']['sequence_length']]
        if (not isinstance(shape, list) or len(shape) != len(prefix) + 1 or shape[:-1] != prefix
                or any(type(value) is not int or value < 1 for value in shape)):
            raise ValueError(f'Missing/invalid parity tensor shape: {name}')
        for key in ("max_abs", "mean_abs", "p99_abs", "min_vector_cosine", "mean_vector_cosine"):
            if not finite(metrics.get(key)):
                raise ValueError(f"Missing/nonfinite metric {name}/{key}")
        if (not .99 <= metrics["min_vector_cosine"] <= 1.00001
                or not .99 <= metrics["mean_vector_cosine"] <= 1.00001
                or any(metrics[key] < 0 for key in ("max_abs", "mean_abs", "p99_abs"))):
            raise ValueError("Invalid cosine")
    widths = {metrics['shape'][-1] for metrics in report['metrics'].values()}
    if len(widths) != 1:
        raise ValueError('Parity tensors disagree on encoder width')
    if set(report.get('runs', {})) != {'jax', 'torch'}:
        raise ValueError('Both physical backend runs required')
    for backend, run in report['runs'].items():
        if (run.get('format') != VERSION or run.get('backend') != backend
                or run.get('scope') != report['scope']
                or run.get('inputs_sha256') != report['scope']['inputs_sha256']
                or not sha256_identity(run.get('output_sha256'))):
            raise ValueError('Complete backend scope/output identity required')
    retrieval = report["retrieval"]
    for key in ("top_1_agreement", "top_3_set_agreement", "max_cosine_score_drift"):
        if not finite(retrieval.get(key)):
            raise ValueError(f"Missing/nonfinite retrieval metric {key}")
    if (report["metrics"]["pool"]["min_vector_cosine"] < .9998
            or retrieval["top_1_agreement"] != 1. or retrieval["top_3_set_agreement"] != 1.
            or not 0 <= retrieval["max_cosine_score_drift"] <= .01):
        raise ValueError("TPU parity thresholds failed")


def validate_release(root: Path, conversion: dict, report: dict) -> None:
    artifacts = artifact_hashes(root)
    if report.get("format") != VERSION:
        raise ValueError("Unsupported release parity format")
    if conversion.get("format") != "flaxchat-yat-pytorch-v2" or conversion.get("artifacts_sha256") != artifacts:
        raise ValueError("Conversion artifact/config/implementation integrity mismatch")
    if conversion["torch_weights_sha256"] != artifacts["model.safetensors"]:
        raise ValueError("Weight identity mismatch")
    if conversion["tokenizer_sha256"] != artifacts["tokenizer.json"]:
        raise ValueError("Tokenizer identity mismatch")
    if report.get("conversion_sha256") != digest(root / "conversion.json") or report.get("artifacts_sha256") != artifacts:
        raise ValueError("Parity receipt belongs to another conversion")
    source_export = json.loads((root / "source-export.json").read_text())
    source_metadata = json.loads((root / "checkpoint-metadata.json").read_text())
    source_manifest = json.loads((root / "checkpoint-manifest.json").read_text())
    if (conversion.get("source_export_sha256") != digest(root / "source-export.json")
            or source_export.get("sha256") != conversion["source_weights_sha256"]
            or source_export.get("source_checkpoint_step") != conversion.get("source_checkpoint_step")
            or source_export.get("source_model_family") != conversion.get("source_model_family")
            or source_export.get("tensors") != conversion.get("tensor_count")
            or canonical_hash(source_metadata) != source_manifest.get("metadata_sha256")
            or canonical_hash(source_manifest.get("identity")) != source_manifest.get("identity_sha256")
            or source_manifest.get("identity") != checkpoint_identity(source_metadata)
            or source_export.get("source_checkpoint_metadata_sha256") != source_manifest["metadata_sha256"]
            or source_export.get("source_checkpoint_manifest_identity_sha256") != source_manifest["identity_sha256"]
            or source_export.get("artifacts_sha256", {}).get("config.json") != artifacts["config.json"]
            or source_export.get("artifacts_sha256", {}).get("tokenizer.json") != artifacts["tokenizer.json"]
            or source_export.get("artifacts_sha256", {}).get("checkpoint-metadata.json") != digest(root / "checkpoint-metadata.json")
            or source_export.get("artifacts_sha256", {}).get("checkpoint-manifest.json") != digest(root / "checkpoint-manifest.json")
            or source_manifest.get("step") != conversion["source_checkpoint_step"]
            or source_metadata.get("model_family") != conversion["source_model_family"]
            or source_metadata.get("resolved_config", {}).get("encoder") != json.loads((root / "config.json").read_text())):
        raise ValueError("Conversion does not match preserved source export/checkpoint lineage")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expanded TPU matrix missing")
    scope = [(case["scope"]["sequence_length"], case["scope"]["batch_size"]) for case in cases]
    maximum = json.loads((root / "config.json").read_text()).get("max_position_embeddings", 512)
    required_lengths = {128, 256, 512}
    if maximum > 512:
        required_lengths.add(maximum)
    required = {(length, batch) for length in required_lengths for batch in (1, 8)}
    if len(scope) != len(set(scope)) or not required <= set(scope):
        raise ValueError(f"Incomplete/duplicate TPU parity matrix; need {sorted(required)}")
    expected_intermediates = intermediate_scope(json.loads((root / "config.json").read_text()))
    width = json.loads((root / "config.json").read_text()).get('hidden_size')
    if type(width) is not int or width < 1:
        raise ValueError('Configured positive encoder width required')
    for case in cases:
        validate_metrics(case)
        if any(case["scope"].get(key) != value for key, value in expected_intermediates.items()):
            raise ValueError("Parity depth differs from published encoder")
        if any(metrics['shape'][-1] != width for metrics in case['metrics'].values()):
            raise ValueError('Parity tensor width differs from published encoder')
        if case.get("conversion_sha256") != report["conversion_sha256"] or case.get("artifacts_sha256") != artifacts:
            raise ValueError("Mixed artifact identities in parity matrix")
        source, target = case["runs"]["jax"], case["runs"]["torch"]
        if (source["model_sha256"]["model.safetensors"] != conversion["source_weights_sha256"]
                or source["model_sha256"]["config.json"] != artifacts["config.json"]
                or source["model_sha256"]["tokenizer.json"] != artifacts["tokenizer.json"]
                or target["model_sha256"] != artifacts
                or source["inputs_sha256"] != target["inputs_sha256"]
                or not source.get("runtime") or not target.get("runtime")
                or not source.get("devices") or not target.get("devices")
                or not source.get("source_sha256") or not target.get("source_sha256")
                or source["inputs_sha256"] != case["scope"].get("inputs_sha256")):
            raise ValueError("Source/target/input/runtime parity binding mismatch")
        for backend in ("jax", "torch"):
            validate_runtime_identity(case["runs"][backend]["runtime"])
            if any(other["runs"][backend]["runtime"] != case["runs"][backend]["runtime"]
                   or other["runs"][backend]["source_sha256"] != case["runs"][backend]["source_sha256"]
                   for other in cases):
                raise ValueError("Mixed runtime/source identities in parity matrix")
        if case["scope"].get("retrieval_protocol") != "8-query/64-corpus-v1":
            raise ValueError("Held-out query/corpus fixture absent")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def checkpoint_identity(metadata):
    return {"resolved_config": metadata.get("resolved_config", metadata.get("model_config")),
            "tokenizer": metadata.get("tokenizer_identity", "unavailable"),
            "data_manifest": metadata.get("data_manifest_identity", "unavailable"),
            "source_revision": metadata.get("source_revision", "unavailable"),
            "source_python_sha256": metadata.get("source_python_sha256", "unavailable")}


def encoder_export_manifest(metadata, manifest):
    """Derive serving leaves from authenticated lineage; exclude only objective alpha.

    Callers authenticate the complete metadata and checkpoint manifest first.
    The original manifest is retained unchanged in every released artifact.
    """
    records = manifest.get('model_state', {})
    objective = metadata.get('resolved_config', {}).get('contrastive_objective')
    alpha_paths = [name for name in records
                   if re.sub(r"\.([A-Za-z_]\w*)", r"['\1']", name)
                   == "['contrastive_raw_alpha']"]
    if objective is None:
        if alpha_paths:
            raise ValueError('Undeclared training-only contrastive alpha')
        return dict(records), []
    from types import SimpleNamespace
    from flaxchat.embedding_stage import contrastive_objective_identity
    if not isinstance(objective, dict) or objective != contrastive_objective_identity(SimpleNamespace(
            contrastive_similarity='yat', yat_infonce_alpha_init=objective.get('alpha_init'))):
        raise ValueError('Unknown contrastive objective export contract')
    if len(alpha_paths) != 1:
        raise ValueError('YAT objective requires exactly one training-only contrastive alpha')
    name = alpha_paths[0]
    record = records[name]
    if (not isinstance(record, dict) or record.get('shape') != []
            or record.get('dtype') != 'float32' or not sha256_identity(record.get('sha256'))):
        raise ValueError('Training-only contrastive alpha must be a committed FP32 scalar')
    exclusions = [{'path': name, 'record': record,
                   'reason': 'training-only-contrastive-objective; encoder-serving-does-not-use-alpha'}]
    return {key: value for key, value in records.items() if key != name}, exclusions


def validate_export(root: Path, export: dict) -> None:
    """Verify portable export against its preserved committed checkpoint lineage."""
    if export.get('source_model_family') not in {'modernbert_contrastive_encoder', 'yat_embedding_finetune'}:
        raise ValueError('Unsupported embedding source family')
    if type(export.get('source_checkpoint_step')) is not int or export['source_checkpoint_step'] < 1:
        raise ValueError('Export requires a positive committed checkpoint step')
    required = ('model.safetensors', 'config.json', 'tokenizer.json', 'checkpoint-metadata.json', 'checkpoint-manifest.json')
    if export.get('artifacts_sha256') != artifact_hashes(root, required):
        raise ValueError('Export artifact/config/lineage identity mismatch; re-export authenticated checkpoint')
    metadata = json.loads((root / 'checkpoint-metadata.json').read_text())
    manifest = json.loads((root / 'checkpoint-manifest.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    if (canonical_hash(metadata) != manifest.get('metadata_sha256')
            or canonical_hash(manifest.get('identity')) != manifest.get('identity_sha256')
            or manifest.get('identity') != checkpoint_identity(metadata)
            or manifest.get('step') != export['source_checkpoint_step']
            or metadata.get('model_family') != export['source_model_family']
            or metadata.get('resolved_config', {}).get('encoder') != config
            or export.get('source_checkpoint_metadata_sha256') != manifest['metadata_sha256']
            or export.get('source_checkpoint_manifest_identity_sha256') != manifest['identity_sha256']
            or export.get('tokenizer_identity') != metadata.get('tokenizer_identity')
            or export['tokenizer_identity'] != digest(root / 'tokenizer.json')
            or export.get('sha256') != digest(root / 'model.safetensors')
            or export.get('bytes') != (root / 'model.safetensors').stat().st_size
            or type(export.get('tensors')) is not int):
        raise ValueError('Export does not match authenticated committed checkpoint metadata')
    serving_manifest, exclusions = encoder_export_manifest(metadata, manifest)
    if (export['tensors'] != len(serving_manifest)
            or export.get('excluded_training_only_tensors', []) != exclusions):
        raise ValueError('Export training-only exclusions differ from authenticated objective')
