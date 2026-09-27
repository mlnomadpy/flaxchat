"""Evaluate saved token classifiers with audited first-subword entity metrics."""

import json
from pathlib import Path

import jax
from flax import nnx
from jax.sharding import Mesh
import numpy as np

from flaxchat.checkpoint import restore_model_from_checkpoint
from flaxchat.common import replicate_on_mesh
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import file_hash
from flaxchat.encoder_tasks import (
    EncoderTokenClassifier,
    token_classification_statistics,
)
from flaxchat.ner import ner_metrics
from flaxchat.runtime import runtime_identity
from flaxchat.training import place_host_batch
from scripts.prepare_encoder_ner import load_rows


def evaluate_ner(a, metadata):
    from scripts.finetune_encoder_classifier import source_identity

    if (
        jax.process_count() != 1
        or a.batch_size < 1
        or a.batch_size % jax.device_count()
    ):
        raise ValueError(
            "NER evaluation requires one host and a device-divisible batch"
        )
    if metadata.get("model_family") != "modernbert_token_classifier":
        raise ValueError("Require a token classifier checkpoint")
    if not a.eval_data or Path(a.report).exists():
        raise ValueError("Require evaluation data and a new report path")
    recipe = metadata["resolved_config"]
    config = EncoderConfig(**recipe["encoder"])
    names = recipe["label_names"]
    inputs, identities = [], set()
    for directory in a.eval_data:
        tokens, labels, _, data = load_rows(directory, config)
        if data["split"] == "train":
            raise ValueError("NER evaluation requires validation or test split")
        if (
            data["tokenizer_sha256"] != metadata["tokenizer_identity"]
            or data["dataset"] != metadata["dataset"]
            or data["revision"] != metadata["dataset_revision"]
            or data["label_names"] != names
            or data["alignment"] != recipe["alignment"]
        ):
            raise ValueError("NER evaluation identity differs from training")
        identity = (data["split"], data["language"])
        if identity in identities:
            raise ValueError("Duplicate NER evaluation split/language")
        identities.add(identity)
        inputs.append(
            (
                directory,
                tokens,
                labels,
                data,
                file_hash(Path(directory) / "manifest.json"),
            )
        )
    if len({split for split, _ in identities}) != 1:
        raise ValueError("Do not mix validation and test in one NER report")
    model = EncoderTokenClassifier(
        ModernBert(config, rngs=nnx.Rngs(0)), recipe["num_labels"], rngs=nnx.Rngs(1)
    )
    restore_model_from_checkpoint(model, a.output, step=metadata["step"])
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))

    @nnx.jit
    def predict(m, x, y):
        logits = m(x)
        loss, _, count = token_classification_statistics(
            logits, y, token_mask=x != config.pad_token_id
        )
        return logits.argmax(axis=-1), loss, count

    datasets, references, predictions, languages = [], [], [], []
    for directory, tokens, labels, data, manifest_hash in inputs:
        gold_rows, predicted_rows = [], []
        total_loss, words = 0.0, 0
        for start in range(0, len(tokens), a.batch_size):
            n = min(a.batch_size, len(tokens) - start)
            x = np.full((a.batch_size, tokens.shape[1]), config.pad_token_id, np.int32)
            y = np.full(x.shape, -100, np.int32)
            x[:n], y[:n] = tokens[start : start + n], labels[start : start + n]
            guessed, loss, count = predict(
                model, place_host_batch(x, mesh), place_host_batch(y, mesh)
            )
            count, loss = int(count), float(loss)
            if not np.isfinite(loss) or count != int((y >= 0).sum()) or count < 1:
                raise ValueError("Invalid NER evaluation batch")
            guessed = np.asarray(guessed)
            for gold, pred in zip(y[:n], guessed[:n], strict=True):
                first = gold >= 0
                gold_rows.append([names[int(v)] for v in gold[first]])
                predicted_rows.append([names[int(v)] for v in pred[first]])
            total_loss += loss * count
            words += count
        if words != data["words"] or len(gold_rows) != data["rows"]:
            raise ValueError("Incomplete NER evaluation coverage")
        if file_hash(Path(directory) / "manifest.json") != manifest_hash:
            raise ValueError("NER manifest changed during evaluation")
        # Revalidate array hashes after inference, before publishing any score.
        load_rows(directory, config)
        row_languages = [data["language"]] * len(gold_rows)
        datasets.append(
            dict(
                data_manifest=data,
                data_manifest_sha256=manifest_hash,
                loss=total_loss / words,
                metrics=ner_metrics(gold_rows, predicted_rows, row_languages),
                references=gold_rows,
                predictions=predicted_rows,
            )
        )
        references.extend(gold_rows)
        predictions.extend(predicted_rows)
        languages.extend(row_languages)
    report = dict(
        scope="end_to_end_encoder_token_classification",
        production_quality_qualified=False,
        checkpoint=a.output,
        checkpoint_step=metadata["step"],
        training_metadata=metadata,
        evaluator_source_sha256=source_identity(),
        runtime=runtime_identity(),
        metrics=ner_metrics(references, predictions, languages),
        datasets=datasets,
    )
    # Exclusive publication prevents overwriting an already observed result.
    with Path(a.report).open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    return report
