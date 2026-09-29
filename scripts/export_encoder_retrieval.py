"""Export retrieval embeddings from a pinned harness checkpoint with weight identity."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from flaxchat.checkpoint import load_checkpoint_metadata, restore_model_from_checkpoint
from flaxchat.encoder import EncoderConfig, ModernBert
from flaxchat.encoder_data import file_hash, load_prepared_rows


def parameter_digest(model):
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(nnx.state(model))[0]:
        array = np.asarray(leaf)
        digest.update(
            json.dumps(
                [jax.tree_util.keystr(path), str(array.dtype), array.shape]
            ).encode()
        )
        flat = array.reshape(-1)
        for start in range(0, flat.size, 262144):
            digest.update(flat[start : start + 262144].tobytes())
    return digest.hexdigest()


class EmbeddingSession:
    """Read-only model lifetime shared by all subsets of one evaluation campaign.

    Keep the private model unchanged. A single compiled callable is reused for
    identical batch/sequence shapes, including padded final batches.
    """

    def __init__(self, checkpoint, *, step=None):
        metadata = load_checkpoint_metadata(checkpoint, step=step)
        if metadata.get("model_family") not in (
            "modernbert", "modernbert_contrastive_encoder"
        ):
            raise ValueError("Encoder or contrastive encoder checkpoint required")
        config = EncoderConfig(**metadata["resolved_config"]["encoder"])
        self._check_runtime(config)
        model = ModernBert(config, rngs=nnx.Rngs(0))
        restore_model_from_checkpoint(
            model,
            checkpoint,
            step=metadata["step"],
            expected_identity={
                "resolved_config": metadata["resolved_config"],
                "tokenizer": metadata["tokenizer_identity"],
            },
        )
        self._checkpoint = checkpoint
        self._origin = dict(checkpoint=str(checkpoint), step=metadata["step"])
        self._initialize(model, metadata, config)

    @staticmethod
    def _check_runtime(config):
        if jax.process_count() != 1:
            raise ValueError("Embedding export currently requires one process")
        if config.attention_backend != "xla":
            raise ValueError("Portable export requires XLA attention")

    @classmethod
    def from_pretrained(cls, snapshot, *, config):
        """Load released weights under an explicitly matched inference config."""
        from scripts.train_encoder import load_pretrained, pretrained_inventory

        cls._check_runtime(config)
        snapshot = Path(snapshot)
        before = {
            name: file_hash(snapshot / name)
            for name in ("config.json", "tokenizer.json")
        }
        raw_config = json.loads((snapshot / "config.json").read_text())
        # Only runtime/loss settings may differ from the released architecture.
        runtime_keys = (
            "compute_dtype",
            "residual_dtype",
            "use_remat",
            "attention_backend",
            "loss_chunk_size",
            "mlm_projection",
            "mlm_loss_backend",
            "mlm_vocab_tile",
        )
        expected = EncoderConfig.from_hf(
            raw_config, **{k: getattr(config, k) for k in runtime_keys}
        )
        if expected != config:
            raise ValueError("Released encoder architecture mismatch")
        tokenizer = snapshot / "tokenizer.json"
        inventory = pretrained_inventory(snapshot, config)
        tokens = json.loads(tokenizer.read_text())
        special = sorted(
            {config.pad_token_id, config.mask_token_id}
            | {t["id"] for t in tokens.get("added_tokens", []) if t.get("special")}
        )
        model = ModernBert(config, rngs=nnx.Rngs(0))
        if load_pretrained(model, snapshot) != inventory or any(
            file_hash(snapshot / name) != digest for name, digest in before.items()
        ):
            raise ValueError("Released snapshot changed during loading")
        self = cls.__new__(cls)
        self._checkpoint = None
        self._origin = dict(released_weights=inventory, snapshot_sha256=before)
        metadata = dict(
            model_family="modernbert",
            step=None,
            tokenizer_identity=before["tokenizer.json"],
            resolved_config=dict(encoder=asdict(config), special_token_ids=special),
        )
        self._initialize(model, metadata, config)
        return self

    def _initialize(self, model, metadata, config):
        self._metadata, self._config = metadata, config
        weights = parameter_digest(model)
        pooling = "mean of nonpadding token states, including special tokens"
        source_root = Path(__file__).resolve().parents[1]
        sources = {
            str(p.relative_to(source_root)): file_hash(p)
            for p in (
                Path(__file__).resolve(),
                source_root / "flaxchat/encoder.py",
                source_root / "flaxchat/yat.py",
                source_root / "flaxchat/checkpoint.py",
                source_root / "scripts/train_encoder.py",
            )
        }
        model_identity = hashlib.sha256(
            json.dumps(
                dict(
                    parameter_sha256=weights,
                    encoder=metadata["resolved_config"]["encoder"],
                    pooling=pooling,
                    source=sources,
                ),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        self._pool = nnx.jit(lambda m, x: m.pool(x))
        self._model = model
        self._weights = weights
        self._pooling = pooling
        self._sources = sources
        self._model_identity = model_identity

    def export(self, queries, corpus, judgments, output, *, batch_size=8):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Positive batch size required")
        if jax.process_count() != 1:
            raise ValueError("Embedding export currently requires one process")
        output = Path(output)
        if output.exists():
            raise ValueError("Refusing existing output")
        judgments = Path(judgments)
        judgment_hash = file_hash(judgments)
        task = json.loads(judgments.read_text())
        if task.get("split") not in ("validation", "test") or not all(
            isinstance(task.get(k), str) and task[k]
            for k in ("dataset", "revision", "tokenization_policy")
        ):
            raise ValueError(
                "Pinned held-out retrieval task and tokenization policy required"
            )
        metadata, config = self._metadata, self._config
        model, pool = self._model, self._pool
        weights, pooling = self._weights, self._pooling
        sources, model_identity = self._sources, self._model_identity
        checkpoint = self._checkpoint
        arrays = []
        identities = {str(judgments): judgment_hash}
        for directory, key in (
            (Path(queries), "query_ids"),
            (Path(corpus), "document_ids"),
        ):
            rows, manifest = load_prepared_rows(directory, config)
            # Contrastive checkpoints pin the tokenizer hash but do not repeat
            # its special-token list in resolved_config.
            if (
                manifest.get("split") != task["split"]
                or manifest.get("dataset") != task["dataset"]
                or manifest.get("revision") != task["revision"]
                or manifest["tokenizer_sha256"] != metadata["tokenizer_identity"]
                or (
                    "special_token_ids" in metadata["resolved_config"]
                    and manifest["special_token_ids"]
                    != metadata["resolved_config"]["special_token_ids"]
                )
            ):
                raise ValueError("Retrieval data/checkpoint identity mismatch")
            ids = task.get(key)
            if (
                not isinstance(ids, list)
                or len(ids) != len(rows)
                or any(not isinstance(i, str) or not i for i in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("One unique ID per input row required")
            for start in range(0, len(rows), 1024):
                if np.any(
                    np.all(rows[start : start + 1024] == config.pad_token_id, axis=1)
                ):
                    raise ValueError("Empty retrieval input row")
            for name in ("manifest.json", "tokens.npy"):
                identities[str(directory / name)] = file_hash(directory / name)
            arrays.append(rows)
        qrels, languages = task.get("qrels", {}), task.get("languages", {})
        documents = set(task["document_ids"])
        if set(qrels) != set(task["query_ids"]) or set(languages) != set(qrels):
            raise ValueError("Every query needs judgments and a language")
        for query, grades in qrels.items():
            if (
                not grades
                or not set(grades) <= documents
                or any(type(g) is not int or g < 0 for g in grades.values())
                or not any(g > 0 for g in grades.values())
                or not isinstance(languages[query], str)
                or not languages[query]
            ):
                raise ValueError("Invalid retrieval judgments or language")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=output.name + "-", dir=output.parent))
        try:
            for name, rows in zip(("queries", "corpus"), arrays, strict=True):
                destination = np.lib.format.open_memmap(
                    staging / f"{name}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(rows), config.hidden_size),
                )
                for start in range(0, len(rows), batch_size):
                    stop = min(len(rows), start + batch_size)
                    batch = np.pad(
                        rows[start:stop],
                        ((0, batch_size - (stop - start)), (0, 0)),
                        constant_values=config.pad_token_id,
                    )
                    values = np.asarray(
                        pool(model, jnp.asarray(batch)), dtype=np.float32
                    )[: stop - start]
                    if not np.isfinite(values).all() or np.any(
                        np.all(values == 0, axis=1)
                    ):
                        raise ValueError("Invalid exported embedding")
                    destination[start:stop] = values
                destination.flush()
                del destination
            if any(file_hash(path) != digest for path, digest in identities.items()):
                raise ValueError("Retrieval input changed during export")
            result = task | dict(
                format="flaxchat-retrieval-embeddings-v1",
                model_identity="sha256:" + model_identity,
                parameter_sha256=weights,
                encoder_config=metadata["resolved_config"]["encoder"],
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                origin=self._origin,
                checkpoint_step=metadata["step"],
                pooling=pooling,
                tokenizer_sha256=metadata["tokenizer_identity"],
                input_sha256=identities,
                queries_sha256=file_hash(staging / "queries.npy"),
                corpus_sha256=file_hash(staging / "corpus.npy"),
                export_source_sha256=sources,
                quality_qualified=False,
            )
            (staging / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
            staging.rename(output)
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def export(checkpoint, queries, corpus, judgments, output, *, batch_size=8):
    """Compatibility wrapper for exporting one subset from a checkpoint."""
    return EmbeddingSession(checkpoint).export(
        queries, corpus, judgments, output, batch_size=batch_size
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "queries", "corpus", "judgments", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    result = export(
        args.checkpoint,
        args.queries,
        args.corpus,
        args.judgments,
        args.output,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "model_identity": result["model_identity"],
                "checkpoint_step": result["checkpoint_step"],
            }
        )
    )


if __name__ == "__main__":
    main()
