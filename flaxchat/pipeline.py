"""Reproducible tokenizer→pretrain→SFT→RL→eval→inference pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import optax

from flaxchat.checkpoint import create_checkpoint_manager, save_checkpoint
from flaxchat.common import compute_init, replicate_on_mesh
from flaxchat.engine import generate_with_cache
from flaxchat.gpt import GPT, attention_backend_metadata
from flaxchat.config import GPTConfig
from flaxchat.tokenizer import HuggingFaceTokenizer


TINYSTORIES_DATASET = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_LICENSE = "cdla-sharing-1.0"
PIPELINE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PipelineConfig:
    sequence_length: int = 128
    embedding_dim: int = 32
    layers: int = 1
    heads: int = 2
    vocab_size: int = 320
    batch_size: int = 2
    pretrain_steps: int = 2
    sft_steps: int = 1
    rl_steps: int = 1
    learning_rate: float = 3e-4
    seed: int = 42
    max_new_tokens: int = 8

    def __post_init__(self):
        if self.sequence_length <= 0 or self.embedding_dim < 24:
            raise ValueError("sequence_length must be positive and embedding_dim must be at least 24")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim must be divisible by heads")
        if min(self.pretrain_steps, self.sft_steps, self.rl_steps) < 1:
            raise ValueError("every training stage must run at least one update")


@nnx.jit
def _language_model_step(model, optimizer, inputs, targets):
    loss, gradients = nnx.value_and_grad(lambda current: current(inputs, targets))(model)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def _preference_step(model, optimizer, inputs, targets, advantages):
    def loss_fn(current):
        logits = current(inputs)
        safe_targets = jnp.maximum(targets, 0)
        token_log_probs = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), safe_targets[..., None], axis=-1
        )[..., 0]
        valid = (targets >= 0).astype(jnp.float32)
        objective = jnp.sum(token_log_probs * advantages[:, None] * valid)
        return -objective / jnp.maximum(jnp.sum(valid), 1.0)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_tinystories(max_train_stories: int, max_validation_stories: int):
    """Load deterministic prefixes of the immutable TinyStories revision."""
    from datasets import load_dataset

    train = load_dataset(
        TINYSTORIES_DATASET,
        split="train",
        revision=TINYSTORIES_REVISION,
        streaming=True,
    )
    validation = load_dataset(
        TINYSTORIES_DATASET,
        split="validation",
        revision=TINYSTORIES_REVISION,
        streaming=True,
    )
    return (
        [record["text"] for record in train.take(max_train_stories)],
        [record["text"] for record in validation.take(max_validation_stories)],
    )


def load_fixture_stories(path: str | Path) -> tuple[list[str], list[str]]:
    """Create deterministic story records from a committed offline corpus."""
    text = Path(path).read_text(encoding="utf-8")
    records = [line.strip() for line in text.splitlines() if line.strip()]
    if len(records) < 4:
        records = [part.strip() for part in text.split(".") if part.strip()]
    if len(records) < 2:
        raise ValueError("Offline corpus must contain at least two non-empty records")
    split = max(1, len(records) // 5)
    return records[split:], records[:split]


def _token_stream(tokenizer, stories):
    bos = tokenizer.get_bos_token_id()
    stream = []
    for story in stories:
        stream.extend(tokenizer.encode(story, prepend=bos))
    return np.asarray(stream, dtype=np.int32)


def _sample_batch(tokens, batch_size, sequence_length, step):
    required = sequence_length + 1
    if len(tokens) < required:
        repeats = (required + len(tokens) - 1) // len(tokens)
        tokens = np.tile(tokens, repeats)
    maximum = len(tokens) - required
    starts = [((step * batch_size + index) * sequence_length) % (maximum + 1)
              for index in range(batch_size)]
    rows = np.stack([tokens[start:start + required] for start in starts])
    return rows[:, :-1], rows[:, 1:]


def _conversation_batch(tokenizer, stories, batch_size, sequence_length):
    inputs, targets = [], []
    for index in range(batch_size):
        story = stories[index % len(stories)]
        conversation = {"messages": [
            {"role": "user", "content": "Story:"},
            {"role": "assistant", "content": story},
        ]}
        ids, mask = tokenizer.render_conversation(
            conversation, max_tokens=sequence_length + 1
        )
        padded = np.full(sequence_length + 1, tokenizer.get_bos_token_id(), np.int32)
        supervised = np.zeros(sequence_length + 1, np.int32)
        length = min(len(ids), sequence_length + 1)
        padded[:length] = ids[:length]
        supervised[:length] = mask[:length]
        target = padded[1:].copy()
        target[supervised[1:] == 0] = -1
        if not np.any(target >= 0):
            raise ValueError(
                "sequence_length is too short to contain an SFT assistant target"
            )
        inputs.append(padded[:-1])
        targets.append(target)
    return np.stack(inputs), np.stack(targets)


def _preference_batch(tokenizer, story, batch_size, sequence_length):
    responses = [story, "The story stopped before anything happened."]
    advantages = np.empty(batch_size, dtype=np.float32)
    inputs, targets = [], []
    for index in range(batch_size):
        response_index = index % 2
        conversation = {"messages": [
            {"role": "user", "content": "Story:"},
            {"role": "assistant", "content": responses[response_index]},
        ]}
        ids, mask = tokenizer.render_conversation(
            conversation, max_tokens=sequence_length + 1
        )
        padded = np.full(sequence_length + 1, tokenizer.get_bos_token_id(), np.int32)
        supervised = np.zeros(sequence_length + 1, np.int32)
        length = min(len(ids), sequence_length + 1)
        padded[:length] = ids[:length]
        supervised[:length] = mask[:length]
        target = padded[1:].copy()
        target[supervised[1:] == 0] = -1
        if not np.any(target >= 0):
            raise ValueError(
                "sequence_length is too short to contain a preference target"
            )
        inputs.append(padded[:-1])
        targets.append(target)
        advantages[index] = 0.5 if response_index == 0 else -0.5
    advantages -= advantages.mean()
    return np.stack(inputs), np.stack(targets), advantages


def run_pipeline(
    train_stories: list[str],
    validation_stories: list[str],
    output_dir: str | Path,
    config: PipelineConfig = PipelineConfig(),
    *,
    dataset_identity: dict | None = None,
) -> dict:
    """Run every public training stage and emit a self-auditing artifact bundle."""
    if not train_stories or not validation_stories:
        raise ValueError("Training and validation story sets must both be non-empty")
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mesh = compute_init()
    device_count = jax.device_count()
    batch_size = max(config.batch_size, device_count)
    batch_size = ((batch_size + device_count - 1) // device_count) * device_count
    data_sharding = NamedSharding(mesh, P("data"))

    tokenizer = HuggingFaceTokenizer.train_from_iterator(
        iter(train_stories), vocab_size=config.vocab_size
    )
    tokenizer_dir = output / "tokenizer"
    tokenizer.save(str(tokenizer_dir))
    tokenizer_hash = _sha256_file(tokenizer_dir / "tokenizer.json")

    model_config = GPTConfig(
        sequence_len=config.sequence_length,
        vocab_size=tokenizer.get_vocab_size(),
        n_layer=config.layers,
        n_head=config.heads,
        n_kv_head=max(1, config.heads // 2),
        n_embd=config.embedding_dim,
        window_pattern="L",
        attention_backend="auto",
    )
    model = GPT(model_config, rngs=nnx.Rngs(config.seed))
    nnx.update(model, replicate_on_mesh(nnx.state(model), mesh))
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        config.learning_rate,
        warmup_steps=1,
        decay_steps=config.pretrain_steps + config.sft_steps + config.rl_steps,
        end_value=config.learning_rate * 0.1,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=0.01),
        wrt=nnx.Param,
    )

    train_tokens = _token_stream(tokenizer, train_stories)
    validation_tokens = _token_stream(tokenizer, validation_stories)
    pretrain_losses: list[float] = []
    sft_losses: list[float] = []
    rl_losses: list[float] = []
    for step in range(config.pretrain_steps):
        inputs, targets = _sample_batch(
            train_tokens, batch_size, config.sequence_length, step
        )
        loss = _language_model_step(
            model,
            optimizer,
            jax.device_put(inputs, data_sharding),
            jax.device_put(targets, data_sharding),
        )
        pretrain_losses.append(float(loss))

    for _ in range(config.sft_steps):
        inputs, targets = _conversation_batch(
            tokenizer, train_stories, batch_size, config.sequence_length
        )
        loss = _language_model_step(
            model,
            optimizer,
            jax.device_put(inputs, data_sharding),
            jax.device_put(targets, data_sharding),
        )
        sft_losses.append(float(loss))

    for _ in range(config.rl_steps):
        inputs, targets, advantages = _preference_batch(
            tokenizer, train_stories[0], batch_size, config.sequence_length
        )
        loss = _preference_step(
            model,
            optimizer,
            jax.device_put(inputs, data_sharding),
            jax.device_put(targets, data_sharding),
            jax.device_put(advantages, NamedSharding(mesh, P("data"))),
        )
        rl_losses.append(float(loss))

    eval_inputs, eval_targets = _sample_batch(
        validation_tokens, batch_size, config.sequence_length, 0
    )
    metrics = {
        "pretrain_loss": pretrain_losses,
        "sft_loss": sft_losses,
        "rl_loss": rl_losses,
        "validation_loss": float(model(
            jax.device_put(eval_inputs, data_sharding),
            jax.device_put(eval_targets, data_sharding),
        )),
    }
    prompt = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    generated_ids = generate_with_cache(
        model,
        prompt_ids,
        max_tokens=config.max_new_tokens,
        temperature=0.0,
        seed=config.seed,
    )
    sample = tokenizer.decode(generated_ids)

    data_payload = json.dumps(
        {"train": train_stories, "validation": validation_stories},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = dataset_identity or {
        "dataset": "committed-offline-fixture",
        "revision": _sha256_bytes(data_payload),
        "license": "repository-test-fixture",
    }
    metadata = {
        "resolved_config": asdict(config),
        "model_config": asdict(model_config),
        "tokenizer_identity": tokenizer_hash,
        "data_manifest_identity": _sha256_bytes(data_payload),
        "source_revision": _source_revision(),
        "dataset": identity,
    }
    checkpoint_dir = output / "checkpoint"
    manager = create_checkpoint_manager(str(checkpoint_dir), async_checkpointing=False)
    final_step = config.pretrain_steps + config.sft_steps + config.rl_steps
    save_checkpoint(
        manager,
        final_step,
        model,
        optimizer,
        metadata,
        training_state={
            "update_step": jnp.asarray(final_step),
            "rng_key": jax.random.key_data(jax.random.key(config.seed)),
        },
    )
    manager.close()

    checkpoint_manifest = checkpoint_dir / str(final_step) / "manifest" / "metadata"

    manifest = {
        "format_version": PIPELINE_FORMAT_VERSION,
        "status": "complete",
        "stages": ["tokenizer", "pretrain", "sft", "rl", "eval", "inference"],
        "dataset": identity,
        "data_sha256": _sha256_bytes(data_payload),
        "tokenizer_sha256": tokenizer_hash,
        "checkpoint_manifest_sha256": _sha256_file(checkpoint_manifest),
        "source_revision": metadata["source_revision"],
        "resolved_config": asdict(config),
        "model_config": asdict(model_config),
        "hardware": {
            "backend": jax.default_backend(),
            "device_count": device_count,
            "device_kind": jax.devices()[0].device_kind,
            "attention": attention_backend_metadata(
                model_config.attention_backend, model_config.sequence_len
            ),
        },
        "metrics": metrics,
        "sample": {"prompt": prompt, "text": sample, "token_ids": generated_ids},
        "wall_time_seconds": time.monotonic() - started,
        "artifacts": {
            "checkpoint": "checkpoint",
            "tokenizer": "tokenizer/tokenizer.json",
        },
    }
    protocol_fields = {
        key: manifest[key]
        for key in (
            "format_version", "stages", "dataset", "data_sha256",
            "tokenizer_sha256", "checkpoint_manifest_sha256", "source_revision",
            "resolved_config", "model_config",
        )
    }
    manifest["protocol_sha256"] = _sha256_bytes(json.dumps(
        protocol_fields, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8"))
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    return manifest
