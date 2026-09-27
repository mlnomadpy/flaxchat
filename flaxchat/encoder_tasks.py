"""Downstream encoder task adapters with explicit masking and normalization."""

from flax import nnx
import jax.numpy as jnp
import optax

from flaxchat.encoder import ModernBert


class EncoderClassifier(nnx.Module):
    """ModernBERT mean-pooling + pretrained prediction head + new classifier.

    Mirrors Transformers' mean-pooling classifier with zero classifier dropout.
    Negative labels are evaluation padding, excluded from loss and accuracy.
    """

    def __init__(self, encoder: ModernBert, num_labels: int, *, rngs):
        if type(num_labels) is not int or num_labels < 2:
            raise ValueError("Classification requires at least two labels")
        self.encoder = encoder
        self.num_labels = num_labels
        self.classifier = nnx.Linear(
            encoder.config.hidden_size,
            num_labels,
            dtype=jnp.float32,
            kernel_init=nnx.initializers.normal(0.02),
            rngs=rngs,
        )

    def __call__(self, tokens):
        return self.classifier(
            self.encoder.prediction_features(self.encoder.pool(tokens))
        )


def classification_statistics(logits, labels):
    if labels.ndim != 1 or logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("Expected logits [batch, classes] and labels [batch]")
    valid = labels >= 0
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), jnp.maximum(labels, 0)
    )
    count = valid.sum()
    total = jnp.where(valid, losses, 0).sum()
    correct = (valid & (logits.argmax(axis=-1) == labels)).sum()
    return total / jnp.maximum(count, 1), correct, count


class EncoderTokenClassifier(nnx.Module):
    """Per-token linear classifier over encoder hidden states, without dropout.

    Dataset preparation must mark ignored special/subword positions with negative
    labels. Padding is independently excluded by the token mask in statistics.
    This adapter does not define word alignment, BIO decoding, or span F1.
    """

    def __init__(self, encoder: ModernBert, num_labels: int, *, rngs):
        if type(num_labels) is not int or num_labels < 2:
            raise ValueError('Token classification requires at least two labels')
        self.encoder = encoder
        self.num_labels = num_labels
        self.classifier = nnx.Linear(
            encoder.config.hidden_size, num_labels, dtype=jnp.float32,
            kernel_init=nnx.initializers.normal(0.02), rngs=rngs)

    def __call__(self, tokens):
        logits = self.classifier(self.encoder.encode(tokens).astype(jnp.float32))
        return jnp.where((tokens != self.encoder.config.pad_token_id)[..., None], logits, 0)

    def statistics(self, tokens, labels):
        return token_classification_statistics(
            self(tokens), labels, token_mask=tokens != self.encoder.config.pad_token_id)


def token_classification_statistics(logits, labels, *, token_mask):
    """Token-weighted mean loss and counts; masked labels receive zero gradient.

    Invalid active class IDs return a nonfinite loss so the existing finite-update
    gate rejects them, rather than silently treating them as a different class.
    All-ignored batches return zero loss/counts. Aggregate by token counts across
    batches; token accuracy does not replace entity-level span F1 for NER.
    """
    if (logits.ndim != 3 or labels.shape != logits.shape[:-1]
            or token_mask.shape != labels.shape or logits.shape[-1] < 2):
        raise ValueError('Expected logits [batch, sequence, classes], matching labels and mask')
    if not jnp.issubdtype(labels.dtype, jnp.integer) or token_mask.dtype != jnp.bool_:
        raise ValueError('Token labels must be integers and token mask must be boolean')
    valid = (labels >= 0) & token_mask
    safe = jnp.clip(labels, 0, logits.shape[-1] - 1)
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), safe)
    count = valid.sum()
    total = jnp.where(valid, losses, 0).sum()
    loss = total / jnp.maximum(count, 1)
    loss = jnp.where(jnp.any(valid & (labels >= logits.shape[-1])), jnp.nan, loss)
    correct = (valid & (logits.argmax(axis=-1) == labels)).sum()
    return loss, correct, count
