"""
BPE Tokenizer — direct port of nanochat's tokenizer.

Three backends:
1) HuggingFace Tokenizer (training + inference)
2) rustbpe + tiktoken (fast training + efficient inference)
3) ByT5-compatible raw UTF-8 bytes (training-free)
"""

import os
import copy
import codecs
import json
import pickle
from functools import lru_cache

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

BYT5_RESERVED_TOKENS = ("<pad>", "</s>", "<unk>")
BYTE_TOKENIZER_FILENAME = "byte_tokenizer.json"


def _render_conversation(tokenizer, conversation, max_tokens=2048):
    """Backend-independent chat rendering with assistant-token supervision."""
    ids, mask = [], []

    def add_tokens(token_ids, mask_value):
        values = [token_ids] if isinstance(token_ids, int) else token_ids
        ids.extend(values)
        mask.extend([mask_value] * len(values))

    messages = copy.deepcopy(conversation["messages"])
    if messages and messages[0]["role"] == "system":
        if len(messages) < 2 or messages[1]["role"] != "user":
            raise ValueError("A system message must be followed by a user message")
        messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
        messages = messages[1:]
    if not messages:
        raise ValueError("Conversation must contain at least one message")

    add_tokens(tokenizer.get_bos_token_id(), 0)
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role:
            raise ValueError(f"Expected {expected_role} message at index {index}")
        content = message["content"]
        if expected_role == "user":
            if not isinstance(content, str):
                raise TypeError("User message content must be text")
            add_tokens(tokenizer.encode_special("<|user_start|>"), 0)
            add_tokens(tokenizer.encode(content), 0)
            add_tokens(tokenizer.encode_special("<|user_end|>"), 0)
            continue

        add_tokens(tokenizer.encode_special("<|assistant_start|>"), 0)
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            part_type = part["type"]
            value_ids = tokenizer.encode(part["text"])
            if part_type == "text":
                add_tokens(value_ids, 1)
            elif part_type == "python":
                add_tokens(tokenizer.encode_special("<|python_start|>"), 1)
                add_tokens(value_ids, 1)
                add_tokens(tokenizer.encode_special("<|python_end|>"), 1)
            elif part_type == "python_output":
                add_tokens(tokenizer.encode_special("<|output_start|>"), 0)
                add_tokens(value_ids, 0)
                add_tokens(tokenizer.encode_special("<|output_end|>"), 0)
            else:
                raise ValueError(f"Unsupported assistant content type: {part_type!r}")
        add_tokens(tokenizer.encode_special("<|assistant_end|>"), 1)
    return ids[:max_tokens], mask[:max_tokens]


def _render_for_completion(tokenizer, conversation):
    conversation = copy.deepcopy(conversation)
    messages = conversation["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Completion conversations must end with an assistant message")
    messages.pop()
    ids, _ = _render_conversation(tokenizer, conversation)
    ids.append(tokenizer.encode_special("<|assistant_start|>"))
    return ids


# ---------------------------------------------------------------------------
# HuggingFace Tokenizer wrapper
# ---------------------------------------------------------------------------
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size, show_progress=True, min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        return [w.content for w in special_tokens_map.values()]

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        assert bos is not None, "Failed to find BOS token"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        return _render_conversation(self, conversation, max_tokens)

    def render_for_completion(self, conversation):
        return _render_for_completion(self, conversation)


# ---------------------------------------------------------------------------
# ByT5 / MrT5-compatible byte tokenizer
# ---------------------------------------------------------------------------
class ByteTokenizer:
    """Training-free UTF-8 byte tokenizer with ByT5-compatible byte IDs.

    ByT5 reserves IDs 0, 1, and 2 for padding, end-of-sequence, and unknown,
    respectively, then maps every byte ``b`` to ``b + 3``. Flaxchat's chat
    control tokens follow the 256 byte IDs. This provides the raw-byte input
    contract used by ByT5 and MrT5; MrT5's learned deletion gate belongs in
    the model architecture and is deliberately not represented here.
    """

    FORMAT_VERSION = 1
    BYTE_OFFSET = len(BYT5_RESERVED_TOKENS)
    BYTE_VOCAB_SIZE = 256

    def __init__(self, special_tokens=None):
        self._special_tokens = tuple(special_tokens or SPECIAL_TOKENS)
        if len(set(self._special_tokens)) != len(self._special_tokens):
            raise ValueError("Byte tokenizer special tokens must be unique")
        reserved = set(BYT5_RESERVED_TOKENS)
        overlap = reserved.intersection(self._special_tokens)
        if overlap:
            raise ValueError(
                f"Byte tokenizer special tokens overlap reserved tokens: {sorted(overlap)}"
            )
        special_offset = self.BYTE_OFFSET + self.BYTE_VOCAB_SIZE
        self._special_to_id = {
            token: special_offset + index
            for index, token in enumerate(self._special_tokens)
        }
        self._id_to_special = {
            token_id: token for token, token_id in self._special_to_id.items()
        }
        self._reserved_to_id = {
            token: index for index, token in enumerate(BYT5_RESERVED_TOKENS)
        }
        self._id_to_reserved = {
            token_id: token for token, token_id in self._reserved_to_id.items()
        }

    @classmethod
    def from_directory(cls, tokenizer_dir):
        config_path = os.path.join(tokenizer_dir, BYTE_TOKENIZER_FILENAME)
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("type") != "byt5_utf8_bytes":
            raise ValueError(f"Unsupported byte tokenizer type: {config.get('type')!r}")
        if config.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported byte tokenizer format version: "
                f"{config.get('format_version')!r}"
            )
        if config.get("byte_offset") != cls.BYTE_OFFSET:
            raise ValueError("Byte tokenizer offset is not ByT5-compatible")
        special_tokens = config.get("special_tokens")
        if not isinstance(special_tokens, list) or not all(
            isinstance(token, str) for token in special_tokens
        ):
            raise ValueError("Byte tokenizer special_tokens must be a list of strings")
        return cls(special_tokens=special_tokens)

    def get_vocab_size(self):
        return self.BYTE_OFFSET + self.BYTE_VOCAB_SIZE + len(self._special_tokens)

    def get_special_tokens(self):
        return [*BYT5_RESERVED_TOKENS, *self._special_tokens]

    def id_to_token(self, token_id):
        if token_id in self._id_to_reserved:
            return self._id_to_reserved[token_id]
        if token_id in self._id_to_special:
            return self._id_to_special[token_id]
        byte_value = token_id - self.BYTE_OFFSET
        if 0 <= byte_value < self.BYTE_VOCAB_SIZE:
            return chr(byte_value)
        raise ValueError(f"Token ID out of range: {token_id}")

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        if text in self._reserved_to_id:
            return self._reserved_to_id[text]
        try:
            return self._special_to_id[text]
        except KeyError as exc:
            raise ValueError(f"Unknown special token: {text!r}") from exc

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        del num_threads
        if not isinstance(text, str):
            raise TypeError("Byte tokenizer input must be text")
        ids = [byte + self.BYTE_OFFSET for byte in text.encode("utf-8")]
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.insert(0, prepend_id)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        if isinstance(text, list):
            return [self._encode_one(item, *args, **kwargs) for item in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        pieces = []
        byte_buffer = bytearray()

        def flush_bytes():
            if byte_buffer:
                pieces.append(bytes(byte_buffer).decode("utf-8", errors="ignore"))
                byte_buffer.clear()

        for token_id in ids:
            byte_value = token_id - self.BYTE_OFFSET
            if 0 <= byte_value < self.BYTE_VOCAB_SIZE:
                byte_buffer.append(byte_value)
                continue
            flush_bytes()
            pieces.append(self.id_to_token(token_id))
        flush_bytes()
        return "".join(pieces)

    def decode_stream(self, ids):
        """Incrementally decode token IDs without dropping split UTF-8 characters."""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        for token_id in ids:
            byte_value = token_id - self.BYTE_OFFSET
            if 0 <= byte_value < self.BYTE_VOCAB_SIZE:
                yield decoder.decode(bytes([byte_value]), final=False)
                continue
            remainder = decoder.decode(b"", final=True)
            if remainder:
                yield remainder
            yield self.id_to_token(token_id)
            decoder.reset()
        remainder = decoder.decode(b"", final=True)
        if remainder:
            yield remainder

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        config_path = os.path.join(tokenizer_dir, BYTE_TOKENIZER_FILENAME)
        payload = {
            "type": "byt5_utf8_bytes",
            "format_version": self.FORMAT_VERSION,
            "byte_offset": self.BYTE_OFFSET,
            "special_tokens": list(self._special_tokens),
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"Saved byte tokenizer config to {config_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        return _render_conversation(self, conversation, max_tokens)

    def render_for_completion(self, conversation):
        return _render_for_completion(self, conversation)


# ---------------------------------------------------------------------------
# rustbpe + tiktoken wrapper
# ---------------------------------------------------------------------------
import rustbpe
import tiktoken


class RustBPETokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe", pat_str=pattern,
            mergeable_ranks=mergeable_ranks, special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """Tokenize a chat conversation, returning (ids, mask)."""
        return _render_conversation(self, conversation, max_tokens)

    def render_for_completion(self, conversation):
        """Render conversation priming the assistant for RL completion."""
        return _render_for_completion(self, conversation)


# ---------------------------------------------------------------------------
# Convenience: load a tokenizer backend from a directory
# ---------------------------------------------------------------------------
def load_tokenizer(tokenizer_dir):
    """Load a tokenizer by its backend-specific persisted artifact."""
    if os.path.isfile(os.path.join(tokenizer_dir, BYTE_TOKENIZER_FILENAME)):
        return ByteTokenizer.from_directory(tokenizer_dir)
    if os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.json")):
        return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    if os.path.isfile(os.path.join(tokenizer_dir, "tokenizer.pkl")):
        return RustBPETokenizer.from_directory(tokenizer_dir)
    raise FileNotFoundError(f"No supported tokenizer artifact in {tokenizer_dir}")


def tokenizer_artifact_path(tokenizer_dir):
    """Return the concrete persisted file used by a supported tokenizer."""
    for filename in (BYTE_TOKENIZER_FILENAME, "tokenizer.json", "tokenizer.pkl"):
        path = os.path.join(tokenizer_dir, filename)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No supported tokenizer artifact in {tokenizer_dir}")


def get_tokenizer():
    from flaxchat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return load_tokenizer(tokenizer_dir)
