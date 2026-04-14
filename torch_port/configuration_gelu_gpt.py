"""HuggingFace-compatible config for the flaxchat GELU GPT."""
from transformers import PretrainedConfig


class GeluGPTConfig(PretrainedConfig):
    model_type = "gelu_gpt"

    def __init__(
        self,
        sequence_len: int = 1024,
        vocab_size: int = 32768,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_head: int = 12,
        n_embd: int = 768,
        window_pattern: str = "SSSL",
        tie_embeddings: bool = True,
        rope_base: float = 100000.0,
        pad_vocab_size_to: int = 64,
        mlp: str = "gelu",
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        self.window_pattern = window_pattern
        self.tie_embeddings = tie_embeddings
        self.rope_base = rope_base
        self.pad_vocab_size_to = pad_vocab_size_to
        self.mlp = mlp
        self.max_position_embeddings = sequence_len * 10
        # HF generation utilities probe these standard attribute names.
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.hidden_size = n_embd
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
