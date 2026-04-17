"""HuggingFace-compatible config for the flaxchat YatNMN-Softplus GPT."""
from transformers import PretrainedConfig


class YatGPTHfConfig(PretrainedConfig):
    model_type = "yatnmn_gpt"

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
        mlp_type: str = "yatnmn-softplus",
        scalar_bias: bool = False,
        softplus_bias: bool = True,
        learnable_epsilon: bool = True,
        epsilon_init: float = 1e-3,
        constant_alpha: bool = False,
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
        self.mlp_type = mlp_type
        self.scalar_bias = scalar_bias
        self.softplus_bias = softplus_bias
        self.learnable_epsilon = learnable_epsilon
        self.epsilon_init = epsilon_init
        self.constant_alpha = constant_alpha
        self.max_position_embeddings = sequence_len * 10
        # HF probes these standard names
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.hidden_size = n_embd
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
