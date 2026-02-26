from dataclasses import dataclass


@dataclass
class RT1Config:
    # === Image ===
    img_size: int = 300
    num_images: int = 6  # history length (6 frames)

    # === Language ===
    lang_model: str = "all-MiniLM-L6-v2"  # sentence-transformers model
    lang_raw_dim: int = 384  # all-MiniLM-L6-v2 output dim
    lang_embed_dim: int = 512  # post-projection dim (matches original USE)
    freeze_lang_encoder: bool = True

    # === FiLM EfficientNet ===
    efficientnet_model: str = "efficientnet_b3"
    use_pretrained: bool = True
    num_film_layers: int = 26  # number of MBConv blocks in EfficientNet-B3

    # === TokenLearner ===
    num_learned_tokens: int = 8  # 81 tokens -> 8 tokens
    token_learner_hidden_dim: int = 64

    # === Transformer ===
    token_dim: int = 512
    num_layers: int = 8  # self-attention layers
    num_heads: int = 8
    d_ff: int = 1024  # feedforward dimension
    dropout: float = 0.1

    # === Action ===
    num_action_bins: int = 256  # discretize each dimension into 256 bins
    action_dims: int = 11  # arm(7) + base(3) + mode(1)

    @property
    def max_seq_len(self) -> int:
        """Transformer input sequence length: num_images * num_learned_tokens."""
        return self.num_images * self.num_learned_tokens  # 6 * 8 = 48
