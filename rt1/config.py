from dataclasses import dataclass


@dataclass
class RT1Config:
    # === Image ===
    img_size: int = 300
    num_images: int = 6  # history length (6 frames)

    # === Language ===
    lang_model: str = "all-MiniLM-L6-v2"  # sentence-transformers 모델
    lang_raw_dim: int = 384  # all-MiniLM-L6-v2 출력 차원
    lang_embed_dim: int = 512  # projection 후 차원 (원본 USE와 동일)
    freeze_lang_encoder: bool = True

    # === FiLM EfficientNet ===
    efficientnet_model: str = "efficientnet_b3"
    use_pretrained: bool = True
    num_film_layers: int = 26  # EfficientNet-B3의 MBConv 블록 수

    # === TokenLearner ===
    num_learned_tokens: int = 8  # 81 tokens -> 8 tokens
    token_learner_hidden_dim: int = 64

    # === Transformer ===
    token_dim: int = 512
    num_layers: int = 8  # self-attention layers
    num_heads: int = 8
    d_ff: int = 1024  # feedforward 차원
    dropout: float = 0.1

    # === Action ===
    num_action_bins: int = 256  # 각 차원을 256개 bin으로 이산화
    action_dims: int = 11  # arm(7) + base(3) + mode(1)

    @property
    def max_seq_len(self) -> int:
        """Transformer 입력 시퀀스 길이: num_images * num_learned_tokens."""
        return self.num_images * self.num_learned_tokens  # 6 * 8 = 48
