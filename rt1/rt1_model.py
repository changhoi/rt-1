import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import SentenceTransformer

from rt1.config import RT1Config
from rt1.film import FiLMEfficientNet
from rt1.token_learner import TokenLearner
from rt1.transformer import CausalTransformer
from rt1.action_tokenizer import ActionHead


class RT1(nn.Module):
    """RT-1: Robotics Transformer for Real-World Control at Scale.

    전체 파이프라인:
        1. Instruction → SentenceTransformer (frozen) → Linear(384→512) → lang_embed
        2. 6 images 각각 → FiLM-EfficientNet(lang_embed) → 81 tokens → TokenLearner → 8 tokens
        3. 6×8 = 48 tokens → CausalTransformer → 48 outputs
        4. 마지막 토큰 → ActionHead → (11, 256) logits
    """

    def __init__(self, config: RT1Config):
        """
        TODO: 구현하기
            1. Language encoder (frozen)
               self.lang_encoder = SentenceTransformer(config.lang_model)
               if config.freeze_lang_encoder:
                   self.lang_encoder.requires_grad_(False)

            2. Language projection (학습됨)
               self.lang_proj = nn.Linear(config.lang_raw_dim, config.lang_embed_dim)
               # 384 → 512

            3. Image tokenizer
               self.film_efficientnet = FiLMEfficientNet(
                   lang_dim=config.lang_embed_dim,
                   pretrained=config.use_pretrained,
               )

            4. Token compression
               self.token_learner = TokenLearner(
                   input_dim=config.token_dim,
                   num_tokens=config.num_learned_tokens,
                   hidden_dim=config.token_learner_hidden_dim,
               )

            5. Transformer
               self.transformer = CausalTransformer(
                   d_model=config.token_dim,
                   nhead=config.num_heads,
                   num_layers=config.num_layers,
                   d_ff=config.d_ff,
                   max_seq_len=config.max_seq_len,
                   dropout=config.dropout,
               )

            6. Action head
               self.action_head = ActionHead(
                   d_model=config.token_dim,
                   action_dims=config.action_dims,
                   num_bins=config.num_action_bins,
               )
        """
        super().__init__()
        self.config = config
        self.lang_encoder = SentenceTransformer(config.lang_model)
        if config.freeze_lang_encoder:
            self.lang_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(config.lang_raw_dim, config.lang_embed_dim)
        self.film_efficientnet = FiLMEfficientNet(
            lang_dim=config.lang_embed_dim,
            pretrained=config.use_pretrained,
        )
        self.token_learner = TokenLearner(
            input_dim=config.token_dim,
            num_tokens=config.num_learned_tokens,
            hidden_dim=config.token_learner_hidden_dim,
        )
        self.transformer = CausalTransformer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.action_head = ActionHead(
            d_model=config.token_dim,
            action_dims=config.action_dims,
            num_bins=config.num_action_bins,
        )

    def encode_instruction(self, instructions: list[str]) -> Tensor:
        """자연어 명령을 512-dim 벡터로 인코딩한다.

        Args:
            instructions: B개의 자연어 명령 문자열 리스트
                ex) ["pick up the red ball", "move to the left"]

        Returns:
            (B, 512) 언어 임베딩

        TODO: 구현하기
            1. SentenceTransformer로 인코딩 (no grad)
               with torch.no_grad():
                   raw = self.lang_encoder.encode(
                       instructions, convert_to_tensor=True
                   )  # (B, 384)

            2. Projection
               lang_embed = self.lang_proj(raw)  # (B, 512)
               return lang_embed
        """
        with torch.no_grad():
            raw = self.lang_encoder.encode(instructions, convert_to_tensor=True)
            raw = raw.to(self.lang_proj.weight.device)
        return self.lang_proj(raw)

    def tokenize_images(self, images: Tensor, lang_embed: Tensor) -> Tensor:
        """6장의 이미지를 48개 토큰으로 변환한다.

        Args:
            images: (B, 6, 3, 300, 300) - 6개 시점의 이미지
            lang_embed: (B, 512)

        Returns:
            (B, 48, 512)

        TODO: 구현하기
            1. 6개 이미지를 하나씩 처리 (메모리 절약)
               all_tokens = []
               for t in range(T):
                   img_t = images[:, t]  # (B, 3, 300, 300)
                   vl_tokens = self.film_efficientnet(img_t, lang_embed)  # (B, 81, 512)
                   compressed = self.token_learner(vl_tokens)  # (B, 8, 512)
                   all_tokens.append(compressed)

            2. 시간 축으로 concat
               tokens = torch.cat(all_tokens, dim=1)  # (B, 48, 512)
               return tokens
        """

        all_tokens = []
        T = images.size(1)
        for t in range(T):
            img_t = images[:, t]
            vl_tokens = self.film_efficientnet(img_t, lang_embed)
            compressed = self.token_learner(vl_tokens)
            all_tokens.append(compressed)

        return torch.cat(all_tokens, dim=1)

    def forward(self, images: Tensor, instructions: list[str]) -> Tensor:
        """RT-1 forward pass.

        Args:
            images: (B, 6, 3, 300, 300)
            instructions: B개의 자연어 명령 리스트

        Returns:
            (B, 11, 256) action logits

        TODO: 구현하기
            1. lang_embed = self.encode_instruction(instructions)  # (B, 512)
            2. tokens = self.tokenize_images(images, lang_embed)   # (B, 48, 512)
            3. transformer_out = self.transformer(tokens)          # (B, 48, 512)
            4. last_token = transformer_out[:, -1, :]              # (B, 512)
            5. action_logits = self.action_head(last_token)        # (B, 11, 256)
            6. return action_logits
        """
        lang_embed = self.encode_instruction(instructions=instructions)
        tokens = self.tokenize_images(images=images, lang_embed=lang_embed)
        output = self.transformer(tokens)
        last_token = output[:, -1, :]
        action_logits = self.action_head(last_token)
        return action_logits

    def compute_loss(self, action_logits: Tensor, action_targets: Tensor) -> Tensor:
        """각 action 차원별 cross-entropy loss의 평균.

        Args:
            action_logits: (B, 11, 256) - 모델 출력
            action_targets: (B, 11) - 정답 bin indices (long)

        Returns:
            scalar loss

        TODO: 구현하기
            각 action 차원(11개)에 대해 cross_entropy를 구하고 평균.

            loss = 0
            for d in range(self.config.action_dims):
                loss += F.cross_entropy(
                    action_logits[:, d, :],  # (B, 256)
                    action_targets[:, d],    # (B,)
                )
            return loss / self.config.action_dims
        """
        loss = torch.tensor(0.0)
        for d in range(self.config.action_dims):
            loss += F.cross_entropy(action_logits[:, d, :], action_targets[:, d])

        return loss / self.config.action_dims
