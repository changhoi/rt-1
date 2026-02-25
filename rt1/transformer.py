import torch
import torch.nn as nn
from torch import Tensor


class CausalTransformer(nn.Module):
    """Decoder-only Transformer with causal masking.

    RT-1의 시퀀스 모델. 48개 토큰(6 images × 8 tokens)을 받아
    causal attention으로 처리한다.

    왜 nn.TransformerEncoder를 쓰는가:
        - "Decoder-only" = self-attention + causal mask (GPT 스타일)
        - PyTorch의 nn.TransformerDecoder는 cross-attention 포함 (encoder-decoder 구조용)
        - nn.TransformerEncoder + causal mask가 decoder-only의 올바른 구현

    Causal mask의 의미:
        - 토큰 t는 토큰 0..t만 볼 수 있다 (미래 불가)
        - 이미지 시점 3의 토큰은 시점 0,1,2,3의 정보만 사용
        - 로봇이 현재+과거 관찰만으로 행동을 결정하는 것과 일치
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 48,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: 토큰 차원 (512)
            nhead: attention head 수 (8)
            num_layers: Transformer 레이어 수 (8)
            d_ff: feedforward 히든 차원 (1024)
            max_seq_len: 최대 시퀀스 길이 (48)
            dropout: dropout 비율

        TODO: 구현하기
            1. Learned positional embedding 정의
               self.pos_embedding = nn.Embedding(max_seq_len, d_model)
               # 또는 nn.Parameter(torch.zeros(1, max_seq_len, d_model))

            2. TransformerEncoder 정의
               encoder_layer = nn.TransformerEncoderLayer(
                   d_model=d_model,
                   nhead=nhead,
                   dim_feedforward=d_ff,
                   dropout=dropout,
                   activation='gelu',
                   batch_first=True,   # 입력이 (B, T, D) 형태
                   norm_first=True,    # Pre-LayerNorm (학습 안정성)
               )
               self.transformer = nn.TransformerEncoder(
                   encoder_layer, num_layers=num_layers
               )

            3. Final LayerNorm
               self.norm = nn.LayerNorm(d_model)
        """
        super().__init__()
        self._pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self._norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: (B, T, D) where T=48, D=512

        Returns:
            (B, T, D) = (B, 48, 512)

        TODO: 구현하기
            1. Positional embedding 추가
               positions = torch.arange(T, device=tokens.device)
               tokens = tokens + self.pos_embedding(positions)  # (B, 48, 512)

            2. Causal mask 생성
               causal_mask = nn.Transformer.generate_square_subsequent_mask(
                   T, device=tokens.device
               )
               # (48, 48) 상삼각 = -inf, 하삼각+대각 = 0

            3. Transformer 통과
               output = self.transformer(tokens, mask=causal_mask)

            4. Final norm
               output = self.norm(output)  # (B, 48, 512)
               return output
        """
        T = tokens.size(1)
        positions = torch.arange(T, device=tokens.device)
        tokens = tokens + self._pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device
        )

        output = self._transformer(tokens, mask=causal_mask)
        return self._norm(output)
