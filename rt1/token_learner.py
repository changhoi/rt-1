import torch
import torch.nn as nn
from torch import Tensor


class TokenLearner(nn.Module):
    """TokenLearner: element-wise spatial attention으로 토큰을 압축한다.

    81개의 vision-language 토큰을 8개의 learned 토큰으로 압축한다.
    각 learned 토큰은 81개 입력 토큰의 가중합(weighted sum)이다.

    참고: Ryoo et al. (2021) "TokenLearner: Adaptive Space-Time Tokenization for Videos"
    """

    def __init__(self, input_dim: int = 512, num_tokens: int = 8, hidden_dim: int = 64):
        """
        Args:
            input_dim: 입력 토큰 차원 (512)
            num_tokens: 출력 토큰 수 (8)
            hidden_dim: attention MLP 히든 차원 (64)

        전체 파라미터 수 ~34K:
            - LayerNorm: 512 * 2 = 1,024
            - Linear(512, 64): 512 * 64 + 64 = 32,832
            - Linear(64, 8): 64 * 8 + 8 = 520
            - Total: ~34,376
        """
        super().__init__()
        self._norm = nn.LayerNorm(input_dim)
        self._attn_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, N, C) where N=81, C=512

        Returns:
            (B, num_tokens, C) = (B, 8, 512)

        TODO: 구현하기
            1. x를 LayerNorm 통과 -> x_norm: (B, 81, 512)
            2. x_norm을 attn_mlp 통과 -> attn: (B, 81, 8)
            3. sigmoid 적용 -> attn: (B, 81, 8)
            4. permute(0, 2, 1) -> attn: (B, 8, 81)
            5. dim=-1로 합이 1이 되도록 normalize (/ sum)
            6. torch.bmm(attn, x) -> (B, 8, 512)

        핵심 아이디어:
            - MLP가 각 위치(81개)에 대해 8개의 attention score를 출력
            - 각 learned token은 전체 81개 위치의 가중합
            - sigmoid + normalize로 soft selection 구현
        """
        x_norm = self._norm(x)  # (B, 81, 512) 정규화

        attn = self._attn_mlp(x_norm)  # (B, 81, 8) 각 위치에 대해 8개 토큰별 점수 출력

        attn = torch.sigmoid(attn)  # 0~1로 변환. 양수를 보장해서 "비율"로 쓸 수 있게 함

        attn = attn.permute(0, 2, 1)  # (B, 8, 81) 각 행이 "하나의 learned token이 81개 위치를 보는 시선"

        attn = attn / attn.sum(dim=-1, keepdim=True)  # 각 행의 합=1로 정규화. softmax와 달리 원래 비율을 보존

        # (B, 8, 81) × (B, 81, 512) = (B, 8, 512)
        # 각 learned token = 81개 위치의 가중평균
        # ex) token_0 = 0.25 × 위치0 + 0.03 × 위치1 + ... + 0.08 × 위치80
        return torch.bmm(attn, x)
