import torch
import torch.nn as nn
from torch import Tensor


class TokenLearner(nn.Module):
    """TokenLearner: compresses tokens via element-wise spatial attention.

    Compresses 81 vision-language tokens into 8 learned tokens.
    Each learned token is a weighted sum of 81 input tokens.

    Ref: Ryoo et al. (2021) "TokenLearner: Adaptive Space-Time Tokenization for Videos"
    """

    def __init__(self, input_dim: int = 512, num_tokens: int = 8, hidden_dim: int = 64):
        """
        Args:
            input_dim: Input token dimension (512)
            num_tokens: Number of output tokens (8)
            hidden_dim: Attention MLP hidden dimension (64)

        Total params ~34K:
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
        """
        x_norm = self._norm(x)  # (B, 81, 512)

        attn = self._attn_mlp(x_norm)  # (B, 81, 8) per-position scores for each token

        attn = torch.sigmoid(attn)  # Map to 0-1 to ensure positivity for ratio computation

        attn = attn.permute(0, 2, 1)  # (B, 8, 81) each row = one learned token's view over 81 positions

        attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize rows to sum=1, preserving original ratios

        # (B, 8, 81) × (B, 81, 512) = (B, 8, 512)
        # Each learned token = weighted average of 81 positions
        return torch.bmm(attn, x)
