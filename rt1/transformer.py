import torch
import torch.nn as nn
from torch import Tensor


class CausalTransformer(nn.Module):
    """Decoder-only Transformer with causal masking.

    Sequence model for RT-1. Processes 48 tokens (6 images × 8 tokens)
    with causal attention.

    Why nn.TransformerEncoder:
        - "Decoder-only" = self-attention + causal mask (GPT-style)
        - PyTorch's nn.TransformerDecoder includes cross-attention (for encoder-decoder)
        - nn.TransformerEncoder + causal mask is the correct decoder-only implementation

    Causal mask:
        - Token t can only attend to tokens 0..t (no future)
        - Tokens from frame 3 can only use information from frames 0,1,2,3
        - Matches the robot deciding actions from current + past observations only
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
            d_model: Token dimension (512)
            nhead: Number of attention heads (8)
            num_layers: Number of Transformer layers (8)
            d_ff: Feedforward hidden dimension (1024)
            max_seq_len: Maximum sequence length (48)
            dropout: Dropout rate
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
        """
        T = tokens.size(1)
        positions = torch.arange(T, device=tokens.device)
        tokens = tokens + self._pos_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device
        )

        output = self._transformer(tokens, mask=causal_mask)
        return self._norm(output)
