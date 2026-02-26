import torch
import torch.nn as nn
from torch import Tensor
import timm


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    Conditions image feature maps with language embeddings.

    Formula: output = (1 + gamma) * x + beta
        - gamma, beta = Linear(lang_embed)
        - zero-init: gamma=0, beta=0 at start → output = x (identity)
        - Preserves pretrained EfficientNet weights initially.

    Ref: Perez et al. (2018) "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    def __init__(self, lang_dim: int, channel_dim: int):
        """
        Args:
            lang_dim: Language embedding dimension (512)
            channel_dim: Feature map channel count (varies per block: 24, 32, 48, ...)
        """
        super().__init__()
        self.gamma_linear = nn.Linear(lang_dim, channel_dim)
        self.beta_linear = nn.Linear(lang_dim, channel_dim)
        nn.init.zeros_(self.gamma_linear.weight)
        nn.init.zeros_(self.gamma_linear.bias)
        nn.init.zeros_(self.beta_linear.weight)
        nn.init.zeros_(self.beta_linear.bias)

    def forward(self, x: Tensor, lang_embed: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) - MBConv block output feature map
            lang_embed: (B, lang_dim) - language embedding (512-dim)

        Returns:
            (B, C, H, W) - FiLM-conditioned feature map
        """
        g = self.gamma_linear(lang_embed)
        b = self.beta_linear(lang_embed)
        g = g.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        return (1 + g) * x + b


class FiLMEfficientNet(nn.Module):
    """FiLM-conditioned EfficientNet-B3 image tokenizer.

    Inserts a FiLM layer after each MBConv block in EfficientNet-B3,
    conditioning the image encoding process with language (early fusion).

    Architecture:
        conv_stem + bn1
        → 26× [MBConv block → FiLM layer]
        → AdaptiveAvgPool2d(9)
        → Conv2d(384 → 512, 1x1)
        → flatten + permute
        → 81 vision-language tokens (9×9×512)
    """

    def __init__(self, lang_dim: int = 512, pretrained: bool = True):
        """
        Args:
            lang_dim: Language embedding dimension (512)
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        self._backbone = timm.create_model("efficientnet_b3", pretrained=pretrained)
        channel_dims = []
        for stage in self._backbone.blocks:  # type: ignore
            for block in stage:
                if hasattr(block, "conv_pwl"):
                    channel_dims.append(block.conv_pwl.out_channels)
                elif hasattr(block, "conv_pw"):
                    channel_dims.append(block.conv_pw.out_channels)

        self.adaptive_pool = nn.AdaptiveAvgPool2d(9)
        self.token_proj = nn.Conv2d(384, 512, kernel_size=1)
        self.film_layers = nn.ModuleList(
            [FiLMLayer(lang_dim, ch) for ch in channel_dims]
        )

    def forward(self, images: Tensor, lang_embed: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, 300, 300) - single-timestep RGB image
            lang_embed: (B, 512) - language embedding

        Returns:
            (B, 81, 512) - 81 vision-language tokens
        """
        x = self._backbone.bn1(self._backbone.conv_stem(images))  # type: ignore

        film_idx = 0
        for stage in self._backbone.blocks:  # type: ignore
            for block in stage:
                x = block(x)
                x = self.film_layers[film_idx](x, lang_embed)
                film_idx += 1

        x = self.adaptive_pool(x)  # (B, 384, 9, 9)
        x = self.token_proj(x)  # (B, 512, 9, 9)
        x = x.flatten(2)  # (B, 512, 81)
        x = x.permute(0, 2, 1)  # (B, 81, 512)
        return x
