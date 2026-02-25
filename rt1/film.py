import torch
import torch.nn as nn
from torch import Tensor
import timm


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (FiLM).

    이미지 feature map을 언어 임베딩으로 조건화한다.

    공식: output = (1 + gamma) * x + beta
        - gamma, beta = Linear(lang_embed)
        - zero-init: 학습 시작 시 gamma=0, beta=0 → output = x (identity)
        - 이렇게 하면 pretrained EfficientNet의 가중치를 초기에 보존할 수 있다.

    참고: Perez et al. (2018) "FiLM: Visual Reasoning with a General Conditioning Layer"
    """

    _gamma_linear: nn.Linear
    _beta_linear: nn.Linear

    def __init__(self, lang_dim: int, channel_dim: int):
        """
        Args:
            lang_dim: 언어 임베딩 차원 (512)
            channel_dim: feature map 채널 수 (블록마다 다름: 24, 32, 48, ...)

        TODO: 구현하기
            1. self.gamma_linear = Linear(lang_dim, channel_dim)
            2. self.beta_linear = Linear(lang_dim, channel_dim)
            3. 두 레이어의 weight와 bias를 모두 0으로 초기화
               → nn.init.zeros_(...)

        왜 zero-init인가:
            gamma_linear(lang) = 0 → gamma = 0
            beta_linear(lang) = 0 → beta = 0
            output = (1 + 0) * x + 0 = x (identity)
            → pretrained 가중치가 처음에 그대로 보존됨
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
            x: (B, C, H, W) - MBConv 블록의 출력 feature map
            lang_embed: (B, lang_dim) - 언어 임베딩 (512-dim)

        Returns:
            (B, C, H, W) - FiLM 적용된 feature map

        TODO: 구현하기
            1. gamma = self.gamma_linear(lang_embed)  # (B, C)
            2. beta = self.beta_linear(lang_embed)    # (B, C)
            3. gamma, beta를 (B, C, 1, 1)로 reshape (H, W에 broadcast하기 위해)
               → .unsqueeze(-1).unsqueeze(-1)
            4. return (1 + gamma) * x + beta
        """
        g = self.gamma_linear(lang_embed)
        b = self.beta_linear(lang_embed)
        g = g.unsqueeze(-1).unsqueeze(-1)
        b = b.unsqueeze(-1).unsqueeze(-1)

        return (1 + g) * x + b


class FiLMEfficientNet(nn.Module):
    """FiLM-conditioned EfficientNet-B3 이미지 토크나이저.

    EfficientNet-B3의 각 MBConv 블록 뒤에 FiLM 레이어를 삽입하여
    이미지 인코딩 과정 자체를 언어로 조건화한다 (early fusion).

    구조:
        conv_stem + bn1
        → 26개 [MBConv block → FiLM layer]
        → AdaptiveAvgPool2d(9)
        → Conv2d(384 → 512, 1x1)
        → flatten + permute
        → 81개 vision-language tokens (9×9×512)
    """

    def __init__(self, lang_dim: int = 512, pretrained: bool = True):
        """
        Args:
            lang_dim: 언어 임베딩 차원 (512)
            pretrained: ImageNet pretrained 가중치 사용 여부
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

        # 인접 칸을 평균으로 합쳐서 9x9로 만들기
        self.adaptive_pool = nn.AdaptiveAvgPool2d(9)
        # 채널 수를 바꾸는 Linear (2D 전용)
        self.token_proj = nn.Conv2d(384, 512, kernel_size=1)
        self.film_layers = nn.ModuleList(
            [FiLMLayer(lang_dim, ch) for ch in channel_dims]
        )

    def forward(self, images: Tensor, lang_embed: Tensor) -> Tensor:
        """
        Args:
            images: (B, 3, 300, 300) - 단일 시점의 RGB 이미지
            lang_embed: (B, 512) - 언어 임베딩

        Returns:
            (B, 81, 512) - 81개 vision-language tokens
        """

        # CNN 구조
        x = self._backbone.bn1(self._backbone.conv_stem(images))  # type: ignore

        film_idx = 0
        for stage in self._backbone.blocks:  # type: ignore
            for block in stage:
                x = block(x)
                # 각 블록에 FiLM 레이어 삽입
                x = self.film_layers[film_idx](x, lang_embed)
                film_idx += 1

        x = self.adaptive_pool(x)  # (B, 384, 9, 9)
        x = self.token_proj(x)  # (B, 512, 9, 9)
        x = x.flatten(2)  # (B, 512, 81)
        x = x.permute(0, 2, 1)  # (B, 81, 512)
        return x
