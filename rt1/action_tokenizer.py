import torch
import torch.nn as nn
from torch import Tensor


class ActionTokenizer:
    """Converts continuous action values to discrete bin indices and back.

    RT-1 discretizes each of 11 action dimensions into 256 uniform bins.

    Action dimensions (11):
        - Arm: x, y, z, roll, pitch, yaw, gripper (7)
        - Base: x, y, yaw (3)
        - Mode: arm / base / terminate (1)
    """

    def __init__(self, action_mins: Tensor, action_maxs: Tensor, num_bins: int = 256):
        """
        Args:
            action_mins: (11,) minimum value per action dimension
            action_maxs: (11,) maximum value per action dimension
            num_bins: Number of bins (default: 256)
        """
        self.action_mins = action_mins
        self.action_maxs = action_maxs
        self.num_bins = num_bins

    def encode(self, actions: Tensor) -> Tensor:
        """Convert continuous actions to bin indices.

        Args:
            actions: (B, 11) continuous action values

        Returns:
            (B, 11) long tensor, each value in [0, num_bins-1]
        """
        actions = torch.clamp(
            actions,
            min=self.action_mins,
            max=self.action_maxs,
        )
        normalized = (actions - self.action_mins) / (
            self.action_maxs - self.action_mins
        )
        return (normalized * (self.num_bins - 1)).long()

    def decode(self, bin_indices: Tensor) -> Tensor:
        """Convert bin indices back to continuous action values (bin center).

        Args:
            bin_indices: (B, 11) long tensor, [0, num_bins-1]

        Returns:
            (B, 11) reconstructed continuous action values
        """
        normalized = bin_indices.float() / (self.num_bins - 1)
        return normalized * (self.action_maxs - self.action_mins) + self.action_mins


class ActionHead(nn.Module):
    """Converts Transformer output to action logits.

    Uses an independent Linear(d_model -> num_bins) for each of 11 action dimensions.
    """

    def __init__(self, d_model: int = 512, action_dims: int = 11, num_bins: int = 256):
        super().__init__()
        self.action_dims = action_dims
        self.num_bins = num_bins
        self._heads = nn.ModuleList(
            [nn.Linear(d_model, num_bins) for _ in range(action_dims)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, d_model) last token output from Transformer

        Returns:
            (B, action_dims, num_bins) = (B, 11, 256)
        """
        return torch.stack([head(x) for head in self._heads], dim=1)
