import torch
import torch.nn as nn
from torch import Tensor


class ActionTokenizer:
    """연속 action 값을 이산 bin index로 변환하고 복원한다.

    RT-1은 11개 action 차원을 각각 256개 bin으로 균등 이산화한다.

    Action dimensions (11):
        - Arm: x, y, z, roll, pitch, yaw, gripper (7)
        - Base: x, y, yaw (3)
        - Mode: arm / base / terminate (1)
    """

    def __init__(self, action_mins: Tensor, action_maxs: Tensor, num_bins: int = 256):
        """
        Args:
            action_mins: (11,) 각 action 차원의 최솟값
            action_maxs: (11,) 각 action 차원의 최댓값
            num_bins: bin 개수 (default: 256)
        """
        self.action_mins = action_mins
        self.action_maxs = action_maxs
        self.num_bins = num_bins

    def encode(self, actions: Tensor) -> Tensor:
        """연속 action을 bin index로 변환한다.

        Args:
            actions: (B, 11) 연속 action 값

        Returns:
            (B, 11) long tensor, 각 값은 [0, num_bins-1] 범위의 bin index

        TODO: 구현하기
            1. actions를 [action_mins, action_maxs] 범위로 clamp
            2. [min, max] 범위를 [0, num_bins-1] 범위로 선형 매핑
            3. long 타입으로 변환

        힌트: normalized = (actions - mins) / (maxs - mins)  # [0, 1]
              bin_index = (normalized * (num_bins - 1)).long()
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
        """bin index를 연속 action 값으로 복원한다 (bin 중심값).

        Args:
            bin_indices: (B, 11) long tensor, [0, num_bins-1]

        Returns:
            (B, 11) 복원된 연속 action 값

        TODO: 구현하기 (encode의 역연산)

        힌트: normalized = bin_indices.float() / (num_bins - 1)  # [0, 1]
              actions = normalized * (maxs - mins) + mins
        """
        noramlized = bin_indices.float() / (self.num_bins - 1)
        return noramlized * (self.action_maxs - self.action_mins) + self.action_mins


class ActionHead(nn.Module):
    """Transformer 출력을 action logits으로 변환한다.

    11개 action 차원에 대해 각각 독립적인 Linear(d_model -> num_bins)를 사용한다.
    """

    def __init__(self, d_model: int = 512, action_dims: int = 11, num_bins: int = 256):
        super().__init__()
        self.action_dims = action_dims
        self.num_bins = num_bins

        # TODO: 11개의 Linear 레이어를 nn.ModuleList로 생성
        #   각 레이어: Linear(d_model, num_bins)
        #
        # 힌트: self.heads = nn.ModuleList([...])
        self._heads = nn.ModuleList(
            [nn.Linear(d_model, num_bins) for _ in range(action_dims)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, d_model) transformer의 마지막 토큰 출력

        Returns:
            (B, action_dims, num_bins) = (B, 11, 256)
            각 action 차원에 대한 256개 bin의 logits

        TODO: 구현하기
            1. 각 head에 x를 통과시켜 (B, 256) logits를 얻는다
            2. 11개 결과를 stack하여 (B, 11, 256)으로 만든다

        힌트: torch.stack([head(x) for head in self.heads], dim=1)
        """
        return torch.stack([head(x) for head in self._heads], dim=1)
