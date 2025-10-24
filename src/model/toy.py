import torch
import torch.nn as nn


class ToyCummulativePrecipitationModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [4, 11, 252, 252] -> [16, 1, 252, 252]
        self.c1 = nn.Conv1d(11, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # # [4, 11, 252, 252] -> [4, 1, 252, 252]
        # x = self.c1(x)

        # # [4, 1, 252, 252] -> [16, 1, 252, 252]
        # x = self.c2(x)

        return torch.rand(1, 16, 1, 252, 252).cuda()