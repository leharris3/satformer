import torch
import torch.nn as nn


class ToyCummulativePrecipitationModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [4, 11, 252, 252] -> [16, 1, 252, 252]
        self.c1 = nn.Conv3d(44, 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C, H, W = x.shape

        # [4, 11, 252, 252] -> [44, 1, 252, 252]
        x = x.reshape(B, T * C, 1, H, W)
        
        # [44, 1, 252, 252] -> [16, 1, 252, 252]
        x = self.c1(x)

        return x