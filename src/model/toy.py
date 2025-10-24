import torch
import torch.nn as nn


class ToyCummulativePrecipitationModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(x: torch.Tensor) -> torch.Tensor:

        # TODO: verify this will work
        return torch.rand(**x.shape)