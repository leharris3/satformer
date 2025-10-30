import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Sequence, List


class MSE(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return F.mse_loss(x, y)
    

class L1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        return F.l1_loss(x, y)