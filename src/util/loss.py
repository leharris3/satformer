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
    

class WeightedL1(nn.Module):
    """
    Simple l1 variant where we weight by magnitude y
    """

    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha=1.0

    def forward(self, preds:torch.Tensor, target:torch.Tensor):
        
        # simply weight by the approx max of the target 
        return F.l1_loss(preds, target) * (self.alpha * torch.logsumexp(target, (0, 1)))

class CategoricalCrossEntropy(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits:torch.Tensor, target:torch.Tensor):
        
        pred = F.softmax(logits)
        return F.binary_cross_entropy(pred, target)
    


if __name__ == "__main__":
    loss = WeightedL1(0, 1)
    y = torch.rand(1, 10)
    y_hat = torch.rand(1, 10)
    loss(y_hat, y)