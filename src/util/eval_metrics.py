import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Callable, List, Union
from torchmetrics.regression import CriticalSuccessIndex, ContinuousRankedProbabilityScore
from torchmetrics.classification import F1Score, MulticlassF1Score
from src.dataloader.dataset_stats import Y_REG_NORM_BINS


class BinnedEvalMetric(nn.Module):

    def __init__(self, f: Callable, bins: Union[torch.Tensor, None]=None):
        """
        args
        ---
        :bins: optional 1D array of class-coefficients to reweight metrics
        """
        
        super().__init__()

        if bins != None:
            assert type(bins) is torch.Tensor, f"Error: bins must be a 1D tensor."
            assert len(bins.shape) == 1, f"Error: bins must be a 1D tensor."
            # assert (bins.sum().item() < 1.01 and bins.sum().item() > 0.99), \
            #     f"Error: bins must be a probability density over outputs;   \
            #     expected bins.sum() == 1.0, got: {bins.sum().item()}"

        self.f    = f
        self.bins = bins

    @torch.no_grad()
    def forward(self, logits:torch.Tensor, target:torch.Tensor) -> float:
        """
        args
        ---
        :logits: unnormalized model class preds
        :target: one-hot categorical label with identical shape to `logits`
        """
        
        assert self.bins.shape[0] == logits.shape[-1] and self.bins.shape[0] == target.shape[-1], \
            f"Error: expected len self.bins to equal len logits/target.shape[-1]"
        
        unnorm: float = self.f(logits, target, reduce="none")
        
        # 1. get class indices
        max_i         = torch.argmax(logits, dim=1, keepdim=True)
        scale         = self.bins[max_i].squeeze()
        
        # [B] * [B]
        return  (scale * unnorm).mean().item()


@torch.no_grad()
def mean_csi(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    :target: one-hot categorical label with identical shape to `logits`
    """

    # [B, N]
    logits = logits.detach().clone()
    target = target.detach().clone()
    
    max_i        = torch.argmax(logits, dim=1, keepdim=True)
    pred         = torch.zeros(logits.shape).to(target.device)
    pred         = pred.scatter(1, max_i, 1.0)

    if reduce == "none":
        csi = CriticalSuccessIndex(0.5, keep_sequence_dim=0).to(target.device)
        return csi(pred, target)
    else:
        csi = CriticalSuccessIndex(0.5).to(target.device)

    return csi(pred, target).item()
    

@torch.no_grad()
def mean_f1(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    :target: one-hot categorical label with identical shape to `logits`
    """

    logits = logits.detach().clone()
    target = target.detach().clone()
    
    max_i        = torch.argmax(logits, dim=1, keepdim=True)
    pred         = torch.zeros(logits.shape).to(target.device)
    pred         = pred.scatter(1, max_i, 1.0)

    pred_idxs = torch.argmax(pred,   dim=-1)
    targ_idxs = torch.argmax(target, dim=-1)

    if reduce == "none":
        f1_metric    = MulticlassF1Score(num_classes=logits.shape[-1], multidim_average="samplewise").to(target.device)
        return f1_metric(pred, target)
    else:
        f1_metric    = MulticlassF1Score(num_classes=logits.shape[-1]).to(target.device)
    
    return f1_metric(pred, target).item()


def mean_crps(logits:torch.Tensor, target:torch.Tensor, reduce="mean") -> float:
    """
    args
    ---
    :logits: unnormalized model class preds
    - [B, N]
    :target: one-hot categorical label with identical shape to `logits`
    - [B, N]
    """

    assert logits.shape == target.shape, f"Shape of logits: {logits.shape} must match target shape: {target.shape}"

    logits = logits.detach().clone()
    target = target.detach().clone()
    
    B, N   = logits.shape

    label  = torch.zeros(logits.shape).to(logits.device)
    max_i  = torch.argmax(target, dim=1)

    # TODO: replace with something efficient/vectorized
    for b, idx in enumerate(max_i): label[b, idx] = 1

    # ground truth degenerate target
    label_cdf = label.cumsum(dim=1)

    # predict cdf from logits
    pred_cdf  = F.softmax(logits, dim=1).cumsum(dim=1)

    mask = (pred_cdf >= label_cdf) * 1

    crps = ((((pred_cdf - label_cdf) * mask) + ((label_cdf - pred_cdf) * ~mask)) ** 2).sum(dim=1)

    if reduce == "mean":
        return crps.mean()
    elif reduce == "none":
        return crps
    else:
        raise Exception()


if __name__ == "__main__":
    
    bins = F.softmax(torch.rand(4), dim=0)
    m = BinnedEvalMetric(mean_crps, bins)
    preds = torch.rand(10, 4)
    target = torch.tensor([[1, 0, 0, 0]] * 10)