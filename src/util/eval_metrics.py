import torch
import torch.nn.functional as F

from torchmetrics.regression import CriticalSuccessIndex, ContinuousRankedProbabilityScore
from torchmetrics.classification import F1Score

from src.dataloader.dataset_stats import Y_REG_NORM_BINS


@torch.no_grad()
def mean_csi(logits:torch.Tensor, target:torch.Tensor) -> float:
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
    csi          = CriticalSuccessIndex(0.5).to(target.device)
    
    return csi(pred, target).item()

@torch.no_grad()
def mean_f1(logits:torch.Tensor, target:torch.Tensor) -> float:
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
    f1_metric    = F1Score(task="multiclass", num_classes=logits.shape[1]).to(target.device)
    
    return f1_metric(pred, target).item()


def mean_crps(logits:torch.Tensor, target:torch.Tensor) -> float:
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

    crps = ((((pred_cdf - label_cdf) * mask) + ((label_cdf - pred_cdf) * ~mask)) ** 2).sum(dim=1).mean()
    return crps


if __name__ == "__main__":
    
    y      = torch.rand(10, 10)
    y_hat  = torch.zeros(10, 10)
    print(mean_crps(y_hat, y))