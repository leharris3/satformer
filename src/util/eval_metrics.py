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
    :target: one-hot categorical label with identical shape to `logits`
    """

    logits = logits.detach().clone()
    target = target.detach().clone()

    max_i  = torch.argmax(target, dim=1,)
    labels = Y_REG_NORM_BINS.cuda(target.device)[max_i]
    
    max_i        = torch.argmax(logits, dim=1,)
    pred         = Y_REG_NORM_BINS.cuda(target.device)[max_i]
    
    # [B] -> [B, 2]
    if len(pred.shape) < 2:
        pred = pred.unsqueeze(1)
        pred = pred.repeat(1, 2)

    crps_metric  = ContinuousRankedProbabilityScore().to(target.device)

    # y:     [B, N]
    # y_hat: [B]    
    return crps_metric(pred, labels).item()

# def mean_crps



# x = torch.tensor([[0.2, 0.7], [0.9, 0.3]])
# y = torch.tensor([[0.4, 0.2], [0.8, 0.6]])
# csi = CriticalSuccessIndex(0.5)
# result = csi(x, y)


# # Example: Binary classification
# num_classes = 2
# f1_metric = F1Score(task="binary", num_classes=num_classes)

# # Example: Multiclass classification
# # num_classes = 5
# # f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro") # or "weighted", "micro"

# # Generate some example predictions and targets
# preds = torch.tensor([0, 1, 0, 1, 0])
# target = torch.tensor([0, 1, 1, 0, 0])

# # Compute F1 score
# f1_score = f1_metric(preds, target)
# print(f"F1 Score: {f1_score}")


# import torch

# # Create an instance of the CRPS metric
# crps_metric = ContinuousRankedProbabilityScore()

# # Generate some example predictions (e.g., from a probabilistic model)
# # These represent the predicted cumulative distribution function (CDF)
# # For simplicity, let's use a sorted tensor as an empirical CDF
# preds = torch.tensor([
#     [0.1, 0.2, 0.5, 0.8, 0.9],  # CDF for sample 1
#     [0.05, 0.3, 0.6, 0.7, 0.95] # CDF for sample 2
# ])

# # Generate some true observations (targets)
# targets = torch.tensor([0.6, 0.4])

# # Compute the CRPS
# score = crps_metric(preds, targets)

# print(f"CRPS: {score.item()}")