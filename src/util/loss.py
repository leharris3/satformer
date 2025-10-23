import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, Sequence, List


class ImageInpaintingL1Loss(nn.Module):
    """
    An inpainting loss where we use our free lunch!
    Include the given signal (i.e., unmasked pixels) in the final model prediction.
    """

    def __init__(self):
        super(ImageInpaintingL1Loss, self).__init__()

    def forward(
        self,
        predicted_image: torch.Tensor,
        target_image: torch.Tensor,
        mask: torch.Tensor,
    ):
        """
        Final loss = || (given_pixels + pred_pixels) - (target) ||
        :param original_image: (B, H, W)
        :param predicted_image: (B, H, W)
        :param target_image: (B, H, W)
        :param mask: (B, H, W)
        """
        # mask = 0: obstructed
        given_pixels = target_image * mask
        pred_pixels = predicted_image * ~mask
        final_prediction = given_pixels + pred_pixels
        return torch.nn.functional.l1_loss(final_prediction, target_image)

    @staticmethod
    def get_final_prediction(
        predicted_image: torch.Tensor, target_image: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns
            (target * mask) + (pred * ~mask)
        """
        # y_sparse [given]
        given_pixels = target_image * mask
        # pred - y_sparse_hat
        pred_pixels = predicted_image * ~mask
        # pred + y_sparse
        final_prediction = given_pixels + pred_pixels
        return final_prediction


class VAELoss(nn.Module):
    def __init__(self):
        """
        Variational Autoencoder Loss Function.
        """
        super(VAELoss, self).__init__()

    def forward(self, output, target, mu, logvar):
        recon_loss = F.mse_loss(output, target, reduction="sum") / target.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.002 * kl_loss


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def focal_loss(
    alpha: Optional[Sequence] = None,
    gamma: float = 0.0,
    reduction: str = "mean",
    ignore_index: int = -100,
    device="cpu",
    dtype=torch.float32,
) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index
    )
    return fl


def vae_loss_function(output, x, mu, logvar):
    # reconstruction loss
    recon_loss = F.mse_loss(output, x, reduction="sum") / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.002 * kl_loss


def _center(x: torch.Tensor) -> torch.Tensor:
    """Zero‑centre each (H, W) map independently."""
    return x - x.mean(dim=(-2, -1), keepdim=True)


def rms_roughness(x: torch.Tensor) -> torch.Tensor:
    # B × H × W  ➜  B
    x = _center(x)
    return torch.sqrt((x**2).mean(dim=(-2, -1)))


def mean_roughness(x: torch.Tensor) -> torch.Tensor:
    # B × H × W  ➜  B
    x = _center(x)
    return x.abs().mean(dim=(-2, -1))


def roughness_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    dataset_min: float,
    dataset_max: float,
    use_metrics: List[str] = ["rms", "mean"],
    weights: List[float] = [1.0, 1.0],
) -> torch.Tensor:
    """
    Surface‑roughness consistency loss.

    Parameters
    ----------
    pred, target : (B, H, W) tensors
        Normalised to [0, 1]. This function rescales them to physical units
        using `dataset_min` / `dataset_max` before computing roughness.
    dataset_min, dataset_max : float
        Global minimum / maximum of the *unnormalised* topography maps.
    use_metrics : list[str]
        Any subset of {"rms", "mean"}.
    weights : list[float]
        Per‑metric weights, same order as `use_metrics`.
    """

    # ------------------------------------------------------------
    # 1) un‑normalise to original scale (e.g. nanometres)
    # ------------------------------------------------------------
    scale       = dataset_max - dataset_min
    pred_phys   = (pred   * scale + dataset_min) * 1e9
    target_phys = (target * scale + dataset_min) * 1e9

    # ------------------------------------------------------------
    # 2) compute roughness metrics
    # ------------------------------------------------------------
    loss_terms: List[torch.Tensor] = []

    if "rms" in use_metrics:
        rms_diff = (rms_roughness(pred_phys) - rms_roughness(target_phys)).abs()
        loss_terms.append(weights[0] * rms_diff)

    if "mean" in use_metrics:
        mean_diff = (mean_roughness(pred_phys) - mean_roughness(target_phys)).abs()
        # if both metrics are used, weights[1] applies; else weights[0]
        w = weights[1] if len(use_metrics) > 1 else weights[0]
        loss_terms.append(w * mean_diff)

    # ------------------------------------------------------------
    # 3) aggregate to a scalar
    # ------------------------------------------------------------
    # -> (B, n_metrics)  ➜   scalar
    return torch.stack(loss_terms, dim=-1).mean()


def rotation_invariant_l1_loss(
    model: torch.nn.Module,
    X: torch.Tensor,
    X_sparse: torch.Tensor,
    _min: float,
    _max: float,
) -> torch.Tensor:
    """
    Average L1 loss between the model’s output and its input over the
    four right‑angle rotations of X (0°, 90°, 180°, 270°).

    Args
    ----
    model : torch.nn.Module
        Any network that maps a tensor shaped like `X` back to itself.
    X : torch.Tensor
        Image‑like tensor with at least (H, W) spatial dims.

    Returns
    -------
    torch.Tensor
        Scalar mean loss (requires_grad=True if model parameters do).
    """
    if X.ndim < 2:
        raise ValueError("X must have at least 2 spatial dimensions.")

    rot_dims = (0, 1) if X.ndim == 2 else (-2, -1)  # pick spatial axes
    
    loss = roughness_loss

    # Pre‑compute the four rotated views: X, R90(X), R180(X), R270(X)
    views = [X_sparse] + [torch.rot90(X_sparse, k, rot_dims) for k in range(1, 4)]

    # Evaluate model and loss for each view, then average
    losses = [loss(model(v), X, _min, _max) for v in views]
    return torch.stack(losses).mean()


def rotation_plus_flip_invariant_loss(
    model: torch.nn.Module,
    X: torch.Tensor,
    X_sparse: torch.Tensor,
    _min: float,
    _max: float,
) -> torch.Tensor:
    """
    Average L1 loss between the model’s output and its input over the
    four right‑angle rotations and horizontal flips of X.

    Args
    ----
    model : torch.nn.Module
        Any network that maps a tensor shaped like `X` back to itself.
    X : torch.Tensor
        Image‑like tensor with at least (H, W) spatial dims.

    Returns
    -------
    torch.Tensor
        Scalar mean loss (requires_grad=True if model parameters do).
    """
    if X.ndim < 2:
        raise ValueError("X must have at least 2 spatial dimensions.")

    rot_dims = (0, 1) if X.ndim == 2 else (-2, -1)  # pick spatial axes
    views = [X] + [torch.rot90(X, k, rot_dims) for k in range(1, 4)]  # rotations
    flipped_views = [torch.flip(v, dims=[rot_dims[-1]]) for v in views]  # flips
    all_views = views + flipped_views

    losses = [roughness_loss(model(v), X, _min, _max) for v in all_views]
    return torch.stack(losses).mean()


if __name__ == "__main__":
    pass
