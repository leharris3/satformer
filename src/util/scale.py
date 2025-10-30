import torch


def scale_zero_to_one(
    X:torch.Tensor, 
    dataset_min:float,
    dataset_max:float,
    ) -> torch.Tensor:
    """
    Map a tensor in range (0, 1) to (from_min, to_max)
    """
    # -> [0, 1]
    X = (X - dataset_min) / (dataset_max - dataset_min)
    return X