import torch
from torch import nn

from .models.pooling import PoolingMethods
from .predictors import Predictors


def extract_features_from_model(
    model: nn.Module,  # Wrapper
    x: Predictors,
    batch_size: int,
    pooling: PoolingMethods,
) -> torch.Tensor:
    """
    Extract features from a model for the given input data using batching and pooling.

    Parameters
    ----------
    model : nn.Module
        The model (wrapper) from which to extract features.
    x : Predictors
        The input data wrapped in a Predictors object.
    batch_size : int
        The number of samples per batch.
    pooling : PoolingMethods
        The pooling method to apply to the model outputs.

    Returns
    -------
    torch.Tensor
        A tensor containing the concatenated features extracted from the model for all input data.
    """
    model_features: list[torch.Tensor] = []

    # Ensure model is in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in x.as_batches(batch_size):
            model_features.append(model(batch, pooling))
    return torch.concat(model_features, dim=0)
