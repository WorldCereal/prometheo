import torch
from torch import nn
from .predictors import Predictors
from .models.pooling import PoolingMethods


def extract_features_from_model(
    model: nn.Module,  # Wrapper
    x: Predictors,
    batch_size: int,
    pooling: PoolingMethods,
) -> torch.Tensor:
    model_features: list[torch.Tensor] = []
    for batch in x.as_batches(batch_size):
        model_features.append(model(batch, pooling))
    return torch.concat(model_features, dim=0)
