import numpy as np
import torch
from typing import List, NamedTuple, Optional, Union
from src.utils import device

NODATAVALUE = 65535


# NEED THOROUGH AND PRECISE DOCUMENTATION AND TESTS!!!
# checks should be implemented at the test level
# here, we need to fix the exted ranges for the bands
S1_bands = ["VV", "VH"]
S1_bands_units = ["dB", "dB"]  # TODO for all other bands
S2_bands = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
]
meteo_bands = ["temperature", "precipitation"]
dem_bands = ["elevation", "slope"]


ArrayTensor = Union[np.ndarray, torch.Tensor]


def to_torchtensor(x: ArrayTensor, device: torch.device = device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device)


class Predictors(NamedTuple):
    s1: Optional[ArrayTensor] = None  # [B, H, W, T, len(S1_bands)]
    s2: Optional[ArrayTensor] = None  # [B, H, W, T, len(S2_bands)]
    meteo: Optional[ArrayTensor] = None  # [B, T, len(meteo_bands)]
    dem: Optional[ArrayTensor] = None  # [B, H, W, len(dem)]
    latlon: Optional[ArrayTensor] = None  # [B, 2]
    # Gabi to try and implement the possibility to learn a linear layer for each aux_input
    aux_inputs: Optional[List[ArrayTensor]] = None
    # Label needs to always be 2D, with temporal dimension
    label: Optional[ArrayTensor] = None
    month: Optional[Union[ArrayTensor, int]] = None
