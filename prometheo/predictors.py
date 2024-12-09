from typing import NamedTuple, Optional, Union, Sequence

import numpy as np
import torch
from torch.utils.data import default_collate

from prometheo.utils import device

NODATAVALUE = 65535


# NEED THOROUGH AND PRECISE DOCUMENTATION AND TESTS!!!
# checks should be implemented at the test level
# here, we need to fix the exted ranges for the bands
S1_BANDS = ["VV", "VH"]
S1_BANDS_UNITS = ["dB", "dB"]  # TODO for all other bands
S2_BANDS = [
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
METEO_BANDS = ["temperature", "precipitation"]
DEM_BANDS = ["elevation", "slope"]


ArrayTensor = Union[np.ndarray, torch.Tensor]
DEFAULT_INT = -1  # Default placeholder for missing integer values


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
    # for now we ignore them
    # aux_inputs: Optional[List[ArrayTensor]] = None
    # Label needs to always be 2D, with temporal dimension
    label: Optional[ArrayTensor] = None  # [B, H, W, T, num_outputs]
    timestamps: Optional[ArrayTensor] = None  # [B, T, D=3], where D=[day, month, year]

    def as_dict(self, ignore_nones: bool = True):
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if ignore_nones and (val is None):
                continue
            else:
                return_dict[field] = val
        return return_dict


def collate_fn(batch: Sequence[Predictors]):
    # we assume that the same values are consistently None
    collated_dict = default_collate([i.as_dict(ignore_nones=True) for i in batch])
    return Predictors(**collated_dict)
