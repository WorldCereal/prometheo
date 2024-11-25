import numpy as np
from typing import List, NamedTuple, Optional, Union

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


class Predictors(NamedTuple):
    s1: Optional[np.ndarray] = None  # [B, H, W, T, len(S1_bands)]
    s2: Optional[np.ndarray] = None  # [B, H, W, T, len(S2_bands)]
    meteo: Optional[np.ndarray] = None  # [B, T, len(meteo_bands)]
    dem: Optional[np.ndarray] = None  # [B, H, W, len(dem)]
    latlon: Optional[np.ndarray] = None
    # Gabi to try and implement the possibility to learn a linear layer for each aux_input
    aux_inputs: Optional[List[np.ndarray]] = None
    # Label needs to always be 2D, with temporal dimension
    label: Optional[np.ndarray] = None
    month: Optional[Union[np.ndarray, int]] = None
