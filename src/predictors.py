import numpy as np
from typing import NamedTuple, Optional, Union

NODATAVALUE = 65535


class Predictors(NamedTuple):
    s1: Optional[np.ndarray] = None
    s2: Optional[np.ndarray] = None
    meteo: Optional[np.ndarray] = None
    dem: Optional[np.ndarray] = None
    latlon: Optional[np.ndarray] = None
    valid_date: Optional[np.ndarray] = None
    label: Optional[Union[np.ndarray, float]] = None
    month: Optional[Union[np.ndarray, int]] = None
