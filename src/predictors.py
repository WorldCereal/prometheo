import numpy as np
from Typing import List, NamedTuple, Optional, Union

NODATAVALUE = 65535


# NEED THOROUGH AND PRECISE DOCUMENTATION AND TESTS!!!
# checks should be implemented at the test level
# here, we need to fix the exted ranges for the bands
S1_bands = [("VV", "db"),()]
meteo_bands = []


class Predictors(NamedTuple):
    s1: Optional[np.ndarray] = None  # VV VH 
    s2: Optional[np.ndarray] = None
    meteo: Optional[np.ndarray] = None
    dem: Optional[np.ndarray] = None
    latlon: Optional[np.ndarray] = None
    # Gabi to try and implement the possibility to learn a linear layer for each aux_input
    aux_inputs: Optional[List[np.ndarray]] = None
    # Label needs to always be 2D, with temporal dimension
    label: Optional[np.ndarray] = None
    month: Optional[Union[np.ndarray,int]] = None
