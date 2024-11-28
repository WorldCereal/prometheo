from typing import Literal, Optional

import numpy as np
import pandas as pd
from src.predictors import NODATAVALUE
from torch.utils.data import Dataset



class DatasetBase(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,  
        num_timesteps: int,
        task_type: Literal["binary", "multiclass", "regression", "ssl"],
        num_outputs: Optional[int] = None,
    ):
        self.dataframe = dataframe.replace({np.nan: NODATAVALUE})
        self.num_timesteps = num_timesteps
        self.task_type = task_type
        self.num_outputs = num_outputs

        assert self.task_type in ["binary", "multiclass", "regression", "ssl"]
        if self.task_type != "ssl":
            assert num_outputs is not None


    def __len__(self):
        return self.dataframe.shape[0]
