from typing import Literal

import pandas as pd
from torch.utils.data import Dataset


class DatasetBase(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_outputs: int,
        num_timesteps: int,
        task_type: Literal["binary", "multiclass", "regression"],
    ):
        self.df = dataframe
        self.num_outputs = num_outputs
        self.num_timesteps = num_timesteps
        self.task_type = task_type

        assert self.task_type in ["binary", "multiclass", "regression"]

    def __len__(self):
        return self.df.shape[0]
