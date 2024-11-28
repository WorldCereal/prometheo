import pandas as pd
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.num_outputs: int
        self.task_type: str

        assert self.task_type in ["binary", "multiclass", "regression"]

    def __len__(self):
        return self.df.shape[0]
