from pathlib import Path
from unittest import TestCase

import pandas as pd
from prometheo.datasets.worldcereal import WorldCerealDataset
from torch.utils.data import DataLoader


def load_dataframe():
    dir = Path(__file__).resolve().parent
    return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")


class TestDataset(TestCase):
    def test_WorldCerealDataset(self):
        df = load_dataframe()
        ds = WorldCerealDataset(df)
        dl = DataLoader(ds, batch_size=1)
        for batch in dl:
            print(batch)
