from pathlib import Path
from unittest import TestCase

import pandas as pd
from torch.utils.data import DataLoader

from prometheo.datasets.worldcereal import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
)
from prometheo.predictors import collate_fn


def load_dataframe():
    dir = Path(__file__).resolve().parent
    return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")


class TestDataset(TestCase):
    def test_WorldCerealDataset(self):
        df = load_dataframe()
        ds = WorldCerealDataset(df)
        dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        for batch in dl:
            # TODO: for now timestamps are UINT16 format because
            # default collate_fn cannot handle timestamp format.
            print(batch)

    def test_WorldCerealLabelledDataset(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True)
        dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        for batch in dl:
            print(batch)
