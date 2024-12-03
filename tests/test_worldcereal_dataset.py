from pathlib import Path
from unittest import TestCase

import pandas as pd
from torch.utils.data import DataLoader

from prometheo.datasets.worldcereal import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
)


def load_dataframe(timestep_freq="month"):
    dir = Path(__file__).resolve().parent
    if timestep_freq == "month":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")
    elif timestep_freq == "dekad":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset_10D.parquet")
    else:
        raise ValueError(f"Invalid timestep frequency `{timestep_freq}`")


class TestDataset(TestCase):
    def test_WorldCerealDataset(self):
        df = load_dataframe()
        ds = WorldCerealDataset(df)
        dl = DataLoader(ds, batch_size=2)
        for batch in dl:
            # TODO: for now timestamps are UINT16 format because
            # default collate_fn cannot handle timestamp format.
            print(batch)

    def test_WorldCerealDataset_10D(self):
        # Test dekadal version of worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        ds = WorldCerealDataset(df, num_timesteps=36, timestep_freq="dekad")
        dl = DataLoader(ds, batch_size=2)
        for batch in dl:
            # TODO: for now timestamps are UINT16 format because
            # default collate_fn cannot handle timestamp format.
            print(batch)

    def test_WorldCerealLabelledDataset(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True)
        dl = DataLoader(ds, batch_size=2)
        for batch in dl:
            print(batch)

    def test_WorldCerealLabelledDataset_10D(self):
        # Test dekadal version of labelled worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        ds = WorldCerealLabelledDataset(
            df, num_timesteps=36, timestep_freq="dekad", augment=True
        )
        dl = DataLoader(ds, batch_size=2)
        for batch in dl:
            print(batch)
