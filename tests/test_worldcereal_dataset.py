from pathlib import Path
from unittest import TestCase

import pandas as pd
from torch.utils.data import DataLoader

from prometheo.datasets.worldcereal import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
)
from prometheo.predictors import DEM_BANDS, METEO_BANDS, S1_BANDS, S2_BANDS, collate_fn


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
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.assertEqual(batch.s1.shape, (batch_size, 1, 1, 12, len(S1_BANDS)))
        self.assertEqual(batch.s2.shape, (batch_size, 1, 1, 12, len(S2_BANDS)))
        self.assertEqual(batch.meteo.shape, (batch_size, 12, len(METEO_BANDS)))
        self.assertEqual(batch.timestamps.shape, (batch_size, 12, 3))
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))

    def test_WorldCerealDataset_10D(self):
        # Test dekadal version of worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        ds = WorldCerealDataset(df, num_timesteps=36, timestep_freq="dekad")
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.assertEqual(batch.s1.shape, (batch_size, 1, 1, 12, len(S1_BANDS)))
        self.assertEqual(batch.s2.shape, (batch_size, 1, 1, 12, len(S2_BANDS)))
        self.assertEqual(batch.meteo.shape, (batch_size, 12, len(METEO_BANDS)))
        self.assertEqual(batch.timestamps.shape, (batch_size, 12, 3))
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))

    def test_WorldCerealLabelledDataset(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True)
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.assertEqual(batch.s1.shape, (batch_size, 1, 1, 12, len(S1_BANDS)))
        self.assertEqual(batch.s2.shape, (batch_size, 1, 1, 12, len(S2_BANDS)))
        self.assertEqual(batch.meteo.shape, (batch_size, 12, len(METEO_BANDS)))
        self.assertEqual(batch.timestamps.shape, (batch_size, 12, 3))
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))
        self.assertEqual(batch.label.shape, (batch_size, 1, 1, 12, 1))

    def test_WorldCerealLabelledDataset_10D(self):
        # Test dekadal version of labelled worldcereal dataset
        # Also test fewer timesteps and more than 1 output
        df = load_dataframe(timestep_freq="dekad")
        num_outputs = 2
        num_timesteps = 24
        ds = WorldCerealLabelledDataset(
            df,
            num_timesteps=num_timesteps,
            timestep_freq="dekad",
            num_outputs=num_outputs,
            augment=True,
        )
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.assertEqual(
            batch.s1.shape, (batch_size, 1, 1, num_timesteps, len(S1_BANDS))
        )
        self.assertEqual(
            batch.s2.shape, (batch_size, 1, 1, num_timesteps, len(S2_BANDS))
        )
        self.assertEqual(
            batch.meteo.shape, (batch_size, num_timesteps, len(METEO_BANDS))
        )
        self.assertEqual(batch.timestamps.shape, (batch_size, num_timesteps, 3))
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))
        self.assertEqual(
            batch.label.shape, (batch_size, 1, 1, num_timesteps, num_outputs)
        )
