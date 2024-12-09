from pathlib import Path
from unittest import TestCase

import pandas as pd
from torch.utils.data import DataLoader

from prometheo.datasets import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
)
from prometheo.models import Presto
from prometheo.predictors import DEM_BANDS, METEO_BANDS, S1_BANDS, S2_BANDS, collate_fn


models_to_test = [Presto]


def load_dataframe(timestep_freq="month"):
    dir = Path(__file__).resolve().parent
    if timestep_freq == "month":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")
    elif timestep_freq == "dekad":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset_10D.parquet")
    else:
        raise ValueError(f"Invalid timestep frequency `{timestep_freq}`")


class TestDataset(TestCase):
    def check_batch(self, batch, batch_size, num_timesteps):
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
        self.assertTrue((batch.timestamps[:, :, 1] <= 12).all())
        self.assertTrue((batch.timestamps[:, :, 1] >= 1).all())
        self.assertTrue((batch.timestamps[:, :, 0] <= 31).all())
        self.assertTrue((batch.timestamps[:, :, 0] >= 1).all())

        # if we are still using this software in 100 years we will need
        # to update this. This should catch errors in year transformations
        # though
        self.assertTrue((batch.timestamps[:, :, 2] >= 1999).all())
        self.assertTrue((batch.timestamps[:, :, 0] <= 2124).all())
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertTrue((batch.latlon[:, 0] >= -90).all())
        self.assertTrue((batch.latlon[:, 0] <= 90).all())
        self.assertTrue((batch.latlon[:, 1] >= -180).all())
        self.assertTrue((batch.latlon[:, 1] <= 180).all())
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))

    def test_WorldCerealDataset(self):
        df = load_dataframe()
        ds = WorldCerealDataset(df)
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_WorldCerealDataset_10D(self):
        # Test dekadal version of worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        num_timesteps = 36
        ds = WorldCerealDataset(df, num_timesteps=num_timesteps, timestep_freq="dekad")
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, num_timesteps)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_WorldCerealLabelledDataset(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True)
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

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
        self.check_batch(batch, batch_size, num_timesteps)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )
