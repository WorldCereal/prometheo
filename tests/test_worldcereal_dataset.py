from unittest import TestCase

import numpy as np
import pandas as pd
import xarray as xr
from resources import load_dataframe
from torch.utils.data import DataLoader

from prometheo.datasets import WorldCerealDataset, WorldCerealLabelledDataset
from prometheo.datasets.worldcereal import (
    get_dekad_timestamp_components,
    get_monthly_timestamp_components,
    run_model_inference,
)
from prometheo.models import Presto
from prometheo.models.presto.wrapper import load_presto_weights
from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    collate_fn,
)
from prometheo.utils import data_dir

models_to_test = [Presto]


class TestDataset(TestCase):
    def check_batch(self, batch, batch_size, num_timesteps, num_outputs=1):
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
        self.assertEqual(batch.latlon.shape, (batch_size, 1, 1, 2))
        self.assertTrue((batch.latlon[:, :, :, 0] >= -90).all())
        self.assertTrue((batch.latlon[:, :, :, 0] <= 90).all())
        self.assertTrue((batch.latlon[:, :, :, 1] >= -180).all())
        self.assertTrue((batch.latlon[:, :, :, 1] <= 180).all())
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))

        if batch.label is not None:
            # Label should either have a single timestep or num_timesteps
            self.assertTrue(
                any(
                    [
                        batch.label.shape
                        == (batch_size, 1, 1, num_timesteps, num_outputs),
                        batch.label.shape == (batch_size, 1, 1, 1, num_outputs),
                    ]
                )
            )

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

    def test_WorldCerealLabelledDataset_time_explicit(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True, time_explicit=True)
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12)

        # Check time-explicit label: need to have 12 timesteps
        self.assertEqual(batch.label.shape, (batch_size, 1, 1, 12, 1))
        self.assertTrue((np.isin(batch.label.unique().numpy(), [0, NODATAVALUE])).all())

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_WorldCerealLabelledDataset_unitemporal(self):
        df = load_dataframe()
        ds = WorldCerealLabelledDataset(df, augment=True)
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12)

        # Check uni-temporal label: need to have 1 timesteps
        self.assertEqual(batch.label.shape, (batch_size, 1, 1, 1, 1))
        self.assertTrue((batch.label.unique().numpy() == 0).all())

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
        self.check_batch(batch, batch_size, num_timesteps, num_outputs=num_outputs)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_get_monthly_timestamp_components(self):
        # Test that the monthly timestamp components are correctly calculated
        start_date = pd.to_datetime("2018-06-15")
        end_date = pd.to_datetime("2019-10-25")
        components = get_monthly_timestamp_components(start_date, end_date)
        for c in components:
            self.assertEqual(len(c), 17)  # We expect to get 17 months

        # Monthly timesteps should always start on the first day of the month
        self.assertTrue((components[0] == 1).all())

        # Check if the month components have been built correctly
        self.assertTrue(
            (
                components[1] == np.concatenate([np.arange(6, 13), np.arange(1, 11)])
            ).all()
        )

        # Check if the year components have been built correctly
        self.assertTrue(
            (
                components[2] == [2018 for _ in range(7)] + [2019 for _ in range(10)]
            ).all()
        )

    def test_get_dekad_timestamp_components(self):
        # Test that the dekad timestamp components are correctly calculated
        start_date = pd.to_datetime("2018-06-15")
        end_date = pd.to_datetime("2019-10-25")
        components = get_dekad_timestamp_components(start_date, end_date)
        for c in components:
            self.assertEqual(len(c), 50)  # We expect to get 50 dekads

        # Check if the day components have been built correctly
        self.assertTrue((components[0] == [11, 21] + [1, 11, 21] * 16).all())

        # Check if the month components have been built correctly
        self.assertTrue(
            (
                components[1]
                == [6, 6]
                + [
                    i
                    for i in (list(range(7, 13)) + list(range(1, 11)))
                    for _ in range(3)
                ]
            ).all()
        )

        # Check if the year components have been built correctly
        self.assertTrue(
            (
                components[2] == [2018 for _ in range(20)] + [2019 for _ in range(30)]
            ).all()
        )

    def test_WorldCerealDatasetTimestamps(self):
        df = load_dataframe()
        ds = WorldCerealDataset(df)

        # Test that the timestamps are correctly transformed
        row = pd.Series.to_dict(ds.dataframe.iloc[0, :])
        timestamps = ds._get_timestamps(row, [ts for ts in range(12)])
        self.assertTrue((timestamps[:, 0] == 1).all())
        self.assertTrue(
            (
                timestamps[:, 1] == np.concatenate([np.arange(8, 13), np.arange(1, 8)])
            ).all()
        )
        self.assertTrue(
            (
                timestamps[:, 2] == [2020 for _ in range(5)] + [2021 for _ in range(7)]
            ).all()
        )


class TestInference(TestCase):
    def test_run_model_inference(self):
        """Test the run_model_inference function. Based on ref features
        generated using this method.
        """
        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")

        # Load a pretrained Presto model
        model_url = str(data_dir / "finetuned_presto_model.pt")
        presto_model = Presto()
        presto_model = load_presto_weights(presto_model, model_url)

        presto_features = run_model_inference(
            arr, presto_model, batch_size=512, epsg=32631
        )

        # Uncomment to regenerate ref features
        # presto_features.to_netcdf(data_dir / "test_presto_inference_features.nc")

        # Load ref features
        ref_presto_features = xr.open_dataarray(
            data_dir / "test_presto_inference_features.nc"
        )

        xr.testing.assert_allclose(
            presto_features,
            ref_presto_features,
            rtol=1e-04,
            atol=1e-04,
        )

        assert presto_features.dims == ref_presto_features.dims
