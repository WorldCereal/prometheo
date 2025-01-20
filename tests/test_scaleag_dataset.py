from unittest import TestCase

import numpy as np
from resources import load_dataframe
from torch.utils.data import DataLoader

from prometheo.datasets import ScaleAgDataset
from prometheo.models import Presto
from prometheo.predictors import DEM_BANDS, METEO_BANDS, S1_BANDS, S2_BANDS, collate_fn

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
        self.assertEqual(batch.latlon.shape, (batch_size, 2))
        self.assertTrue((batch.latlon[:, 0] >= -90).all())
        self.assertTrue((batch.latlon[:, 0] <= 90).all())
        self.assertTrue((batch.latlon[:, 1] >= -180).all())
        self.assertTrue((batch.latlon[:, 1] <= 180).all())
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

    def test_ScaleAgDataset_30D(self):
        df = load_dataframe()
        num_timesteps = 12
        ds = ScaleAgDataset(
            df, num_timesteps=num_timesteps, compositing_window="monthly"
        )
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

    def test_ScaleAgDataset_10D(self):
        # Test dekadal version of worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        num_timesteps = 36
        ds = ScaleAgDataset(df, num_timesteps=num_timesteps, compositing_window="dekad")
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

    def test_ScaleAgDataset_unitemporal_labelled(self):
        df = load_dataframe()
        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="binary",
            target_name="LANDCOVER_LABEL",
            positive_labels=[10, 11, 12, 13],
            compositing_window="monthly",
        )
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12)

        # Check uni-temporal label: need to have 1 timestep
        self.assertEqual(batch.label.shape, (batch_size, 1, 1, 1, 1))
        self.assertTrue((batch.label.unique().numpy() == [0, 1]).all())

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_ScaleAgDataset_num_outputs(self):
        df = load_dataframe()
        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="binary",
            target_name="LANDCOVER_LABEL",
            positive_labels=[10, 11, 12, 13],
            compositing_window="monthly",
        )
        # Check that num_output is properly set for the binary case
        self.assertTrue(ds.num_outputs == 1)

        df["regression_label"] = np.random.rand(len(df))
        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="regression",
            target_name="regression_label",
            compositing_window="monthly",
        )
        # Check that num_output is properly set for the regression case
        self.assertTrue(ds.num_outputs == 1)

        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="multiclass",
            target_name="LANDCOVER_LABEL",
            compositing_window="monthly",
        )
        # Check that num_output is properly set for the multiclass case
        self.assertTrue(ds.num_outputs == len(df.LANDCOVER_LABEL.unique()))

        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="ssl",
            compositing_window="monthly",
        )
        # Check that num_output is properly set for ssl case
        self.assertTrue(ds.num_outputs is None)

    def test_ScaleAGLabelledDataset_10D(self):
        # Test dekadal version of labelled worldcereal dataset
        df = load_dataframe(timestep_freq="dekad")
        num_timesteps = 36
        ds = ScaleAgDataset(
            df,
            num_timesteps=num_timesteps,
            task_type="binary",
            target_name="LANDCOVER_LABEL",
            positive_labels=[10, 11, 12, 13],
            compositing_window="dekad",
        )
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, num_timesteps, num_outputs=1)

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_ScaleAGLabelledDataset_regression(self):
        # Test dekadal version of labelled worldcereal dataset
        df = load_dataframe()
        df["LANDCOVER_LABEL"] = df["LANDCOVER_LABEL"].astype(float)
        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="regression",
            target_name="LANDCOVER_LABEL",
            compositing_window="monthly",
        )
        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12, num_outputs=1)
        self.assertTrue(
            (
                (batch.label.max().numpy() == 1) and (batch.label.min().numpy() == 0)
            ).all()
        )

        # test that assertion on type of regression target works
        df_str = df.copy()
        df_str.LANDCOVER_LABEL = df_str.LANDCOVER_LABEL.astype(str)
        with self.assertRaises(AssertionError) as regression_error:
            ScaleAgDataset(
                df_str,
                task_type="regression",
                target_name="LANDCOVER_LABEL",
            )
            self.assertEqual(
                str(regression_error.exception),
                "Regression target must be of type float",
            )

        for model_cls in models_to_test:
            model = model_cls()
            output = model(batch)
            self.assertEqual(
                output.shape[0],
                batch_size,
                f"Forward pass failed for {model.__class__.__name__}",
            )

    def test_ScaleAGLabelledDataset_multiclass(self):
        df = load_dataframe()
        num_outputs = len(df.LANDCOVER_LABEL.unique())
        ds = ScaleAgDataset(
            df,
            num_timesteps=12,
            task_type="multiclass",
            target_name="LANDCOVER_LABEL",
            compositing_window="monthly",
        )

        batch_size = 2
        dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.check_batch(batch, batch_size, 12, num_outputs=num_outputs)

        # checking that the the class mapping to indices is correct
        self.assertTrue((batch.label.unique().numpy() == [0, 1]).all())
        self.assertTrue(
            (ds.dataframe.iloc[:batch_size].LANDCOVER_LABEL == [99, 13]).all()
        )

    def test_ScaleAGDataset_10D_Timestamps(self):
        # Test that the dekad timestamp components are correctly calculated
        df = load_dataframe(timestep_freq="dekad")
        num_timesteps = 50
        ds = ScaleAgDataset(df, num_timesteps, compositing_window="dekad")
        row = ds.dataframe.iloc[0]
        row.start_date = "2018-06-15"
        row.end_date = "2019-10-25"
        components = ds.get_date_array(row)
        for c in components.T:
            self.assertEqual(len(c), num_timesteps)  # We expect to get 50 dekads

        # Check if the day components have been built correctly
        self.assertTrue((components[:, 0] == [11, 21] + [1, 11, 21] * 16).all())

        # Check if the month components have been built correctly
        self.assertTrue(
            (
                components[:, 1]
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
                components[:, 2]
                == [2018 for _ in range(20)] + [2019 for _ in range(30)]
            ).all()
        )

    def test_ScaleAGDatasetTimestamps(self):
        df = load_dataframe()
        num_timesteps = 12
        ds = ScaleAgDataset(df, num_timesteps, compositing_window="monthly")

        # Test that the timestamps are correctly transformed
        row = ds.dataframe.iloc[0, :]
        timestamps = ds.get_date_array(row)
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
