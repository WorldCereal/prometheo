import unittest
from abc import ABC

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader

from prometheo.datasets.worldcereal import (
    WorldCerealDataset,
    WorldCerealLabelledDataset,
    align_to_composite_window,
    get_dekad_timestamp_components,
    get_monthly_timestamp_components,
    run_model_inference,
)
from prometheo.models import Presto
from prometheo.models.presto.wrapper import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    load_presto_weights,
)
from prometheo.predictors import collate_fn
from prometheo.utils import data_dir

models_to_test = [Presto]


class WorldCerealDatasetBaseTest(unittest.TestCase, ABC):
    __test__ = False  # Explicitly mark this class as not a test

    def setUp(self, num_timesteps, timestep_freq, start_date, end_date):
        """Set up test data for datasets tests."""
        self.num_samples = 5
        self.num_timesteps = num_timesteps
        self.timestep_freq = timestep_freq

        # Create a dataframe with the required columns
        data = {
            "lat": [45.1, 45.2, 45.3, 45.4, 45.5],
            "lon": [5.1, 5.2, 5.3, 5.4, 5.5],
            "start_date": [start_date] * self.num_samples,
            "end_date": [end_date] * self.num_samples,
            "available_timesteps": [self.num_timesteps] * self.num_samples,
            "valid_position": [self.num_timesteps // 2] * self.num_samples,
        }

        for ts in range(self.num_timesteps + (6 if timestep_freq is None else 18)):
            for band_template in [
                "OPTICAL-B02-ts{}-10m",
                "OPTICAL-B03-ts{}-10m",
                "OPTICAL-B04-ts{}-10m",
                "OPTICAL-B08-ts{}-10m",
                "OPTICAL-B05-ts{}-20m",
                "OPTICAL-B06-ts{}-20m",
                "OPTICAL-B07-ts{}-20m",
                "OPTICAL-B8A-ts{}-20m",
                "OPTICAL-B11-ts{}-20m",
                "OPTICAL-B12-ts{}-20m",
            ]:
                data[band_template.format(ts)] = [1000 + ts * 10] * self.num_samples

            for band_template in ["SAR-VH-ts{}-20m", "SAR-VV-ts{}-20m"]:
                data[band_template.format(ts)] = [0.01 + ts * 0.001] * self.num_samples

            for band_template in [
                "METEO-precipitation_flux-ts{}-100m",
                "METEO-temperature_mean-ts{}-100m",
            ]:
                data[band_template.format(ts)] = [10 + ts] * self.num_samples

        data["DEM-alt-20m"] = [100] * self.num_samples
        data["DEM-slo-20m"] = [5] * self.num_samples
        data["finetune_class"] = [
            "cropland",
            "not_cropland",
            "cropland",
            "cropland",
            "not_cropland",
        ]

        self.df = pd.DataFrame(data)

        self.base_ds = WorldCerealDataset(
            self.df,
            num_timesteps=self.num_timesteps,
            timestep_freq=self.timestep_freq,
            augment=True,
        )
        self.binary_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            timestep_freq=self.timestep_freq,
            num_timesteps=self.num_timesteps,
        )
        self.multiclass_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="multiclass",
            num_outputs=4,
            classes_list=["cropland", "not_cropland", "other1", "other2"],
            timestep_freq=self.timestep_freq,
            num_timesteps=self.num_timesteps,
        )
        self.time_explicit_ds = WorldCerealLabelledDataset(
            self.df,
            task_type="binary",
            num_outputs=1,
            time_explicit=True,
            timestep_freq=self.timestep_freq,
            num_timesteps=self.num_timesteps,
        )

    def test_dataset_length(self):
        """Test that dataset length matches dataframe length."""
        self.assertEqual(len(self.base_ds), self.num_samples)
        self.assertEqual(len(self.binary_ds), self.num_samples)
        self.assertEqual(len(self.multiclass_ds), self.num_samples)

    def test_get_timestep_positions(self):
        """Test getting timestep positions works correctly."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(row)

        # Check we got the right number of timesteps
        self.assertEqual(len(timestep_positions), self.num_timesteps)

        # Check the valid position is in the timestep positions
        self.assertIn(valid_position, timestep_positions)

        # Test with augmentation - ensure we have enough timesteps for augmentation
        # Modify the row to have a larger number of available timesteps
        augmented_row = row.copy()
        augmented_row["available_timesteps"] = (
            self.num_timesteps + 4
        )  # Increase number of available timesteps

        # Test with augmentation
        timestep_positions, valid_position = self.base_ds.get_timestep_positions(
            augmented_row
        )
        self.assertEqual(len(timestep_positions), self.num_timesteps)
        self.assertIn(valid_position, timestep_positions)

    def test_batch_shapes(self):
        """Test input initialization creates correct array shapes."""
        batch_size = 2
        dl = DataLoader(self.base_ds, batch_size=batch_size, collate_fn=collate_fn)
        batch = next(iter(dl))
        self.assertEqual(
            batch.s1.shape, (batch_size, 1, 1, self.num_timesteps, len(S1_BANDS))
        )
        self.assertEqual(
            batch.s2.shape, (batch_size, 1, 1, self.num_timesteps, len(S2_BANDS))
        )
        self.assertEqual(
            batch.meteo.shape, (batch_size, self.num_timesteps, len(METEO_BANDS))
        )
        self.assertEqual(batch.timestamps.shape, (batch_size, self.num_timesteps, 3))
        self.assertEqual(batch.dem.shape, (batch_size, 1, 1, len(DEM_BANDS)))

    def test_initialize_inputs(self):
        """Test input initialization creates correct array shapes."""
        s1, s2, meteo, dem = self.base_ds.initialize_inputs()

        # Check shapes
        self.assertEqual(s1.shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(s2.shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(meteo.shape, (self.num_timesteps, len(METEO_BANDS)))
        self.assertEqual(dem.shape, (1, 1, len(DEM_BANDS)))

        # Check all initialized with NODATAVALUE
        self.assertTrue(np.all(s1 == NODATAVALUE))
        self.assertTrue(np.all(s2 == NODATAVALUE))
        self.assertTrue(np.all(meteo == NODATAVALUE))
        self.assertTrue(np.all(dem == NODATAVALUE))

    def test_get_inputs(self):
        """Test getting inputs from a row."""
        row = pd.Series.to_dict(self.df.iloc[0, :])
        timestep_positions, _ = self.base_ds.get_timestep_positions(row)

        inputs = self.base_ds.get_inputs(row, timestep_positions)

        # Check all required keys are in the inputs
        self.assertIn("s1", inputs)
        self.assertIn("s2", inputs)
        self.assertIn("meteo", inputs)
        self.assertIn("dem", inputs)
        self.assertIn("latlon", inputs)
        self.assertIn("timestamps", inputs)

        # Check shapes
        self.assertEqual(inputs["s1"].shape, (1, 1, self.num_timesteps, len(S1_BANDS)))
        self.assertEqual(inputs["s2"].shape, (1, 1, self.num_timesteps, len(S2_BANDS)))
        self.assertEqual(inputs["meteo"].shape, (self.num_timesteps, len(METEO_BANDS)))
        self.assertEqual(inputs["dem"].shape, (1, 1, len(DEM_BANDS)))
        self.assertEqual(inputs["latlon"].shape, (1, 1, 2))
        self.assertEqual(inputs["timestamps"].shape, (self.num_timesteps, 3))

        # Check data has been filled in (not all NODATAVALUE)
        self.assertTrue(np.any(inputs["s1"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["s2"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["meteo"] != NODATAVALUE))
        self.assertTrue(np.any(inputs["dem"] != NODATAVALUE))

    def test_getitem(self):
        """Test __getitem__ returns correct type."""
        item = self.base_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")

        # Test labelled dataset
        item = self.binary_ds[0]
        self.assertEqual(type(item).__name__, "Predictors")
        self.assertTrue(hasattr(item, "label"))

    def test_labels(self):
        """Test binary and multiclass labelled dataset returns correct labels."""
        binary_item = self.binary_ds[0]
        self.assertEqual(binary_item.label[0, 0, 0, 0], 1)

        multiclass_item = self.multiclass_ds[4]
        self.assertEqual(multiclass_item.label[0, 0, 0, 0], 1)

    def test_time_explicit_label(self):
        """Test time explicit labelled dataset returns correct label shape."""
        item = self.time_explicit_ds[0]
        # Label should have temporal dimension
        self.assertEqual(item.label.shape, (1, 1, self.num_timesteps, 1))

        # For time_explicit, valid_position should have a value, other positions should be NODATAVALUE
        row = pd.Series.to_dict(self.df.iloc[0, :])
        _, valid_position = self.time_explicit_ds.get_timestep_positions(row)

        # Check only one position has a value
        valid_values = (item.label != NODATAVALUE).sum()
        self.assertEqual(valid_values, 1)


class TestWorldCerealMonthlyDataset(WorldCerealDatasetBaseTest):
    __test__ = True  # Explicitly mark this class as a test

    def setUp(self):
        super().setUp(
            num_timesteps=12,
            timestep_freq="month",
            start_date="2021-01-01",
            end_date="2022-01-01",
        )

    def test_get_timestamps(self):
        """Test timestamps for monthly dataset."""
        row = pd.Series.to_dict(self.base_ds.dataframe.iloc[0, :])
        ref_timestamps = np.array(
            (
                [
                    [1, 1, 2021],
                    [1, 2, 2021],
                    [1, 3, 2021],
                    [1, 4, 2021],
                    [1, 5, 2021],
                    [1, 6, 2021],
                    [1, 7, 2021],
                    [1, 8, 2021],
                    [1, 9, 2021],
                    [1, 10, 2021],
                    [1, 11, 2021],
                    [1, 12, 2021],
                ]
            )
        )

        computed_timestamps = self.base_ds._get_timestamps(
            row, self.base_ds.get_timestep_positions(row)[0]
        )

        np.testing.assert_array_equal(computed_timestamps, ref_timestamps)


class TestWorldCerealDekadalDataset(WorldCerealDatasetBaseTest):
    __test__ = True  # Explicitly mark this class as a test

    def setUp(self):
        super().setUp(
            num_timesteps=36,
            timestep_freq="dekad",
            start_date="2022-07-17",
            end_date="2023-09-28",
        )

    def test_get_timestamps(self):
        """Test timestamps for dekadal dataset."""
        row = pd.Series.to_dict(self.base_ds.dataframe.iloc[0, :])
        ref_timestamps = np.array(
            [
                [11, 7, 2022],
                [21, 7, 2022],
                [1, 8, 2022],
                [11, 8, 2022],
                [21, 8, 2022],
                [1, 9, 2022],
                [11, 9, 2022],
                [21, 9, 2022],
                [1, 10, 2022],
                [11, 10, 2022],
                [21, 10, 2022],
                [1, 11, 2022],
                [11, 11, 2022],
                [21, 11, 2022],
                [1, 12, 2022],
                [11, 12, 2022],
                [21, 12, 2022],
                [1, 1, 2023],
                [11, 1, 2023],
                [21, 1, 2023],
                [1, 2, 2023],
                [11, 2, 2023],
                [21, 2, 2023],
                [1, 3, 2023],
                [11, 3, 2023],
                [21, 3, 2023],
                [1, 4, 2023],
                [11, 4, 2023],
                [21, 4, 2023],
                [1, 5, 2023],
                [11, 5, 2023],
                [21, 5, 2023],
                [1, 6, 2023],
                [11, 6, 2023],
                [21, 6, 2023],
                [1, 7, 2023],
            ]
        )

        computed_timestamps = self.base_ds._get_timestamps(
            row, self.base_ds.get_timestep_positions(row)[0]
        )

        np.testing.assert_array_equal(computed_timestamps, ref_timestamps)


class TestTimeUtilities(unittest.TestCase):
    def test_align_to_composite_window(self):
        """Test aligning dates to composite window."""
        # Test with dekad frequency
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-01-24", "D")
        aligned_start = align_to_composite_window(start_date, "dekad")
        aligned_end = align_to_composite_window(end_date, "dekad")

        # Should align to first dekad of January
        self.assertEqual(aligned_start, np.datetime64("2021-01-01", "D"))
        self.assertEqual(aligned_end, np.datetime64("2021-01-21", "D"))

        # Test with monthly frequency
        start_date = np.datetime64("2021-01-15", "D")
        end_date = np.datetime64("2021-02-10", "D")
        aligned_start = align_to_composite_window(start_date, "month")
        aligned_end = align_to_composite_window(end_date, "month")

        # Should align to first day of month
        self.assertEqual(aligned_start, np.datetime64("2021-01-01", "D"))
        self.assertEqual(aligned_end, np.datetime64("2021-02-01", "D"))

    def test_get_timestamp_components(self):
        """Test getting month timestamp components."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-12-24", "D")

        days, months, years = get_monthly_timestamp_components(start_date, end_date)

        # Should have 12 months
        self.assertEqual(len(days), 12)
        self.assertEqual(len(months), 12)
        self.assertEqual(len(years), 12)

        # All days should be 1 (first day of month)
        self.assertTrue(np.all(days == 1))

        # Months should be 1-12
        self.assertTrue(np.all(months == np.arange(1, 13)))

        # All years should be 2021
        self.assertTrue(np.all(years == 2021))

    def test_get_dekad_timestamp_components(self):
        """Test getting dekad timestamp components."""
        start_date = np.datetime64("2021-01-03", "D")
        end_date = np.datetime64("2021-01-24", "D")

        days, months, years = get_dekad_timestamp_components(start_date, end_date)

        # Should have 3 dekads per month
        self.assertEqual(len(days), 3)

        # Days should be 1, 11, 21 for first month
        self.assertTrue(np.all(days == np.array([1, 11, 21])))

        # All months should be 1 (January)
        self.assertTrue(np.all(months == 1))

        # All years should be 2021
        self.assertTrue(np.all(years == 2021))


class TestInference(unittest.TestCase):
    def test_run_model_inference(self):
        """Test the run_model_inference function. Based on reference features
        generated using the following code at commit 6f0f7d6 :

        arr = xr.open_dataarray(data_dir / "test_inference_array.nc")
        model_url = str(data_dir / "finetuned_presto_model.pt")
                presto_features = get_presto_features(
                    arr, model_url, batch_size=512, epsg=32631
                )
        presto_features.to_netcdf(data_dir / "test_presto_inference_features.nc")

        Features were regenerated at commit 6f0f7d6 since this fixed a bug with dtypes
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


if __name__ == "__main__":
    unittest.main()
