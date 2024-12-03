from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd
from datasets.base import DatasetBase
from src.predictors import (
    NODATAVALUE,
    Predictors,
    S1_bands,
    S2_bands,
    dem_bands,
    meteo_bands,
)


class ScaleAGDataset(DatasetBase):
    BAND_MAPPING = {
        "OPTICAL-B02-ts{}-10m": "B2",
        "OPTICAL-B03-ts{}-10m": "B3",
        "OPTICAL-B04-ts{}-10m": "B4",
        "OPTICAL-B05-ts{}-20m": "B5",
        "OPTICAL-B06-ts{}-20m": "B6",
        "OPTICAL-B07-ts{}-20m": "B7",
        "OPTICAL-B08-ts{}-10m": "B8",
        "OPTICAL-B8A-ts{}-20m": "B8A",
        "OPTICAL-B11-ts{}-20m": "B11",
        "OPTICAL-B12-ts{}-20m": "B12",
        "SAR-VH-ts{}-20m": "VH",
        "SAR-VV-ts{}-20m": "VV",
        "METEO-precipitation_flux-ts{}-100m": "precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature",
        "DEM-alt-20m": "elevation",
        "DEM-slo-20m": "slope",
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_timesteps: int,
        task_type: Literal["regression", "binary", "multiclass", "ssl"],
        num_outputs: int,
        target_name: Optional[str] = None,
        compositing_window: Literal["dek", "monthly"] = "monthly",
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        super().__init__(
            dataframe,
            num_timesteps,
            task_type,
            num_outputs,
        )
        self.target_name = target_name
        self.compositing_window = compositing_window

        # bound label values to valid range if upper and lower bounds are provided
        if task_type == "regression":
            if upper_bound is None or lower_bound is None:
                upper_bound = self.dataframe[target_name].max()
                lower_bound = self.dataframe[target_name].min()
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound
            self.dataframe[target_name] = self.dataframe[target_name].clip(
                lower=lower_bound, upper=upper_bound
            )

        # most of downstream classifiers expect target to be provided as [0, num_classes - 1]
        if self.task_type == "multiclass":
            self.class_to_index = {
                label: idx
                for idx, label in enumerate(self.dataframe[target_name].unique())
            }
            self.index_to_class = {
                idx: label
                for idx, label in enumerate(self.dataframe[target_name].unique())
            }

    def get_predictors(self, row: pd.Series) -> Predictors:

        row_d = pd.Series.to_dict(row)
        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["start_date"], "%Y-%m-%d").month

        # initialize sensor arrays filled with NODATAVALUE
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        meteo = np.full(
            (self.num_timesteps, len(meteo_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        dem = np.full((1, 1, len(dem_bands)), fill_value=NODATAVALUE, dtype=np.float32)

        # iterate over all bands and fill the corresponding arrays. convert to presto units if necessary
        for df_val, presto_val in self.BAND_MAPPING.items():
            # retrieve ts for each band column df_val
            values = np.array(
                [float(row_d[df_val.format(t)]) for t in range(self.num_timesteps)]
            )
            values = np.nan_to_num(values, nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            if presto_val in S2_bands:
                s2[..., S2_bands.index(presto_val)] = values
            elif presto_val in S1_bands:
                s1 = self.openeo_to_presto_units(s1, presto_val, values, idx_valid)
            elif presto_val == "precipitation":
                meteo = self.openeo_to_presto_units(
                    meteo, presto_val, values, idx_valid
                )
            elif presto_val == "temperature":
                meteo = self.openeo_to_presto_units(
                    meteo, presto_val, values, idx_valid
                )
            elif presto_val in dem_bands:
                dem = self.openeo_to_presto_units(dem, presto_val, values, idx_valid)

        predictors_dict = {
            "s1": s1,
            "s2": s2,
            "meteo": meteo,
            "dem": dem,
            "latlon": latlon,
            "month": (
                self.get_month_array(row) if self.compositing_window == "dek" else month
            ),
        }
        if self.task_type != "ssl":
            predictors_dict["label"] = self.get_target(row_d)

        return Predictors(**predictors_dict)

    def __getitem__(self, idx):
        # Get the sample
        row = self.dataframe.iloc[idx, :]
        return self.get_predictors(row)

    def get_month_array(self, row: pd.Series) -> np.ndarray:
        start_date, end_date = datetime.strptime(
            row.start_date, "%Y-%m-%d"
        ), datetime.strptime(row.end_date, "%Y-%m-%d")

        # Calculate the step size for 10-day intervals and create a list of dates
        step = int((end_date - start_date).days / (self.num_timesteps - 1))
        date_vector = [
            start_date + timedelta(days=i * step) for i in range(self.num_timesteps)
        ]

        # Ensure last date is not beyond the end date
        if date_vector[-1] > end_date:
            date_vector[-1] = end_date

        return np.array([d.month - 1 for d in date_vector])

    def get_target(self, row_d: pd.Series) -> np.ndarray:
        target = int(row_d[self.target_name])
        if self.task_type == "regression":
            target = self.normalize_target(target)
        # convert classes to indices for multiclass
        elif self.task_type == "multiclass":
            target = self.class_to_index[target]
        return np.array(target)

    def normalize_target(self, target):
        return (target - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def revert_to_original_units(self, target_norm):
        return target_norm * (self.upper_bound - self.lower_bound) + self.lower_bound

    def openeo_to_presto_units(self, band_array, presto_band, values, idx_valid):
        if presto_band in S1_bands:
            # convert to dB
            idx_valid = idx_valid & (values > 0)
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            band_array[..., S1_bands.index(presto_band)] = values * idx_valid

        elif presto_band == "precipitation":
            # scaling, and AgERA5 is in mm, Presto expects m
            values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            band_array[..., meteo_bands.index(presto_band)] = values * idx_valid

        elif presto_band == "temperature":
            # remove scaling. conversion to celsius is done in the normalization
            values[idx_valid] = values[idx_valid] / 100
            band_array[..., meteo_bands.index(presto_band)] = values * idx_valid
        elif presto_band in dem_bands:
            band_array[..., dem_bands.index(presto_band)] = values[0]

        return band_array
