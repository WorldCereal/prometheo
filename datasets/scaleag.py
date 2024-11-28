from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple

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
    }
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_outputs: int,
        num_timesteps: int,  #
        task_type: Literal["regression", "binary", "multiclass"],
        target_name: str,
        compositing_window: Literal["dek", "monthly"] = "monthly",
    ):
        super().__init__(dataframe, num_outputs, num_timesteps, task_type)
        self.df = dataframe.replace({np.nan: NODATAVALUE})
        self.target_name = target_name
        self.task_type = task_type
        self.num_timesteps = num_timesteps
        self.compositing_window = compositing_window

        if self.task_type == "multiclass":
            self.class_to_index = {
                label: idx for idx, label in enumerate(dataframe[target_name].unique())
            }
            self.index_to_class = {
                idx: label for idx, label in enumerate(dataframe[target_name].unique())
            }

    def get_target(self, row_d: pd.Series) -> int:
        return int(row_d[self.target_name])

    def get_predictors(self, row: pd.Series) -> Predictors:
        row_d = pd.Series.to_dict(row)
        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)
        month = datetime.strptime(row_d["date"], "%Y-%m-%d").month

        s1, s2, meteo, dem = [], [], [], []

        for df_val, presto_val in self.BAND_MAPPING.items():
            # retrieve ts for each band column df_val
            values = np.array(
                [float(row_d[df_val.format(t)]) for t in range(self.num_timesteps)]
            )
            values = np.nan_to_num(values, nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            if presto_val in S2_bands:
                s2.append(values)
            elif presto_val in S1_bands:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1.append(values)
            elif presto_val == "precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo.append(values)
            elif presto_val == "temperature":
                # remove scaling. conversion to celsius is done in the normalization
                values[idx_valid] = values[idx_valid] / 100
                meteo.append(values)
        for df_val, presto_val in self.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            dem.append(values)
        return Predictors(
            s1=np.array(s1),
            s2=np.array(s2),
            meteo=np.array(meteo),
            dem=np.array(dem),
            latlon=latlon,
            label=self.get_target(row_d),
            month=(
                self.get_month_array(row) if self.compositing_window == "dek" else month
            ),
        )

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
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
