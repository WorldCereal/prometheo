import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from base import DatasetBase

from src.predictors import (NODATAVALUE, Predictors, S1_bands, S2_bands,
                            dem_bands, meteo_bands)

logger = logging.getLogger("__main__")


class WorldCerealBase(DatasetBase):
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

    @classmethod
    def get_timestep_positions(
        cls, row_d: Dict, num_timesteps: int, augment: bool = False, is_ssl: bool = False, MIN_EDGE_BUFFER: int = 2
    ) -> List[int]:
        available_timesteps = int(row_d["available_timesteps"])

        if is_ssl:
            if available_timesteps == num_timesteps:
                valid_position = int(num_timesteps // 2)
            else:
                valid_position = int(
                    np.random.choice(
                        range(
                            num_timesteps // 2, (available_timesteps - num_timesteps // 2)
                        ),
                        1,
                    )
                )
            center_point = valid_position
        else:
            valid_position = int(row_d["valid_position"])
            if not augment:
                #  check if the valid position is too close to the start_date and force shifting it
                if valid_position < num_timesteps // 2:
                    center_point = num_timesteps // 2
                #  or too close to the end_date
                elif valid_position > (available_timesteps - num_timesteps // 2):
                    center_point = available_timesteps - num_timesteps // 2
                else:
                    # Center the timesteps around the valid position
                    center_point = valid_position
            else:
                # Shift the center point but make sure the resulting range
                # well includes the valid position

                min_center_point = max(
                    num_timesteps // 2,
                    valid_position + MIN_EDGE_BUFFER - num_timesteps // 2,
                )
                max_center_point = min(
                    available_timesteps - num_timesteps // 2,
                    valid_position - MIN_EDGE_BUFFER + num_timesteps // 2,
                )

                center_point = np.random.randint(
                    min_center_point, max_center_point + 1
                )  # max_center_point included

        last_timestep = min(available_timesteps, center_point + num_timesteps // 2)
        first_timestep = max(0, last_timestep - num_timesteps)
        timestep_positions = list(range(first_timestep, last_timestep))

        if len(timestep_positions) != num_timesteps:
            raise ValueError(
                f"Acquired timestep positions do not have correct length: \
required {num_timesteps}, got {len(timestep_positions)}"
            )
        assert (
            valid_position in timestep_positions
        ), f"Valid position {valid_position} not in timestep positions {timestep_positions}"
        return timestep_positions

    @classmethod
    def get_predictors(
        cls,
        row: pd.Series,
        num_timesteps: int,
        augment: bool = False,
        is_ssl: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        timestep_positions = cls.get_timestep_positions(row_d, num_timesteps=num_timesteps, augment=augment, is_ssl=is_ssl)

        if num_timesteps == 12:
            initial_start_date_position = pd.to_datetime(row_d["start_date"]).month
        elif num_timesteps > 12:
            # get the correct index of the start_date based on num_timesteps`
            # e.g. if num_timesteps is 36 (dekadal setup), we should take the correct
            # 10-day interval that the start_date falls into
            # TODO: 1) this needs to go into a separate function
            # 2) definition of valid_position and timestep_ind
            #  should also be changed accordingly
            year = pd.to_datetime(row_d["start_date"]).year
            year_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
            bins = pd.cut(year_dates, bins=num_timesteps, labels=False)
            initial_start_date_position = bins[
                np.where(year_dates == pd.to_datetime(row_d["start_date"]))[0][0]
            ]
        else:
            raise ValueError(
                f"num_timesteps must be at least 12. Currently it is {num_timesteps}"
            )

        # make sure that month for encoding gets shifted according to
        # the selected timestep positions. Also ensure circular indexing
        month = (initial_start_date_position - 1 + timestep_positions[0]) % num_timesteps

        # adding workaround for compatibility between Phase I and Phase II datasets.
        # (in Phase II, the relevant attribute name was changed to valid_time)
        # once we fully move to Phase II data, this should be replaced to valid_tome only.
        if "valid_date" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_date"], "%Y-%m-%d").month - 1
        elif "valid_time" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_time"], "%Y-%m-%d").month - 1
        else:
            logger.error("Dataset does not contain neither valid_date, nor valid_time attribute.")

        s1 = np.full((1, 1, num_timesteps, len(S1_bands)), fill_value=NODATAVALUE, dtype=np.float32) # [B, H, W, T, len(S1_bands)]
        s2 = np.full((1, 1, num_timesteps, len(S2_bands)), fill_value=NODATAVALUE, dtype=np.float32) # [B, H, W, T, len(S2_bands)]
        meteo = np.full((num_timesteps, len(meteo_bands)), fill_value=NODATAVALUE, dtype=np.float32) # [B, T, len(meteo_bands)]
        dem = np.full((1, 1, len(dem_bands)), fill_value=NODATAVALUE, dtype=np.float32) # [B, H, W, len(dem_bands)]

        for df_val, presto_val in cls.BAND_MAPPING.items():
            values = np.array([float(row_d[df_val.format(t)]) for t in timestep_positions])
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            
            if presto_val in S2_bands:
                s2[..., S2_bands.index(presto_val)] = values * idx_valid
            elif presto_val in S1_bands:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_bands.index(presto_val)] = values * idx_valid
            elif presto_val == "precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., meteo_bands.index(presto_val)] = values * idx_valid
            elif presto_val == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., meteo_bands.index(presto_val)] = values * idx_valid
            elif presto_val in dem_bands:
                values = values[0]  # dem is not temporal
                dem[..., dem_bands.index(presto_val)] = values
            else:
                raise ValueError(f"Unknown band {presto_val}")
            
        return Predictors(
            s1=s1,
            s2=s2,
            meteo=meteo,
            dem=dem,
            latlon=latlon,
            aux_inputs=[valid_month],
            month=month,
            )


    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx, :]
        is_ssl = self.task_type == "ssl"
        return self.get_predictors(row, num_timesteps=self.num_timesteps, is_ssl=is_ssl)