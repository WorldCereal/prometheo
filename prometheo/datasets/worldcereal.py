import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from prometheo.predictors import (
    NODATAVALUE,
    Predictors,
    S1_bands,
    S2_bands,
    dem_bands,
    meteo_bands,
)

logger = logging.getLogger("__main__")


class WorldCerealDataset(Dataset):
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
        num_timesteps: int = 12,
        timestep_freq: str = "month",
        task_type: Literal["ssl", "binary", "multiclass", "regression"] = "ssl",
        num_outputs: Optional[int] = None,
        augment: bool = False,
    ):
        """WorldCereal base dataset. This dataset is typically used for
        self-supervised learning.

        Parameters
        ----------
        dataframe : pd.DataFrame
            input dataframe containing the data
        num_timesteps : int, optional
            number of timesteps for a sample, by default 12
        timestep_freq : str, optional. Should be one of ['month', 'dekad']
            frequency of the timesteps, by default "month"
        task_type : str, optional. One of ['ssl', 'binary', 'multiclass', 'regression']
            type of the task, by default self-supervised learning "ssl"
        num_outputs : int, optional
            number of outputs for the task, by default None. If task_type is 'ssl',
            the value of this parameter is ignored.
        augment : bool, optional
            whether to augment the data, by default False
        """
        self.dataframe = dataframe.replace({np.nan: NODATAVALUE})
        self.num_timesteps = num_timesteps

        if timestep_freq not in ["month", "dekad"]:
            raise NotImplementedError(
                f"timestep_freq should be one of ['month', 'dekad']. Got `{timestep_freq}`"
            )
        self.timestep_freq = timestep_freq
        self.task_type = task_type
        self.num_outputs = num_outputs
        self.is_ssl = task_type == "ssl"
        self.augment = augment

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, _ = self.get_timestep_positions(row)
        return Predictors(**self.get_inputs(row, timestep_positions))

    def get_timestep_positions(
        self,
        row_d: Dict,
        MIN_EDGE_BUFFER: int = 2,
    ) -> Tuple[List[int], int]:

        available_timesteps = int(row_d["available_timesteps"])
        valid_position = int(row_d["valid_position"])

        # Get the center point to use for extracting a sequence of timesteps
        center_point = self._get_center_point(
            available_timesteps, valid_position, self.augment, MIN_EDGE_BUFFER
        )

        # Determine the timestep positions to extract
        last_timestep = min(available_timesteps, center_point + self.num_timesteps // 2)
        first_timestep = max(0, last_timestep - self.num_timesteps)
        timestep_positions = list(range(first_timestep, last_timestep))

        # Sanity check to make sure we will extract the correct number of timesteps
        if len(timestep_positions) != self.num_timesteps:
            raise ValueError(
                (
                    "Acquired timestep positions do not have correct length: "
                    f"required {self.num_timesteps}, got {len(timestep_positions)}"
                )
            )

        # Sanity check to make sure valid_position is still within the extracted timesteps
        assert (
            valid_position in timestep_positions
        ), f"Valid position {valid_position} not in timestep positions {timestep_positions}"

        return timestep_positions, valid_position

    def _get_center_point(
        self, available_timesteps, valid_position, augment, min_edge_buffer
    ):
        """Helper method to decide on the center point based on which to
        extract the timesteps."""

        if not augment:
            #  check if the valid position is too close to the start_date and force shifting it
            if valid_position < self.num_timesteps // 2:
                center_point = self.num_timesteps // 2
            #  or too close to the end_date
            elif valid_position > (available_timesteps - self.num_timesteps // 2):
                center_point = available_timesteps - self.num_timesteps // 2
            else:
                # Center the timesteps around the valid position
                center_point = valid_position
        else:
            if self.is_ssl:
                # Take a random center point enabling horizontal jittering
                center_point = int(
                    np.random.choice(
                        range(
                            self.num_timesteps // 2,
                            (available_timesteps - self.num_timesteps // 2),
                        ),
                        1,
                    )
                )
            else:
                # Randomly shift the center point but make sure the resulting range
                # well includes the valid position

                min_center_point = max(
                    self.num_timesteps // 2,
                    valid_position + min_edge_buffer - self.num_timesteps // 2,
                )
                max_center_point = min(
                    available_timesteps - self.num_timesteps // 2,
                    valid_position - min_edge_buffer + self.num_timesteps // 2,
                )

                center_point = np.random.randint(
                    min_center_point, max_center_point + 1
                )  # max_center_point included

        return center_point

    def _get_timestamps(self, row_d: Dict, timestep_positions: List[int]) -> np.ndarray:
        start_date = pd.to_datetime(row_d["start_date"])
        end_date = pd.to_datetime(row_d["end_date"])
        if self.timestep_freq == "month":
            timestamps = pd.date_range(start=start_date, end=end_date, freq="MS")
        elif self.timestep_freq == "dekad":
            timestamps = get_dekad_date_range(start_date, end_date)
        else:
            raise NotImplementedError()
        return timestamps[timestep_positions].to_numpy()

    def get_inputs(self, row_d: Dict, timestep_positions: List[int]) -> dict:

        # Get latlons
        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        # Get timestamps belonging to each timestep
        timestamps = self._get_timestamps(row_d, timestep_positions)

        # Initialize inputs
        s1, s2, meteo, dem = self.initialize_inputs()

        # Fill inputs
        for src_attr, dst_atr in self.BAND_MAPPING.items():
            values = np.array(
                [float(row_d[src_attr.format(t)]) for t in timestep_positions]
            )
            idx_valid = values != NODATAVALUE
            if dst_atr in S2_bands:
                s2[..., S2_bands.index(dst_atr)] = values
            elif dst_atr in S1_bands:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_bands.index(dst_atr)] = values
            elif dst_atr == "precipitation":
                # scaling, and AgERA5 is in mm, prometheo convention expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., meteo_bands.index(dst_atr)] = values
            elif dst_atr == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., meteo_bands.index(dst_atr)] = values
            elif dst_atr in dem_bands:
                values = values[0]  # dem is not temporal
                dem[..., dem_bands.index(dst_atr)] = values
            else:
                raise ValueError(f"Unknown band {dst_atr}")

        return dict(
            s1=s1,
            s2=s2,
            meteo=meteo,
            dem=dem,
            latlon=latlon,
            # timestamps=timestamps,
            timestamps=timestamps.astype(
                np.int64
            ),  # TODO: parse true datetimes once DataLoader can handle this!!
        )

    def initialize_inputs(self):
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S1_bands)]
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S2_bands)]
        meteo = np.full(
            (self.num_timesteps, len(meteo_bands)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [T, len(meteo_bands)]
        dem = np.full(
            (1, 1, len(dem_bands)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [H, W, len(dem_bands)]

        return s1, s2, meteo, dem


class WorldCerealLabelledDataset(WorldCerealDataset):

    FILTER_LABELS = [0, 10]

    def __init__(
        self,
        dataframe,
        task_type: Literal["binary", "multiclass", "regression"] = "binary",
        num_outputs: int = 1,
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
        **kwargs,
    ):

        assert task_type in [
            "binary",
            "multiclass",
            "regression",
        ], f"Invalid task type `{task_type}` for labelled dataset"

        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        super().__init__(
            dataframe, task_type=task_type, num_outputs=num_outputs, **kwargs
        )
        self.croptype_list = croptype_list
        self.return_hierarchical_labels = return_hierarchical_labels

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, valid_position = self.get_timestep_positions(row)
        inputs = self.get_inputs(row, timestep_positions)
        label = self.get_label(
            row, "cropland", valid_position=valid_position - timestep_positions[0]
        )

        return Predictors(**inputs, label=label)

    def initialize_label(self, dtype=np.uint16):
        label = np.full(
            (1, 1, self.num_timesteps, self.num_outputs),
            fill_value=NODATAVALUE,
            dtype=dtype,
        )  # [H, W, T, num_outputs]

        return label

    def get_label(
        self,
        row_d: Dict,
        task_type: str = "cropland",
        valid_position: Optional[int] = None,
    ) -> np.ndarray:
        """Get the label for the given row. Label is a 2D array based number
        of timesteps and number of outputs.

        Parameters
        ----------
        row_d : Dict
            input row as a dictionary
        task_type : str, optional
            task type to infer labels from, by default "cropland"
        valid_position : int, optional
            validity position of the label, by default None.
            If provided, only the label at the corresponding timestep will be
            set while other timesteps will be set to NODATAVALUE.

        Returns
        -------
        np.ndarray
            label array
        """

        label = self.initialize_label()
        valid_idx = valid_position or np.arange(self.num_timesteps)

        if task_type == "cropland":
            label[0, 0, valid_idx, :] = int(row_d["LANDCOVER_LABEL"] == 11)
        if task_type == "croptype":
            if self.return_hierarchical_labels:
                label[0, 0, valid_idx, :] = [
                    row_d["landcover_name"],
                    row_d["downstream_class"],
                ]
            elif len(self.croptype_list) == 0:
                label[0, 0, valid_idx, :] = row_d["downstream_class"]
            else:
                label[0, 0, valid_idx, :] = np.array(
                    row_d[self.croptype_list].astype(int).values
                )
        return label


def get_dekad_date_range(start, end):
    return pd.DatetimeIndex(
        [_dekad_startdate_from_date(t) for t in _dekad_index(start, end)]
    )


def _dekad_index(begin, end):
    """Creates a pandas datetime index on a dekadal basis.
    Returns end date for each dekad.
    Based on: https://pytesmo.readthedocs.io/en/7.1/_modules/pytesmo/timedate/dekad.html  # NOQA

    Parameters
    ----------
    begin : datetime
        Datetime index start date.
    end : datetime, optional
        Datetime index end date, set to current date if None.

    Returns
    -------
    dtindex : pandas.DatetimeIndex
        Dekadal datetime index.
    """

    import calendar

    begin = pd.to_datetime(begin)
    end = pd.to_datetime(end)

    mon_begin = datetime(begin.year, begin.month, 1)
    mon_end = datetime(end.year, end.month, 1)

    daterange = pd.date_range(mon_begin, mon_end, freq="MS")

    dates = []

    for i, dat in enumerate(daterange):
        lday = calendar.monthrange(dat.year, dat.month)[1]
        if i == 0 and begin.day > 1:
            if begin.day < 11:
                if daterange.size == 1:
                    if end.day < 11:
                        dekads = [10]
                    elif end.day >= 11 and end.day < 21:
                        dekads = [10, 20]
                    else:
                        dekads = [10, 20, lday]
                else:
                    dekads = [10, 20, lday]
            elif begin.day >= 11 and begin.day < 21:
                if daterange.size == 1:
                    if end.day < 21:
                        dekads = [20]
                    else:
                        dekads = [20, lday]
                else:
                    dekads = [20, lday]
            else:
                dekads = [lday]
        elif i == (len(daterange) - 1) and end.day < 21:
            if end.day < 11:
                dekads = [10]
            else:
                dekads = [10, 20]
        else:
            dekads = [10, 20, lday]

        for j in dekads:
            dates.append(datetime(dat.year, dat.month, j))

    dtindex = pd.DatetimeIndex(dates)

    return dtindex


def _dekad_startdate_from_date(dt_in):
    """
    dekadal startdate that a date falls in
    Based on: https://pytesmo.readthedocs.io/en/7.1/_modules/pytesmo/timedate/dekad.html  # NOQA

    Parameters
    ----------
    run_dt: datetime.datetime

    Returns
    -------
    startdate: datetime.datetime
        startdate of dekad
    """
    if dt_in.day <= 10:
        startdate = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
    if dt_in.day >= 11 and dt_in.day <= 20:
        startdate = datetime(dt_in.year, dt_in.month, 11, 0, 0, 0)
    if dt_in.day >= 21:
        startdate = datetime(dt_in.year, dt_in.month, 21, 0, 0, 0)
    return startdate
