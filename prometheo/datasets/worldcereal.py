import functools
from datetime import datetime
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from einops import rearrange
from pyproj import Transformer
from torch import nn
from torch.utils.data import Dataset

from prometheo.infer import extract_features_from_model
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)


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
        assert valid_position in timestep_positions, (
            f"Valid position {valid_position} not in timestep positions {timestep_positions}"
        )

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
        start_date = datetime.strptime(row_d["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(row_d["end_date"], "%Y-%m-%d")

        if self.timestep_freq == "month":
            days, months, years = get_monthly_timestamp_components(start_date, end_date)
        elif self.timestep_freq == "dekad":
            days, months, years = get_dekad_timestamp_components(start_date, end_date)
        else:
            raise NotImplementedError()

        return np.stack(
            [
                days[timestep_positions],
                months[timestep_positions],
                years[timestep_positions],
            ],
            axis=1,
        )

    def get_inputs(self, row_d: Dict, timestep_positions: List[int]) -> dict:
        # Get latlons
        latlon = rearrange(np.array([row_d["lat"], row_d["lon"]], dtype=np.float32), "d -> 1 1 d")

        # Get timestamps belonging to each timestep
        timestamps = self._get_timestamps(row_d, timestep_positions)

        # Initialize inputs
        s1, s2, meteo, dem = self.initialize_inputs()

        # Fill inputs
        for src_attr, dst_atr in self.BAND_MAPPING.items():
            keys = [src_attr.format(t) for t in timestep_positions]
            values = np.array([float(row_d[key]) for key in keys], dtype=np.float32)
            idx_valid = values != NODATAVALUE
            if dst_atr in S2_BANDS:
                s2[..., S2_BANDS.index(dst_atr)] = values
            elif dst_atr in S1_BANDS:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
                s1[..., S1_BANDS.index(dst_atr)] = values
            elif dst_atr == "precipitation":
                # scaling, and AgERA5 is in mm, prometheo convention expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr == "temperature":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
                meteo[..., METEO_BANDS.index(dst_atr)] = values
            elif dst_atr in DEM_BANDS:
                values = values[0]  # dem is not temporal
                dem[..., DEM_BANDS.index(dst_atr)] = values
            else:
                raise ValueError(f"Unknown band {dst_atr}")
        return dict(
            s1=s1, s2=s2, meteo=meteo, dem=dem, latlon=latlon, timestamps=timestamps
        )

    def initialize_inputs(self):
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S1_BANDS)]
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [H, W, T, len(S2_BANDS)]
        meteo = np.full(
            (self.num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [T, len(METEO_BANDS)]
        dem = np.full(
            (1, 1, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [H, W, len(DEM_BANDS)]

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
        time_explicit: bool = False,
        **kwargs,
    ):
        """Labelled version of WorldCerealDataset for supervised training.
        Additional arguments are explained below.

        Parameters
        ----------
        num_outputs : int, optional
            number of outputs to supervise training on, by default 1
        croptype_list : List, optional
            TODO: explain, by default []
        return_hierarchical_labels : bool, optional
            TODO: explain, by default False
        time_explicit : bool, optional
            if True, labels respect the full temporal dimension
            to have temporally explicit outputs, by default False
        """
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
        self.time_explicit = time_explicit
        self.return_hierarchical_labels = return_hierarchical_labels

    def __getitem__(self, idx):
        row = pd.Series.to_dict(self.dataframe.iloc[idx, :])
        timestep_positions, valid_position = self.get_timestep_positions(row)
        inputs = self.get_inputs(row, timestep_positions)
        label = self.get_label(
            row, "cropland", valid_position=valid_position - timestep_positions[0]
        )

        return Predictors(**inputs, label=label)

    def initialize_label(self):
        tsteps = self.num_timesteps if self.time_explicit else 1
        label = np.full(
            (1, 1, tsteps, self.num_outputs),
            fill_value=NODATAVALUE,
            dtype=np.int32,
        )  # [H, W, T or 1, num_outputs]

        return label

    def get_label(
        self,
        row_d: Dict,
        task_type: str = "cropland",
        valid_position: Optional[int] = None,
    ) -> np.ndarray:
        """Get the label for the given row. Label is a 2D array based on
        the number of timesteps and number of outputs. If time_explicit is False,
        the number of timesteps will be set to 1.

        Parameters
        ----------
        row_d : Dict
            input row as a dictionary
        task_type : str, optional
            task type to infer labels from, by default "cropland"
        valid_position : int, optional
            validity position of the label, by default None.
            If provided and `time_explicit` is True,
            only the label at the corresponding timestep will be
            set while other timesteps will be set to NODATAVALUE.

        Returns
        -------
        np.ndarray
            label array
        """

        label = self.initialize_label()
        if not self.time_explicit:
            # We have only one label for the whole sequence
            valid_idx = 0
        else:
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


def generate_month_sequence(start_date: datetime, end_date: datetime) -> np.ndarray:
    """Helper function to generate a sequence of months between start_date and end_date.
    This is much faster than using a pd.date_range().

    Parameters
    ----------
    start_date : datetime
        start of the sequence
    end_date : datetime
        end of the sequence

    Returns
    -------
    array contaning the sequence of months

    """
    start = np.datetime64(start_date, "M")  # Truncate to month start
    end = np.datetime64(end_date, "M")  # Truncate to month start
    timestamps = np.arange(start, end + 1, dtype="datetime64[M]")

    return timestamps


def get_monthly_timestamp_components(
    start_date: datetime, end_date: datetime
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to generate day/month/year components for
    a sequence of months between start_date and end_date.

    Parameters
    ----------
    start_date : datetime
        start of the sequence
    end_date : datetime
        end of the sequence

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        tuple of day, month, year components for monthly sequence
    """

    timestamps = generate_month_sequence(start_date, end_date)
    years = timestamps.astype("datetime64[Y]").astype(int) + 1970
    months = timestamps.astype("datetime64[M]").astype(int) % 12 + 1
    days = np.ones_like(years)  # All days are 1 for "MS" frequency

    return days, months, years


def get_dekad_timestamp_components(start_date, end_date):
    timestamps = np.array(
        [_dekad_startdate_from_date(t) for t in _dekad_timestamps(start_date, end_date)]
    )
    years = np.array([t.year for t in timestamps])
    months = np.array([t.month for t in timestamps])
    days = np.array([t.day for t in timestamps])

    return days, months, years


def _dekad_timestamps(begin, end):
    """Creates a temporal sequence on a dekadal basis.
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

    daterange = generate_month_sequence(begin, end)

    dates = []

    for i, dat in enumerate(daterange):
        year, month = int(str(dat)[:4]), int(str(dat)[5:7])
        lday = calendar.monthrange(year, month)[1]
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
            dates.append(datetime(year, month, j))

    return dates


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


def _predictor_from_xarray(arr: xr.DataArray, epsg: int) -> Predictors:
    def _get_timestamps() -> np.ndarray:
        timestamps = arr.t.values
        years = timestamps.astype("datetime64[Y]").astype(int) + 1970
        months = timestamps.astype("datetime64[M]").astype(int) % 12 + 1
        days = timestamps.astype("datetime64[D]").astype("datetime64[M]")
        days = (timestamps - days).astype(int) + 1

        components = np.stack(
            [
                days,
                months,
                years,
            ],
            axis=1,
        )

        return components[None, ...]  # Add batch dimension

    def _initialize_eo_inputs():
        num_timesteps = arr.t.size
        h, w = arr.y.size, arr.x.size
        s1 = np.full(
            (1, h, w, num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, H, W, T, len(S1_BANDS)]
        s2 = np.full(
            (1, h, w, num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, H, W, T, len(S2_BANDS)]
        meteo = np.full(
            (1, num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )  # [B, T, len(METEO_BANDS)]
        dem = np.full(
            (1, h, w, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32
        )  # [B, H, W, len(DEM_BANDS)]

        return s1, s2, meteo, dem

    # TODO: remove temporary band renaming due to old data file
    arr["bands"] = arr.bands.where(arr.bands != "temperature_2m", "temperature")
    arr["bands"] = arr.bands.where(arr.bands != "total_precipitation", "precipitation")

    # Initialize EO inputs
    s1, s2, meteo, dem = _initialize_eo_inputs()

    # Fill EO inputs
    for band in S2_BANDS + S1_BANDS + METEO_BANDS + DEM_BANDS:
        if band not in arr.bands.values:
            print(f"Band {band} not found in the input data, skipping.")
            continue  # skip bands that are not present in the data
        values = arr.sel(bands=band).values
        idx_valid = values != NODATAVALUE
        if band in S2_BANDS:
            s2[..., S2_BANDS.index(band)] = rearrange(
                values, "t x y -> 1 y x t"
            )  # TODO check if this is correct
        elif band in S1_BANDS:
            # convert to dB
            idx_valid = idx_valid & (values > 0)
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            s1[..., S1_BANDS.index(band)] = rearrange(values, "t x y -> 1 y x t")
        elif band == "precipitation":
            # scaling, and AgERA5 is in mm, prometheo convention expects m
            values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            meteo[..., METEO_BANDS.index(band)] = rearrange(values[:, 0, 0], "t -> 1 t")
        elif band == "temperature":
            # remove scaling
            values[idx_valid] = values[idx_valid] / 100
            meteo[..., METEO_BANDS.index(band)] = rearrange(values[:, 0, 0], "t -> 1 t")
        elif band in DEM_BANDS:
            values = values[0]  # dem is not temporal
            dem[..., DEM_BANDS.index(band)] = rearrange(values, "x y -> 1 y x")
        else:
            raise ValueError(f"Unknown band {band}")

    # Extract the latlons
    # EPSG:4326 is the supported crs for presto
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    x, y = np.meshgrid(arr.x, arr.y)
    lon, lat = transformer.transform(x, y)

    predictors_dict = {
        "s1": s1,
        "s2": s2,
        "meteo": meteo,
        "latlon": rearrange(
            np.stack([lat, lon]), "c x y -> y x c"
        ),  # TODO make explicit once #36 is tackled
        "dem": dem,
        "timestamps": _get_timestamps(),
    }

    return Predictors(**predictors_dict)


def generate_predictor(x: pd.DataFrame | xr.DataArray, epsg: int) -> Predictors:
    if isinstance(x, xr.DataArray):
        return _predictor_from_xarray(x, epsg)
    raise NotImplementedError


@functools.lru_cache(maxsize=6)
def compile_encoder(presto_encoder: nn.Module) -> Callable:
    """Helper function that compiles the encoder of a Presto model
    and performs a warm-up on dummy data. The lru_cache decorator
    ensures caching on compute nodes to be able to actually benefit
    from the compilation process.

    Parameters
    ----------
    presto_encoder : nn.Module
        Encoder part of Presto model to compile

    """

    presto_encoder = torch.compile(presto_encoder)  # type: ignore

    for _ in range(3):
        presto_encoder(
            torch.rand((1, 12, 17)),
            torch.ones((1, 12)).long(),
            torch.rand(1, 2),
        )

    return presto_encoder


def run_model_inference(
    inarr: pd.DataFrame | xr.DataArray,
    model: nn.Module,  # Wrapper
    epsg: int = 4326,
    batch_size: int = 8192,
) -> np.ndarray | xr.DataArray:
    """
    Runs a forward pass of the model on the input data

    Args:
        inarr (xr.DataArray or pd.DataFrame): Input data as xarray DataArray or pandas DataFrame.
        model (nn.Module): A Prometheo compatible (wrapper) model.
        epsg (int) : EPSG code describing the coordinates.
        batch_size (int): Batch size to be used for Presto inference.

    Returns:
        xr.DataArray or np.ndarray: Model output as xarray DataArray or numpy ndarray.
    """

    predictor = generate_predictor(inarr, epsg)
    # fixing the pooling method to keep the function signature the same
    # as in presto-worldcereal but this could be an input argument too
    features = (
        extract_features_from_model(model, predictor, batch_size, PoolingMethods.GLOBAL)
        .cpu()
        .numpy()
    )

    # todo - return the output tensors to the right shape, either xarray or df
    if isinstance(inarr, pd.DataFrame):
        return features
    else:
        features = rearrange(
            features, "1 y x 1 c -> x y c", x=len(inarr.x), y=len(inarr.y)
        )
        features_da = xr.DataArray(
            features,
            dims=["x", "y", "bands"],
            coords={"x": inarr.x, "y": inarr.y},
        )

        return features_da
