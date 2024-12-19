from datetime import datetime
from typing import Any, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)


class ScaleAgDataset(Dataset):
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
        num_timesteps: int = 36,
        task_type: Literal["regression", "binary", "multiclass", "ssl"] = "ssl",
        num_outputs: Optional[int] = None,
        target_name: Optional[str] = None,
        positive_labels: Optional[Union[List[Any], Any]] = None,
        compositing_window: Literal["dekad", "monthly"] = "dekad",
        time_explicit: bool = False,
        upper_bound: Optional[float] = None,
        lower_bound: Optional[float] = None,
    ):
        """
        Initialize the dataset object.
        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input dataframe containing the dataset.
        num_timesteps : int, optional
            Number of timesteps to consider, by default 36.
        task_type : Literal["regression", "binary", "multiclass", "ssl"], optional
            Type of task to perform, by default self-supervised-learning "ssl".
        num_outputs : Optional[int], optional
            Number of output classes, by default None.
            - for binary and regression tasks, independently on the value provided, num_outputs is enforced to be 1.
            - for multiclass task, if this value is not provided, num_outputs is set to be the number of unique classes in the target column.
        target_name : Optional[str], optional
            Name of the target column, by default None.
        positive_labels : Optional[Union[List[Any], Any]], optional
            Positive labels for binary classification, by default None.
        compositing_window : Literal["dekad", "monthly"], optional
            Compositing window type, by default "dekad".
        time_explicit : bool, optional
            Defines how to handle time dimension for the label predictor.
            If True, itr indicates each example is associated with time-dependent labels.
            Hence, the time dimension for the label predictor will be set accordingly.
            If False, the label time dimension is set to 1.
            By default False.
        upper_bound : Optional[float], optional
            Upper bound for target values in regression tasks, by default None.
            If no upper bound is provided, the maximum value of the target column is used.
        lower_bound : Optional[float], optional
            Lower bound for target values in regression tasks, by default None.
            If no lower bound is provided, the minimum value of the target column is used.
        """
        self.dataframe = dataframe.replace({np.nan: NODATAVALUE})
        self.num_timesteps = num_timesteps
        self.task_type = task_type
        self.target_name = target_name
        self.positive_labels = positive_labels
        self.num_outputs = (
            None if task_type == "ssl" else self.set_num_outputs(num_outputs)
        )
        self.compositing_window = compositing_window
        self.time_explicit = time_explicit

        # assess label type and bound label values to valid range if upper and lower bounds are provided
        if task_type == "regression":
            assert self.dataframe[target_name].dtype in [
                np.float32,
                np.float64,
            ], "Regression target must be of type float"

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
            # series allows direct mapping in case target is a list of values
            self.class_to_index = pd.Series(
                {
                    label: idx
                    for idx, label in enumerate(self.dataframe[target_name].unique())
                }
            )
            self.index_to_class = pd.Series(
                {
                    idx: label
                    for idx, label in enumerate(self.dataframe[target_name].unique())
                }
            )

        if self.task_type == "binary" and positive_labels is not None:
            # mapping for binary classification. this gives the user the flexibility to indicate which labels
            # or set of labels should be appointed as positive class
            self.binary_mapping = pd.Series(
                {
                    bin_cl: 1 if bin_cl in positive_labels else 0
                    for bin_cl in self.dataframe[target_name].unique()
                }
            )

    def set_num_outputs(self, num_outputs: Optional[int]) -> Optional[int]:
        if self.task_type in ["binary", "regression"]:
            if num_outputs != 1 and num_outputs is not None:
                logger.warning(
                    "Number of outputs for regression and binary tasks is always set to 1."
                    f"Ignoring provided value {num_outputs}."
                )
            return 1
        else:
            if num_outputs is None:
                logger.warning(
                    "Number of outputs for multiclass task not provided."
                    "Setting to the number of classes found in the dataset."
                )
                return len(self.dataframe[self.target_name].unique())
            else:
                return num_outputs

    def get_predictors(self, row: pd.Series) -> Predictors:
        row_d = pd.Series.to_dict(row)
        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        # initialize sensor arrays filled with NODATAVALUE
        s1, s2, meteo, dem = self.initialize_inputs()

        # iterate over all bands and fill the corresponding arrays. convert to presto units if necessary
        for src_attr, dst_attr in self.BAND_MAPPING.items():
            # retrieve ts for each band column df_val
            values = np.array(
                [float(row_d[src_attr.format(t)]) for t in range(self.num_timesteps)]
            )
            values = np.nan_to_num(values, nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            if dst_attr in S2_BANDS:
                s2[..., S2_BANDS.index(dst_attr)] = values
            elif dst_attr in S1_BANDS:
                s1 = self.openeo_to_prometheo_units(s1, dst_attr, values, idx_valid)
            elif dst_attr == "precipitation":
                meteo = self.openeo_to_prometheo_units(
                    meteo, dst_attr, values, idx_valid
                )
            elif dst_attr == "temperature":
                meteo = self.openeo_to_prometheo_units(
                    meteo, dst_attr, values, idx_valid
                )
            elif dst_attr in DEM_BANDS:
                dem = self.openeo_to_prometheo_units(dem, dst_attr, values, idx_valid)

        predictors_dict = {
            "s1": s1,
            "s2": s2,
            "meteo": meteo,
            "dem": dem,
            "latlon": latlon,
            "timestamps": self.get_date_array(row),
        }
        if self.task_type != "ssl":
            predictors_dict["label"] = self.get_label(row_d)

        return Predictors(**predictors_dict)

    def __getitem__(self, idx):
        # Get the sample
        row = self.dataframe.iloc[idx, :]
        return self.get_predictors(row)

    def __len__(self):
        return len(self.dataframe)

    def _get_correct_date(self, dt_in):
        """
        Determine the correct date based on the input date and compositing window.
        Args:
            dt_in (datetime): The input date.
        Returns:
            datetime: The corrected date based on the compositing window.
        Raises:
            ValueError: If the compositing window is not "dekad" or "monthly".
        Notes:
        - For "dekad" compositing window:
            - If the day of the month is between 1 and 10, the correct date is the 1st of the month.
            - If the day of the month is between 11 and 20, the correct date is the 11th of the month.
            - If the day of the month is between 21 and the end of the month, the correct date is the 21st of the month.
        - For "monthly" compositing window, the correct date is always the 1st of the month.
        """

        if self.compositing_window == "dekad":
            if dt_in.day <= 10:
                correct_date = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
            if dt_in.day >= 11 and dt_in.day <= 20:
                correct_date = datetime(dt_in.year, dt_in.month, 11, 0, 0, 0)
            if dt_in.day >= 21:
                correct_date = datetime(dt_in.year, dt_in.month, 21, 0, 0, 0)
        elif self.compositing_window == "monthly":
            correct_date = datetime(dt_in.year, dt_in.month, 1, 0, 0, 0)
        else:
            raise ValueError(f"Unknown compositing window: {self.compositing_window}")

        return correct_date

    def _get_following_date(self, dt_in):
        """
        Calculate the following date based on the given compositing window.
        Args:
            dt_in (datetime): The input date.
        Returns:
            datetime: The calculated following date.
        Raises:
            ValueError: f the compositing window is not "dekad" or "monthly".
        """

        if self.compositing_window == "dekad":
            if dt_in.day < 21:
                return datetime(dt_in.year, dt_in.month, dt_in.day + 10, 0, 0, 0)
            else:
                month = dt_in.month + 1 if dt_in.month < 12 else 1
                year = dt_in.year + 1 if month == 1 else dt_in.year
                return datetime(year, month, 1, 0, 0, 0)
        elif self.compositing_window == "monthly":
            month = dt_in.month + 1 if dt_in.month < 12 else 1
            year = dt_in.year + 1 if month == 1 else dt_in.year
            return datetime(year, month, 1, 0, 0, 0)
        else:
            raise ValueError(f"Unknown compositing window: {self.compositing_window}")

    def get_date_array(self, row):
        # set correct initial and end dates
        start_date = pd.to_datetime(row.start_date)
        end_date = pd.to_datetime(row.end_date)
        start_date = self._get_correct_date(start_date)
        end_date = self._get_correct_date(end_date)

        # generate date vector based on compositing window
        date_vector = [start_date]
        while start_date != end_date:
            start_date = self._get_following_date(start_date)
            date_vector.append(start_date)

        # truncate date vector on the end side if it exceeds the number of timesteps
        # this strategy currently does not give the freedom to chose which timestamps to keep
        # it might need to be adjusted in the future
        if self.num_timesteps < len(date_vector):
            date_vector = date_vector[: self.num_timesteps]
            logger.warning(
                "The number of timesteps is smaller than the number of available dates. "
                f"Replace end date {end_date.date()} to {date_vector[-1].date()}."
            )

        return np.stack(
            [
                np.array([d.day for d in date_vector]),
                np.array([d.month for d in date_vector]),
                np.array([d.year for d in date_vector]),
            ],
            axis=1,
        )

    def get_label(
        self, row_d: pd.Series, valid_positions: Optional[int] = None
    ) -> np.ndarray:
        target = np.array(row_d[self.target_name])
        if self.time_explicit:
            raise NotImplementedError("Time explicit labels not yet implemented")
        time_dim = 1
        valid_idx = valid_positions or np.arange(time_dim)

        labels = np.full(
            (1, 1, time_dim, self.num_outputs),
            fill_value=NODATAVALUE,
            dtype=np.int32,
        )
        if self.task_type == "regression":
            target = self.normalize_target(target)

        elif self.task_type == "binary":
            if self.positive_labels is not None:
                target = self.binary_mapping[target]
            assert target in [
                0,
                1,
            ], f"Invalid target value: {target}. Target must be either 0 or 1. Please provide pos_labels list."

        # convert classes to indices for multiclass
        elif self.task_type == "multiclass":
            target = self.class_to_index[target]
            if target.size > 1:
                target = target.to_numpy()
        labels[0, 0, valid_idx, :] = target
        return labels

    def normalize_target(self, target):
        return (target - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def revert_to_original_units(self, target_norm):
        return target_norm * (self.upper_bound - self.lower_bound) + self.lower_bound

    def openeo_to_prometheo_units(self, band_array, band, values, idx_valid):
        if band in S1_BANDS:
            # convert to dB
            idx_valid = idx_valid & (values > 0)
            values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            band_array[..., S1_BANDS.index(band)] = values
        elif band == "precipitation":
            # scaling, and AgERA5 is in mm, Presto expects m
            values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            band_array[..., METEO_BANDS.index(band)] = values
        elif band == "temperature":
            # remove scaling. conversion to celsius is done in the normalization
            values[idx_valid] = values[idx_valid] / 100
            band_array[..., METEO_BANDS.index(band)] = values
        elif band in DEM_BANDS:
            band_array[..., DEM_BANDS.index(band)] = values[0]
        else:
            raise ValueError(f"Unknown band {band}")

        return band_array

    def initialize_inputs(self):
        s1 = np.full(
            (1, 1, self.num_timesteps, len(S1_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        s2 = np.full(
            (1, 1, self.num_timesteps, len(S2_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        meteo = np.full(
            (self.num_timesteps, len(METEO_BANDS)),
            fill_value=NODATAVALUE,
            dtype=np.float32,
        )
        dem = np.full((1, 1, len(DEM_BANDS)), fill_value=NODATAVALUE, dtype=np.float32)
        return s1, s2, meteo, dem
