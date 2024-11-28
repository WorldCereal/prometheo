import logging
from datetime import datetime, timedelta
from math import modf
from pathlib import Path
from random import sample
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset


from .src.predictors import Predictors
from .datasets.base import DatasetBase

from .dataops import (BANDS, BANDS_GROUPS_IDX, MIN_EDGE_BUFFER, NDVI_INDEX,
                      NODATAVALUE, NORMED_BANDS, S1_S2_ERA5_SRTM, S2_RGB_INDEX,
                      DynamicWorld2020_2021, S2_NIR_10m_INDEX)
from .masking import BAND_EXPANSION, MaskedExample, MaskParamsNoDw
from .utils import DEFAULT_SEED, data_dir, get_class_mappings, load_world_df

logger = logging.getLogger("__main__")

IDX_TO_BAND_GROUPS = {}
for band_group_idx, (key, val) in enumerate(BANDS_GROUPS_IDX.items()):
    for idx in val:
        IDX_TO_BAND_GROUPS[NORMED_BANDS[idx]] = band_group_idx


class WorldCerealBase(DatasetBase):
    NUM_TIMESTEPS = 12
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
        "METEO-precipitation_flux-ts{}-100m": "total_precipitation",
        "METEO-temperature_mean-ts{}-100m": "temperature_2m",
    }
    STATIC_BAND_MAPPING = {"DEM-alt-20m": "elevation", "DEM-slo-20m": "slope"}

    @classmethod
    def get_timestep_positions(
        cls, row_d: Dict, augment: bool = False, is_ssl: bool = False
    ) -> List[int]:
        available_timesteps = int(row_d["available_timesteps"])

        if is_ssl:
            if available_timesteps == cls.NUM_TIMESTEPS:
                valid_position = int(cls.NUM_TIMESTEPS // 2)
            else:
                valid_position = int(
                    np.random.choice(
                        range(
                            cls.NUM_TIMESTEPS // 2, (available_timesteps - cls.NUM_TIMESTEPS // 2)
                        ),
                        1,
                    )
                )
            center_point = valid_position
        else:
            valid_position = int(row_d["valid_position"])
            if not augment:
                #  check if the valid position is too close to the start_date and force shifting it
                if valid_position < cls.NUM_TIMESTEPS // 2:
                    center_point = cls.NUM_TIMESTEPS // 2
                #  or too close to the end_date
                elif valid_position > (available_timesteps - cls.NUM_TIMESTEPS // 2):
                    center_point = available_timesteps - cls.NUM_TIMESTEPS // 2
                else:
                    # Center the timesteps around the valid position
                    center_point = valid_position
            else:
                # Shift the center point but make sure the resulting range
                # well includes the valid position

                min_center_point = max(
                    cls.NUM_TIMESTEPS // 2,
                    valid_position + MIN_EDGE_BUFFER - cls.NUM_TIMESTEPS // 2,
                )
                max_center_point = min(
                    available_timesteps - cls.NUM_TIMESTEPS // 2,
                    valid_position - MIN_EDGE_BUFFER + cls.NUM_TIMESTEPS // 2,
                )

                center_point = np.random.randint(
                    min_center_point, max_center_point + 1
                )  # max_center_point included

        last_timestep = min(available_timesteps, center_point + cls.NUM_TIMESTEPS // 2)
        first_timestep = max(0, last_timestep - cls.NUM_TIMESTEPS)
        timestep_positions = list(range(first_timestep, last_timestep))

        if len(timestep_positions) != cls.NUM_TIMESTEPS:
            raise ValueError(
                f"Acquired timestep positions do not have correct length: \
required {cls.NUM_TIMESTEPS}, got {len(timestep_positions)}"
            )
        assert (
            valid_position in timestep_positions
        ), f"Valid position {valid_position} not in timestep positions {timestep_positions}"
        return timestep_positions

    @classmethod
    def row_to_arrays(
        cls,
        row: pd.Series,
        task_type: str = "cropland",
        croptype_list: List = [],
        augment: bool = False,
        is_ssl: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
        # https://stackoverflow.com/questions/45783891/is-there-a-way-to-speed-up-the-pandas-getitem-getitem-axis-and-get-label
        # This is faster than indexing the series every time!
        row_d = pd.Series.to_dict(row)

        latlon = np.array([row_d["lat"], row_d["lon"]], dtype=np.float32)

        timestep_positions = cls.get_timestep_positions(row_d, augment=augment, is_ssl=is_ssl)

        if cls.NUM_TIMESTEPS == 12:
            initial_start_date_position = pd.to_datetime(row_d["start_date"]).month
        elif cls.NUM_TIMESTEPS > 12:
            # get the correct index of the start_date based on NUM_TIMESTEPS`
            # e.g. if NUM_TIMESTEPS is 36 (dekadal setup), we should take the correct
            # 10-day interval that the start_date falls into
            # TODO: 1) this needs to go into a separate function
            # 2) definition of valid_position and timestep_ind
            #  should also be changed accordingly
            year = pd.to_datetime(row_d["start_date"]).year
            year_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31")
            bins = pd.cut(year_dates, bins=cls.NUM_TIMESTEPS, labels=False)
            initial_start_date_position = bins[
                np.where(year_dates == pd.to_datetime(row_d["start_date"]))[0][0]
            ]
        else:
            raise ValueError(
                f"NUM_TIMESTEPS must be at least 12. Currently it is {cls.NUM_TIMESTEPS}"
            )

        # make sure that month for encoding gets shifted according to
        # the selected timestep positions. Also ensure circular indexing
        month = (initial_start_date_position - 1 + timestep_positions[0]) % cls.NUM_TIMESTEPS

        # adding workaround for compatibility between Phase I and Phase II datasets.
        # (in Phase II, the relevant attribute name was changed to valid_time)
        # once we fully move to Phase II data, this should be replaced to valid_tome only.
        if "valid_date" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_date"], "%Y-%m-%d").month - 1
        elif "valid_time" in row_d.keys():
            valid_month = datetime.strptime(row_d["valid_time"], "%Y-%m-%d").month - 1
        else:
            logger.error("Dataset does not contain neither valid_date, nor valid_time attribute.")

        eo_data = np.zeros((cls.NUM_TIMESTEPS, len(BANDS)))
        # an assumption we make here is that all timesteps for a token
        # have the same masking
        mask = np.zeros((cls.NUM_TIMESTEPS, len(BANDS_GROUPS_IDX)))

        for df_val, presto_val in cls.BAND_MAPPING.items():
            values = np.array([float(row_d[df_val.format(t)]) for t in timestep_positions])
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(values, nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            if presto_val in ["VV", "VH"]:
                # convert to dB
                idx_valid = idx_valid & (values > 0)
                values[idx_valid] = 20 * np.log10(values[idx_valid]) - 83
            elif presto_val == "total_precipitation":
                # scaling, and AgERA5 is in mm, Presto expects m
                values[idx_valid] = values[idx_valid] / (100 * 1000.0)
            elif presto_val == "temperature_2m":
                # remove scaling
                values[idx_valid] = values[idx_valid] / 100
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid
            eo_data[:, BANDS.index(presto_val)] = values * idx_valid

        for df_val, presto_val in cls.STATIC_BAND_MAPPING.items():
            # this occurs for the DEM values in one point in Fiji
            values = np.nan_to_num(row_d[df_val], nan=NODATAVALUE)
            idx_valid = values != NODATAVALUE
            eo_data[:, BANDS.index(presto_val)] = values * idx_valid
            mask[:, IDX_TO_BAND_GROUPS[presto_val]] += ~idx_valid

        # check if the visual bands mask is True
        # or nir mask, and adjust the NDVI mask accordingly
        mask[:, NDVI_INDEX] = np.logical_or(mask[:, S2_RGB_INDEX], mask[:, S2_NIR_10m_INDEX])

        return (
            s1: Optional[np.ndarray] = None,
            s2: Optional[np.ndarray] = None,
            meteo: Optional[np.ndarray] = None,
            dem: Optional[np.ndarray] = None,
            latlon: Optional[np.ndarray] = None,
            aux_inputs: Optional[List[np.ndarray]] = None,
            label: Optional[np.ndarray] = None,
            month: Optional[Union[np.ndarray,int]] = None
        )


    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        
        s1, s2, meteo, dem, latlon, valid_month, month = self.row_to_arrays(row)

        return Predictors(
            s1=s1,
            s2=s2,
            meteo=meteo,
            dem=dem,
            latlon=latlon,
            aux_inputs=[valid_month],
            month=month
            )


    @staticmethod
    def map_croptypes(
        df: pd.DataFrame,
        finetune_classes="CROPTYPE0",
        downstream_classes="CROPTYPE19",
    ) -> pd.DataFrame:

        CLASS_MAPPINGS = get_class_mappings()

        wc2ewoc_map = pd.read_csv(data_dir / "croptype_mappings" / "wc2eurocrops_map.csv")
        wc2ewoc_map["ewoc_code"] = wc2ewoc_map["ewoc_code"].str.replace("-", "").astype(int)

        ewoc_map = pd.read_csv(data_dir / "croptype_mappings" / "eurocrops_map_wcr_edition.csv")
        ewoc_map = ewoc_map[ewoc_map["ewoc_code"].notna()]
        ewoc_map["ewoc_code"] = ewoc_map["ewoc_code"].str.replace("-", "").astype(int)
        ewoc_map = ewoc_map.apply(lambda x: x[: x.last_valid_index()].ffill(), axis=1)
        ewoc_map.set_index("ewoc_code", inplace=True)

        df.loc[df["CROPTYPE_LABEL"] == 0, "CROPTYPE_LABEL"] = np.nan
        df["CROPTYPE_LABEL"] = df["CROPTYPE_LABEL"].fillna(df["LANDCOVER_LABEL"])

        df["ewoc_code"] = df["CROPTYPE_LABEL"].map(wc2ewoc_map.set_index("croptype")["ewoc_code"])
        df["landcover_name"] = df["ewoc_code"].map(ewoc_map["landcover_name"])
        df["cropgroup_name"] = df["ewoc_code"].map(ewoc_map["cropgroup_name"])
        df["croptype_name"] = df["ewoc_code"].map(ewoc_map["croptype_name"])

        df["downstream_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS[downstream_classes].items()}
        )
        df["finetune_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS[finetune_classes].items()}
        )
        df["balancing_class"] = df["ewoc_code"].map(
            {int(k): v for k, v in CLASS_MAPPINGS["CROPTYPE19"].items()}
        )

        return df


class WorldCerealMaskedDataset(WorldCerealBase):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        mask_params: MaskParamsNoDw,
        task_type: str = "cropland",
        croptype_list: List = [],
        is_ssl: bool = True,
    ):
        super().__init__(dataframe)
        self.mask_params = mask_params
        self.task_type = task_type
        self.croptype_list = croptype_list
        self.is_ssl = is_ssl

    def __getitem__(self, idx):
        # Get the sample
        row = self.df.iloc[idx, :]
        eo, real_mask_per_token, latlon, month, valid_month = self.row_to_arrays(
            row, task_type=self.task_type, croptype_list=self.croptype_list, is_ssl=self.is_ssl
        )

        mask_eo, x_eo, y_eo, strat = self.mask_params.mask_data(
            self.normalize_and_mask(eo), real_mask_per_token
        )
        real_mask_per_variable = np.repeat(real_mask_per_token, BAND_EXPANSION, axis=1)

        dynamic_world = np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount)
        mask_dw = np.full(self.NUM_TIMESTEPS, True)
        y_dw = dynamic_world.copy()
        return MaskedExample(
            mask_eo,
            mask_dw,
            x_eo,
            y_eo,
            dynamic_world,
            y_dw,
            month,
            latlon,
            strat,
            real_mask_per_variable,
        )


def filter_remove_noncrops(df: pd.DataFrame) -> pd.DataFrame:
    labels_to_exclude = [
        0,
        991,
        7900,
        9900,
        9998,  # unspecified cropland
        1910,
        1900,
        1920,
        1000,  # cereals, too generic
        11,
        9910,
        6212,  # temporary crops, too generic
        7920,
        9520,
        3400,
        3900,  # generic and other classes
        4390,
        4000,
        4300,  # generic and other classes
    ]
    df = df[(df["LANDCOVER_LABEL"] == 11) & (~df["CROPTYPE_LABEL"].isin(labels_to_exclude))]
    df.reset_index(inplace=True)
    return df


class WorldCerealLabelledDataset(WorldCerealBase):
    # 0: no information, 10: could be both annual or perennial
    FILTER_LABELS = [0, 10]

    def __init__(
        self,
        dataframe: pd.DataFrame,
        countries_to_remove: Optional[List[str]] = None,
        years_to_remove: Optional[List[int]] = None,
        balance: bool = False,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
        augment: bool = False,
        mask_ratio: float = 0.0,
    ):
        dataframe = dataframe.loc[~dataframe.LANDCOVER_LABEL.isin(self.FILTER_LABELS)]

        if countries_to_remove is not None:
            dataframe = self.join_with_world_df(dataframe)
            for country in countries_to_remove:
                assert dataframe.name.str.contains(
                    country
                ).any(), f"Tried removing {country} but it is not in the dataframe"
            dataframe = dataframe[(~dataframe.name.isin(countries_to_remove))]
        if years_to_remove is not None:
            dataframe["end_date"] = pd.to_datetime(dataframe.end_date)
            dataframe = dataframe[(~dataframe.end_date.dt.year.isin(years_to_remove))]

        self._class_weights: Optional[np.ndarray] = None
        self.task_type = task_type
        self.croptype_list = croptype_list
        self.return_hierarchical_labels = return_hierarchical_labels
        self.augment = augment
        if augment:
            logger.info(
                "Augmentation is enabled. \
The horizontal jittering of the selected window will be performed."
            )
        self.mask_ratio = mask_ratio
        self.mask_params = MaskParamsNoDw(
            (
                "group_bands",
                "random_timesteps",
                "chunk_timesteps",
                "random_combinations",
            ),
            mask_ratio,
        )

        super().__init__(dataframe)
        if balance:
            if self.task_type == "cropland":
                logger.info("Balancing is enabled. Underrepresented class will be upsampled.")
                neg_indices, pos_indices = [], []
                for loc_idx, (_, row) in enumerate(self.df.iterrows()):
                    target = self.target_crop(
                        row, self.task_type, self.croptype_list, return_hierarchical_labels
                    )
                    if target == 0:
                        neg_indices.append(loc_idx)
                    else:
                        pos_indices.append(loc_idx)
                if len(pos_indices) > len(neg_indices):
                    self.indices = (
                        pos_indices + (len(pos_indices) // len(neg_indices)) * neg_indices
                    )
                elif len(neg_indices) > len(pos_indices):
                    self.indices = (
                        neg_indices + (len(neg_indices) // len(pos_indices)) * pos_indices
                    )
                else:
                    self.indices = neg_indices + pos_indices
            if self.task_type == "croptype":
                classes_lst = self.df["balancing_class"].unique()

                # optimal_class_size = self.df["balancing_class"].value_counts().max()
                optimal_class_size = 10000
                balancing_coeff = 1.5

                logger.info(
                    f"Balancing is enabled. Underrepresented classes will be randomly \
upsampled until they reach size {optimal_class_size}, but no more than {balancing_coeff} \
times of the initial class size."
                )

                balanced_inds = []
                for tclass in classes_lst:
                    tclass_sample_ids = self.df[self.df["balancing_class"] == tclass].index
                    tclass_loc_idx = [self.df.index.get_loc(xx) for xx in tclass_sample_ids]
                    if len(tclass_loc_idx) < optimal_class_size:

                        if balancing_coeff > 0:
                            if (optimal_class_size / len(tclass_loc_idx)) > balancing_coeff:
                                samples_to_add = np.random.choice(
                                    tclass_loc_idx, size=int(len(tclass_loc_idx) / balancing_coeff)
                                )
                                tclass_loc_idx.extend(list(samples_to_add))
                            else:
                                tclass_loc_idx = tclass_loc_idx * (
                                    optimal_class_size // len(tclass_loc_idx)
                                )
                        else:
                            tclass_loc_idx = tclass_loc_idx * (
                                optimal_class_size // len(tclass_loc_idx)
                            )

                    balanced_inds.extend(tclass_loc_idx)
                self.indices = balanced_inds
        else:
            self.indices = [i for i in range(len(self.df))]

    @staticmethod
    def target_crop(
        row_d: pd.Series,
        task_type: str = "cropland",
        croptype_list: List = [],
        return_hierarchical_labels: bool = False,
    ) -> Union[int, np.ndarray, List]:

        _target: Union[int, np.ndarray, List]
        if task_type == "cropland":
            _target = int(row_d["LANDCOVER_LABEL"] == 11)
        if task_type == "croptype":
            if return_hierarchical_labels:
                _target = [row_d["landcover_name"], row_d["downstream_class"]]
            elif len(croptype_list) == 0:
                _target = row_d["downstream_class"]
            else:
                _target = np.array(row_d[croptype_list].astype(int).values)
        return _target

    @staticmethod
    def multiply_list_length_by_float(input_list: List, multiplier: float) -> List:
        decimal_part, integer_part = modf(multiplier)
        sublist = sample(input_list, k=int(len(input_list) * decimal_part))
        return input_list * int(integer_part) + sublist

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the sample
        df_index = self.indices[idx]
        row = self.df.iloc[df_index, :]
        eo, mask_per_token, latlon, month, valid_month = self.row_to_arrays(
            row, self.task_type, self.croptype_list, augment=self.augment
        )

        if self.mask_ratio > 0:
            mask_per_variable, normed_eo, _, _ = self.mask_params.mask_data(
                self.normalize_and_mask(eo), mask_per_token
            )
        else:
            normed_eo = self.normalize_and_mask(eo)
            mask_per_variable = np.repeat(mask_per_token, BAND_EXPANSION, axis=1)

        target = self.target_crop(
            row, self.task_type, self.croptype_list, self.return_hierarchical_labels
        )

        return (
            normed_eo,
            target,
            np.ones(self.NUM_TIMESTEPS) * (DynamicWorld2020_2021.class_amount),
            latlon,
            month,
            valid_month,
            mask_per_variable,
        )

    @property
    def class_weights(self) -> np.ndarray:
        from sklearn.utils.class_weight import compute_class_weight

        if self._class_weights is None:
            ys: Union[List, np.ndarray]
            ys = []
            for _, row in self.df.iterrows():
                ys.append(
                    self.target_crop(
                        row, self.task_type, self.croptype_list, self.return_hierarchical_labels
                    )
                )
            self._class_weights = compute_class_weight(
                class_weight="balanced", classes=np.unique(ys), y=ys
            )
        return self._class_weights

