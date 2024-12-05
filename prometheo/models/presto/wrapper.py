from .single_file_presto import (
    BANDS_GROUPS_IDX,
    BANDS,
    NUM_DYNAMIC_WORLD_CLASSES,
    BANDS_ADD,
    BANDS_DIV,
    Encoder,
    FinetuningHead
)
import numpy as np
from prometheo.predictors import (
    Predictors,
    S1_BANDS,
    NODATAVALUE,
    METEO_BANDS,
    DEM_BANDS,
    S2_BANDS,
    to_torchtensor,
)
from einops import repeat
import warnings
from torch import nn
from typing import Union, Optional
from functools import lru_cache
import torch

from prometheo.utils import device
from pathlib import Path
import requests
import io

default_model_path = Path(__file__).parent / "default_model.pt"


# the mapper defines the mapping of predictor bands to presto bands
mapper = {
    "S1": {
        "predictor": [S1_BANDS.index("VV"), S1_BANDS.index("VH")],
        "presto": [BANDS.index("VV"), BANDS.index("VH")],
    },
    "S2": {
        "predictor": [
            S2_BANDS.index("B2"),
            S2_BANDS.index("B3"),
            S2_BANDS.index("B4"),
            S2_BANDS.index("B5"),
            S2_BANDS.index("B6"),
            S2_BANDS.index("B7"),
            S2_BANDS.index("B8"),
            S2_BANDS.index("B8A"),
            S2_BANDS.index("B11"),
            S2_BANDS.index("B12"),
        ],
        "presto": [
            BANDS.index("B2"),
            BANDS.index("B3"),
            BANDS.index("B4"),
            BANDS.index("B5"),
            BANDS.index("B6"),
            BANDS.index("B7"),
            BANDS.index("B8"),
            BANDS.index("B8A"),
            BANDS.index("B11"),
            BANDS.index("B12"),
        ],
    },
    "meteo": {
        "predictor": [
            METEO_BANDS.index("temperature"),
            METEO_BANDS.index("precipitation"),
        ],
        "presto": [BANDS.index("temperature_2m"), BANDS.index("total_precipitation")],
    },
    "dem": {
        "predictor": [DEM_BANDS.index("elevation"), DEM_BANDS.index("slope")],
        "presto": [BANDS.index("elevation"), BANDS.index("slope")],
    },
}


def calculate_ndvi(input_array):
    r"""
    Given an input array of shape [timestep, bands] or [batches, timesteps, shapes]
    where bands == len(bands), returns an array of shape
    [timestep, bands + 1] where the extra band is NDVI,
    (b08 - b04) / (b08 + b04)
    """
    band_1, band_2 = "B8", "B4"

    num_dims = len(input_array.shape)
    if num_dims == 2:
        band_1_np = input_array[:, BANDS.index(band_1)]
        band_2_np = input_array[:, BANDS.index(band_2)]
    elif num_dims == 3:
        band_1_np = input_array[:, :, BANDS.index(band_1)]
        band_2_np = input_array[:, :, BANDS.index(band_2)]
    else:
        raise ValueError(f"Expected num_dims to be 2 or 3 - got {num_dims}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in true_divide"
        )
        # suppress the following warning
        # RuntimeWarning: invalid value encountered in true_divide
        # for cases where near_infrared + red == 0
        # since this is handled in the where condition
        if isinstance(band_1_np, np.ndarray):
            x = np.where(
                (band_1_np + band_2_np) > 0,
                (band_1_np - band_2_np) / (band_1_np + band_2_np),
                0,
            )
        else:
            x = torch.where(
                (band_1_np + band_2_np) > 0,
                (band_1_np - band_2_np) / (band_1_np + band_2_np),
                0,
            )
    mask = (
        (band_1_np == NODATAVALUE)
        | (band_2_np == NODATAVALUE)
        | ((band_1_np + band_2_np) == 0)
    )
    return x, mask


def normalize(x: np.ndarray, mask: np.ndarray):
    if isinstance(x, np.ndarray):
        x = ((x + BANDS_ADD) / BANDS_DIV).astype(np.float32)
    else:
        x = (x + torch.tensor(BANDS_ADD)) / torch.tensor(BANDS_DIV)

    ndvi, ndvi_mask = calculate_ndvi(x)

    if len(x.shape) == 2:
        x[:, BANDS.index("NDVI")] = ndvi
        mask[:, BANDS.index("NDVI")] = ndvi_mask

    else:
        x[:, :, BANDS.index("NDVI")] = ndvi
        mask[:, :, BANDS.index("NDVI")] = ndvi_mask
    return x, mask


def dataset_to_model(x: Predictors):
    batch_sizes = [v.shape[0] for v in [x.s1, x.s2, x.meteo, x.dem] if v is not None]
    timesteps = [v.shape[-2] for v in [x.s1, x.s2, x.meteo] if v is not None]
    if len(timesteps) == 0:
        raise ValueError("One of s1, s2, meteo must be not None")
    if len(batch_sizes) == 0:
        raise ValueError("One of s1, s2, dem, meteo must be not None")
    if not all(v == batch_sizes[0] for v in batch_sizes):
        raise ValueError("dim 0 (batch size) must be consistent for s1, s2, dem, meteo")
    if not all(v == timesteps[0] for v in timesteps):
        raise ValueError("dim -2 (timesteps) must be consistent for s1, s2, meteo")

    batch_size, timesteps = batch_sizes[0], timesteps[0]
    total_bands = sum([len(v) for _, v in BANDS_GROUPS_IDX.items()])

    mask, output = (
        np.ones((batch_size, timesteps, total_bands)),
        np.zeros((batch_size, timesteps, total_bands)),
    )

    if x.s1 is not None:
        h, w = x.s1.shape[1], x.s1.shape[2]
        if (h != 1) or (w != 1):
            raise ValueError("Presto does not support h, w > 1")
        # for some reason, doing
        # x.s1[ :, 0, 0, :, mapper["S1"]["predictor"]]
        # directly yields an array of shape [bands, batch_size, timesteps]
        # but splitting it like this doesn't. I am not sure why
        s1_hw = x.s1[:, 0, 0, :, :]
        s1_hw_bands = s1_hw[:, :, mapper["S1"]["predictor"]]
        output[:, :, mapper["S1"]["presto"]] = s1_hw_bands
        mask[:, :, mapper["S1"]["presto"]] = s1_hw_bands == NODATAVALUE

    if x.s2 is not None:
        h, w = x.s2.shape[1], x.s2.shape[2]
        if (h != 1) or (w != 1):
            raise ValueError("Presto does not support h, w > 1")

        s2_hw = x.s2[:, 0, 0, :, :]
        s2_hw_bands = s2_hw[:, :, mapper["S2"]["predictor"]]
        output[:, :, mapper["S2"]["presto"]] = s2_hw_bands
        mask[:, :, mapper["S2"]["presto"]] = s2_hw_bands == NODATAVALUE

    if x.meteo is not None:
        output[:, :, mapper["meteo"]["presto"]] = x.meteo[
            :, :, mapper["meteo"]["predictor"]
        ]
        mask[:, :, mapper["meteo"]["presto"]] = (
            x.meteo[:, :, mapper["meteo"]["predictor"]] == NODATAVALUE
        )

    if x.dem is not None:
        h, w = x.dem.shape[1], x.dem.shape[2]
        if (h != 1) or (w != 1):
            raise ValueError("Presto does not support h, w > 1")
        dem_with_time = repeat(
            x.dem[:, 0, 0, mapper["dem"]["predictor"]], "b d -> b t d", t=timesteps
        )
        output[:, :, mapper["dem"]["presto"]] = dem_with_time
        mask[:, :, mapper["dem"]["presto"]] = dem_with_time == NODATAVALUE

    dynamic_world = np.ones((batch_size, timesteps)) * NUM_DYNAMIC_WORLD_CLASSES

    output, mask = normalize(output, mask)
    return output, mask, dynamic_world


class PretrainedPrestoWrapper(nn.Module):
    def __init__(self, num_outputs: Optional[int] = None, regression: Optional[bool] = None):
        super().__init__()
        self.presto = Encoder()
        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.presto.requires_grad_(True)
        # but don't unfreeze the position encoder, which
        # shouldn't be trainable
        self.presto.pos_embed.requires_grad_(False)
        self.presto.month_embed.requires_grad_(False)

        head_variables = [num_outputs, regression]
        if len([x for x in head_variables if x is not None]) not in [0, len(head_variables)]:
            raise ValueError("num_outputs and regression must both be None or not None")

        self.head: Optional[nn.Module] = None
        if num_outputs is not None:
            if regression is None:
                raise ValueError("regression cannot be None if num_outputs is not None")
            self.head = FinetuningHead(hidden_size=self.presto.embedding_size, num_outputs=num_outputs, regression=regression)

    @classmethod
    @lru_cache(maxsize=6)
    def load_pretrained(
        cls,
        model_path: Union[str, Path] = default_model_path,
        strict: bool = True,
    ):
        model = cls()
        if isinstance(model_path, str) and (model_path.startswith("http")):
            response = requests.get(model_path)
            presto_model_layers = torch.load(
                io.BytesIO(response.content), map_location=device
            )
            model.presto.load_state_dict(presto_model_layers, strict=strict)
        else:
            model.presto.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )

        return model

    def forward(self, x: Predictors):
        s1_s2_era5_srtm, mask, dynamic_world = dataset_to_model(x)

        # labels should have shape [B, T or 1, outputs].
        # need some way to communicate global vs time if
        # they are not passed as part of the predictors.
        if x.label is not None:
            if x.label.shape[1] == dynamic_world.shape[1]:
                eval_pooling = "time"
            else:
                if x.label.shape[1] != 1:
                    raise ValueError(f"Unexpected label shape {x.label.shape}")
                eval_pooling = "global"
        else:
            eval_pooling = "global"

        if x.month is None:
            raise ValueError("Presto requires input months")

        embeddings = self.presto(
            x=to_torchtensor(s1_s2_era5_srtm, device=device).float(),
            dynamic_world=to_torchtensor(dynamic_world, device=device).long(),
            latlons=to_torchtensor(x.latlon, device=device).float(),
            mask=to_torchtensor(mask, device=device).long(),
            month=to_torchtensor(x.month[:, :, 1], device=device),
            eval_pooling=eval_pooling
        )
        if self.head is not None:
            return self.head(embeddings)
        else:
            return embeddings
