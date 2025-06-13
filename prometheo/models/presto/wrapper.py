import io
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import requests
import torch
from einops import repeat, rearrange
from torch import nn

from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
    to_torchtensor,
    ArrayTensor
)
from prometheo.utils import device

from .single_file_presto import (
    BANDS,
    BANDS_ADD,
    BANDS_DIV,
    BANDS_GROUPS_IDX,
    NUM_DYNAMIC_WORLD_CLASSES,
    FinetuningHead,
    Presto,
    get_sinusoid_encoding_table,
)

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
    hs = [v.shape[1] for v in [x.s1, x.s2, x.dem] if v is not None]
    ws = [v.shape[2] for v in [x.s1, x.s2, x.dem] if v is not None]
    if len(timesteps) == 0:
        raise ValueError("One of s1, s2, meteo must be not None")
    if len(hs) == 0:
        raise ValueError("One of s1, s2, dem must be not None")
    if not len(set(batch_sizes)) == 1:
        raise ValueError("dim 0 (batch size) must be consistent for s1, s2, dem, meteo")
    if not len(set(timesteps)) == 1:
        raise ValueError("dim -2 (timesteps) must be consistent for s1, s2, meteo")
    if not len(set(hs)) == 1:
        raise ValueError("dim 1 (height) must be consistent for s1, s2, dem")
    h = hs[0]
    if not len(set(ws)) == 1:
        raise ValueError("dim 2 (width) must be consistent for s1, s2, dem")
    w = ws[0]

    batch_size, timesteps = batch_sizes[0], timesteps[0]
    total_bands = sum([len(v) for _, v in BANDS_GROUPS_IDX.items()])

    mask, output = (
        np.ones((batch_size, h, w, timesteps, total_bands)),
        np.zeros((batch_size, h, w, timesteps, total_bands)),
    )

    if x.s1 is not None:
        # for some reason, doing
        # x.s1[ :, 0, 0, :, mapper["S1"]["predictor"]]
        # directly yields an array of shape [bands, batch_size, timesteps]
        # but splitting it like this doesn't. I am not sure why
        s1_hw = x.s1[:, :, :, :, :]
        s1_hw_bands = s1_hw[:, :, :, :, mapper["S1"]["predictor"]]
        output[:, :, :, :, mapper["S1"]["presto"]] = s1_hw_bands
        mask[:, :, :, :, mapper["S1"]["presto"]] = s1_hw_bands == NODATAVALUE

    if x.s2 is not None:
        s2_hw = x.s2[:, :, :, :, :]
        s2_hw_bands = s2_hw[:, :, :, :, mapper["S2"]["predictor"]]
        output[:, :, :, :, mapper["S2"]["presto"]] = s2_hw_bands
        mask[:, :, :, :, mapper["S2"]["presto"]] = s2_hw_bands == NODATAVALUE

    if x.dem is not None:
        dem_with_time = repeat(
            x.dem[:, :, :, mapper["dem"]["predictor"]], "b h w d -> b h w t d", t=timesteps
        )
        output[:, :, :, :, mapper["dem"]["presto"]] = dem_with_time
        mask[:, :, :, :, mapper["dem"]["presto"]] = dem_with_time == NODATAVALUE

    if x.meteo is not None:
        meteo_with_hw = repeat(
            x.meteo[:, :, mapper["meteo"]["predictor"]], "b t d -> b h w t d", h=h, w=w
        )
        output[:, :, :, :, mapper["meteo"]["presto"]] = meteo_with_hw
        mask[:, :, :, :, mapper["meteo"]["presto"]] = (meteo_with_hw == NODATAVALUE)

    dynamic_world = np.ones((batch_size * h * w, timesteps)) * NUM_DYNAMIC_WORLD_CLASSES

    latlon: ArrayTensor | None = None
    if x.latlon is not None:
        latlon = repeat(x.latlon, "b d -> b h w d", h=h, w=w)
        latlon = rearrange(latlon, "b h w d -> (b h w) d")


    timestamps: ArrayTensor | None = None
    if x.timestamps is not None:
        timestamps = repeat(x.timestamps, "b t d -> b h w t d", h=h, w=w)
        timestamps = rearrange(timestamps, "b h w t d -> (b h w) t d")

    output = rearrange(output, "b h w t d -> (b h w) t d")
    mask = rearrange(mask, "b h w t d -> (b h w) t d")

    output, mask = normalize(output, mask)
    return output, mask, dynamic_world, latlon, timestamps, h, w


@lru_cache(maxsize=6)
def load_presto_weights(
    presto_model: Presto,
    weights_path: Union[str, Path] = default_model_path,
    strict: bool = True,
):
    """Load pretrained weights into a Presto model.

    Parameters
    ----------
    presto_model : Presto
        The Presto model to load the pretrained weights into.
    weights_path : Union[str, Path], optional
        The path to the pretrained weights file. If not provided, the default model path will be used.
    strict : bool, optional
        Whether to strictly enforce that the keys in the pretrained weights match the keys in the model.
        If True, an error will be raised if there are any missing or unexpected keys. If False, missing or
        unexpected keys will be ignored. Default is True.

    Returns
    -------
    Presto
        The Presto model with the pretrained weights loaded.

    Raises
    ------
    FileNotFoundError
        If the specified model path does not exist.
    """
    presto_model.to(device)
    if isinstance(weights_path, str) and (weights_path.startswith("http")):
        response = requests.get(weights_path)
        presto_model_layers = torch.load(
            io.BytesIO(response.content), map_location=device
        )
        presto_model.load_state_dict(presto_model_layers, strict=strict)
    else:
        presto_model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=strict
        )

    return presto_model


class PretrainedPrestoWrapper(nn.Module):
    def __init__(
        self,
        num_outputs: Optional[int] = None,
        regression: Optional[bool] = None,
        pretrained_model_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize the Presto model through a prometheo wrapper.

        Parameters
        ----------
        num_outputs : Optional[int], optional
            The number of output units of the model, by default None if no head
            should be used.
        regression : Optional[bool], optional
            Whether the model performs regression or not, by default None.
            Needs to be specified when num_outputs is not None.
        pretrained_model_path : Union[str, Path], optional
            The path to the pretrained model, by default None.

        Raises
        ------
        ValueError
            If both num_outputs and regression are not None or both are None.

        """
        super().__init__()

        # Construct original Presto model with the default configuration
        presto = Presto.construct()

        # Load pretrained model before making any adaptations
        if pretrained_model_path is not None:
            presto = load_presto_weights(presto, pretrained_model_path, strict=False)

        # Extract the encoder from the original Presto model
        self.encoder = presto.encoder

        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # but don't unfreeze the month encoder, which
        # shouldn't be trainable
        self.encoder.month_embed.requires_grad_(False)

        max_sequence_length = 72
        old_pos_embed_device = self.encoder.pos_embed.device
        self.encoder.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                max_sequence_length,
                self.encoder.pos_embed.shape[-1],
                device=old_pos_embed_device,
            ),
            requires_grad=False,
        )
        pos_embed = get_sinusoid_encoding_table(
            self.encoder.pos_embed.shape[1], self.encoder.pos_embed.shape[-1]
        )
        self.encoder.pos_embed.data.copy_(pos_embed.to(device=old_pos_embed_device))
        self.encoder.pos_embed.requires_grad_(False)

        head_variables = [num_outputs, regression]
        if len([x for x in head_variables if x is not None]) not in [
            0,
            len(head_variables),
        ]:
            raise ValueError("num_outputs and regression must both be None or not None")

        self.head: Optional[nn.Module] = None
        if num_outputs is not None:
            if regression is None:
                raise ValueError("regression cannot be None if num_outputs is not None")
            self.head = FinetuningHead(
                hidden_size=self.encoder.embedding_size,
                num_outputs=num_outputs,
                regression=regression,
            )

    def forward(
        self, x: Predictors, eval_pooling: Literal["global", "time", None] = "global"
    ):
        """
        If x.label is not None, then we infer the output pooling from the labels (time or global).
        If x.label is None, then we default to the eval_pooling argument passed to forward.

        Encoder output pooling can be three options:

        eval_pooling = "global" -> global pooling (DEFAULT)
        eval_pooling = "time" -> time pooling, for time-explicit embeddings
        eval_pooling = None -> no pooling (for SSL)

        Note: in case a finetuning head is part of Presto, it does not include activation.

        """

        s1_s2_era5_srtm, mask, dynamic_world, latlon, timestamps, h, w= dataset_to_model(x)

        # labels should have shape [B, H, W, T or 1, num_outputs].
        # need some way to communicate global vs time if
        # they are not passed as part of the predictors.
        if x.label is not None:
            if x.label.shape[3] == dynamic_world.shape[1]:
                eval_pooling = "time"
            else:
                if x.label.shape[1] != 1:
                    raise ValueError(f"Unexpected label shape {x.label.shape}")
                eval_pooling = "global"

        if x.timestamps is None:
            raise ValueError("Presto requires input timestamps")

        model_device = self.encoder.pos_embed.device
        embeddings = self.encoder(
            x=to_torchtensor(s1_s2_era5_srtm, device=model_device).float(),
            dynamic_world=to_torchtensor(dynamic_world, device=model_device).long(),
            latlons=to_torchtensor(latlon, device=model_device).float(),
            mask=to_torchtensor(mask, device=model_device).long(),
            # presto wants 0 indexed months, not 1 indexed months
            month=to_torchtensor(timestamps[:, :, 1] - 1, device=model_device),
            eval_pooling=eval_pooling,
        )

        # Need to reintroduce spatial and temporal dims according to prometheo convention
        if eval_pooling == "global":
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(embeddings, "(b h w) d -> b h w d", b=b, h=h, w=w)
            # add the time dimension back
            embeddings = torch.unsqueeze(embeddings, 3)
        elif eval_pooling == "time":
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(embeddings, "(b h w) t d -> b h w t d", b=b, h=h, w=w)
        else:
            if ((h != 1)) or ((w != 1)):
                raise ValueError("h w != 1 unsupported for SSL")
            pass  # In case of no pooling we assume SSL and don't change embeddings

        if self.head is not None:
            return self.head(embeddings)
        else:
            return embeddings
