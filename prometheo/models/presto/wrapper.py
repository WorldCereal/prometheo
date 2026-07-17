import io
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import requests
import torch
from einops import rearrange, repeat
from torch import nn

from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    ArrayTensor,
    Predictors,
)
from prometheo.utils import device

from ..pooling import PoolingMethods
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
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
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

    ndvi, _ = calculate_ndvi(x)

    # Derive the NDVI mask from the upstream B4 / B8 masks rather than
    # recomputing it from the (already-normalized) values. The recomputed
    # ndvi_mask checks `band == NODATAVALUE` on values that have been
    # divided by 10000, so masked positions look "valid" with NDVI = 0.
    b4_idx, b8_idx, ndvi_idx = (
        BANDS.index("B4"),
        BANDS.index("B8"),
        BANDS.index("NDVI"),
    )
    if isinstance(x, np.ndarray):
        ndvi_mask_from_input = np.logical_or(mask[..., b4_idx], mask[..., b8_idx])
    else:
        ndvi_mask_from_input = torch.logical_or(
            mask[..., b4_idx].bool(), mask[..., b8_idx].bool()
        )

    if len(x.shape) == 2:
        x[:, ndvi_idx] = ndvi
        mask[:, ndvi_idx] = ndvi_mask_from_input
    else:
        x[:, :, ndvi_idx] = ndvi
        mask[:, :, ndvi_idx] = ndvi_mask_from_input
    return x, mask


def dataset_to_model(x: Predictors, device: Optional[torch.device] = None):
    batch_sizes = [v.shape[0] for v in [x.s1, x.s2, x.meteo, x.dem] if v is not None]
    timesteps = [v.shape[-2] for v in [x.s1, x.s2, x.meteo] if v is not None]
    hs = [v.shape[1] for v in [x.s1, x.s2, x.dem, x.latlon] if v is not None]
    ws = [v.shape[2] for v in [x.s1, x.s2, x.dem, x.latlon] if v is not None]
    if len(timesteps) == 0:
        raise ValueError("One of s1, s2, meteo must be not None")
    if len(hs) == 0:
        raise ValueError("One of s1, s2, dem must be not None")
    if not len(set(batch_sizes)) == 1:
        raise ValueError("dim 0 (batch size) must be consistent for s1, s2, dem, meteo")
    if not len(set(timesteps)) == 1:
        raise ValueError("dim -2 (timesteps) must be consistent for s1, s2, meteo")
    if not len(set(hs)) == 1:
        raise ValueError("dim 1 (height) must be consistent for s1, s2, dem, latlon")
    h = hs[0]
    if not len(set(ws)) == 1:
        raise ValueError("dim 2 (width) must be consistent for s1, s2, dem, latlon")
    w = ws[0]

    batch_size, timesteps = batch_sizes[0], timesteps[0]
    total_bands = sum([len(v) for _, v in BANDS_GROUPS_IDX.items()])

    if device is not None:
        # Torch-native path: assemble and normalize on `device` in float32.
        # This avoids a per-batch float64 CPU allocation and serialization with the GPU.
        s1 = _to_device_tensor(x.s1, device) if x.s1 is not None else None
        s2 = _to_device_tensor(x.s2, device) if x.s2 is not None else None
        meteo = _to_device_tensor(x.meteo, device) if x.meteo is not None else None
        dem = _to_device_tensor(x.dem, device) if x.dem is not None else None

        output = torch.zeros(
            (batch_size, h, w, timesteps, total_bands),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.ones(
            (batch_size, h, w, timesteps, total_bands),
            dtype=torch.bool,
            device=device,
        )

        def _fill(group: str, values: torch.Tensor):
            values = values.to(torch.float32)
            presto_idx = mapper[group]["presto"]
            output[..., presto_idx] = values
            mask[..., presto_idx] = values == NODATAVALUE

        if s1 is not None:
            _fill("S1", s1[..., mapper["S1"]["predictor"]])
        if s2 is not None:
            _fill("S2", s2[..., mapper["S2"]["predictor"]])
        if dem is not None:
            dem_with_time = (
                dem[..., mapper["dem"]["predictor"]]
                .unsqueeze(3)
                .expand(-1, -1, -1, timesteps, -1)
            )
            _fill("dem", dem_with_time)
        if meteo is not None:
            _fill("meteo", meteo[..., mapper["meteo"]["predictor"]])

        dynamic_world = torch.full(
            (batch_size * h * w, timesteps),
            NUM_DYNAMIC_WORLD_CLASSES,
            dtype=torch.long,
            device=device,
        )

        latlon: Union[ArrayTensor, None] = None
        if x.latlon is not None:
            latlon = rearrange(
                _to_device_tensor(x.latlon, device).to(torch.float32),
                "b h w d -> (b h w) d",
            )

        timestamps: Union[ArrayTensor, None] = None
        if x.timestamps is not None:
            timestamps = _to_device_tensor(x.timestamps, device).long()
            timestamps = repeat(timestamps, "b t d -> b h w t d", h=h, w=w)
            timestamps = rearrange(timestamps, "b h w t d -> (b h w) t d")

        output = rearrange(output, "b h w t d -> (b h w) t d")
        mask = rearrange(mask, "b h w t d -> (b h w) t d")

        bands_add, bands_div = _normalize_constants(device, torch.float32)
        output = (output + bands_add) / bands_div

        b4_idx, b8_idx, ndvi_idx = (
            BANDS.index("B4"),
            BANDS.index("B8"),
            BANDS.index("NDVI"),
        )
        band_1 = output[..., b8_idx]
        band_2 = output[..., b4_idx]
        denom = band_1 + band_2
        ndvi = torch.where(
            denom > 0, (band_1 - band_2) / denom, torch.zeros_like(denom)
        )
        output[..., ndvi_idx] = ndvi
        mask[..., ndvi_idx] = mask[..., b8_idx] | mask[..., b4_idx]

        return output, mask.long(), dynamic_world, latlon, timestamps, h, w

    else:
        # NumPy path (CPU, used e.g. during inference without a GPU).
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
                x.dem[:, :, :, mapper["dem"]["predictor"]],
                "b h w d -> b h w t d",
                t=timesteps,
            )
            output[:, :, :, :, mapper["dem"]["presto"]] = dem_with_time
            mask[:, :, :, :, mapper["dem"]["presto"]] = dem_with_time == NODATAVALUE

        if x.meteo is not None:
            meteo_hw = x.meteo[:, :, :, :, :]
            meteo_hw_bands = meteo_hw[:, :, :, :, mapper["meteo"]["predictor"]]
            output[:, :, :, :, mapper["meteo"]["presto"]] = meteo_hw_bands
            mask[:, :, :, :, mapper["meteo"]["presto"]] = meteo_hw_bands == NODATAVALUE

        dynamic_world = (
            np.ones((batch_size * h * w, timesteps)) * NUM_DYNAMIC_WORLD_CLASSES
        )

        latlon: Union[ArrayTensor, None] = None
        if x.latlon is not None:
            latlon = rearrange(x.latlon, "b h w d -> (b h w) d")

        timestamps: Union[ArrayTensor, None] = None
        if x.timestamps is not None:
            timestamps = repeat(x.timestamps, "b t d -> b h w t d", h=h, w=w)
            timestamps = rearrange(timestamps, "b h w t d -> (b h w) t d")

        output = rearrange(output, "b h w t d -> (b h w) t d")
        mask = rearrange(mask, "b h w t d -> (b h w) t d")

        output, mask = normalize(output, mask)
        return output, mask, dynamic_world, latlon, timestamps, h, w


_NORMALIZE_CONSTANTS: dict = {}


def _normalize_constants(device: torch.device, dtype: torch.dtype):
    """Cache (BANDS_ADD, BANDS_DIV) tensors per device/dtype."""
    key = (device, dtype)
    if key not in _NORMALIZE_CONSTANTS:
        _NORMALIZE_CONSTANTS[key] = (
            torch.tensor(BANDS_ADD, device=device, dtype=dtype),
            torch.tensor(BANDS_DIV, device=device, dtype=dtype),
        )
    return _NORMALIZE_CONSTANTS[key]


def _to_device_tensor(v, device: torch.device, non_blocking: bool = True):
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    return v.to(device, non_blocking=non_blocking)


@lru_cache(maxsize=6)
def load_presto_weights(
    presto_model: Presto,
    weights_path: Union[str, Path] = default_model_path,
    strict: bool = True,
) -> Presto:
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
        latlon_dropout: float = 0.0,
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
        latlon_dropout : float, optional
            Probability with which the latlon token is masked out during
            training. If 0 (default), the latlon token is always kept.

        Raises
        ------
        ValueError
            If both num_outputs and regression are not None or both are None.

        """
        super().__init__()

        # Construct original Presto model with the default configuration
        presto = Presto.construct(latlon_dropout=latlon_dropout)

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
        self,
        x: Predictors,
        eval_pooling: Union[PoolingMethods, None] = PoolingMethods.GLOBAL,
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

        model_device = self.encoder.pos_embed.device
        s1_s2_era5_srtm, mask, dynamic_world, latlon, timestamps, h, w = (
            dataset_to_model(x, device=model_device)
        )

        # labels should have shape [B, H, W, T or 1, num_outputs].
        # need some way to communicate global vs time if
        # they are not passed as part of the predictors.
        if x.label is not None:
            if x.label.shape[3] == dynamic_world.shape[1]:
                eval_pooling = PoolingMethods.TIME
            else:
                if x.label.shape[1] != 1:
                    raise ValueError(f"Unexpected label shape {x.label.shape}")
                eval_pooling = PoolingMethods.GLOBAL

        if x.timestamps is None:
            raise ValueError("Presto requires input timestamps")

        embeddings = self.encoder(
            x=s1_s2_era5_srtm,
            dynamic_world=dynamic_world,
            latlons=latlon,
            mask=mask,
            # presto wants 0 indexed months, not 1 indexed months
            month=timestamps[:, :, 1] - 1,
            eval_pooling=eval_pooling.value if eval_pooling is not None else None,
        )

        # Need to reintroduce spatial and temporal dims according to prometheo convention
        if eval_pooling == PoolingMethods.GLOBAL:
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(embeddings, "(b h w) d -> b h w d", b=b, h=h, w=w)
            # add the time dimension back
            embeddings = torch.unsqueeze(embeddings, 3)
        elif eval_pooling == PoolingMethods.TIME:
            b = int(embeddings.shape[0] / (h * w))
            embeddings = rearrange(
                embeddings, "(b h w) t d -> b h w t d", b=b, h=h, w=w
            )
        else:
            if (h != 1) or (w != 1):
                raise ValueError("h w != 1 unsupported for SSL")
            pass  # In case of no pooling we assume SSL and don't change embeddings

        if self.head is not None:
            if eval_pooling is None:
                raise ValueError("Can't use the head without a pooling method")
            return self.head(embeddings)
        else:
            return embeddings
