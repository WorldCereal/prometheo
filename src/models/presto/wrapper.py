from .single_file_presto import (
    BANDS_GROUPS_IDX,
    BANDS,
    NUM_DYNAMIC_WORLD_CLASSES,
    BANDS_ADD,
    BANDS_DIV,
)
import numpy as np
from src.predictors import (
    Predictors,
    S1_bands,
    NODATAVALUE,
    meteo_bands,
    dem_bands,
    S2_bands,
)
from einops import repeat
import warnings


# the mapper defines the mapping of predictor bands to presto bands
mapper = {
    "S1": {
        "predictor": [S1_bands.index("VV"), S1_bands.index("VH")],
        "presto": [BANDS.index("VV"), BANDS.index("VH")],
    },
    "S2": {
        "predictor": [
            S2_bands.index("B2"),
            S2_bands.index("B3"),
            S2_bands.index("B4"),
            S2_bands.index("B5"),
            S2_bands.index("B6"),
            S2_bands.index("B7"),
            S2_bands.index("B8"),
            S2_bands.index("B8A"),
            S2_bands.index("B11"),
            S2_bands.index("B12"),
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
            meteo_bands.index("temperature"),
            meteo_bands.index("precipitation"),
        ],
        "presto": [BANDS.index("temperature_2m"), BANDS.index("total_precipitation")],
    },
    "dem": {
        "predictor": [dem_bands.index("elevation"), dem_bands.index("slope")],
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
        x = np.where(
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
    x = ((x + BANDS_ADD) / BANDS_DIV).astype(np.float32)
    ndvi, ndvi_mask = calculate_ndvi(x)

    if len(x.shape) == 2:
        x[:, BANDS.index("NDVI")] = ndvi
        mask[:, BANDS.index("NDVI")] = ndvi_mask

    else:
        x[:, :, BANDS.index("NDVI")] = ndvi
        mask[:, :, BANDS.index("NDVI")] = ndvi_mask
    return x


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

    mask, x = (
        np.ones(batch_size, timesteps, total_bands),
        np.zeros(batch_size, timesteps, total_bands),
    )

    if x.s1 is not None:
        h, w = x.s1.shape[1], x.s1.shape[2]
        if (h != 1) or (w != 1):
            raise ValueError("Presto does not support h, w > 1")

        x[:, :, mapper["S1"]["presto"]] = x.s1[:, 0, 0, :, mapper["S1"]["predictor"]]
        mask[:, :, mapper["S1"]["presto"]] = (
            x.s1[:, 0, 0, :, mapper["S1"]["predictor"]] == NODATAVALUE
        )

    if x.s2 is not None:
        h, w = x.s2.shape[1], x.s2.shape[2]
        if (h != 1) or (w != 1):
            raise ValueError("Presto does not support h, w > 1")

        x[:, :, mapper["S2"]["presto"]] = x.s2[:, 0, 0, :, mapper["S2"]["predictor"]]
        mask[:, :, mapper["S2"]["presto"]] = (
            x.s2[:, 0, 0, :, mapper["S2"]["predictor"]] == NODATAVALUE
        )

    if x.meteo is not None:
        x[:, :, mapper["meteo"]["presto"]] = x.meteo[:, :, mapper["meteo"]["predictor"]]
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
        mask[:, :, mapper["dem"]["presto"]] = dem_with_time == NODATAVALUE

    dynamic_world = np.ones(batch_size, timesteps) * NUM_DYNAMIC_WORLD_CLASSES

    x, mask = normalize(x, mask)
    return x, mask, dynamic_world
