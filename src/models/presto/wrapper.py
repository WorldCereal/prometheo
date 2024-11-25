from .single_file_presto import BANDS_GROUPS_IDX, BANDS
import numpy as np
from src.predictors import Predictors, S1_bands, NODATAVALUE


mapper = {
    "S1": {
        "predictor": [S1_bands.index("VV"), S1_bands.index("VH")],
        "presto": [BANDS.index("VV"), BANDS.index("VH")],
    }
}


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

    # if dynamic_world is None:
    #     dynamic_world = np.ones(batch_size, timesteps) * (DynamicWorld2020_2021.class_amount)

    # if normalize:
    #     # normalize includes x = x[:, keep_indices]
    #     x = S1_S2_ERA5_SRTM.normalize(x)
    #     if s2_bands is not None:
    #         if ("B8" in s2_bands) and ("B4" in s2_bands):
    #             mask[:, NORMED_BANDS.index("NDVI")] = 0
    # else:
    #     x = x[:, keep_indices]
    # return x, mask, dynamic_world
