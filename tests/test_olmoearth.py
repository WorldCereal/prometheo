import sys
import types
import unittest
from typing import ClassVar

import numpy as np
import torch
from einops import repeat

from prometheo.models.olmoearth.wrapper import (
    PretrainedOlmoEarthWrapper,
    S2_OLMOEARTH_TO_PROMETHEO,
    dataset_to_olmoearth_sample,
)
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import NODATAVALUE, S1_BANDS, S2_BANDS, Predictors


class FakeModality:
    SENTINEL2_L2A = "sentinel2_l2a"
    SENTINEL1 = "sentinel1"


class FakeNormalizer:
    calls: ClassVar[list[tuple[object, np.ndarray]]] = []

    def __init__(self, std_multiplier):
        self.std_multiplier = std_multiplier

    def normalize(self, modality, data):
        self.calls.append((modality, data.copy()))
        return data + 1


class FakeSample:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEncoder:
    embedding_size = 3

    def __call__(self, sample, patch_size, input_res, fast_pass):
        b, h, w, t, _ = sample.sentinel2_l2a.shape
        base = torch.arange(b * h * w * t * self.embedding_size, dtype=torch.float32)
        tokens = base.reshape(b, h, w, t, 1, self.embedding_size)
        return {"tokens_and_masks": types.SimpleNamespace(sentinel2_l2a=tokens, sentinel1=tokens + 2)}


class FakeModel:
    def __init__(self):
        self.encoder = FakeEncoder()


class FakeModelID:
    OLMOEARTH_V1_2_NANO = "nano-id"
    OLMOEARTH_V1_2_TINY = "tiny-id"


class OlmoEarthDependencyPatch(unittest.TestCase):
    def setUp(self):
        FakeNormalizer.calls = []
        self.saved_modules = dict(sys.modules)
        top = types.ModuleType("olmoearth_pretrain_minimal")
        top.ModelID = FakeModelID
        top.Normalizer = FakeNormalizer
        top.loaded = []

        def load_model_from_id(model_id, load_weights=True):
            top.loaded.append((model_id, load_weights))
            return FakeModel()

        top.load_model_from_id = load_model_from_id

        constants = types.ModuleType(
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants"
        )
        constants.Modality = FakeModality
        datatypes = types.ModuleType(
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes"
        )
        datatypes.MaskedOlmoEarthSample = FakeSample
        sys.modules["olmoearth_pretrain_minimal"] = top
        sys.modules["olmoearth_pretrain_minimal.olmoearth_pretrain_v1"] = types.ModuleType(
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1"
        )
        sys.modules["olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils"] = types.ModuleType(
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils"
        )
        sys.modules[constants.__name__] = constants
        sys.modules[datatypes.__name__] = datatypes
        self.top = top

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.saved_modules)


class TestOlmoEarthAdapter(OlmoEarthDependencyPatch):
    def test_dataset_to_sample_reorders_bands_masks_and_zero_indexes_months(self):
        b, h, w, t = 1, 1, 1, 2
        s2 = np.zeros((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        for band_idx in range(len(S2_BANDS)):
            s2[..., band_idx] = band_idx
        s2[:, :, :, 1, S2_BANDS.index("B2")] = NODATAVALUE
        s1 = np.zeros((b, h, w, t, len(S1_BANDS)), dtype=np.float32)
        s1[..., S1_BANDS.index("VV")] = 10
        s1[..., S1_BANDS.index("VH")] = 20
        timestamps = np.array([[[1, 1, 2024], [1, 12, 2024]]])

        sample = dataset_to_olmoearth_sample(
            Predictors(s1=s1, s2=s2, timestamps=timestamps), model_device=torch.device("cpu")
        )

        self.assertEqual(sample.sentinel2_l2a.shape, (b, h, w, t, 12))
        # Normalizer adds one after reordering, so compare against original band indices + 1.
        expected_first_timestep = [
            S2_BANDS.index(prometheo) + 1 for _, prometheo in S2_OLMOEARTH_TO_PROMETHEO
        ]
        self.assertEqual(sample.sentinel2_l2a[0, 0, 0, 0].tolist(), expected_first_timestep)
        self.assertEqual(sample.sentinel2_l2a_mask[0, 0, 0].tolist(), [0, 3])
        self.assertEqual(sample.sentinel1[0, 0, 0, 0].tolist(), [11, 21])
        self.assertEqual(sample.timestamps[0, :, 1].tolist(), [0, 11])

    def test_wrapper_loads_latest_nano_by_default_and_pools_global_outputs(self):
        b, h, w, t = 2, 2, 1, 4
        timestamps = repeat(np.array([[1, m + 1, 2024] for m in range(t)]), "t d -> b t d", b=b)
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            timestamps=timestamps,
        )

        model = PretrainedOlmoEarthWrapper(load_weights=False)
        output = model(x, eval_pooling=PoolingMethods.GLOBAL)

        self.assertEqual(self.top.loaded, [("nano-id", False)])
        self.assertEqual(output.shape, (b, h, w, 1, 3))

    def test_wrapper_time_pooling_preserves_temporal_dimension(self):
        b, h, w, t = 1, 1, 1, 3
        timestamps = repeat(np.array([[1, m + 1, 2024] for m in range(t)]), "t d -> b t d", b=b)
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            timestamps=timestamps,
        )

        model = PretrainedOlmoEarthWrapper(model_id="OLMOEARTH_V1_2_TINY", load_weights=False)
        output = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(self.top.loaded, [("tiny-id", False)])
        self.assertEqual(output.shape, (b, h, w, t, 3))


if __name__ == "__main__":
    unittest.main()
