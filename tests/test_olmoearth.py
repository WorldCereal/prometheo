import importlib.util
import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models.olmoearth.wrapper import (
    PretrainedOlmoEarthWrapper,
    S2_OLMOEARTH_TO_PROMETHEO,
    dataset_to_olmoearth_sample,
)
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import DEM_BANDS, NODATAVALUE, S1_BANDS, S2_BANDS, Predictors


@unittest.skipIf(
    importlib.util.find_spec("olmoearth_pretrain_minimal") is None,
    "olmoearth-pretrain-minimal optional dependency is not installed",
)
class TestOlmoEarthAdapter(unittest.TestCase):
    def test_requires_at_least_one_of_s1_or_s2(self):
        b, h, w, t = 1, 1, 1, 2
        timestamps = np.array([[[1, 1, 2024], [1, 2, 2024]]])

        # Neither Sentinel-1 nor Sentinel-2 -> error.
        with self.assertRaises(ValueError):
            dataset_to_olmoearth_sample(
                Predictors(timestamps=timestamps), model_device=torch.device("cpu")
            )

        # Sentinel-1 only is valid; the sample carries S1 and no S2.
        s1 = np.ones((b, h, w, t, len(S1_BANDS)), dtype=np.float32)
        sample = dataset_to_olmoearth_sample(
            Predictors(s1=s1, timestamps=timestamps), model_device=torch.device("cpu")
        )
        self.assertIsNotNone(sample.sentinel1)
        self.assertIsNone(sample.sentinel2_l2a)

    def test_dataset_to_sample_reorders_bands_masks_and_zero_indexes_months(self):
        b, h, w, t = 1, 1, 1, 2
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        # Knock out one band from two different S2 band sets at timestep 1. This
        # exercises both the prometheo -> OlmoEarth band reordering and the
        # per-band-set masking: B2 -> B02 (band set 0), B5 -> B05 (band set 1).
        s2[:, :, :, 1, S2_BANDS.index("B2")] = NODATAVALUE
        s2[:, :, :, 1, S2_BANDS.index("B5")] = NODATAVALUE
        s1 = np.ones((b, h, w, t, len(S1_BANDS)), dtype=np.float32)
        timestamps = np.array([[[1, 1, 2024], [1, 12, 2024]]])

        sample = dataset_to_olmoearth_sample(
            Predictors(s1=s1, s2=s2, timestamps=timestamps),
            model_device=torch.device("cpu"),
        )

        # OlmoEarth's default tokenization splits S2 into 3 band sets, so the mask
        # carries a trailing band-set axis rather than collapsing to one token.
        self.assertEqual(
            sample.sentinel2_l2a.shape, (b, h, w, t, len(S2_OLMOEARTH_TO_PROMETHEO))
        )
        self.assertEqual(sample.sentinel2_l2a_mask.shape, (b, h, w, t, 3))
        # timestep 0 fully present; at timestep 1 band sets 0 and 1 are MISSING (3).
        self.assertEqual(
            sample.sentinel2_l2a_mask[0, 0, 0].tolist(), [[0, 0, 0], [3, 3, 0]]
        )
        # S1 is a single band set.
        self.assertEqual(sample.sentinel1_mask.shape, (b, h, w, t, 1))
        # Months are zero-indexed for OlmoEarth (1, 12 -> 0, 11).
        self.assertEqual(sample.timestamps[0, :, 1].tolist(), [0, 11])

    def test_dem_is_ignored_with_warning(self):
        # SRTM is a decode-only modality for OlmoEarth, so Predictors.dem must not
        # reach the encoder input.
        b, h, w, t = 1, 1, 1, 1
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            dem=np.ones((b, h, w, len(DEM_BANDS)), dtype=np.float32),
            timestamps=np.array([[[1, 1, 2024]]]),
        )

        with self.assertWarns(UserWarning):
            sample = dataset_to_olmoearth_sample(x, model_device=torch.device("cpu"))

        self.assertIsNone(sample.srtm)
        self.assertIsNone(sample.srtm_mask)

    def test_mask_trailing_axis_follows_tokenization_config(self):
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.tokenization import (
            ModalityTokenization,
            TokenizationConfig,
        )

        b, h, w, t = 1, 1, 1, 1
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            timestamps=np.array([[[1, 1, 2024]]]),
        )

        # Default grouping -> 3 band sets.
        default = dataset_to_olmoearth_sample(x, tokenization_config=None)
        self.assertEqual(default.sentinel2_l2a_mask.shape[-1], 3)

        # A config that gives each band its own token -> 12 band sets. The mask's
        # trailing axis must follow the config, not a hardcoded assumption.
        per_band = TokenizationConfig(
            overrides={
                "sentinel2_l2a": ModalityTokenization(
                    band_groups=[[olmoearth] for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
                )
            }
        )
        grouped = dataset_to_olmoearth_sample(x, tokenization_config=per_band)
        self.assertEqual(grouped.sentinel2_l2a_mask.shape[-1], len(S2_OLMOEARTH_TO_PROMETHEO))

    def test_wrapper_global_pooling_collapses_time(self):
        # Spatial dims must be divisible by the patch size; a 16x16 input with
        # patch_size=8 yields a 2x2 patch grid.
        b, h, w, t = 1, 16, 16, 2
        timestamps = repeat(
            np.array([[1, m + 1, 2024] for m in range(t)]), "t d -> b t d", b=b
        )
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            timestamps=timestamps,
        )

        model = PretrainedOlmoEarthWrapper(load_weights=False, patch_size=8)
        output = model(x, eval_pooling=PoolingMethods.GLOBAL)

        self.assertEqual(output.shape[:4], (b, h // model.patch_size, w // model.patch_size, 1))

    def test_wrapper_time_pooling_preserves_temporal_dimension(self):
        b, h, w, t = 1, 16, 16, 3
        timestamps = repeat(
            np.array([[1, m + 1, 2024] for m in range(t)]), "t d -> b t d", b=b
        )
        x = Predictors(
            s2=np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32),
            timestamps=timestamps,
        )

        model = PretrainedOlmoEarthWrapper(load_weights=False, patch_size=8)
        output = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(output.shape[:4], (b, h // model.patch_size, w // model.patch_size, t))


if __name__ == "__main__":
    unittest.main()
