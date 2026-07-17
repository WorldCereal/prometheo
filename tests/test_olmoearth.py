import importlib.util
import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models.olmoearth.wrapper import (
    MissingBandStrategy,
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

    def test_replace_b1_b9_b8a_fills_b8a_and_unmasks(self):
        # B8A is consistently missing in the data; replace_b1_b9_b8a should fill
        # it with B8 (normalized with B8's stats) and stop marking its band set
        # (band set 1) MISSING.
        b, h, w, t = 1, 1, 1, 1
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        # Give B8 a distinct value so we can tell the substitution happened, and
        # knock out B8A entirely to simulate the data issue.
        s2[:, :, :, :, S2_BANDS.index("B8")] = 1234.0
        s2[:, :, :, :, S2_BANDS.index("B8A")] = NODATAVALUE
        timestamps = np.array([[[1, 1, 2024]]])
        x = Predictors(s2=s2, timestamps=timestamps)

        oe_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
        b8_idx = oe_bands.index("B08")
        b8a_idx = oe_bands.index("B8A")

        # Without the workaround: B8A is missing, so band set 1 is MISSING (3).
        baseline = dataset_to_olmoearth_sample(x, model_device=torch.device("cpu"))
        self.assertEqual(baseline.sentinel2_l2a_mask[0, 0, 0, 0, 1].item(), 3)

        # With the workaround: band set 1 is present (0) and the B8A slot equals
        # the B8-normalized B08 slot rather than a normalized NODATAVALUE.
        fixed = dataset_to_olmoearth_sample(
            x,
            model_device=torch.device("cpu"),
            replace_b1_b9_b8a=MissingBandStrategy.INTERPOLATE,
        )
        self.assertEqual(fixed.sentinel2_l2a_mask[0, 0, 0, 0, 1].item(), 0)
        self.assertTrue(
            torch.allclose(
                fixed.sentinel2_l2a[..., b8a_idx], fixed.sentinel2_l2a[..., b8_idx]
            )
        )

    def test_zero_strategy_fills_b8a_with_zeros_and_unmasks(self):
        # The 'zero' strategy also unmasks B8A's band set, but the encoder sees
        # zeros in the B8A slot rather than interpolated B08 values.
        b, h, w, t = 1, 1, 1, 1
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        s2[:, :, :, :, S2_BANDS.index("B8A")] = NODATAVALUE
        x = Predictors(s2=s2, timestamps=np.array([[[1, 1, 2024]]]))

        oe_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
        sample = dataset_to_olmoearth_sample(
            x, model_device=torch.device("cpu"), replace_b1_b9_b8a="zero"
        )

        self.assertEqual(sample.sentinel2_l2a_mask[0, 0, 0, 0, 1].item(), 0)
        # Zero in *normalized* space — the value the encoder actually sees.
        self.assertTrue(
            torch.equal(
                sample.sentinel2_l2a[..., oe_bands.index("B8A")],
                torch.zeros_like(sample.sentinel2_l2a[..., 0]),
            )
        )

    def test_bool_true_is_deprecated_alias_for_interpolate(self):
        b, h, w, t = 1, 1, 1, 1
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        s2[:, :, :, :, S2_BANDS.index("B8A")] = NODATAVALUE
        x = Predictors(s2=s2, timestamps=np.array([[[1, 1, 2024]]]))

        with self.assertWarns(DeprecationWarning):
            legacy = dataset_to_olmoearth_sample(
                x, model_device=torch.device("cpu"), replace_b1_b9_b8a=True
            )
        enum = dataset_to_olmoearth_sample(
            x,
            model_device=torch.device("cpu"),
            replace_b1_b9_b8a=MissingBandStrategy.INTERPOLATE,
        )
        self.assertTrue(torch.equal(legacy.sentinel2_l2a, enum.sentinel2_l2a))
        self.assertTrue(
            torch.equal(legacy.sentinel2_l2a_mask, enum.sentinel2_l2a_mask)
        )

    def test_replace_b1_b9_b8a_drops_b1_b9_under_v1_tokenization(self):
        # OlmoEarth v1 gives B01/B09 their own band set (band set 2), so when
        # they are missing that band set alone is masked MISSING and dropped —
        # replace_b1_b9_b8a must not fabricate values there.
        b, h, w, t = 1, 1, 1, 1
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        s2[:, :, :, :, S2_BANDS.index("B1")] = NODATAVALUE
        s2[:, :, :, :, S2_BANDS.index("B9")] = NODATAVALUE
        x = Predictors(s2=s2, timestamps=np.array([[[1, 1, 2024]]]))

        # The default tokenization is the v1 grouping; neither strategy may
        # fabricate values for a band set that can simply be dropped.
        for strategy in MissingBandStrategy:
            sample = dataset_to_olmoearth_sample(
                x, model_device=torch.device("cpu"), replace_b1_b9_b8a=strategy
            )

            # Band sets 0 and 1 present, the atmospheric band set MISSING (3).
            self.assertEqual(
                sample.sentinel2_l2a_mask[0, 0, 0, 0].tolist(), [0, 0, 3]
            )

    def test_replace_b1_b9_b8a_fills_b1_b9_under_v1_2_tokenization(self):
        from olmoearth_pretrain_minimal import ModelID, load_model_from_id

        # v1.2 tokenizes all twelve S2 bands as a single band set, so missing
        # B01/B09 would mark the whole Sentinel-2 modality MISSING. With the
        # workaround, B01 is filled from B02 and B09 from B08 (each normalized
        # with the source band's stats) and the band set stays present.
        model = load_model_from_id(ModelID.OLMOEARTH_V1_2_NANO, load_weights=False)
        tokenization_config = model.encoder.tokenization_config

        b, h, w, t = 1, 1, 1, 1
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        s2[:, :, :, :, S2_BANDS.index("B2")] = 567.0
        s2[:, :, :, :, S2_BANDS.index("B8")] = 1234.0
        s2[:, :, :, :, S2_BANDS.index("B1")] = NODATAVALUE
        s2[:, :, :, :, S2_BANDS.index("B9")] = NODATAVALUE
        x = Predictors(s2=s2, timestamps=np.array([[[1, 1, 2024]]]))

        baseline = dataset_to_olmoearth_sample(
            x, model_device=torch.device("cpu"), tokenization_config=tokenization_config
        )
        self.assertEqual(baseline.sentinel2_l2a_mask.shape[-1], 1)
        self.assertEqual(baseline.sentinel2_l2a_mask[0, 0, 0, 0, 0].item(), 3)

        fixed = dataset_to_olmoearth_sample(
            x,
            model_device=torch.device("cpu"),
            tokenization_config=tokenization_config,
            replace_b1_b9_b8a="interpolate",
        )
        self.assertEqual(fixed.sentinel2_l2a_mask[0, 0, 0, 0, 0].item(), 0)
        oe_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
        self.assertTrue(
            torch.allclose(
                fixed.sentinel2_l2a[..., oe_bands.index("B01")],
                fixed.sentinel2_l2a[..., oe_bands.index("B02")],
            )
        )
        self.assertTrue(
            torch.allclose(
                fixed.sentinel2_l2a[..., oe_bands.index("B09")],
                fixed.sentinel2_l2a[..., oe_bands.index("B08")],
            )
        )

        # The 'zero' strategy also keeps the band set present, but the missing
        # bands (B8A, B01, B09) become zeros in normalized space.
        zeroed = dataset_to_olmoearth_sample(
            x,
            model_device=torch.device("cpu"),
            tokenization_config=tokenization_config,
            replace_b1_b9_b8a=MissingBandStrategy.ZERO,
        )
        self.assertEqual(zeroed.sentinel2_l2a_mask[0, 0, 0, 0, 0].item(), 0)
        for band in ["B8A", "B01", "B09"]:
            self.assertTrue(
                torch.equal(
                    zeroed.sentinel2_l2a[..., oe_bands.index(band)],
                    torch.zeros_like(zeroed.sentinel2_l2a[..., 0]),
                )
            )

    def test_wrapper_replace_b1_b9_b8a_forward_pass(self):
        # End to end: the default wrapper model is v1.2, whose single S2 band
        # set would go entirely MISSING without the workaround.
        b, h, w, t = 1, 16, 16, 2
        timestamps = repeat(
            np.array([[1, m + 1, 2024] for m in range(t)]), "t d -> b t d", b=b
        )
        s2 = np.ones((b, h, w, t, len(S2_BANDS)), dtype=np.float32)
        s2[..., S2_BANDS.index("B1")] = NODATAVALUE
        s2[..., S2_BANDS.index("B9")] = NODATAVALUE
        x = Predictors(s2=s2, timestamps=timestamps)

        for strategy in MissingBandStrategy:
            model = PretrainedOlmoEarthWrapper(
                load_weights=False, patch_size=8, replace_b1_b9_b8a=strategy
            )
            output = model(x, eval_pooling=PoolingMethods.GLOBAL)

            self.assertEqual(
                output.shape[:4], (b, h // model.patch_size, w // model.patch_size, 1)
            )
            self.assertTrue(torch.isfinite(output).all())

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
