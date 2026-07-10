import importlib.util
import os
import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models import OlmoEarth
from prometheo.models.olmoearth.wrapper import (
    S1_OLMOEARTH_TO_PROMETHEO,
    S2_OLMOEARTH_TO_PROMETHEO,
    dataset_to_olmoearth_sample,
)
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import DEM_BANDS, NODATAVALUE, S1_BANDS, S2_BANDS, Predictors


@unittest.skipIf(
    importlib.util.find_spec("olmoearth_pretrain_minimal") is None,
    "olmoearth-pretrain-minimal optional dependency is not installed",
)
class TestRealOlmoEarthIntegration(unittest.TestCase):
    def _predictors(self, h=16, w=16):
        b, t = 1, 2
        return Predictors(
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)).astype("float32"),
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)).astype("float32"),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)).astype("float32"),
            timestamps=repeat(
                np.array([[1, month + 1, 2024] for month in range(t)], dtype="int64"),
                "t d -> b t d",
                b=b,
            ),
        )

    def test_real_olmoearth_forward_without_weights(self):
        x = self._predictors()
        model = OlmoEarth(load_weights=False, patch_size=8)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)
        time_embeddings = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(global_embeddings.shape, (1, 2, 2, 1, 128))
        self.assertEqual(time_embeddings.shape, (1, 2, 2, 2, 128))

    def test_real_olmoearth_forward_sentinel1_only(self):
        b, h, w, t = 1, 16, 16, 2
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)).astype("float32"),
            timestamps=repeat(
                np.array([[1, month + 1, 2024] for month in range(t)], dtype="int64"),
                "t d -> b t d",
                b=b,
            ),
        )
        model = OlmoEarth(load_weights=False, patch_size=8)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)

        self.assertEqual(global_embeddings.shape, (1, 2, 2, 1, 128))

    def test_real_olmoearth_preserves_spatial_patch_grid(self):
        x = self._predictors(h=32, w=32)
        model = OlmoEarth(load_weights=False, patch_size=8)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)
        time_embeddings = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(global_embeddings.shape, (1, 4, 4, 1, 128))
        self.assertEqual(time_embeddings.shape, (1, 4, 4, 2, 128))

    def test_wrapper_forward_matches_raw_library_pipeline(self):
        """End-to-end there-and-back: the wrapper's public output must equal the
        embeddings obtained by driving the *same* model with a sample built
        independently through the raw olmoearth_pretrain_minimal API.

        This is a characterization test — the reference reconstructs the sample
        from the raw API, so it pins forward's encoder-invocation params
        (patch_size / input_res / fast_pass) and the extraction + pooling at the
        value level, which the shape-only tests above do not.
        """
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
            Modality,
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
            MaskValue,
        )

        b, h, w, t = 1, 16, 16, 2
        rng = np.random.default_rng(0)
        s2 = rng.random((b, h, w, t, len(S2_BANDS))).astype("float32")
        s1 = rng.random((b, h, w, t, len(S1_BANDS))).astype("float32")
        timestamps = repeat(
            np.array([[1, month + 1, 2024] for month in range(t)], dtype="int64"),
            "t d -> b t d",
            b=b,
        )
        x = Predictors(s2=s2, s1=s1, timestamps=timestamps)

        # eval() disables band dropout so both paths are deterministic; both share
        # the same underlying (randomly initialised) weights via wrapper.model.
        wrapper = OlmoEarth(load_weights=False, patch_size=8)
        wrapper.eval()

        # --- wrapper path (the code under test) ---
        with torch.no_grad():
            wrapper_out = wrapper(x, eval_pooling=PoolingMethods.TIME)

        # --- reference path: build the canonical sample with the raw API ---
        model = wrapper.model
        tokenization_config = model.encoder.tokenization_config
        normalizer = Normalizer(std_multiplier=2.0)

        s2_ref = s2[..., [S2_BANDS.index(p) for _, p in S2_OLMOEARTH_TO_PROMETHEO]]
        s1_ref = s1[..., [S1_BANDS.index(p) for _, p in S1_OLMOEARTH_TO_PROMETHEO]]
        s2_ref = normalizer.normalize(Modality.SENTINEL2_L2A, s2_ref)
        s1_ref = normalizer.normalize(Modality.SENTINEL1, s1_ref)

        timestamps_ref = timestamps.copy()
        timestamps_ref[:, :, 1] -= 1  # OlmoEarth expects zero-indexed months

        online = MaskValue.ONLINE_ENCODER.value
        s2_mask = np.full(
            (b, h, w, t, tokenization_config.get_num_bandsets("sentinel2_l2a")),
            online,
            dtype=np.int64,
        )
        s1_mask = np.full(
            (b, h, w, t, tokenization_config.get_num_bandsets("sentinel1")),
            online,
            dtype=np.int64,
        )
        reference_sample = MaskedOlmoEarthSample(
            timestamps=torch.from_numpy(timestamps_ref).long(),
            sentinel2_l2a=torch.from_numpy(s2_ref).float(),
            sentinel2_l2a_mask=torch.from_numpy(s2_mask).long(),
            sentinel1=torch.from_numpy(s1_ref).float(),
            sentinel1_mask=torch.from_numpy(s1_mask).long(),
        )

        with torch.no_grad():
            reference_encoder_output = model.encoder(
                reference_sample,
                patch_size=wrapper.patch_size,
                input_res=wrapper.input_res,
                fast_pass=wrapper._resolve_fast_pass(reference_sample),
            )
        reference_embeddings, reference_validity = wrapper._extract_embeddings(
            reference_encoder_output
        )
        reference_out = wrapper._apply_pooling(
            reference_embeddings, reference_validity, PoolingMethods.TIME
        )

        self.assertEqual(wrapper_out.shape, reference_out.shape)
        torch.testing.assert_close(wrapper_out, reference_out)

    def test_missing_timestep_excluded_from_global_pooling(self):
        """A fully-nodata timestep must not affect the GLOBAL-pooled embedding.

        With only timestep 0 valid, GLOBAL pooling (masked mean over time) should
        equal timestep 0 of the TIME-pooled output. Under the old naive mean it
        would instead be the average of the valid and the missing timestep.
        """
        b, h, w = 1, 16, 16
        rng = np.random.default_rng(2)
        valid = rng.random((b, h, w, 1, len(S2_BANDS))).astype("float32")
        s2 = np.concatenate([valid, np.full_like(valid, NODATAVALUE)], axis=3)
        timestamps = repeat(
            np.array([[1, 6, 2024], [1, 7, 2024]], dtype="int64"), "t d -> b t d", b=b
        )
        x = Predictors(s2=s2, timestamps=timestamps)

        wrapper = OlmoEarth(load_weights=False, patch_size=8)
        wrapper.eval()
        with torch.no_grad():
            global_out = wrapper(x, eval_pooling=PoolingMethods.GLOBAL)
            time_out = wrapper(x, eval_pooling=PoolingMethods.TIME)

        # GLOBAL pooled over the single valid timestep == that timestep's embedding.
        torch.testing.assert_close(global_out[:, :, :, 0], time_out[:, :, :, 0])

    def test_fast_pass_auto_selected_from_missing_tokens(self):
        # fast_pass=True only when every token is present, else False.
        wrapper = OlmoEarth(load_weights=False, patch_size=8)
        tokenization_config = wrapper.model.encoder.tokenization_config

        b, h, w, t = 1, 16, 16, 2
        timestamps = repeat(
            np.array([[1, 6, 2024], [1, 7, 2024]], dtype="int64"), "t d -> b t d", b=b
        )
        clean = Predictors(
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)).astype("float32"),
            timestamps=timestamps,
        )
        dirty_s2 = np.random.rand(b, h, w, t, len(S2_BANDS)).astype("float32")
        dirty_s2[:, :, :, 1, :] = NODATAVALUE
        dirty = Predictors(s2=dirty_s2, timestamps=timestamps)

        clean_sample = dataset_to_olmoearth_sample(
            clean, tokenization_config=tokenization_config
        )
        dirty_sample = dataset_to_olmoearth_sample(
            dirty, tokenization_config=tokenization_config
        )

        self.assertTrue(wrapper._resolve_fast_pass(clean_sample))
        self.assertFalse(wrapper._resolve_fast_pass(dirty_sample))

    @unittest.skipUnless(
        os.environ.get("PROMETHEO_TEST_OLMOEARTH_WEIGHTS") == "1",
        "set PROMETHEO_TEST_OLMOEARTH_WEIGHTS=1 to download and test real weights",
    )
    def test_real_olmoearth_forward_with_weights(self):
        x = self._predictors()
        model = OlmoEarth(load_weights=True, patch_size=8)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)

        self.assertEqual(global_embeddings.shape, (1, 2, 2, 1, 128))


if __name__ == "__main__":
    unittest.main()
