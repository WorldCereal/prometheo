import importlib.util
import os
import unittest

import numpy as np
from einops import repeat

from prometheo.models import OlmoEarth
from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import S1_BANDS, S2_BANDS, Predictors


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
            timestamps=repeat(
                np.array([[1, month + 1, 2024] for month in range(t)], dtype="int64"),
                "t d -> b t d",
                b=b,
            ),
        )

    def test_real_olmoearth_forward_without_weights(self):
        x = self._predictors()
        model = OlmoEarth(load_weights=False)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)
        time_embeddings = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(global_embeddings.shape, (1, 2, 2, 1, 128))
        self.assertEqual(time_embeddings.shape, (1, 2, 2, 2, 128))

    def test_real_olmoearth_preserves_spatial_patch_grid(self):
        x = self._predictors(h=32, w=32)
        model = OlmoEarth(load_weights=False)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)
        time_embeddings = model(x, eval_pooling=PoolingMethods.TIME)

        self.assertEqual(global_embeddings.shape, (1, 4, 4, 1, 128))
        self.assertEqual(time_embeddings.shape, (1, 4, 4, 2, 128))

    @unittest.skipUnless(
        os.environ.get("PROMETHEO_TEST_OLMOEARTH_WEIGHTS") == "1",
        "set PROMETHEO_TEST_OLMOEARTH_WEIGHTS=1 to download and test real weights",
    )
    def test_real_olmoearth_forward_with_weights(self):
        x = self._predictors()
        model = OlmoEarth(load_weights=True)

        global_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)

        self.assertEqual(global_embeddings.shape, (1, 2, 2, 1, 128))


if __name__ == "__main__":
    unittest.main()
