import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models import Presto
from prometheo.predictors import DEM_BANDS, METEO_BANDS, S1_BANDS, S2_BANDS, Predictors


class TestPresto(unittest.TestCase):
    def test_forward_from_predictor_with_all(self):
        b, t, h, w = 8, 4, 1, 1

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, 2),
            label=np.ones((b, 1, 1, 1, 1)),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(
            output_embeddings.shape, (b, h, w, 1, model.encoder.embedding_size)
        )

    def test_forward_from_predictor_with_s1_only(self):
        b, t, h, w = 8, 4, 1, 1

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            latlon=np.random.rand(b, 2),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto(eval_pooling="time")
        output_embeddings = model(x)
        self.assertEqual(
            output_embeddings.shape, (b, h, w, t, model.encoder.embedding_size)
        )

        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(
            output_embeddings.shape, (b, h, w, 1, model.encoder.embedding_size)
        )

    def test_add_masked_tokens_with_zeros(self):
        model = Presto()
        # [b=1, t=3, d=2]
        input_x = repeat(
            torch.tensor([[1, 2, 3]]).float(),
            "b t -> b t d",
            d=model.encoder.embedding_size,
        )
        input_mask = torch.tensor([[0, 1, 0]])
        x, orig_indices, upd_mask = model.encoder.mask_tokens(input_x, input_mask)
        filled_x = model.encoder.add_masked_tokens_with_zeros(x, orig_indices, upd_mask)
        self.assertTrue(
            torch.equal(
                filled_x,
                repeat(
                    torch.tensor([[1, 0, 3]]).float(),
                    "b t -> b t d",
                    d=model.encoder.embedding_size,
                ),
            )
        )

    def test_forward_from_predictor_per_time(self):
        b, t, h, w = 8, 4, 1, 1

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, 2),
            label=np.ones((b, h, w, t, 1)),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(output_embeddings.shape, (b, h, w, t, model.encoder.embedding_size))
