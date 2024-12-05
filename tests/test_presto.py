import unittest

import numpy as np
import torch
from einops import repeat
from prometheo.models import Presto
from prometheo.predictors import Predictors, S1_BANDS, S2_BANDS, DEM_BANDS, METEO_BANDS


class TestPresto(unittest.TestCase):
    def test_forward_from_predictor_with_all(self):
        b, t, h, w = 8, 4, 1, 1

        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, 2),
            label=np.ones((b, 1, 1)),
            month=6,
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(output_embeddings.shape, (b, model.presto.embedding_size))

    def test_forward_from_predictor_with_s1_only(self):
        b, t, h, w = 8, 4, 1, 1

        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            latlon=np.random.rand(b, 2),
            label=np.ones((b, 1, 1)),
            month=6,
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(output_embeddings.shape, (b, model.presto.embedding_size))

    def test_add_masked_tokens_with_zeros(self):
        model = Presto()
        # [b=1, t=3, d=2]
        input_x = repeat(
            torch.tensor([[1, 2, 3]]).float(),
            "b t -> b t d",
            d=model.presto.embedding_size,
        )
        input_mask = torch.tensor([[0, 1, 0]])
        x, orig_indices, upd_mask = model.presto.mask_tokens(input_x, input_mask)
        filled_x = model.presto.add_masked_tokens_with_zeros(x, orig_indices, upd_mask)
        self.assertTrue(
            torch.equal(
                filled_x,
                repeat(
                    torch.tensor([[1, 0, 3]]).float(),
                    "b t -> b t d",
                    d=model.presto.embedding_size,
                ),
            )
        )

    def test_forward_from_predictor_per_time(self):
        b, t, h, w = 8, 4, 1, 1

        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, 2),
            label=np.ones((b, t, 1)),
            month=6,
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(output_embeddings.shape, (b, t, model.presto.embedding_size))
        output_embeddings = model(x)
        self.assertEqual(output_embeddings.shape, (b, t, model.presto.embedding_size))
