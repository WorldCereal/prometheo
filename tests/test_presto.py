import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models import Presto
from prometheo.models.pooling import PoolingMethods
from prometheo.models.presto.single_file_presto import BANDS
from prometheo.models.presto.wrapper import dataset_to_model
from prometheo.predictors import (
    DEM_BANDS,
    METEO_BANDS,
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
)


class TestPresto(unittest.TestCase):
    def test_forward_from_predictor_with_all(self):
        b, t, h, w = 8, 4, 1, 1

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, h, w, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, h, w, 2),
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
            latlon=np.random.rand(b, h, w, 2),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x, eval_pooling=PoolingMethods.TIME)
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
            meteo=np.random.rand(b, h, w, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, h, w, 2),
            label=np.ones((b, h, w, t, 1)),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x)
        self.assertEqual(
            output_embeddings.shape, (b, h, w, t, model.encoder.embedding_size)
        )

    def test_forward_from_predictor_nopooling(self):
        b, t, h, w = 8, 4, 1, 1

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, h, w, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, h, w, 2),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x, eval_pooling=None)
        self.assertEqual(
            output_embeddings[0].shape, (b, 34, model.encoder.embedding_size)
        )

    def test_forward_from_predictor_hw_global(self):
        b, t, h, w = 8, 4, 2, 2

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, h, w, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, h, w, 2),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x, eval_pooling=PoolingMethods.GLOBAL)
        self.assertEqual(
            # t = 1 since we do global pooling
            output_embeddings.shape,
            (b, h, w, 1, model.encoder.embedding_size),
        )

    def test_forward_from_predictor_hw_time(self):
        b, t, h, w = 8, 4, 2, 2

        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s1=np.random.rand(b, h, w, t, len(S1_BANDS)),
            s2=np.random.rand(b, h, w, t, len(S2_BANDS)),
            meteo=np.random.rand(b, h, w, t, len(METEO_BANDS)),
            dem=np.random.rand(b, h, w, len(DEM_BANDS)),
            latlon=np.random.rand(b, h, w, 2),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        model = Presto()
        output_embeddings = model(x, eval_pooling=PoolingMethods.TIME)
        self.assertEqual(
            output_embeddings.shape, (b, h, w, t, model.encoder.embedding_size)
        )


class TestNormalize(unittest.TestCase):
    def test_ndvi_mask_propagated_at_masked_s2_timesteps(self):
        # Regression: when S2 is masked at the input, the NDVI position
        # emitted by dataset_to_model must also be masked. A previous bug
        # recomputed the NDVI mask from already-normalized values, so masked
        # positions arrived at the encoder as a "valid" NDVI = 0 phantom token.
        b, t, h, w = 1, 4, 1, 1
        s2 = np.full((b, h, w, t, len(S2_BANDS)), 2000.0, dtype=np.float32)
        s2[..., S2_BANDS.index("B8")] = 4000.0
        s2[..., S2_BANDS.index("B4")] = 1000.0
        # Mask timesteps 1 and 2 entirely.
        s2[:, :, :, 1:3, :] = NODATAVALUE

        s1 = np.full((b, h, w, t, len(S1_BANDS)), -10.0, dtype=np.float32)
        meteo = np.full((b, h, w, t, len(METEO_BANDS)), 15.0, dtype=np.float32)
        dem = np.full((b, h, w, len(DEM_BANDS)), 100.0, dtype=np.float32)
        latlon = np.zeros((b, h, w, 2), dtype=np.float32)
        timestamps = np.stack(
            [np.array([1, 1 + ts, 2024], dtype=np.int32) for ts in range(t)]
        )[None, ...]

        x = Predictors(
            s1=s1,
            s2=s2,
            meteo=meteo,
            dem=dem,
            latlon=latlon,
            timestamps=timestamps,
        )
        _, mask, *_ = dataset_to_model(x)

        ndvi_idx = BANDS.index("NDVI")
        # Shape after rearrange: [B*H*W, T, total_bands]
        ndvi_mask_per_timestep = mask[:, :, ndvi_idx].astype(bool)
        # Timesteps 1 and 2 had S2 fully masked -> NDVI must be masked there.
        self.assertTrue(ndvi_mask_per_timestep[0, 1])
        self.assertTrue(ndvi_mask_per_timestep[0, 2])
        # Timesteps 0 and 3 had real S2 -> NDVI must be unmasked there.
        self.assertFalse(ndvi_mask_per_timestep[0, 0])
        self.assertFalse(ndvi_mask_per_timestep[0, 3])
