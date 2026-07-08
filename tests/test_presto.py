import unittest

import numpy as np
import torch
from einops import repeat

from prometheo.models import Presto
from prometheo.models.pooling import PoolingMethods
from prometheo.models.presto.single_file_presto import BANDS, BANDS_GROUPS_IDX
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


class TestLatlonDropout(unittest.TestCase):
    @staticmethod
    def _make_predictors(latlon, s1, timestamps):
        return Predictors(s1=s1, latlon=latlon, timestamps=timestamps)

    def _fixed_inputs(self):
        b, t, h, w = 4, 4, 1, 1
        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        rng = np.random.RandomState(0)
        s1 = rng.rand(b, h, w, t, len(S1_BANDS))
        timestamps = repeat(timestamps_per_instance, "t d -> b t d", b=b)
        latlon_a = np.zeros((b, h, w, 2))
        latlon_b = np.full((b, h, w, 2), 45.0)
        return s1, timestamps, latlon_a, latlon_b

    def test_full_dropout_masks_latlon_during_training(self):
        # With latlon_dropout=1.0 in train mode the latlon token is always
        # masked, so the embeddings must be invariant to the latlon values.
        s1, timestamps, latlon_a, latlon_b = self._fixed_inputs()
        model = Presto(latlon_dropout=1.0)
        model.train()
        out_a = model(
            self._make_predictors(latlon_a, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        out_b = model(
            self._make_predictors(latlon_b, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        self.assertTrue(torch.allclose(out_a, out_b))

    def test_no_dropout_keeps_latlon(self):
        # Without dropout the latlon token is kept, so different latlons must
        # produce different embeddings.
        s1, timestamps, latlon_a, latlon_b = self._fixed_inputs()
        model = Presto(latlon_dropout=0.0)
        model.train()
        out_a = model(
            self._make_predictors(latlon_a, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        out_b = model(
            self._make_predictors(latlon_b, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_dropout_disabled_in_eval_mode(self):
        # Dropout only applies while training; in eval mode the latlon token is
        # kept even with latlon_dropout=1.0.
        s1, timestamps, latlon_a, latlon_b = self._fixed_inputs()
        model = Presto(latlon_dropout=1.0)
        model.eval()
        out_a = model(
            self._make_predictors(latlon_a, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        out_b = model(
            self._make_predictors(latlon_b, s1, timestamps),
            eval_pooling=PoolingMethods.GLOBAL,
        )
        self.assertFalse(torch.allclose(out_a, out_b))

    def test_invalid_dropout_raises(self):
        with self.assertRaises(ValueError):
            Presto(latlon_dropout=1.5)


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


class TestFullyMaskedTimestepPooling(unittest.TestCase):
    """Regression tests for the NaN ``val_loss`` seen in finetuning.

    Causal chain (why it started after the last two prometheo upgrades):

    * The NDVI mask-propagation fix (commit e57125c) makes the NDVI token
      correctly *masked* at cloud-masked S2 timesteps. Previously it survived
      as a phantom "valid" NDVI = 0 token.
    * ``dataset_to_model`` always emits ``dynamic_world`` as the missing class
      (so it is always masked), and any modality that is absent (S1/meteo/DEM)
      is masked too.
    * Therefore, when S2 is the only modality present, a fully cloud-masked S2
      timestep now has *zero* valid tokens. In ``eval_pooling="time"`` the
      masked mean divides the (zero) token sum by the (zero) valid-token count
      -> 0 / 0 -> NaN embeddings -> NaN ``val_loss``.
    * The encoder now clamps that denominator to >= 1, so a fully-masked
      timestep pools to a finite (zero) embedding instead of NaN.
    """

    @staticmethod
    def _s2_only_predictors(mask_timesteps):
        # S2 is the only modality -> S1 / meteo / DEM (SRTM) and dynamic_world
        # are all masked, so S2 groups + NDVI are the only tokens that can be
        # valid. This is what makes a cloud-masked timestep *fully* masked.
        b, t, h, w = 2, 4, 1, 1
        rng = np.random.RandomState(0)
        s2 = rng.uniform(500, 5000, size=(b, h, w, t, len(S2_BANDS))).astype(
            np.float32
        )
        for ts in mask_timesteps:
            s2[:, :, :, ts, :] = NODATAVALUE  # fully mask this timestep
        timestamps_per_instance = np.array([[2020, m + 1, 1] for m in range(t)])
        x = Predictors(
            s2=s2,
            latlon=np.zeros((b, h, w, 2), dtype=np.float32),
            timestamps=repeat(timestamps_per_instance, "t d -> b t d", b=b),
        )
        return x, (b, t, h, w)

    @staticmethod
    def _valid_group_counts(mask):
        # Mirror the encoder: a channel-group token is valid at a timestep iff
        # *all* of its bands are valid (the encoder takes ``max`` over the band
        # masks, i.e. the group is masked if any of its bands is masked).
        # Returns the number of valid channel-group tokens per [sample, timestep].
        n, t, _ = mask.shape
        counts = np.zeros((n, t), dtype=int)
        for _, idxs in BANDS_GROUPS_IDX.items():
            group_masked = mask[:, :, idxs].any(axis=-1)
            counts += (~group_masked).astype(int)
        return counts

    def test_masked_s2_timestep_is_fully_masked_due_to_ndvi(self):
        # Confirms the *cause*: with only S2 present, the NDVI mask fix removes
        # the last surviving token at a cloud-masked timestep, driving the
        # valid-token count to 0 (the 0/0 precondition for the pooling NaN).
        masked_ts, clear_ts = [1, 2], [0, 3]
        x, _ = self._s2_only_predictors(masked_ts)
        _, mask, *_ = dataset_to_model(x)
        mask = np.asarray(mask).astype(bool)  # [n, t, total_bands]
        ndvi_idx = BANDS.index("NDVI")

        # 1) NDVI-update behaviour: NDVI masked exactly at cloud-masked S2 steps.
        for ts in masked_ts:
            self.assertTrue(mask[:, ts, ndvi_idx].all())
        for ts in clear_ts:
            self.assertFalse(mask[:, ts, ndvi_idx].any())

        # 2) Those timesteps have zero valid tokens -> 0/0 in the masked mean.
        counts = self._valid_group_counts(mask)
        for ts in masked_ts:
            self.assertTrue((counts[:, ts] == 0).all())
        for ts in clear_ts:
            self.assertTrue((counts[:, ts] > 0).all())

        # 3) Causality: NDVI is the *only* token whose masking drops the count
        #    to 0. Simulating the pre-fix "phantom valid" NDVI token restores a
        #    single valid token (denominator 1 -> no NaN), which pins the NaN on
        #    the NDVI mask introduced by the mask-propagation fix.
        premask = mask.copy()
        premask[:, masked_ts, ndvi_idx] = False
        pre_counts = self._valid_group_counts(premask)
        for ts in masked_ts:
            self.assertTrue((pre_counts[:, ts] == 1).all())

    def test_time_pooling_is_finite_with_fully_masked_timestep(self):
        # The actual regression: before the denominator clamp, the fully-masked
        # timesteps pooled to 0/0 = NaN and propagated into val_loss. With the
        # clamp they pool to a finite (zero) embedding.
        masked_ts = [1, 2]
        x, (b, t, h, w) = self._s2_only_predictors(masked_ts)
        model = Presto()
        model.eval()
        out = model(x, eval_pooling=PoolingMethods.TIME)
        self.assertEqual(out.shape, (b, h, w, t, model.encoder.embedding_size))
        self.assertTrue(torch.isfinite(out).all())
