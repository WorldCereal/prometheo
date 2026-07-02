from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import NODATAVALUE, S1_BANDS, S2_BANDS, Predictors, to_torchtensor
from prometheo.utils import device

DEFAULT_MODEL_ID = "OLMOEARTH_V1_2_NANO"
S2_OLMOEARTH_TO_PROMETHEO = [
    ("B02", "B2"),
    ("B03", "B3"),
    ("B04", "B4"),
    ("B08", "B8"),
    ("B05", "B5"),
    ("B06", "B6"),
    ("B07", "B7"),
    ("B8A", "B8A"),
    ("B11", "B11"),
    ("B12", "B12"),
    ("B01", "B1"),
    ("B09", "B9"),
]
S1_OLMOEARTH_TO_PROMETHEO = [("vv", "VV"), ("vh", "VH")]


def _missing_dependency_error() -> ImportError:
    return ImportError(
        "OlmoEarth support requires the optional dependency "
        "`olmoearth-pretrain-minimal`. Install it with "
        "`pip install prometheo[olmoearth]` or install the package directly."
    )


def _get_model_id(model_id: Union[str, object]):
    try:
        from olmoearth_pretrain_minimal import ModelID
    except ImportError as exc:
        raise _missing_dependency_error() from exc

    if isinstance(model_id, str):
        return getattr(ModelID, model_id)
    return model_id


def _make_modality_mask(values: np.ndarray) -> np.ndarray:
    return np.where(np.any(values == NODATAVALUE, axis=-1), 3, 0).astype(np.int64)


def _normalizer_and_sample_types():
    try:
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
            Modality,
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )
    except ImportError as exc:
        raise _missing_dependency_error() from exc
    return Normalizer, Modality, MaskedOlmoEarthSample


def dataset_to_olmoearth_sample(x: Predictors, model_device: torch.device = device):
    """Convert PromethEO predictors into an OlmoEarth masked sample.

    The first implementation supports the modalities PromethEO already models
    cleanly for OlmoEarth v1.2: Sentinel-2 L2A, Sentinel-1, and timestamps.
    Landsat is intentionally omitted because `Predictors` has no Landsat field.
    """

    if x.timestamps is None:
        raise ValueError("OlmoEarth requires input timestamps")
    if x.s2 is None:
        raise ValueError("OlmoEarth requires sentinel2_l2a input via Predictors.s2")

    Normalizer, Modality, MaskedOlmoEarthSample = _normalizer_and_sample_types()
    normalizer = Normalizer(std_multiplier=2.0)

    sample_kwargs = {
        "timestamps": to_torchtensor(np.array(x.timestamps, copy=True), model_device).long()
    }
    # PromethEO timestamps are [day, month, year] with 1-indexed months; upstream
    # OlmoEarth examples expect the month column to be zero-indexed.
    sample_kwargs["timestamps"][:, :, 1] -= 1

    if x.s2 is not None:
        s2_indices = [S2_BANDS.index(prometheo) for _, prometheo in S2_OLMOEARTH_TO_PROMETHEO]
        s2 = np.asarray(x.s2)[..., s2_indices].astype(np.float32)
        s2_mask = _make_modality_mask(s2)
        s2 = normalizer.normalize(Modality.SENTINEL2_L2A, s2)
        sample_kwargs["sentinel2_l2a"] = to_torchtensor(s2, model_device).float()
        sample_kwargs["sentinel2_l2a_mask"] = to_torchtensor(s2_mask, model_device).long()

    if x.s1 is not None:
        s1_indices = [S1_BANDS.index(prometheo) for _, prometheo in S1_OLMOEARTH_TO_PROMETHEO]
        s1 = np.asarray(x.s1)[..., s1_indices].astype(np.float32)
        s1_mask = _make_modality_mask(s1)
        s1 = normalizer.normalize(Modality.SENTINEL1, s1)
        sample_kwargs["sentinel1"] = to_torchtensor(s1, model_device).float()
        sample_kwargs["sentinel1_mask"] = to_torchtensor(s1_mask, model_device).long()

    return MaskedOlmoEarthSample(**sample_kwargs)


class _FinetuningHead(nn.Module):
    def __init__(self, hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PretrainedOlmoEarthWrapper(nn.Module):
    def __init__(
        self,
        model_id: Union[str, object] = DEFAULT_MODEL_ID,
        load_weights: bool = True,
        num_outputs: Optional[int] = None,
        patch_size: int = 8,
        input_res: int = 10,
        fast_pass: bool = True,
    ):
        super().__init__()
        try:
            from olmoearth_pretrain_minimal import load_model_from_id
        except ImportError as exc:
            raise _missing_dependency_error() from exc

        self.model = load_model_from_id(_get_model_id(model_id), load_weights=load_weights)
        self.patch_size = patch_size
        self.input_res = input_res
        self.fast_pass = fast_pass
        self.head: Optional[nn.Module] = None
        if num_outputs is not None:
            hidden_size = getattr(getattr(self.model, "encoder", self.model), "embedding_size", None)
            if hidden_size is None:
                hidden_size = getattr(getattr(self.model, "encoder", self.model), "embed_dim", None)
            if hidden_size is None:
                raise ValueError("num_outputs requires the OlmoEarth encoder to expose embedding_size or embed_dim")
            self.head = _FinetuningHead(int(hidden_size), num_outputs)

    def forward(
        self,
        x: Predictors,
        eval_pooling: Union[PoolingMethods, None] = PoolingMethods.GLOBAL,
    ):
        if x.label is not None:
            if x.label.shape[3] == x.timestamps.shape[1]:
                eval_pooling = PoolingMethods.TIME
            else:
                eval_pooling = PoolingMethods.GLOBAL

        model_device = next(self.parameters(), torch.empty(0, device=device)).device
        sample = dataset_to_olmoearth_sample(x, model_device=model_device)
        encoder_output = self.model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=self.fast_pass,
        )
        embeddings = self._extract_embeddings(encoder_output)
        embeddings = self._apply_pooling(embeddings, eval_pooling)
        if self.head is not None:
            if eval_pooling is None:
                raise ValueError("Can't use the head without a pooling method")
            return self.head(embeddings)
        return embeddings

    def _extract_embeddings(self, encoder_output):
        if isinstance(encoder_output, dict) and "tokens_and_masks" in encoder_output:
            tokens_and_masks = encoder_output["tokens_and_masks"]
            modality_embeddings = []
            for modality_name in ["sentinel2_l2a", "sentinel1"]:
                modality_tokens = getattr(tokens_and_masks, modality_name, None)
                if modality_tokens is None:
                    continue
                if modality_tokens.ndim == 6:
                    # [B, patch_h, patch_w, T, bandsets, D] -> [B, patch_h, patch_w, T, D]
                    modality_tokens = modality_tokens.mean(dim=4)
                modality_embeddings.append(modality_tokens)
            if modality_embeddings:
                return torch.stack(modality_embeddings, dim=0).mean(dim=0)
            raise ValueError("OlmoEarth encoder output did not include supported S1/S2 tokens")
        if isinstance(encoder_output, tuple):
            return encoder_output[0]
        return encoder_output

    def _apply_pooling(self, embeddings: torch.Tensor, eval_pooling):
        if eval_pooling is None:
            return embeddings
        if embeddings.ndim == 6:
            # OlmoEarth token shape: [B, patch_h, patch_w, T, bandsets, D].
            embeddings = embeddings.mean(dim=4)
        if embeddings.ndim == 5:
            # PromethEO convention: preserve the spatial token grid and pool time only for GLOBAL.
            if eval_pooling == PoolingMethods.GLOBAL:
                return embeddings.mean(dim=3, keepdim=True)
            if eval_pooling == PoolingMethods.TIME:
                return embeddings
        if embeddings.ndim == 3:
            # Token embeddings [B, tokens, D]. Use a conservative global mean.
            pooled = embeddings.mean(dim=1)
            if eval_pooling == PoolingMethods.GLOBAL:
                return pooled[:, None, None, None, :]
        raise ValueError(
            f"Unsupported OlmoEarth encoder output shape {tuple(embeddings.shape)} "
            f"for pooling mode {eval_pooling}"
        )
