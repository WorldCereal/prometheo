from __future__ import annotations

import warnings
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from prometheo.models.pooling import PoolingMethods
from prometheo.predictors import (
    NODATAVALUE,
    S1_BANDS,
    S2_BANDS,
    Predictors,
    to_torchtensor,
)
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


@lru_cache(maxsize=1)
def _olmoearth() -> SimpleNamespace:
    """Lazily import the optional ``olmoearth_pretrain_minimal`` dependency.

    Imports live here (rather than at module top level) so PromethEO can be
    used without OlmoEarth installed. The result is cached, so the import
    machinery only runs once and every call site shares a single, friendly
    error path.
    """
    try:
        from olmoearth_pretrain_minimal import (
            ModelID,
            Normalizer,
            load_model_from_id,
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.tokenization import (
            TokenizationConfig,
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import (
            Modality,
        )
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample, MaskValue
        )
    except ImportError as exc:
        raise _missing_dependency_error() from exc

    return SimpleNamespace(
        ModelID=ModelID,
        Normalizer=Normalizer,
        load_model_from_id=load_model_from_id,
        Modality=Modality,
        MaskedOlmoEarthSample=MaskedOlmoEarthSample,
        MaskValue=MaskValue,
        TokenizationConfig=TokenizationConfig,
    )


def _get_model_id(model_id: Union[str, object]):
    if isinstance(model_id, str):
        return getattr(_olmoearth().ModelID, model_id)
    return model_id


def _make_modality_mask(
    values: np.ndarray, modality_name: str, tokenization_config
) -> np.ndarray:
    """Build a per-band-set OlmoEarth mask for one modality.

    OlmoEarth tokenizes each modality into one or more *band sets* (as described
    by the model's ``TokenizationConfig``), and every band set becomes its own
    token with its own mask entry. The encoder therefore expects a mask whose
    trailing dimension indexes band sets — shape ``[..., num_band_sets]``.
    Collapsing every band into a single value (the previous behaviour) drops that
    axis, so a model that splits a modality across multiple band sets would be
    masked incorrectly.

    A band set is marked ``MISSING`` if any of its bands is ``NODATAVALUE`` at a
    given (spatial, temporal) location, and ``ONLINE_ENCODER`` otherwise.

    Args:
        values: Modality array ``[..., bands]`` with bands already ordered to
            match the modality's canonical OlmoEarth band order.
        modality_name: OlmoEarth modality name, e.g. ``"sentinel2_l2a"``.
        tokenization_config: The model's ``TokenizationConfig``, used to resolve
            which band indices belong to each band set.
    """
    maskvalue = _olmoearth().MaskValue
    bandset_indices = tokenization_config.get_bandset_indices(modality_name)
    per_bandset = [
        np.where(
            np.any(values[..., idxs] == NODATAVALUE, axis=-1),
            maskvalue.MISSING.value,
            maskvalue.ONLINE_ENCODER.value,
        )
        for idxs in bandset_indices
    ]
    return np.stack(per_bandset, axis=-1).astype(np.int64)


def dataset_to_olmoearth_sample(
    x: Predictors,
    model_device: torch.device = device,
    tokenization_config=None,
    replace_b8a_with_b8: bool = False,
):
    """Convert PromethEO predictors into an OlmoEarth masked sample.

    The first implementation supports the modalities PromethEO already models
    cleanly for OlmoEarth v1.2: Sentinel-2 L2A, Sentinel-1, and timestamps.
    Landsat is intentionally omitted because `Predictors` has no Landsat field.
    SRTM (``Predictors.dem``) is a decode-only modality for OlmoEarth (it is
    reconstructed by the decoder but never fed to the encoder), so it is not
    passed to the encoder; if supplied, it is ignored with a warning.

    Args:
        x: The PromethEO predictors to convert.
        model_device: Device to place the resulting tensors on.
        tokenization_config: The model's ``TokenizationConfig``, used to build
            per-band-set masks. If ``None``, the OlmoEarth default tokenization
            is used.
        replace_b8a_with_b8: Work around a data issue where Sentinel-2 B8A is
            consistently missing. When ``True``, the B8A band is filled with the
            (present) B8 band before masking and normalization. OlmoEarth groups
            B8A into a band set with five other bands, so a missing B8A would
            otherwise mark that whole band set MISSING and drop it from the
            encoder. See the S2 handling below for details.
    """

    if x.timestamps is None:
        raise ValueError("OlmoEarth requires input timestamps")
    if x.s2 is None and x.s1 is None:
        raise ValueError(
            "OlmoEarth requires at least one of Sentinel-2 (Predictors.s2) or "
            "Sentinel-1 (Predictors.s1)"
        )

    oe = _olmoearth()
    normalizer = oe.Normalizer(std_multiplier=2.0)
    if tokenization_config is None:
        tokenization_config = oe.TokenizationConfig()

    sample_kwargs = {
        "timestamps": to_torchtensor(np.array(x.timestamps, copy=True), model_device).long()
    }
    # PromethEO timestamps are [day, month, year] with 1-indexed months; upstream
    # OlmoEarth examples expect the month column to be zero-indexed.
    sample_kwargs["timestamps"][:, :, 1] -= 1

    # At least one of Sentinel-2 / Sentinel-1 is present (guarded above); both optional.
    if x.s2 is not None:
        s2_indices = [S2_BANDS.index(prometheo) for _, prometheo in S2_OLMOEARTH_TO_PROMETHEO]
        s2 = np.asarray(x.s2)[..., s2_indices].astype(np.float32)
        if replace_b8a_with_b8:
            # Data-quality workaround: B8A is consistently missing, which would
            # mark its whole band set MISSING and drop it. Substitute the present
            # B8 band. Copying the *raw* B8 values into the B8A slot before the
            # mask is built lets that band set be treated as present wherever B8
            # is (and still MISSING if B8 or the band set's other bands are gone).
            oe_s2_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
            b8_idx = oe_s2_bands.index("B08")
            b8a_idx = oe_s2_bands.index("B8A")
            s2[..., b8a_idx] = s2[..., b8_idx]
        s2_mask = _make_modality_mask(s2, oe.Modality.SENTINEL2_L2A.name, tokenization_config)
        s2 = normalizer.normalize(oe.Modality.SENTINEL2_L2A, s2)
        if replace_b8a_with_b8:
            # The requirement is to normalize the substituted values with B8's
            # stats. ``normalize`` used B8A's stats for the B8A slot, so overwrite
            # it with the already-B8-normalized B08 column.
            s2[..., b8a_idx] = s2[..., b8_idx]
        sample_kwargs["sentinel2_l2a"] = to_torchtensor(s2, model_device).float()
        sample_kwargs["sentinel2_l2a_mask"] = to_torchtensor(s2_mask, model_device).long()

    if x.s1 is not None:
        s1_indices = [S1_BANDS.index(prometheo) for _, prometheo in S1_OLMOEARTH_TO_PROMETHEO]
        s1 = np.asarray(x.s1)[..., s1_indices].astype(np.float32)
        s1_mask = _make_modality_mask(
            s1, oe.Modality.SENTINEL1.name, tokenization_config
        )
        s1 = normalizer.normalize(oe.Modality.SENTINEL1, s1)
        sample_kwargs["sentinel1"] = to_torchtensor(s1, model_device).float()
        sample_kwargs["sentinel1_mask"] = to_torchtensor(s1_mask, model_device).long()

    if x.dem is not None:
        # SRTM is decode-only for OlmoEarth; the encoder never sees it, so we drop
        # it rather than feed it in and get wrong results.
        warnings.warn(
            "OlmoEarth treats SRTM as a decode-only modality, so it is not passed "
            "to the encoder; Predictors.dem is ignored.",
            stacklevel=2,
        )

    return oe.MaskedOlmoEarthSample(**sample_kwargs)


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
        patch_size: int = 1,
        input_res: int = 10,
        fast_pass: Optional[bool] = None,
        replace_b8a_with_b8: bool = False,
    ):
        """
        Args:
            fast_pass: OlmoEarth's inference fast path skips all mask handling
                (nodata/MISSING tokens are neither removed nor attention-masked).
                It is only correct when every token is present. Leave as ``None``
                (the default) to select it automatically per input — fast path
                when nothing is masked, masked path otherwise. Pass a bool to
                force it.
            replace_b8a_with_b8: Work around a data issue where Sentinel-2 B8A is
                consistently missing. When ``True``, the B8A band is filled with
                the (present) B8 band (normalized with B8's stats) and its mask is
                no longer marked MISSING on account of B8A. See
                ``dataset_to_olmoearth_sample``.
        """
        super().__init__()
        load_model_from_id = _olmoearth().load_model_from_id
        self.model = load_model_from_id(_get_model_id(model_id), load_weights=load_weights)
        self.patch_size = patch_size
        self.input_res = input_res
        self.fast_pass = fast_pass
        self.replace_b8a_with_b8 = replace_b8a_with_b8
        self.head: Optional[nn.Module] = None
        if num_outputs is not None:
            hidden_size = getattr(getattr(self.model, "encoder", self.model), "embedding_size", None)
            if hidden_size is None:
                raise ValueError("num_outputs requires the OlmoEarth encoder to expose embedding_size")
            self.head = _FinetuningHead(int(hidden_size), num_outputs)

    def forward(
        self,
        x: Predictors,
        eval_pooling: Union[PoolingMethods, None] = PoolingMethods.GLOBAL,
    ):
        if x.timestamps is None:
            raise ValueError("OlmoEarth requires input timestamps")

        if x.label is not None:
            if x.label.shape[3] == x.timestamps.shape[1]:
                eval_pooling = PoolingMethods.TIME
            else:
                eval_pooling = PoolingMethods.GLOBAL

        model_device = next(self.parameters(), torch.empty(0, device=device)).device
        tokenization_config = getattr(self.model.encoder, "tokenization_config", None)
        sample = dataset_to_olmoearth_sample(
            x,
            model_device=model_device,
            tokenization_config=tokenization_config,
            replace_b8a_with_b8=self.replace_b8a_with_b8,
        )
        encoder_output = self.model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=self._resolve_fast_pass(sample),
        )
        embeddings, validity = self._extract_embeddings(encoder_output)
        embeddings = self._apply_pooling(embeddings, validity, eval_pooling)
        if self.head is not None:
            if eval_pooling is None:
                raise ValueError("Can't use the head without a pooling method")
            return self.head(embeddings)
        return embeddings

    def _resolve_fast_pass(self, sample) -> bool:
        """Decide whether to use OlmoEarth's fast path for this sample.

        The fast path skips mask handling entirely, so it is only correct when
        every token is present. If any modality mask contains a non-
        ``ONLINE_ENCODER`` (e.g. MISSING) entry, fall back to the masked path so
        nodata tokens are excluded from attention. An explicit ``self.fast_pass``
        bool overrides the auto-detection.
        """
        if self.fast_pass is not None:
            return self.fast_pass
        online = _olmoearth().MaskValue.ONLINE_ENCODER.value
        for name, value in sample.as_dict().items():
            if name.endswith("_mask") and value is not None:
                if not bool((value == online).all()):
                    return False
        return True

    def _extract_embeddings(self, encoder_output):
        """Reduce the per-modality encoder tokens to ``[B, patch_h, patch_w, T, D]``.

        MISSING tokens are excluded from every mean (over band sets and over
        modalities), mirroring OlmoEarth's own ``pool_unmasked_tokens``. Returns
        the embeddings plus a ``[B, patch_h, patch_w, T]`` validity tensor
        (1.0 where at least one non-missing token contributed), which downstream
        pooling uses to exclude missing timesteps.
        """
        if isinstance(encoder_output, dict) and "tokens_and_masks" in encoder_output:
            tokens_and_masks = encoder_output["tokens_and_masks"]
            online = _olmoearth().MaskValue.ONLINE_ENCODER.value
            modality_embeddings, modality_validities = [], []
            for modality_name in ["sentinel2_l2a", "sentinel1"]:
                tokens = getattr(tokens_and_masks, modality_name, None)
                if tokens is None:
                    continue
                # tokens [B, ph, pw, T, bandsets, D]; mask [B, ph, pw, T, bandsets].
                # Masked mean over band sets, dropping MISSING tokens.
                mask = getattr(tokens_and_masks, f"{modality_name}_mask")
                valid = (mask == online).to(tokens.dtype)
                count = valid.sum(dim=4)
                emb = (tokens * valid.unsqueeze(-1)).sum(dim=4) / count.clamp(min=1.0).unsqueeze(-1)
                modality_embeddings.append(emb)
                modality_validities.append((count > 0).to(tokens.dtype))
            if modality_embeddings:
                emb_stack = torch.stack(modality_embeddings, dim=0)   # [M, B, ph, pw, T, D]
                valid_stack = torch.stack(modality_validities, dim=0)  # [M, B, ph, pw, T]
                count = valid_stack.sum(dim=0)                        # [B, ph, pw, T]
                combined = (emb_stack * valid_stack.unsqueeze(-1)).sum(dim=0) / count.clamp(min=1.0).unsqueeze(-1)
                combined_valid = (count > 0).to(emb_stack.dtype)
                return combined, combined_valid
            raise ValueError("OlmoEarth encoder output did not include supported S1/S2 tokens")
        raise ValueError(
            f"Unexpected OlmoEarth encoder output type {type(encoder_output).__name__}; "
            "expected a dict with a 'tokens_and_masks' entry"
        )

    def _apply_pooling(self, embeddings: torch.Tensor, validity: torch.Tensor, eval_pooling):
        if eval_pooling is None:
            return embeddings
        if embeddings.ndim == 5:
            # PromethEO convention: preserve the spatial token grid; pool time only for GLOBAL.
            if eval_pooling == PoolingMethods.TIME:
                return embeddings
            if eval_pooling == PoolingMethods.GLOBAL:
                # Masked mean over time, excluding missing timesteps.
                weights = validity.unsqueeze(-1)  # [B, ph, pw, T, 1]
                count = validity.sum(dim=3, keepdim=True).clamp(min=1.0).unsqueeze(-1)
                return (embeddings * weights).sum(dim=3, keepdim=True) / count
        raise ValueError(
            f"Unsupported OlmoEarth embedding shape {tuple(embeddings.shape)} "
            f"for pooling mode {eval_pooling}"
        )
