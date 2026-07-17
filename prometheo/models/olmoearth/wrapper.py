from __future__ import annotations

import warnings
from enum import Enum
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional, Union

from pathlib import Path
import numpy as np
import torch
from torch import nn
import requests
import io

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
# Spectrally-nearest present band for each consistently-missing S2 band:
# B8A (865nm) <- B08 (842nm), B01 (443nm) <- B02 (490nm), B09 (945nm) <- B08
# (its true neighbour B8A is itself missing in the data).
S2_B8A_SUBSTITUTE = ("B8A", "B08")
S2_B1_B9_SUBSTITUTES = [("B01", "B02"), ("B09", "B08")]


class MissingBandStrategy(str, Enum):
    """How to fill the consistently-missing Sentinel-2 bands (B8A, B01, B09).

    INTERPOLATE: fill each missing band from its spectrally-nearest present
        band (normalized with the source band's stats).
    ZERO: feed the encoder zeros for the missing bands (zero in normalized
        space, i.e. the value the model sees is 0.0).

    Either strategy stops the missing bands from marking their band set
    MISSING; see ``dataset_to_olmoearth_sample``.
    """

    INTERPOLATE = "interpolate"
    ZERO = "zero"


def _resolve_missing_band_strategy(
    value: Union[MissingBandStrategy, str, bool, None],
) -> Optional[MissingBandStrategy]:
    """Normalize ``replace_b1_b9_b8a`` to a strategy (or None for off).

    Booleans are the parameter's legacy form: True maps to INTERPOLATE (the
    original behaviour) with a deprecation warning, False to off.
    """
    if value is None or value is False:
        return None
    if value is True:
        warnings.warn(
            "replace_b1_b9_b8a=True is deprecated; pass "
            "MissingBandStrategy.INTERPOLATE (or 'interpolate') instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return MissingBandStrategy.INTERPOLATE
    return MissingBandStrategy(value)


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
            MaskedOlmoEarthSample,
            MaskValue,
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


def _b1_b9_share_bandset_with_other_bands(tokenization_config) -> bool:
    """Whether the model tokenizes B01/B09 together with other S2 bands.

    OlmoEarth v1 gives the 60m atmospheric bands (B01, B09) a band set of
    their own, so when both are missing only that band set is marked MISSING
    and dropped by the encoder — no other band is affected and no
    substitution is needed. v1.1 and v1.2 tokenize all twelve S2 bands as a
    single band set, so missing B01/B09 would mark the whole Sentinel-2
    modality MISSING at every location; there, substitution is required.
    """
    oe_s2_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
    atmospheric = {oe_s2_bands.index("B01"), oe_s2_bands.index("B09")}
    bandsets = tokenization_config.get_bandset_indices(
        _olmoearth().Modality.SENTINEL2_L2A.name
    )
    return any(
        (atmospheric & set(idxs)) and (set(idxs) - atmospheric) for idxs in bandsets
    )


@lru_cache(maxsize=6)
def load_olmoearth_weights(
    olmoearth_model: nn.Module,
    weights_path: Union[str, Path],
    strict: bool = True,
) -> nn.Module:
    """Load pretrained weights into an OlmoEarth model.

    Parameters
    ----------
    olmoearth_model : nn.Module
        The OlmoEarth model to load the pretrained weights into.
    weights_path : Union[str, Path], optional
        The path to the pretrained weights file. If not provided, the default model path will be used.
    strict : bool, optional
        Whether to strictly enforce that the keys in the pretrained weights match the keys in the model.
        If True, an error will be raised if there are any missing or unexpected keys. If False, missing or
        unexpected keys will be ignored. Default is True.

    Returns
    -------
    nn.Module
        The Presto model with the pretrained weights loaded.

    Raises
    ------
    FileNotFoundError
        If the specified model path does not exist.
    """
    olmoearth_model.to(device)
    if isinstance(weights_path, str) and (weights_path.startswith("http")):
        response = requests.get(weights_path)
        olmoearth_model_layers = torch.load(
            io.BytesIO(response.content), map_location=device
        )
        olmoearth_model.load_state_dict(olmoearth_model_layers, strict=strict)
    else:
        olmoearth_model.load_state_dict(
            torch.load(weights_path, map_location=device), strict=strict
        )

    return olmoearth_model


def dataset_to_olmoearth_sample(
    x: Predictors,
    model_device: torch.device = device,
    tokenization_config=None,
    replace_b1_b9_b8a: Union[MissingBandStrategy, str, bool, None] = None,
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
        replace_b1_b9_b8a: Work around a data issue where the Sentinel-2 bands
            B8A, B01 and B09 are consistently missing. A missing band marks its
            whole band set MISSING and drops it from the encoder, so a
            ``MissingBandStrategy`` (or its string value) fills the missing
            bands instead: ``INTERPOLATE`` copies each from a
            spectrally-adjacent present band (normalized with the source
            band's stats) — B8A from B08, B01 from B02, B09 from B08 — while
            ``ZERO`` feeds the encoder zeros for them. B01/B09 are only filled
            when the model tokenizes them together with other bands
            (v1.1/v1.2); under v1 tokenization they form their own band set,
            so they are left masked MISSING and simply dropped by the encoder.
            ``None`` (the default) applies no workaround; the legacy ``True``
            is a deprecated alias for ``INTERPOLATE``.
    """

    if x.timestamps is None:
        raise ValueError("OlmoEarth requires input timestamps")
    if x.s2 is None and x.s1 is None:
        raise ValueError(
            "OlmoEarth requires at least one of Sentinel-2 (Predictors.s2) or "
            "Sentinel-1 (Predictors.s1)"
        )

    missing_band_strategy = _resolve_missing_band_strategy(replace_b1_b9_b8a)
    oe = _olmoearth()
    normalizer = oe.Normalizer(std_multiplier=2.0)
    if tokenization_config is None:
        tokenization_config = oe.TokenizationConfig()

    sample_kwargs = {
        "timestamps": to_torchtensor(
            np.array(x.timestamps, copy=True), model_device
        ).long()
    }
    # PromethEO timestamps are [day, month, year] with 1-indexed months; upstream
    # OlmoEarth examples expect the month column to be zero-indexed.
    sample_kwargs["timestamps"][:, :, 1] -= 1

    # At least one of Sentinel-2 / Sentinel-1 is present (guarded above); both optional.
    if x.s2 is not None:
        s2_indices = [
            S2_BANDS.index(prometheo) for _, prometheo in S2_OLMOEARTH_TO_PROMETHEO
        ]
        s2 = np.asarray(x.s2)[..., s2_indices].astype(np.float32)
        # Data-quality workarounds for consistently-missing bands that share a
        # band set with present bands (a missing band marks its whole band set
        # MISSING and drops it from the encoder). Overwriting the target slot
        # *before* the mask is built stops it from marking its band set
        # MISSING: INTERPOLATE copies the raw source values, so the band set
        # is treated as present wherever the source is (and still MISSING if
        # the source or the band set's other bands are gone); ZERO writes a
        # raw placeholder, so the target never affects the mask at all.
        # B01/B09 are only substituted when the tokenization groups them with
        # other bands (v1.1/v1.2); under v1 they form their own band set,
        # which is simply masked MISSING and dropped.
        substitutions = []
        if missing_band_strategy is not None:
            substitutions.append(S2_B8A_SUBSTITUTE)
            if _b1_b9_share_bandset_with_other_bands(tokenization_config):
                substitutions.extend(S2_B1_B9_SUBSTITUTES)
        oe_s2_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
        for target, source in substitutions:
            if missing_band_strategy is MissingBandStrategy.INTERPOLATE:
                s2[..., oe_s2_bands.index(target)] = s2[..., oe_s2_bands.index(source)]
            else:  # ZERO
                s2[..., oe_s2_bands.index(target)] = 0.0
        s2_mask = _make_modality_mask(
            s2, oe.Modality.SENTINEL2_L2A.name, tokenization_config
        )
        s2 = normalizer.normalize(oe.Modality.SENTINEL2_L2A, s2)
        for target, source in substitutions:
            if missing_band_strategy is MissingBandStrategy.INTERPOLATE:
                # The requirement is to normalize substituted values with the
                # *source* band's stats. ``normalize`` used the target's stats
                # for the target slot, so overwrite it with the
                # already-normalized source column.
                s2[..., oe_s2_bands.index(target)] = s2[..., oe_s2_bands.index(source)]
            else:
                # ZERO means the *encoder* sees zeros, so the zero goes in
                # after normalization (a raw 0 would normalize to a nonzero
                # value).
                s2[..., oe_s2_bands.index(target)] = 0.0
        sample_kwargs["sentinel2_l2a"] = to_torchtensor(s2, model_device).float()
        sample_kwargs["sentinel2_l2a_mask"] = to_torchtensor(
            s2_mask, model_device
        ).long()

    if x.s1 is not None:
        s1_indices = [
            S1_BANDS.index(prometheo) for _, prometheo in S1_OLMOEARTH_TO_PROMETHEO
        ]
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
        replace_b1_b9_b8a: Union[MissingBandStrategy, str, bool, None] = None,
    ):
        """
        Args:
            fast_pass: OlmoEarth's inference fast path skips all mask handling
                (nodata/MISSING tokens are neither removed nor attention-masked).
                It is only correct when every token is present. Leave as ``None``
                (the default) to select it automatically per input — fast path
                when nothing is masked, masked path otherwise. Pass a bool to
                force it.
            replace_b1_b9_b8a: Work around a data issue where the Sentinel-2
                bands B8A, B01 and B09 are consistently missing. Pass a
                ``MissingBandStrategy`` (or its string value):
                ``INTERPOLATE`` fills B8A with the (present) B8 band, and —
                for models that tokenize B01/B09 together with the other S2
                bands (v1.1/v1.2) — B01 with B02 and B09 with B08 (each
                normalized with the source band's stats); ``ZERO`` feeds the
                encoder zeros for those bands instead. Either way the missing
                bands no longer mark their band set MISSING. For models that
                give B01/B09 their own band set (v1), that band set is left
                masked MISSING and dropped by the encoder. ``None`` (the
                default) applies no workaround; the legacy ``True`` is a
                deprecated alias for ``INTERPOLATE``. See
                ``dataset_to_olmoearth_sample``.
        """
        super().__init__()
        load_model_from_id = _olmoearth().load_model_from_id
        model = load_model_from_id(_get_model_id(model_id), load_weights=load_weights)
        self.encoder = model.encoder
        self.patch_size = patch_size
        self.input_res = input_res
        self.fast_pass = fast_pass
        self.replace_b1_b9_b8a = _resolve_missing_band_strategy(replace_b1_b9_b8a)
        self.head: Optional[nn.Module] = None
        if num_outputs is not None:
            hidden_size = getattr(self.encoder, "embedding_size", None)
            if hidden_size is None:
                raise ValueError(
                    "num_outputs requires the OlmoEarth encoder to expose embedding_size"
                )
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
        tokenization_config = getattr(self.encoder, "tokenization_config", None)
        sample = dataset_to_olmoearth_sample(
            x,
            model_device=model_device,
            tokenization_config=tokenization_config,
            replace_b1_b9_b8a=self.replace_b1_b9_b8a,
        )
        encoder_output = self.encoder(
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
                emb = (tokens * valid.unsqueeze(-1)).sum(dim=4) / count.clamp(
                    min=1.0
                ).unsqueeze(-1)
                modality_embeddings.append(emb)
                modality_validities.append((count > 0).to(tokens.dtype))
            if modality_embeddings:
                emb_stack = torch.stack(
                    modality_embeddings, dim=0
                )  # [M, B, ph, pw, T, D]
                valid_stack = torch.stack(
                    modality_validities, dim=0
                )  # [M, B, ph, pw, T]
                count = valid_stack.sum(dim=0)  # [B, ph, pw, T]
                combined = (emb_stack * valid_stack.unsqueeze(-1)).sum(
                    dim=0
                ) / count.clamp(min=1.0).unsqueeze(-1)
                combined_valid = (count > 0).to(emb_stack.dtype)
                return combined, combined_valid
            raise ValueError(
                "OlmoEarth encoder output did not include supported S1/S2 tokens"
            )
        raise ValueError(
            f"Unexpected OlmoEarth encoder output type {type(encoder_output).__name__}; "
            "expected a dict with a 'tokens_and_masks' entry"
        )

    def _apply_pooling(
        self, embeddings: torch.Tensor, validity: torch.Tensor, eval_pooling
    ):
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
