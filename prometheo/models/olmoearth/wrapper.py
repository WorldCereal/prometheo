from __future__ import annotations

import warnings
from enum import Enum
from functools import lru_cache
from types import SimpleNamespace
from typing import Mapping, Optional, Union

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
# Central wavelengths (nm) of the Sentinel-2 bands (OlmoEarth naming), used to
# pick the spectrally-nearest source band when interpolating a missing band.
S2_BAND_WAVELENGTHS_NM = {
    "B01": 443,
    "B02": 490,
    "B03": 560,
    "B04": 665,
    "B05": 705,
    "B06": 740,
    "B07": 783,
    "B08": 842,
    "B8A": 865,
    "B09": 945,
    "B11": 1610,
    "B12": 2190,
}

class MissingBandStrategy(str, Enum):
    """How to fill a Sentinel-2 band's missing values so its band set stays
    present. Only NODATA entries are filled; real values are kept.

    INTERPOLATE: fill from the spectrally-nearest band that is not itself
        being replaced (normalized with the source band's stats).
    ZERO: feed the encoder zeros for the missing entries (zero in normalized
        space, i.e. the value the model sees is 0.0).

    Either strategy stops the missing entries from marking their band set
    MISSING; see ``dataset_to_olmoearth_sample``.
    """

    INTERPOLATE = "interpolate"
    ZERO = "zero"


def _resolve_band_replacements(
    replace_bands: Optional[Mapping[str, Union[MissingBandStrategy, str]]],
) -> dict[str, MissingBandStrategy]:
    """Normalize ``replace_bands`` to ``{olmoearth band: strategy}``.

    Keys may use either the OlmoEarth ("B01") or PromethEO ("B1") band
    spelling.
    """
    if not replace_bands:
        return {}
    aliases = {}
    for olmoearth, prometheo in S2_OLMOEARTH_TO_PROMETHEO:
        aliases[olmoearth] = olmoearth
        aliases[prometheo] = olmoearth
    resolved = {}
    for band, strategy in replace_bands.items():
        if band not in aliases:
            raise ValueError(
                f"Unknown Sentinel-2 band {band!r} in replace_bands; expected "
                f"one of {sorted(aliases)}"
            )
        resolved[aliases[band]] = MissingBandStrategy(strategy)
    return resolved


def _interpolation_source(band: str, replaced_bands) -> str:
    """The spectrally-nearest S2 band that is not itself being replaced."""
    candidates = [b for b in S2_BAND_WAVELENGTHS_NM if b not in replaced_bands]
    if not candidates:
        raise ValueError(
            "Cannot interpolate a Sentinel-2 band when every band is being "
            "replaced; no source band is left to interpolate from"
        )
    target_wavelength = S2_BAND_WAVELENGTHS_NM[band]
    return min(
        candidates, key=lambda b: abs(S2_BAND_WAVELENGTHS_NM[b] - target_wavelength)
    )


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


def _shares_bandset_with_unreplaced_bands(
    band: str, replaced_bands, tokenization_config
) -> bool:
    """Whether ``band`` is tokenized together with a band that is not itself
    being replaced.

    Replacing a band is only useful when its band set contains genuinely
    present bands — otherwise the whole band set is missing anyway, and it is
    cleaner to leave it masked MISSING (the encoder simply drops it) than to
    fabricate every band in it. E.g. OlmoEarth v1 gives the 60m atmospheric
    bands (B01, B09) a band set of their own, so when both are replaced that
    band set is just dropped; v1.1/v1.2 tokenize all twelve S2 bands as a
    single band set, so missing B01/B09 would mark the whole Sentinel-2
    modality MISSING and substitution is required.
    """
    oe_s2_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
    replaced_indices = {oe_s2_bands.index(b) for b in replaced_bands}
    band_index = oe_s2_bands.index(band)
    bandsets = tokenization_config.get_bandset_indices(
        _olmoearth().Modality.SENTINEL2_L2A.name
    )
    return any(
        band_index in idxs and (set(idxs) - replaced_indices) for idxs in bandsets
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
    replace_bands: Optional[Mapping[str, Union[MissingBandStrategy, str]]] = None,
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
        replace_bands: Work around missing Sentinel-2 bands. A missing band
            marks its whole band set MISSING and drops it from the encoder,
            so this maps each affected band (OlmoEarth "B01" or PromethEO
            "B1" spelling) to a ``MissingBandStrategy`` (or its string value)
            that fills its NODATA entries instead — locations where the band
            has real data are left untouched. ``INTERPOLATE`` copies the
            spectrally-nearest band that is not itself being replaced
            (normalized with the source band's stats), while ``ZERO`` feeds
            the encoder zeros. A band is only filled when the model tokenizes
            it together with bands that are not being replaced; when a band
            set consists entirely of replaced bands (e.g. B01/B09 under v1
            tokenization), it is left masked MISSING and simply dropped by
            the encoder. ``None`` (the default) applies no workaround.
    """

    if x.timestamps is None:
        raise ValueError("OlmoEarth requires input timestamps")
    if x.s2 is None and x.s1 is None:
        raise ValueError(
            "OlmoEarth requires at least one of Sentinel-2 (Predictors.s2) or "
            "Sentinel-1 (Predictors.s1)"
        )

    band_replacements = _resolve_band_replacements(replace_bands)
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
        # Data-quality workarounds for missing bands that share a band set
        # with present bands (a missing band marks its whole band set MISSING
        # and drops it from the encoder). Only NODATA entries of a replaced
        # band are filled — locations where the band has real data keep it.
        # Filling *before* the mask is built stops the filled entries from
        # marking their band set MISSING: INTERPOLATE copies the raw source
        # values, so the band set is treated as present wherever the source
        # is (and still MISSING if the source or the band set's other bands
        # are gone); ZERO writes a raw placeholder, so missing entries never
        # affect the mask at all. Bands whose band set consists entirely of
        # replaced bands are skipped: that band set is simply masked MISSING
        # and dropped (e.g. B01/B09 under v1 tokenization, where they form
        # their own band set).
        substitutions = {
            band: strategy
            for band, strategy in band_replacements.items()
            if _shares_bandset_with_unreplaced_bands(
                band, band_replacements, tokenization_config
            )
        }
        sources = {
            band: _interpolation_source(band, band_replacements)
            for band, strategy in substitutions.items()
            if strategy is MissingBandStrategy.INTERPOLATE
        }
        oe_s2_bands = [olmoearth for olmoearth, _ in S2_OLMOEARTH_TO_PROMETHEO]
        missing_by_band = {
            band: s2[..., oe_s2_bands.index(band)] == NODATAVALUE
            for band in substitutions
        }

        def fill_missing_entries(values: np.ndarray) -> None:
            # Sources are never themselves replaced bands, so the fill order
            # doesn't matter.
            for band, strategy in substitutions.items():
                target = oe_s2_bands.index(band)
                if strategy is MissingBandStrategy.INTERPOLATE:
                    fill = values[..., oe_s2_bands.index(sources[band])]
                else:  # ZERO
                    fill = np.float32(0.0)
                values[..., target] = np.where(
                    missing_by_band[band], fill, values[..., target]
                )

        fill_missing_entries(s2)
        s2_mask = _make_modality_mask(
            s2, oe.Modality.SENTINEL2_L2A.name, tokenization_config
        )
        s2 = normalizer.normalize(oe.Modality.SENTINEL2_L2A, s2)
        # Run the fill again on the normalized array: interpolated entries must
        # be normalized with the *source* band's stats (``normalize`` used the
        # target's stats for the target slot), and ZERO means the *encoder*
        # sees zeros, so the zero goes in after normalization (a raw 0 would
        # normalize to a nonzero value). Entries with real data are untouched
        # either way.
        fill_missing_entries(s2)
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
        replace_bands: Optional[Mapping[str, Union[MissingBandStrategy, str]]] = None,
    ):
        """
        Args:
            fast_pass: OlmoEarth's inference fast path skips all mask handling
                (nodata/MISSING tokens are neither removed nor attention-masked).
                It is only correct when every token is present. Leave as ``None``
                (the default) to select it automatically per input — fast path
                when nothing is masked, masked path otherwise. Pass a bool to
                force it.
            replace_bands: Work around missing Sentinel-2 bands, which would
                otherwise mark their band set MISSING and drop it from the
                encoder. Maps each affected band (OlmoEarth "B01" or
                PromethEO "B1" spelling) to a ``MissingBandStrategy`` (or its
                string value) that fills its NODATA entries — real values are
                kept: ``INTERPOLATE`` fills from the spectrally-nearest band
                that is not itself being replaced (normalized with the source
                band's stats), ``ZERO`` feeds the encoder zeros. E.g.
                ``replace_bands={"B8A": "interpolate", "B01": "zero", "B09": "zero"}``.
                Bands whose whole band set is being replaced (e.g. B01/B09
                under v1 tokenization) are left masked MISSING and dropped by
                the encoder instead. ``None`` (the default) applies no
                workaround. See ``dataset_to_olmoearth_sample``.
        """
        super().__init__()
        load_model_from_id = _olmoearth().load_model_from_id
        model = load_model_from_id(_get_model_id(model_id), load_weights=load_weights)
        self.encoder = model.encoder
        self.patch_size = patch_size
        self.input_res = input_res
        self.fast_pass = fast_pass
        self.replace_bands = _resolve_band_replacements(replace_bands)
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
            replace_bands=self.replace_bands,
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
