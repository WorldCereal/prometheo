# OlmoEarth tiny support investigation

## Current target

As of 2026-07-01, the newest publicly advertised OlmoEarth release is **v1.2**. The smallest v1.2 checkpoint is **`allenai/OlmoEarth-v1_2-Nano`** (5.5M encoder parameters), but the smallest model explicitly named **Tiny** is **`allenai/OlmoEarth-v1_2-Tiny`** (12.5M encoder parameters). For a "latest version in its tiniest form" integration, PromethEO should therefore support `ModelID.OLMOEARTH_V1_2_NANO` by default, while also exposing `ModelID.OLMOEARTH_V1_2_TINY` for users who specifically ask for Tiny.

Primary upstream sources checked:

- `allenai/olmoearth_pretrain_minimal`: minimal inference/loading package, latest GitHub release `v0.0.6` on 2026-06-25.
- `allenai/OlmoEarth-v1_2-Tiny`: Hugging Face model card; ViT-Tiny, 12M parameters, Sentinel-1/Sentinel-2/Landsat images or time series.
- Ai2 v1.1 blog, updated 2026-06-29: says OlmoEarth v1.2 is available, adds RoPE, and ships Nano/Tiny/Small/Base.

## Upstream interface relevant to PromethEO

The minimal package is the best dependency boundary for PromethEO because it contains only code needed to initialize/load models and avoids training/evaluation dependencies.

Expected loading API:

```python
from olmoearth_pretrain_minimal import ModelID, load_model_from_id

model = load_model_from_id(ModelID.OLMOEARTH_V1_2_NANO, load_weights=True)
```

Expected sample construction API:

```python
from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
    MaskedOlmoEarthSample,
)

sample = MaskedOlmoEarthSample(
    timestamps=timestamps,
    sentinel2_l2a=sentinel2_l2a,
    sentinel2_l2a_mask=sentinel2_l2a_mask,
    sentinel1=sentinel1,
    sentinel1_mask=sentinel1_mask,
)
output = model.encoder(sample, patch_size=8, input_res=10, fast_pass=True)
```

Important details:

- Multitemporal modalities use shape `(batch, height, width, time, bands)`.
- Single-temporal modalities use shape `(batch, height, width, bands)`.
- Timestamps are required, with months expected as zero-indexed integers.
- Sentinel-2 L2A band order is `B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09`.
- Sentinel-1 band order is `vv, vh`.
- Landsat band order is `B8, B1, B2, B3, B4, B5, B6, B7, B9, B10, B11`.
- OlmoEarth has its own `Normalizer`; PromethEO should not reuse Presto normalization.

## Fit against PromethEO today

PromethEO's common model contract already matches the broad shape convention OlmoEarth needs: datasets produce a `Predictors` object with modalities such as `s1`, `s2`, `dem`, `latlon`, `timestamps`, and optional `label`, and models consume `Predictors` and return embeddings and optional head outputs. `Predictors.s1` and `Predictors.s2` already use `[B, H, W, T, bands]`, which matches OlmoEarth's multitemporal shape convention.

The main gaps are:

1. **Dependency**: add `olmoearth-pretrain-minimal` as an optional dependency, not a core dependency, because it is model-specific and may have explicit torch extras.
2. **New wrapper module**: add `prometheo/models/olmoearth/wrapper.py` with an `OlmoEarth` or `PretrainedOlmoEarthWrapper` class.
3. **Model export**: update `prometheo/models/__init__.py` to export `OlmoEarth` without breaking the existing `Presto` import.
4. **Band remapping**: map PromethEO Sentinel-2 bands to OlmoEarth L2A order. PromethEO currently has 13 S2 bands, including `B10`, while OlmoEarth Sentinel-2 L2A expects 12 bands and excludes `B10`.
5. **Mask conversion**: produce OlmoEarth masks from PromethEO's `NODATAVALUE` convention. The README sample shows per-pixel/time masks shaped `(B, H, W, T)` for a modality, not per-band masks.
6. **Timestamps**: adapt PromethEO timestamps from `[day, month, year]` to the zero-indexed month convention expected by OlmoEarth. Unlike Presto, this should happen in the OlmoEarth adapter rather than by mutating the input.
7. **Pooling contract**: identify the exact encoder output shape for v1.2 Nano/Tiny and implement PromethEO pooling modes (`GLOBAL`, `TIME`, and possibly unpooled tokens). This requires a small live smoke test once the package is installable in CI/dev.
8. **Task head**: reuse PromethEO's existing task-head pattern or introduce a small generic head so downstream fine-tuning works like Presto.
9. **Licensing/docs**: document OlmoEarth Artifact License constraints separately from PromethEO's license.

## Proposed implementation plan

### Phase 1: dependency and adapter skeleton

- Add an optional extra such as:

```toml
[project.optional-dependencies]
olmoearth = [
  "olmoearth-pretrain-minimal>=0.0.6",
]
```

- Add `prometheo/models/olmoearth/__init__.py` and `wrapper.py`.
- Keep imports lazy inside the wrapper so users without the extra can still import `prometheo.models.Presto`.
- Default model ID to `OLMOEARTH_V1_2_NANO`; accept a string/enum override for `OLMOEARTH_V1_2_TINY`.

### Phase 2: `Predictors` to `MaskedOlmoEarthSample`

- Implement a converter analogous to Presto's `dataset_to_model`, but OlmoEarth-specific.
- Use S2 mapping:

| OlmoEarth | PromethEO |
| --- | --- |
| B02 | B2 |
| B03 | B3 |
| B04 | B4 |
| B08 | B8 |
| B05 | B5 |
| B06 | B6 |
| B07 | B7 |
| B8A | B8A |
| B11 | B11 |
| B12 | B12 |
| B01 | B1 |
| B09 | B9 |

- Use S1 mapping `vv -> VV`, `vh -> VH`.
- Do not support Landsat initially because `Predictors` has no Landsat field.
- Map PromethEO `dem` elevation to OlmoEarth `srtm`; PromethEO `slope` has no direct OlmoEarth SRTM band and is intentionally ignored.

### Phase 3: output and tests

- Add unit tests that monkeypatch `load_model_from_id` and the normalizer so tests do not download weights.
- Test band order, mask construction, timestamp month conversion, missing optional modalities, and output reshaping.
- Add one optional integration test marked/skipped unless `PROMETHEO_TEST_OLMOEARTH=1` and weights are available.

## Open questions before coding full support

- What is the exact v1.2 encoder output schema for `fast_pass=True` and for multi-patch images? The README only shows how to call it, not the resulting shape.
- Should PromethEO's public model be named `OlmoEarth` and default to Nano, or `OlmoEarthTiny` and load Tiny despite Nano being smaller?
- Should the first version support SRTM/elevation? PromethEO can map elevation, but not necessarily with OlmoEarth's expected preprocessing and missing-value semantics.
- Should `olmoearth-pretrain-minimal` be optional with no torch extra, or should PromethEO expose separate CPU/CUDA extras once upstream packaging requirements are confirmed?

## Recommendation

Implement a thin optional wrapper first, defaulting to `OLMOEARTH_V1_2_NANO` and supporting S2, optional S1, optional DEM elevation-as-SRTM, and timestamps. Keep Landsat and derived-map support out of the first pass because PromethEO's `Predictors` schema does not currently represent them. The integration is medium-sized but low-risk if it remains optional and has dependency-free unit tests around the adapter.
