# PromethEO

From pretrained EO representations to simple, reusable building blocks.

## Essence
Two things only:
1. Datasets produce `Predictors` (a typed container of modalities: `s1`, `s2`, `meteo`, `dem`, `latlon`, `timestamps`, optional `label`).
2. Models consume `Predictors` and return embeddings (per time, pooled, or token sequence) and optional task head outputs.

Everything else (training loop, collation, pooling) is lightweight glue so you can swap either side.

## What It Supports
- Pixel time series (H=W=1)
- Multi‑temporal image patches / stacks (H,W > 1)
- Any subset of modalities (missing ones are `None`)

## Included Examples
- Dataset: `WorldCerealDataset` / `WorldCerealLabelledDataset` – illustrative only. Downstream projects should implement their own dataset class that returns a `Predictors` instance per sample.
- Model: `Presto` – pixel‑centric pretrained encoder (works for patches too). Additional models just need a thin wrapper.

## Minimal Flow
```python
from torch.utils.data import DataLoader
from prometheo.datasets import WorldCerealLabelledDataset
from prometheo.models import Presto
from prometheo import finetune
from prometheo.finetune import Hyperparams
from torch.nn import BCEWithLogitsLoss

train_ds = WorldCerealLabelledDataset(df, task_type="binary", num_outputs=1)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl   = DataLoader(train_ds, batch_size=8)

model = Presto(num_outputs=1)
finetune.run_finetuning(
    model,
    train_dl,
    val_dl,
    experiment_name="demo",
    output_dir="./outputs",
    loss_fn=BCEWithLogitsLoss(),
    hyperparams=Hyperparams(max_epochs=3, batch_size=8)
)
```

### Inference
```python
from prometheo.datasets import WorldCerealDataset
from prometheo.models import Presto
from prometheo.models.presto import load_presto_weights

sample = WorldCerealDataset(df)[0]
model  = load_presto_weights(Presto(), "weights.pt")
emb    = model(sample)  # (B, H, W, T|1, D)
```

## Add Your Own Dataset
Return a `Predictors` object in `__getitem__` – nothing else required.

## Add Your Own Model
Wrap it so `forward(predictors, eval_pooling=...) -> tensor or (tokens, mask, meta)`.

## Install
```bash
pip install .
```
Dev mode:
```bash
pip install -e .[dev]
```

## License
See `LICENSE`.

## Acknowledgement
Built around an initial WorldCereal (crop mapping) use case; intentionally task‑agnostic.

