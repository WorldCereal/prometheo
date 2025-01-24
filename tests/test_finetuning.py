import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
from loguru import logger
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW, lr_scheduler

from prometheo import finetune
from prometheo.datasets import WorldCerealLabelledDataset
from prometheo.finetune import Hyperparams
from prometheo.models import Presto
from prometheo.models.presto import param_groups_lrd


def load_dataframe(timestep_freq="month"):
    dir = Path(__file__).resolve().parent
    if timestep_freq == "month":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")
    elif timestep_freq == "dekad":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset_10D.parquet")
    else:
        raise ValueError(f"Invalid timestep frequency `{timestep_freq}`")


class TestFinetuning(TestCase):
    def test_WorldCerealFinetuneCropNoCrop(self):
        """Minimal finetuning test as much as possible making use of defaults"""
        df = load_dataframe()
        train_ds = WorldCerealLabelledDataset(df)
        val_ds = WorldCerealLabelledDataset(df)

        # Construct the model with finetuning head
        model = Presto(num_outputs=train_ds.num_outputs, regression=False)

        # Reduce epochs for testing purposes
        hyperparams = Hyperparams(max_epochs=1, batch_size=2, patience=1, num_workers=2)

        # Run finetuning
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            finetune.run_finetuning(
                model,
                train_ds,
                val_ds,
                experiment_name="test",
                output_dir=output_dir,
                loss_fn=BCEWithLogitsLoss(),
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()

    def test_WorldCerealFinetuneCropNoCrop_TimeExplicit(self):
        """More advanced finetuning test with custom optimizer and scheduler
        and making time-explicit predictions"""
        df = load_dataframe()
        train_ds = WorldCerealLabelledDataset(df, time_explicit=True)
        val_ds = WorldCerealLabelledDataset(df, time_explicit=True)

        # Construct the model with finetuning head
        model = Presto(num_outputs=train_ds.num_outputs, regression=False)

        # Set the parameters
        hyperparams = Hyperparams(max_epochs=1, batch_size=2, patience=1, num_workers=2)
        parameters = param_groups_lrd(model)
        optimizer = AdamW(parameters, lr=hyperparams.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # Run finetuning
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            finetune.run_finetuning(
                model,
                train_ds,
                val_ds,
                experiment_name="test_advanced",
                output_dir=output_dir,
                loss_fn=BCEWithLogitsLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()
