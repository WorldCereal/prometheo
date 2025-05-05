import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
from loguru import logger

from prometheo import ssl
from prometheo.datasets import WorldCerealDataset
from prometheo.models import SSLPresto
from prometheo.models.presto.masking import MASK_STRATEGIES, MaskParamsNoDw


def load_dataframe(timestep_freq="month"):
    dir = Path(__file__).resolve().parent
    if timestep_freq == "month":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset.parquet")
    elif timestep_freq == "dekad":
        return pd.read_parquet(dir / "resources" / "worldcereal_dataset_10D.parquet")
    else:
        raise ValueError(f"Invalid timestep frequency `{timestep_freq}`")


class TestSSL(TestCase):
    def test_WorldCerealPrestoSSL(self):
        """Minimal SSL test for Presto"""
        df = load_dataframe()
        train_ds = WorldCerealDataset(df)
        val_ds = WorldCerealDataset(df)

        # Set mask parameters
        mask_params = MaskParamsNoDw(
            strategies=MASK_STRATEGIES, ratio=0.75, num_timesteps=12
        )

        # Construct the model with finetuning head
        model = SSLPresto(mask_params=mask_params)

        # Reduce epochs for testing purposes
        hyperparams = ssl.Hyperparams(nr_epochs=1, batch_size=2, num_workers=2)

        # Run SSL
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            ssl.run_ssl(
                model,
                train_ds,
                val_ds,
                experiment_name="test",
                output_dir=output_dir,
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()

    def test_WorldCerealPrestoSSL_10D(self):
        """Minimal SSL test for Presto, now on 10-day dataset"""
        df = load_dataframe(timestep_freq="dekad")
        num_timesteps = 36
        train_ds = WorldCerealDataset(
            df, timestep_freq="dekad", num_timesteps=num_timesteps
        )
        val_ds = WorldCerealDataset(
            df, timestep_freq="dekad", num_timesteps=num_timesteps
        )

        # Set mask parameters
        mask_params = MaskParamsNoDw(
            strategies=MASK_STRATEGIES, ratio=0.75, num_timesteps=num_timesteps
        )

        # Construct the model with finetuning head
        model = SSLPresto(mask_params=mask_params)

        # Reduce epochs for testing purposes
        hyperparams = ssl.Hyperparams(nr_epochs=1, batch_size=2, num_workers=2)

        # Run SSL
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            ssl.run_ssl(
                model,
                train_ds,
                val_ds,
                experiment_name="test",
                output_dir=output_dir,
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()
