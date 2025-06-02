import tempfile
from unittest import TestCase

import pandas as pd
from loguru import logger
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from prometheo import finetune
from prometheo.datasets import WorldCerealLabelledDataset
from prometheo.finetune import Hyperparams
from prometheo.models import Presto
from prometheo.models.presto import param_groups_lrd


class TestFinetuning(TestCase):
    def setUp(self):
        """Set up mockup datasets for finetuning tests."""
        self.num_samples = 5
        self.num_timesteps = 12

        # Create mockup dataframe
        data = {
            "lat": [45.1, 45.2, 45.3, 45.4, 45.5],
            "lon": [5.1, 5.2, 5.3, 5.4, 5.5],
            "start_date": ["2021-01-01"] * self.num_samples,
            "end_date": ["2022-01-01"] * self.num_samples,
            "available_timesteps": [self.num_timesteps] * self.num_samples,
            "valid_position": [6] * self.num_samples,
        }

        for ts in range(self.num_timesteps):
            # Add optical bands
            for band_template in [
                "OPTICAL-B02-ts{}-10m",
                "OPTICAL-B03-ts{}-10m",
                "OPTICAL-B04-ts{}-10m",
                "OPTICAL-B05-ts{}-20m",
                "OPTICAL-B06-ts{}-20m",
                "OPTICAL-B07-ts{}-20m",
                "OPTICAL-B08-ts{}-10m",
                "OPTICAL-B8A-ts{}-20m",
                "OPTICAL-B11-ts{}-20m",
                "OPTICAL-B12-ts{}-20m",
            ]:
                data[band_template.format(ts)] = [1000 + ts * 10] * self.num_samples

            # Add SAR bands
            for band_template in [
                "SAR-VH-ts{}-20m",
                "SAR-VV-ts{}-20m",
            ]:
                data[band_template.format(ts)] = [0.01 + ts * 0.001] * self.num_samples

            # Add METEO bands
            for band_template in [
                "METEO-precipitation_flux-ts{}-100m",
                "METEO-temperature_mean-ts{}-100m",
            ]:
                data[band_template.format(ts)] = [10 + ts] * self.num_samples

        data["DEM-alt-20m"] = [100] * self.num_samples
        data["DEM-slo-20m"] = [5] * self.num_samples
        data["finetune_class"] = [
            "cropland",
            "not_cropland",
            "cropland",
            "cropland",
            "not_cropland",
        ]

        self.df = pd.DataFrame(data)

        # Multiclass version
        data["finetune_class"] = [
            "cropland",
            "shrub",
            "shrub",
            "tree",
            "builtup",
        ]

        self.df_multiclass = pd.DataFrame(data)

    def test_WorldCerealFinetuneCropNoCrop(self):
        """Minimal finetuning test using mockup datasets."""
        train_ds = WorldCerealLabelledDataset(self.df)
        val_ds = WorldCerealLabelledDataset(self.df)

        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

        # Construct the model with finetuning head
        model = Presto(num_outputs=train_ds.num_outputs, regression=False)

        # Reduce epochs for testing purposes
        hyperparams = Hyperparams(max_epochs=1, batch_size=2, patience=1, num_workers=2)

        # Run finetuning
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            finetune.run_finetuning(
                model,
                train_dl,
                val_dl,
                experiment_name="test",
                output_dir=output_dir,
                loss_fn=BCEWithLogitsLoss(),
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()

    def test_WorldCerealFinetuneCropNoCrop_TimeExplicit(self):
        """Finetuning test with time-explicit predictions using mockup datasets."""
        train_ds = WorldCerealLabelledDataset(self.df, time_explicit=True)
        val_ds = WorldCerealLabelledDataset(self.df, time_explicit=True)

        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

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
                train_dl,
                val_dl,
                experiment_name="test_advanced",
                output_dir=output_dir,
                loss_fn=BCEWithLogitsLoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()

    def test_WorldCerealFinetuneCropType(self):
        """Finetuning test with multiclass target using mockup datasets."""
        train_ds = WorldCerealLabelledDataset(
            self.df_multiclass,
            task_type="multiclass",
            num_outputs=self.df_multiclass["finetune_class"].nunique(),
            classes_list=self.df_multiclass["finetune_class"].unique().tolist(),
        )
        val_ds = WorldCerealLabelledDataset(
            self.df_multiclass,
            task_type="multiclass",
            num_outputs=self.df_multiclass["finetune_class"].nunique(),
            classes_list=self.df_multiclass["finetune_class"].unique().tolist(),
        )

        train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=2, shuffle=False)

        # Construct the model with finetuning head
        model = Presto(num_outputs=train_ds.num_outputs, regression=False)

        # Reduce epochs for testing purposes
        hyperparams = Hyperparams(max_epochs=1, batch_size=2, patience=1, num_workers=2)

        # Run finetuning
        with tempfile.TemporaryDirectory(dir=".") as output_dir:
            finetune.run_finetuning(
                model,
                train_dl,
                val_dl,
                experiment_name="test",
                output_dir=output_dir,
                loss_fn=CrossEntropyLoss(),
                hyperparams=hyperparams,
            )

            # Explicitly remove handlers to avoid errors with autodelete of temp folder
            logger.remove()
