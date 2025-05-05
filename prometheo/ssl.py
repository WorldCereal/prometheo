import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from prometheo.models.presto import adjust_learning_rate, param_groups_weight_decay
from prometheo.predictors import collate_fn
from prometheo.utils import DEFAULT_SEED, device, initialize_logging, seed_everything


@dataclass
class Hyperparams:
    lr: float = 2e-5
    nr_epochs: int = 20
    batch_size: int = 2048
    num_workers: int = 8
    warmup_epochs: int = 2
    weight_decay: float = 0.05
    val_per_n_steps: int = -1
    min_learning_rate: float = 0.0
    max_learning_rate: float = 0.0001


def _train_loop(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    hyperparams: Hyperparams,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    experiment_name: str,
) -> Path:
    """Perform the training loop for pretraining a model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_dl : torch.utils.data.DataLoader
        The data loader for the training dataset.
    val_dl : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    hyperparams : dict
        A dictionary containing hyperparameters for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating the model's parameters.
    output_dir : Path
        The directory where the model checkpoints will be saved during training.
    experiment_name: str
        Name of the current experiment, used for naming the models.

    Returns
    -------
    Path
        Path to the best model.
    """

    # Configure output path for models
    model_dir = output_dir / Path("models")
    model_dir.mkdir(exist_ok=True, parents=True)

    if hyperparams.val_per_n_steps == -1:
        hyperparams.val_per_n_steps = len(train_dl)

    lowest_validation_loss = None
    best_val_epoch = 0
    training_step = 0
    num_validations = 0

    with tqdm(range(hyperparams.nr_epochs), desc="Epoch") as tqdm_epoch:
        for epoch in tqdm_epoch:
            # ------------------------ Training ----------------------------------------
            total_eo_train_loss = 0.0
            num_updates_being_captured = 0
            train_size = 0
            model.train()

            for epoch_step, batch in enumerate(
                tqdm(train_dl, desc="Train", leave=False)
            ):
                # zero the parameter gradients
                optimizer.zero_grad()

                # Adjust learning rate
                lr = adjust_learning_rate(
                    optimizer,
                    epoch_step / len(train_dl) + epoch,
                    hyperparams.warmup_epochs,
                    hyperparams.nr_epochs,
                    hyperparams.max_learning_rate,
                    hyperparams.min_learning_rate,
                )

                # Run a pass through the model
                loss, y_pred, mask = model(batch)
                loss.backward()
                optimizer.step()

                # Update tracking metrics
                current_batch_size = y_pred.shape[0]
                total_eo_train_loss += loss.item()
                num_updates_being_captured += 1
                train_size += current_batch_size
                training_step += 1

                # ------------------------ Validation --------------------------------------
                if training_step % hyperparams.val_per_n_steps == 0:
                    total_eo_val_loss = 0.0
                    num_val_updates_captured = 0
                    # val_size = 0
                    model.eval()

                    with torch.no_grad():
                        for batch in tqdm(val_dl, desc="Validate"):
                            loss, y_pred, mask = model(batch)
                            current_batch_size = y_pred.shape[0]
                            total_eo_val_loss += loss.item()
                            num_val_updates_captured += 1

                    # ------------------------ Metrics + Logging -------------------------------
                    # train_loss now reflects the value against which we calculate gradients
                    train_eo_loss = total_eo_train_loss / num_updates_being_captured
                    val_eo_loss = total_eo_val_loss / num_val_updates_captured

                    # if (
                    #     "train_size" not in training_config
                    #     and "val_size" not in training_config
                    # ):
                    #     training_config["train_size"] = train_size
                    #     training_config["val_size"] = val_size
                    #     if wandb_enabled:
                    #         wandb.config.update(training_config)

                    # to_log = {
                    #     "train_eo_loss": train_eo_loss,
                    #     "val_eo_loss": val_eo_loss,
                    #     "training_step": training_step,
                    #     "epoch": epoch,
                    #     "lr": lr,
                    # }
                    tqdm_epoch.set_postfix(loss=val_eo_loss, lr=lr)

                    if (
                        lowest_validation_loss is None
                        or val_eo_loss < lowest_validation_loss
                    ):
                        lowest_validation_loss = val_eo_loss
                        best_val_epoch = epoch

                        best_model_path = (
                            model_dir / f"{experiment_name}_epoch{epoch}.pt"
                        )
                        logger.info(f"Saving best model to: {best_model_path}")
                        torch.save(model.state_dict(), best_model_path)

                    # reset training logging
                    total_eo_train_loss = 0.0
                    num_updates_being_captured = 0
                    train_size = 0
                    num_validations += 1

                    # if wandb_enabled:
                    #     wandb.log(to_log)

                    description = (
                        f"Train metric: {train_eo_loss:.3f},"
                        f" Val metric: {val_eo_loss:.3f},"
                        f" Best Val Loss: {lowest_validation_loss:.3f}"
                        f" Current LR: {lr:.5f}"
                    )

                    if best_val_epoch < epoch:
                        description += (
                            f" (no improvement for {epoch - best_val_epoch} epochs)"
                        )
                    else:
                        description += " (improved)"

                    model.train()

            logger.info(
                f"PROGRESS after Epoch {epoch + 1}/{hyperparams.nr_epochs}: {description}"
            )  # Only log to file if console filters on "PROGRESS"

    return best_model_path


def _setup(output_dir: Path, experiment_name: Union[str, Path], setup_logging: bool):
    """Set up the output directory and logging for the SSL process.

    Parameters
    ----------
    output_dir : Path
        The path to the output directory where the pretrained model and logs will be saved.
    experiment_name : Union[str, Path]
        The name of the experiment to use as name for the log file
    setup_logging : bool
        Whether to set up logging for the SSL process. Disable if logging has already been
        setup elsewhere.

    Notes
    -----
    This function creates the output directory if it doesn't exist.

    If `setup_logging` is True, the function also creates a subdirectory named 'logs' inside the output
    directory for storing logs.

    """

    output_dir.mkdir(exist_ok=True, parents=True)

    # Set logging dir
    model_logging_dir = output_dir / "logs"
    model_logging_dir.mkdir(exist_ok=True)

    if setup_logging:
        initialize_logging(
            log_file=model_logging_dir / f"{experiment_name}.log",
            console_filter_keyword="PROGRESS",
        )
    logger.info(f"Using output dir: {output_dir.resolve()}")


def run_ssl(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    experiment_name: str,
    output_dir: Union[Path, str],
    optimizer: Union[torch.optim.Optimizer, None] = None,
    hyperparams: Hyperparams = Hyperparams(),
    seed: int = DEFAULT_SEED,
    setup_logging: bool = True,
):
    """Runs the SSL process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be pretrained.
    train_ds : Dataset
        The training dataset.
    val_ds : Dataset
        The validation dataset.
    experiment_name : str
        The name of the experiment.
    output_dir : Union[Path, str]
        The directory where the pretrained model and logs will be saved.
    optimizer : Union[torch.optim.Optimizer, None], optional
        The optimizer used for training. If None, AdamW optimizer will be used with default learning rate.
    hyperparams : Hyperparams, optional
        The hyperparameters for training. Default is Hyperparams().
    seed : int, optional
        The random seed for reproducibility. Default is DEFAULT_SEED.
    setup_logging : bool, optional
        Whether to set up logging for the SSL process. Default is True.

    Returns
    -------
    torch.nn.Module
        The pretrained model.
    """

    # TODO: add wandb

    # Set seed
    seed_everything(seed)

    # Set up optimizer
    if optimizer is None:
        param_groups = param_groups_weight_decay(model, hyperparams.weight_decay)
        optimizer = AdamW(
            param_groups, lr=hyperparams.max_learning_rate, betas=(0.9, 0.95)
        )

    # Setup directories and initialize logging
    output_dir = Path(output_dir)
    _setup(output_dir, experiment_name, setup_logging)

    # Set model path
    ssl_pretrained_model_path = output_dir / f"{experiment_name}.pt"
    if ssl_pretrained_model_path.is_file():
        raise FileExistsError(
            f"Model file {ssl_pretrained_model_path} already exists. Choose a different directory or experiment name."
        )

    # Move model to device
    model.to(device)

    # Setup dataloaders
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dl = DataLoader(
        train_ds,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        num_workers=hyperparams.num_workers,
        generator=generator,
        collate_fn=collate_fn,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
        collate_fn=collate_fn,
    )

    # Run the SSL loop
    best_model_path = _train_loop(
        model,
        train_dl,
        val_dl,
        hyperparams,
        optimizer,
        output_dir,
        experiment_name,
    )

    # Copy the best model to final location
    shutil.copy(best_model_path, ssl_pretrained_model_path)

    logger.info("SSL done!")

    return ssl_pretrained_model_path
