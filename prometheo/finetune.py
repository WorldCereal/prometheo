from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from loguru import logger
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prometheo.predictors import NODATAVALUE, Predictors
from prometheo.utils import (  # config_dir,; data_dir,; default_model_path,
    DEFAULT_SEED,
    device,
    initialize_logging,
    seed_everything,
)


@dataclass
class Hyperparams:
    lr: float = 2e-5
    max_epochs: int = 100
    batch_size: int = 256
    patience: int = 20
    num_workers: int = 8


def _train_loop(
    model: torch.nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    hyperparams: Hyperparams,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    """Perform the training loop for fine-tuning a model.

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
    loss_fn : torch.nn.Module
        The loss function to be used for training.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating the model's parameters.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        The learning rate scheduler used for adjusting the learning rate during training.

    Returns
    -------
    torch.nn.Module
        The trained model.
    """
    train_loss = []
    val_loss = []
    val_acc = []
    val_f1 = []
    best_loss = None
    best_model_dict = None
    epochs_since_improvement = 0

    for epoch in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            optimizer.zero_grad()

            batch = batch.move_predictors_to_device(device)
            assert isinstance(batch, Predictors)

            preds = model(batch)
            if not isinstance(batch.label, torch.Tensor):
                raise ValueError(
                    "Predictors.label must be a torch.Tensor during training"
                )

            targets = batch.label.to(device)
            if preds.dim() > 1 and preds.size(-1) > 1:
                # multiclass case: targets should be class indices
                # predictions are multiclass logits
                targets = targets.long().squeeze(axis=-1)
            else:
                # binary case
                targets = targets.float()

            # Compute loss
            loss = loss_fn(
                preds[targets != NODATAVALUE], targets[targets != NODATAVALUE]
            )

            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        all_preds, all_y = [], []

        for batch in val_dl:
            with torch.no_grad():
                batch = batch.move_predictors_to_device(device)
                assert isinstance(batch, Predictors)
                preds = model(batch)

                if not isinstance(batch.label, torch.Tensor):
                    raise ValueError(
                        "Predictors.label must be a torch.Tensor during training"
                    )

                targets = batch.label.to(device)

                if preds.dim() > 1 and preds.size(-1) > 1:
                    # multiclass case: targets should be class indices
                    # predictions are multiclass logits
                    targets = targets.long().squeeze(axis=-1)
                else:
                    # binary case
                    targets = targets.float()

                preds = preds[targets != NODATAVALUE]
                targets = targets[targets != NODATAVALUE]
                all_preds.append(preds)
                all_y.append(targets)

        # val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))  # Duplicate??

        # --- compute validation loss + metrics ---
        from sklearn.metrics import f1_score

        val_preds = torch.cat(all_preds)
        val_targets = torch.cat(all_y)
        current_val_loss = loss_fn(val_preds, val_targets).item()
        val_loss.append(current_val_loss)

        # derive discrete predictions & true labels
        if val_preds.dim() > 1 and val_preds.size(-1) > 1:
            # multiclass
            pred_labels = val_preds.argmax(dim=-1).cpu()
            true_labels = val_targets.cpu()
        else:
            # binary
            # TODO: what if this is regression?
            probs = torch.sigmoid(val_preds).cpu()
            pred_labels = probs.gt(0.5).long().squeeze(-1)
            true_labels = val_targets.cpu()

        # accuracy & # macro F1
        current_val_acc = pred_labels.eq(true_labels).float().mean().item()
        current_val_f1 = f1_score(
            true_labels.numpy(), pred_labels.numpy(), average="macro", zero_division=0
        )
        val_acc.append(current_val_acc)
        val_f1.append(current_val_f1)

        if best_loss is None:
            best_loss = val_loss[-1]
            best_model_dict = deepcopy(model.state_dict())
        else:
            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                best_model_dict = deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= hyperparams.patience:
                    logger.info("Early stopping!")
                    break

        description = (
            f"Epoch {epoch + 1}/{hyperparams.max_epochs} | "
            f"Train Loss: {train_loss[-1]:.4f} | "
            f"Val Loss: {current_val_loss:.4f} | "
            f"Val Acc: {current_val_acc:.3f} | "
            f"Val Macro F1: {current_val_f1:.3f} | "
            f"Best Loss: {best_loss:.4f}"
        )

        if epochs_since_improvement > 0:
            description += f" (no improvement for {epochs_since_improvement} epochs)"
        else:
            description += " (improved)"

        pbar.set_description(description)
        pbar.set_postfix(lr=scheduler.get_last_lr()[0])
        logger.info(
            f"PROGRESS after Epoch {epoch + 1}/{hyperparams.max_epochs}: {description}"
        )  # Only log to file if console filters on "PROGRESS"

    assert best_model_dict is not None

    model.load_state_dict(best_model_dict)
    model.eval()

    return model


def _setup(output_dir: Path, experiment_name: Union[str, Path], setup_logging: bool):
    """Set up the output directory and logging for the fine-tuning process.

    Parameters
    ----------
    output_dir : Path
        The path to the output directory where the fine-tuned model and logs will be saved.
    experiment_name : Union[str, Path]
        The name of the experiment to use as name for the log file
    setup_logging : bool
        Whether to set up logging for the fine-tuning process. Disable if logging has already been
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


def run_finetuning(
    model: torch.nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    experiment_name: str,
    output_dir: Union[Path, str],
    loss_fn: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
    hyperparams: Hyperparams = Hyperparams(),
    seed: int = DEFAULT_SEED,
    setup_logging: bool = True,
):
    """Runs the finetuning process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be finetuned.
    train_ds : DataLoader
        The training dataloader.
    val_ds : DataLoader
        The validation dataloader.
    experiment_name : str
        The name of the experiment.
    output_dir : Union[Path, str]
        The directory where the finetuned model and logs will be saved.
    loss_fn : torch.nn.Module
        The loss function used for training.
    optimizer : Union[torch.optim.Optimizer, None], optional
        The optimizer used for training. If None, AdamW optimizer will be used with default learning rate.
    scheduler : Union[torch.optim.lr_scheduler.LRScheduler, None], optional
        The learning rate scheduler used for training. If None, ExponentialLR scheduler will be used with default gamma.
    hyperparams : Hyperparams, optional
        The hyperparameters for training. Default is Hyperparams().
    seed : int, optional
        The random seed for reproducibility. Default is DEFAULT_SEED.
    setup_logging : bool, optional
        Whether to set up logging for the finetuning process. Default is True.
    Returns
    -------
    torch.nn.Module
        The finetuned model.
    """

    # TODO: add wandb

    # Set seed
    seed_everything(seed)

    # Set up optimizer and scheduler
    assert (optimizer is None) == (scheduler is None), (
        "`optimizer` and `scheduler` must either both be None or both not None."
    )
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=hyperparams.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Setup directories and initialize logging
    output_dir = Path(output_dir)
    _setup(output_dir, experiment_name, setup_logging)

    # Set model path
    finetuned_model_path = output_dir / f"{experiment_name}.pt"
    if finetuned_model_path.is_file():
        raise FileExistsError(
            f"Model file {finetuned_model_path} already exists. Choose a different directory or experiment name."
        )

    # Move model to device
    model.to(device)

    # Run the finetuning loop
    finetuned_model = _train_loop(
        model, train_dl, val_dl, hyperparams, loss_fn, optimizer, scheduler
    )

    # Save the best model
    torch.save(finetuned_model.state_dict(), finetuned_model_path)

    logger.info("Finetuning done")

    return finetuned_model
