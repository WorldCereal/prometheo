import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from prometheo.utils import (  # config_dir,; data_dir,; default_model_path,
    DEFAULT_SEED,
    device,
    initialize_logging,
    seed_everything,
)

logger = logging.getLogger("__main__")


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
    best_loss = None
    best_model_dict = None
    epochs_since_improvement = 0

    for _ in (pbar := tqdm(range(hyperparams.max_epochs), desc="Finetuning")):
        model.train()
        epoch_train_loss = 0.0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            optimizer.zero_grad()
            preds = model(batch)
            loss = loss_fn(preds, batch.label.float())
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_loss.append(epoch_train_loss / len(train_dl))

        model.eval()
        all_preds, all_y = [], []

        for batch in val_dl:
            with torch.no_grad():
                preds = model(batch)
                all_preds.append(preds)
                all_y.append(batch.label.float())

        val_loss.append(loss_fn(torch.cat(all_preds), torch.cat(all_y)))

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

        pbar.set_description(
            f"Train metric: {train_loss[-1]:.3f}, Val metric: {val_loss[-1]:.3f}, \
Best Val Loss: {best_loss:.3f} \
(no improvement for {epochs_since_improvement} epochs)"
        )

    assert best_model_dict is not None

    model.load_state_dict(best_model_dict)
    model.eval()

    return model


def _setup(output_dir: Path):
    """Set up the output directory and logging for the fine-tuning process.

    Parameters
    ----------
    output_dir : Path
        The path to the output directory where the fine-tuned model and logs will be saved.

    Raises
    ------
    FileExistsError
        If the output directory already exists and is not empty.

    Notes
    -----
    This function creates the output directory if it doesn't exist. If the directory already exists,
    it checks if it's empty. If the directory exists and is not empty, a `FileExistsError` is raised.

    The function also creates a subdirectory named 'logs' inside the output directory for storing logs.
    The logging is initialized with the 'logs' directory as the logging directory.

    """

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory {output_dir} exists and is not empty. Please choose a different directory."
        )
    else:
        output_dir.mkdir(exist_ok=True, parents=True)

    # Set logging dir
    model_logging_dir = output_dir / "logs"
    model_logging_dir.mkdir()
    initialize_logging(model_logging_dir)
    logger.info(f"Using output dir: {output_dir}")


def run_finetuning(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    experiment_name: str,
    output_dir: Union[Path, str],
    loss_fn: torch.nn.Module,
    optimizer: Union[torch.optim.Optimizer, None] = None,
    scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
    hyperparams: Hyperparams = Hyperparams(),
    seed: int = DEFAULT_SEED,
):
    """Runs the finetuning process.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be finetuned.
    train_ds : Dataset
        The training dataset.
    val_ds : Dataset
        The validation dataset.
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

    Returns
    -------
    torch.nn.Module
        The finetuned model.
    """

    # TODO: add wandb

    # Set seed
    seed_everything(seed)

    # Set up optimizer and scheduler
    assert (optimizer is None) == (
        scheduler is None
    ), "`optimizer` and `scheduler` must either both be None or both not None."
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=hyperparams.lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Setup directories and initialize logging
    output_dir = Path(output_dir)
    _setup(output_dir)

    # Set model path
    finetuned_model_path = output_dir / f"{experiment_name}.pt"

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
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        num_workers=hyperparams.num_workers,
    )

    # Run the finetuning loop
    finetuned_model = _train_loop(
        model, train_dl, val_dl, hyperparams, loss_fn, optimizer, scheduler
    )

    # Save the best model
    torch.save(finetuned_model.state_dict(), finetuned_model_path)

    logger.info("Finetuning done")

    return finetuned_model
