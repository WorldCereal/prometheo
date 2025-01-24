import sys
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger

DEFAULT_SEED: int = 42


if not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


# From https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def seed_everything(seed: int = DEFAULT_SEED):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def initialize_logging(
    log_file: Optional[Union[str, Path]] = None,
    level="INFO",
    console_filter_keyword: Optional[str] = None,
):
    # Remove the default console handler if necessary
    logger.remove()

    # Custom format
    custom_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - "
        "<level>{message}</level>"
    )

    # Re-add console handler with optional filtering
    logger.add(
        sys.stdout,
        level=level,
        format=custom_format,
        filter=lambda record: console_filter_keyword not in record["message"]
        if console_filter_keyword
        else None,
    )

    # File handler
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=custom_format)

    logger.info(
        "Logging setup complete. Logging to: {} and console.",
        log_file or "console only",
    )
