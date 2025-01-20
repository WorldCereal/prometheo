from pathlib import Path

import pandas as pd


def load_dataframe(timestep_freq="month"):
    dir = Path(__file__).resolve().parent
    if timestep_freq == "month":
        return pd.read_parquet(dir / "worldcereal_dataset.parquet")
    elif timestep_freq == "dekad":
        return pd.read_parquet(dir / "worldcereal_dataset_10D.parquet")
    else:
        raise ValueError(f"Invalid timestep frequency `{timestep_freq}`")
