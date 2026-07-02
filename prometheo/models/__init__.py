from importlib import import_module
from typing import Any

__all__ = ["Presto", "OlmoEarth"]


def __getattr__(name: str) -> Any:
    if name == "Presto":
        return import_module("prometheo.models.presto.wrapper").PretrainedPrestoWrapper
    if name == "OlmoEarth":
        return import_module("prometheo.models.olmoearth.wrapper").PretrainedOlmoEarthWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
