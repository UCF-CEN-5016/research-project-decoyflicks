"""
delf/python/datasets/__init__.py

Dataset registry and factory utilities.

This module provides:
- DatasetBase: abstract base class for dataset implementations.
- register_dataset: decorator to register dataset classes by name.
- get_dataset_class: retrieve a registered dataset class.
- create_dataset: instantiate a dataset by name with given kwargs.
- list_datasets: list all registered dataset names.

Designed to be small, dependency-free, and friendly to lazy imports.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

# Internal registry mapping dataset names to dataset classes.
_registry: Dict[str, Type["DatasetBase"]] = {}


class DatasetBase(ABC):
    """Abstract base class for datasets.

    Subclasses should implement 'load' and may override 'info'.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the dataset. Subclasses can accept arbitrary args/kwargs."""
        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def load(self) -> Any:
        """Load or prepare the dataset and return a representation (e.g., generator, list, object)."""
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        """Return metadata about the dataset. Default implementation returns constructor args."""
        return {"args": self._args, "kwargs": self._kwargs}


def register_dataset(name: str) -> Callable[[Type[DatasetBase]], Type[DatasetBase]]:
    """Decorator to register a DatasetBase subclass under a canonical name.

    Example:
        @register_dataset("my_dataset")
        class MyDataset(DatasetBase):
            ...

    If a dataset with the same name already exists, it will be overwritten.
    """

    def _decorator(cls: Type[DatasetBase]) -> Type[DatasetBase]:
        if not issubclass(cls, DatasetBase):
            raise TypeError("Only subclasses of DatasetBase can be registered.")
        _registry[name] = cls
        return cls

    return _decorator


def register_dataset_class(name: str, cls: Type[DatasetBase]) -> None:
    """Register a dataset class programmatically.

    Equivalent to using the decorator form.
    """
    if not issubclass(cls, DatasetBase):
        raise TypeError("Only subclasses of DatasetBase can be registered.")
    _registry[name] = cls


def get_dataset_class(name: str, try_lazy_import: bool = True) -> Optional[Type[DatasetBase]]:
    """Retrieve a registered dataset class by name.

    If not found and try_lazy_import is True, attempt to import a module
    named 'delf.python.datasets.<name>' to allow for lazy registration by module import.
    Returns None if the class is still not found.
    """
    cls = _registry.get(name)
    if cls is not None:
        return cls

    if try_lazy_import:
        module_name = f"delf.python.datasets.{name}"
        try:
            importlib.import_module(module_name)
        except Exception:
            # Ignore import errors; return None below.
            pass
        cls = _registry.get(name)
    return cls


def create_dataset(name: str, *args: Any, **kwargs: Any) -> DatasetBase:
    """Instantiate and return a dataset by registered name.

    Raises KeyError if the dataset name is not registered.
    """
    cls = get_dataset_class(name)
    if cls is None:
        raise KeyError(f"Dataset '{name}' is not registered.")
    return cls(*args, **kwargs)


def list_datasets() -> List[str]:
    """Return a sorted list of registered dataset names."""
    return sorted(_registry.keys())


def clear_registry() -> None:
    """Clear the dataset registry. Useful for tests or dynamic reconfiguration."""
    _registry.clear()


# Public API
__all__ = [
    "DatasetBase",
    "register_dataset",
    "register_dataset_class",
    "get_dataset_class",
    "create_dataset",
    "list_datasets",
    "clear_registry",
]