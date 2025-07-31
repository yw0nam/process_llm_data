"""Dataset module for loading and managing datasets.

AIDEV-NOTE: Dataset registry and base classes for extensible dataset handling.
Imports loaders and processors to register them automatically.
"""

# Import processors to register processing functions
from . import processors

# Import format converters to register processing functions
from . import format_converters
from .base import BaseDataset

# Import loaders to register them
from .loaders import huggingface, local_files
from .registry import DatasetRegistry, register_dataset, register_processing_function

__all__ = [
    "BaseDataset",
    "DatasetRegistry",
    "register_dataset",
    "register_processing_function",
]
