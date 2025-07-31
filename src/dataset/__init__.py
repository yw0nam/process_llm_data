"""Dataset management module.

AIDEV-NOTE: Core dataset module providing registry, loaders, and format converters.
All dataset loading and processing functionality is centralized here.
"""

from .base import BaseDataset
from .registry import DatasetRegistry

# Import processors to ensure they are registered
from . import processors

__all__ = [
    "BaseDataset",
    "DatasetRegistry",
]
