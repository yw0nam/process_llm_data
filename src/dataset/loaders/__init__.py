"""Dataset loaders package.

AIDEV-NOTE: Collection of dataset loaders implementing the registry pattern.
Each loader is a plugin that can be dynamically loaded.
"""

from .huggingface import HuggingFaceDataset
from .local_files import LocalFilesDataset

__all__ = [
    "HuggingFaceDataset",
    "LocalFilesDataset",
]
