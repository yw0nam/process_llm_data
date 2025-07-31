"""Base dataset interface for all dataset implementations.

AIDEV-NOTE: Abstract base class defining the interface for all dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDataset(ABC):
    """Abstract base class for all dataset implementations."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize dataset with configuration.

        Args:
            config: Dataset-specific configuration
        """
        self.config = config
        self.name = config.get("name", "unknown")
        self.source_path = config.get("source_path")

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load the dataset and return as pandas DataFrame.

        Returns:
            DataFrame with standardized columns
        """
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate that the loaded data meets requirements.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset metadata
        """
        return {
            "name": self.name,
            "source_path": self.source_path,
            "config": self.config,
        }

    def sample(self, data: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
        """
        Sample n rows from the dataset.

        Args:
            data: DataFrame to sample from
            n: Number of samples (None for all data)

        Returns:
            Sampled DataFrame
        """
        if n is None or n >= len(data):
            return data

        return data.sample(n=n, random_state=42).reset_index(drop=True)
