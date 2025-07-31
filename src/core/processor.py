"""
Base processor interface for all data processors.

This module defines the abstract base class that all data processors must implement.
It provides a consistent interface for processing datasets with validation and error handling.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from .exceptions import DataValidationError, ProcessorError

# AIDEV-NOTE: Abstract processor interface supporting async operations and validation

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the processor with configuration.

        Args:
            config: Configuration dictionary containing processor settings
        """
        self.config = config
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @abstractmethod
    def process(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process datasets and return combined result.

        Args:
            datasets: Dictionary mapping dataset names to DataFrames

        Returns:
            Processed DataFrame ready for training

        Raises:
            ProcessorError: If processing fails
        """
        pass

    def validate_input(self, datasets: dict[str, pd.DataFrame]) -> bool:
        """
        Validate input datasets before processing.

        Args:
            datasets: Dictionary mapping dataset names to DataFrames

        Returns:
            True if validation passes

        Raises:
            DataValidationError: If validation fails
        """
        if not datasets:
            raise DataValidationError("No datasets provided for processing")

        for name, df in datasets.items():
            if df.empty:
                raise DataValidationError(f"Dataset '{name}' is empty")

        return True

    def validate_output(self, data: pd.DataFrame) -> bool:
        """
        Validate processed output format.

        Args:
            data: Processed DataFrame to validate

        Returns:
            True if validation passes

        Raises:
            DataValidationError: If validation fails
        """
        if data.empty:
            raise DataValidationError("Processed data is empty")

        required_columns = self.get_required_output_columns()
        missing_columns = set(required_columns) - set(data.columns)

        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        return True

    @abstractmethod
    def get_required_output_columns(self) -> list[str]:
        """
        Return list of required output columns.

        Returns:
            List of column names that must be present in output
        """
        pass

    def process_with_validation(
        self, datasets: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Process datasets with input/output validation.

        Args:
            datasets: Dictionary mapping dataset names to DataFrames

        Returns:
            Validated processed DataFrame

        Raises:
            ProcessorError: If processing fails
            DataValidationError: If validation fails
        """
        try:
            # Validate input
            self.validate_input(datasets)
            self.logger.info(f"Processing {len(datasets)} datasets")

            # Process data
            result = self.process(datasets)

            # Validate output
            self.validate_output(result)
            self.logger.info(f"Successfully processed {len(result)} records")

            return result

        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            if isinstance(e, ProcessorError | DataValidationError):
                raise
            else:
                raise ProcessorError(f"Unexpected error during processing: {e}") from e
