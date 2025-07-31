"""
Data processing pipeline orchestrator.

This module contains the main pipeline class that coordinates dataset loading,
processing, and output generation. It serves as the central coordinator for
the entire data processing workflow.
"""

import logging
from typing import Any

import pandas as pd

from .exceptions import ConfigurationError, ProcessingError
from .processor import BaseProcessor

# AIDEV-NOTE: Main pipeline orchestrator with plugin support for scalable processing

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """Factory for creating processors based on configuration."""

    _processors: dict[str, type[BaseProcessor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register processor classes.

        Args:
            name: Processor name for registration
        """

        def decorator(processor_class: type[BaseProcessor]):
            cls._processors[name] = processor_class
            return processor_class

        return decorator

    @classmethod
    def create_processor(
        cls, process_type: str, version: str, config: dict[str, Any]
    ) -> BaseProcessor:
        """
        Create processor instance based on type and version.

        Args:
            process_type: Type of processing ('instruction' or 'preference')
            version: Version of processor to use
            config: Configuration for processor

        Returns:
            Configured processor instance

        Raises:
            ConfigurationError: If processor not found
        """
        processor_key = f"{process_type}_{version}"

        if processor_key not in cls._processors:
            raise ConfigurationError(
                f"Processor '{processor_key}' not found. Available: {list(cls._processors.keys())}"
            )

        return cls._processors[processor_key](config)

    @classmethod
    def list_processors(cls) -> list[str]:
        """Return list of registered processors."""
        return list(cls._processors.keys())


class DataPipeline:
    """Main pipeline orchestrator with plugin support."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )
        self.processor_factory = ProcessorFactory()

    def run(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Execute the full data processing pipeline.

        Args:
            datasets: Dictionary of datasets to process

        Returns:
            Processed DataFrame ready for training

        Raises:
            ProcessingError: If pipeline execution fails
        """
        try:
            self.logger.info("Starting data processing pipeline")

            # Create processor
            processor = self.processor_factory.create_processor(
                process_type=self.config["process_type"],
                version=self.config["version"],
                config=self.config.get("processor_config", {}),
            )

            # Process datasets
            result = processor.process_with_validation(datasets)

            self.logger.info(
                f"Pipeline completed successfully. Processed {len(result)} records"
            )
            return result

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            if isinstance(e, ProcessingError):
                raise
            else:
                raise ProcessingError(f"Pipeline execution failed: {e}") from e

    def validate_config(self) -> bool:
        """
        Validate pipeline configuration.

        Returns:
            True if configuration is valid

        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_keys = ["process_type", "version"]

        for key in required_keys:
            if key not in self.config:
                raise ConfigurationError(f"Missing required configuration key: {key}")

        if self.config["process_type"] not in ["instruction", "preference"]:
            raise ConfigurationError(
                f"Invalid process_type: {self.config['process_type']}"
            )

        return True
