"""Dataset registry system for dynamic dataset loading.

AIDEV-NOTE: Registry pattern implementation for extensible dataset management.
Supports plugin-based dataset loading with lazy initialization and custom processing.
"""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd

from .base import BaseDataset

logger = logging.getLogger(__name__)

# Type variable for dataset classes
DatasetType = TypeVar("DatasetType", bound=BaseDataset)

# Type alias for processing functions
ProcessingFunction = Callable[[pd.DataFrame], pd.DataFrame]


class DatasetRegistry:
    """Registry for managing available datasets and their loaders.

    AIDEV-NOTE: Central registry using decorator pattern for dataset registration.
    Supports lazy loading, plugin-based architecture, and custom processing functions.
    """

    _datasets: dict[str, type[BaseDataset]] = {}
    _instances: dict[str, BaseDataset] = {}
    _processing_functions: dict[str, ProcessingFunction] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register dataset loaders.

        Args:
            name: Unique name for the dataset

        Returns:
            Decorator function that registers the dataset class

        Example:
            @DatasetRegistry.register("smoltalk")
            class SmolTalkDataset(BaseDataset):
                pass
        """

        def decorator(dataset_class: type[DatasetType]) -> type[DatasetType]:
            if name in cls._datasets:
                logger.warning(f"Dataset '{name}' already registered, overwriting")

            cls._datasets[name] = dataset_class
            logger.info(f"Registered dataset: {name} -> {dataset_class.__name__}")
            return dataset_class

        return decorator

    @classmethod
    def register_processing_function(cls, dataset_name: str):
        """Decorator to register custom processing functions for datasets.

        Args:
            dataset_name: Name of the dataset this processing function is for

        Returns:
            Decorator function that registers the processing function

        Example:
            @DatasetRegistry.register_processing_function("smoltalk")
            def process_smoltalk_data(df: pd.DataFrame) -> pd.DataFrame:
                # Custom processing logic
                return df
        """

        def decorator(processing_func: ProcessingFunction) -> ProcessingFunction:
            if dataset_name in cls._processing_functions:
                logger.warning(
                    f"Processing function for '{dataset_name}' already registered, overwriting"
                )

            cls._processing_functions[dataset_name] = processing_func
            logger.info(
                f"Registered processing function: {dataset_name} -> {processing_func.__name__}"
            )
            return processing_func

        return decorator

    @classmethod
    def create_dataset(cls, name: str, config: dict[str, Any]) -> BaseDataset:
        """Create a dataset instance by name.

        Args:
            name: Name of the registered dataset
            config: Configuration dictionary for the dataset

        Returns:
            Initialized dataset instance

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not registered. Available: {available}")

        dataset_class = cls._datasets[name]

        # Create new instance with config
        instance = dataset_class(config)
        logger.info(f"Created dataset instance: {name}")

        return instance

    @classmethod
    def process_dataset(
        cls, dataset_name: str, data: pd.DataFrame, use_custom_processing: bool = True
    ) -> pd.DataFrame:
        """Process dataset using registered processing function if available.

        Args:
            dataset_name: Name of the dataset/processing function
            data: Raw data to process
            use_custom_processing: Whether to use custom processing if available

        Returns:
            Processed DataFrame

        Note:
            Processing functions are independent of dataset registration.
            You can process data with any registered processing function.
        """
        # Check if there's a custom processing function
        if use_custom_processing and dataset_name in cls._processing_functions:
            processing_func = cls._processing_functions[dataset_name]
            logger.info(f"Using custom processing function for: {dataset_name}")
            try:
                processed_data = processing_func(data)
                logger.info(f"Successfully processed {len(processed_data)} rows")
                return processed_data
            except Exception as e:
                logger.error(f"Error in custom processing for {dataset_name}: {e}")
                raise
        else:
            logger.info(f"No custom processing function found for: {dataset_name}")
            return data

    @classmethod
    def get_processing_function(cls, dataset_name: str) -> ProcessingFunction | None:
        """Get the processing function for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Processing function if registered, None otherwise
        """
        return cls._processing_functions.get(dataset_name)

    @classmethod
    def has_processing_function(cls, dataset_name: str) -> bool:
        """Check if a dataset has a custom processing function.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if processing function is registered, False otherwise
        """
        return dataset_name in cls._processing_functions

    @classmethod
    def get_dataset(
        cls, name: str, config: dict[str, Any], use_cache: bool = True
    ) -> BaseDataset:
        """Get a dataset instance, using cache if enabled.

        Args:
            name: Name of the registered dataset
            config: Configuration dictionary for the dataset
            use_cache: Whether to use cached instance if available

        Returns:
            Dataset instance (cached or new)
        """
        cache_key = f"{name}_{hash(str(sorted(config.items())))}"

        if use_cache and cache_key in cls._instances:
            logger.debug(f"Using cached dataset instance: {name}")
            return cls._instances[cache_key]

        instance = cls.create_dataset(name, config)

        if use_cache:
            cls._instances[cache_key] = instance

        return instance

    @classmethod
    def list_datasets(cls) -> list[str]:
        """List all registered dataset names.

        Returns:
            List of registered dataset names
        """
        return list(cls._datasets.keys())

    @classmethod
    def list_processing_functions(cls) -> list[str]:
        """List all registered processing function names.

        Returns:
            List of dataset names that have processing functions
        """
        return list(cls._processing_functions.keys())

    @classmethod
    def get_dataset_info(cls, name: str) -> dict[str, Any]:
        """Get information about a registered dataset.

        Args:
            name: Name of the registered dataset

        Returns:
            Dictionary with dataset information

        Raises:
            ValueError: If dataset name is not registered
        """
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not registered. Available: {available}")

        dataset_class = cls._datasets[name]
        has_processing = name in cls._processing_functions
        processing_func_name = None

        if has_processing:
            processing_func_name = cls._processing_functions[name].__name__

        return {
            "name": name,
            "class": dataset_class.__name__,
            "module": dataset_class.__module__,
            "docstring": dataset_class.__doc__,
            "has_processing_function": has_processing,
            "processing_function_name": processing_func_name,
        }

    @classmethod
    def clear_cache(cls):
        """Clear all cached dataset instances."""
        cls._instances.clear()
        logger.info("Dataset instance cache cleared")

    @classmethod
    def unregister(cls, name: str):
        """Unregister a dataset (mainly for testing).

        Args:
            name: Name of the dataset to unregister
        """
        if name in cls._datasets:
            del cls._datasets[name]
            logger.info(f"Unregistered dataset: {name}")

        # Remove processing function if exists
        if name in cls._processing_functions:
            del cls._processing_functions[name]
            logger.info(f"Unregistered processing function for: {name}")

        # Clear related cached instances
        keys_to_remove = [
            key for key in cls._instances.keys() if key.startswith(f"{name}_")
        ]
        for key in keys_to_remove:
            del cls._instances[key]

    @classmethod
    def unregister_processing_function(cls, dataset_name: str):
        """Unregister a processing function for a dataset.

        Args:
            dataset_name: Name of the dataset
        """
        if dataset_name in cls._processing_functions:
            del cls._processing_functions[dataset_name]
            logger.info(f"Unregistered processing function for: {dataset_name}")

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a dataset is registered.

        Args:
            name: Name to check

        Returns:
            True if dataset is registered, False otherwise
        """
        return name in cls._datasets


# Convenience functions for external use
def register_dataset(name: str):
    """Convenience function to register a dataset.

    Args:
        name: Unique name for the dataset

    Returns:
        Decorator function
    """
    return DatasetRegistry.register(name)


def register_processing_function(dataset_name: str):
    """Convenience function to register a processing function.

    Args:
        dataset_name: Name of the dataset this processing function is for

    Returns:
        Decorator function
    """
    return DatasetRegistry.register_processing_function(dataset_name)
