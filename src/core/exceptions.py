"""
Custom exceptions for the LLM data processing pipeline.

This module defines the exception hierarchy used throughout the processing pipeline.
All custom exceptions inherit from ProcessingError to enable proper error handling.
"""

# AIDEV-NOTE: Custom exception hierarchy for type-safe error handling


class ProcessingError(Exception):
    """Base exception for all processing-related errors."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class DataValidationError(ProcessingError):
    """Raised when data validation fails."""

    pass


class DatasetLoadError(ProcessingError):
    """Raised when dataset loading fails."""

    pass


class DatasetError(ProcessingError):
    """Raised for general dataset-related errors."""

    pass


class ConfigurationError(ProcessingError):
    """Raised when configuration is invalid or missing."""

    pass


class ProcessorError(ProcessingError):
    """Raised when data processing fails."""

    pass


class RegistryError(ProcessingError):
    """Raised when dataset registry operations fail."""

    pass


class FormatError(ProcessingError):
    """Raised when data format is invalid."""

    pass
