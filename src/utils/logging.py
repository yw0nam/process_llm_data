"""Logging configuration and utilities.

AIDEV-NOTE: Centralized logging setup with support for file and console output.
"""

import logging
import sys
from pathlib import Path

from ..config.schemas import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Set up logging based on configuration.

    Args:
        config: Logging configuration

    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger("process_llm_data")
    logger.setLevel(getattr(logging, config.level))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    if config.level == "DEBUG" or hasattr(sys, "_getframe"):
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, defaults to module name

    Returns:
        Logger instance
    """
    if name is None:
        name = "process_llm_data"

    return logging.getLogger(name)
