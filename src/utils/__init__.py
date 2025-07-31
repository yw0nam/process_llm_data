"""Utility modules for data processing.

AIDEV-NOTE: Centralized utilities for file I/O, formatting, logging and parallel processing.
"""

from .formatting import (
    format_instruction_data,
    format_message,
    format_messages,
    format_preference_data,
    validate_instruction_format,
    validate_preference_format,
)
from .io import (
    DatasetSaver,
    ensure_directory,
    load_hf_dataset,
    read_json,
    read_jsonl,
    save_dataset,
    write_json,
    write_jsonl,
)
from .logging import setup_logging
from .parallel import ParallelProcessor, process_in_batches, safe_parallel_map

__all__ = [
    # I/O utilities
    "DatasetSaver",
    "ensure_directory",
    "load_hf_dataset",
    "read_json",
    "read_jsonl",
    "save_dataset",
    "write_json",
    "write_jsonl",
    # Formatting utilities
    "format_instruction_data",
    "format_message",
    "format_messages",
    "format_preference_data",
    "validate_instruction_format",
    "validate_preference_format",
    # Logging utilities
    "setup_logging",
    # Parallel processing utilities
    "ParallelProcessor",
    "process_in_batches",
    "safe_parallel_map",
]
