"""Custom instruction dataset processor.

AIDEV-NOTE: Example of dedicated processor module for complex instruction processing.
This demonstrates how to create specialized processors for specific dataset types.
"""

import logging
from typing import Any, Dict, List

import pandas as pd
from src.dataset.registry import DatasetRegistry

logger = logging.getLogger(__name__)


class CustomInstructionProcessor:
    """Custom processor for instruction datasets with advanced transformations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize processor with configuration.

        Args:
            config: Configuration dictionary for processor settings
        """
        self.config = config or {}
        self.min_length = self.config.get("min_instruction_length", 10)
        self.max_length = self.config.get("max_instruction_length", 2048)

    def validate_instruction(self, instruction: str) -> bool:
        """Validate instruction quality.

        Args:
            instruction: Instruction text to validate

        Returns:
            True if instruction meets quality criteria
        """
        if not instruction or not instruction.strip():
            return False

        length = len(instruction.strip())
        return self.min_length <= length <= self.max_length

    def process_instruction_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process instruction dataset with advanced formatting.

        Args:
            df: Raw instruction DataFrame

        Returns:
            Processed DataFrame with standardized instruction format
        """
        processed_df = df.copy()

        # Validate and filter instructions
        if "instruction" in processed_df.columns:
            mask = processed_df["instruction"].apply(self.validate_instruction)
            processed_df = processed_df[mask]
            logger.info(f"Filtered {len(df) - len(processed_df)} invalid instructions")

        # Standardize response format
        if "response" in processed_df.columns:
            processed_df["response"] = processed_df["response"].astype(str).str.strip()

        # Add instruction metadata
        processed_df["instruction_length"] = (
            processed_df.get("instruction", "").astype(str).str.len()
        )
        processed_df["response_length"] = (
            processed_df.get("response", "").astype(str).str.len()
        )
        processed_df["processor_version"] = "custom_v1.0"

        return processed_df


# Register the processing function using the class
@DatasetRegistry.register_processing_function("custom_instruction")
def process_custom_instruction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process custom instruction dataset using dedicated processor.

    Args:
        df: Raw DataFrame from instruction dataset

    Returns:
        Processed DataFrame with instruction-specific transformations
    """
    logger.info("Processing custom instruction dataset")

    # Use the dedicated processor class
    processor = CustomInstructionProcessor()
    processed_df = processor.process_instruction_format(df)

    # Add common metadata
    processed_df["source_type"] = "custom_instruction"
    processed_df["processed_at"] = pd.Timestamp.now()

    logger.info(f"Custom instruction processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df
