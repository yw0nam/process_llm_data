"""Example dataset processing functions.

AIDEV-NOTE: Demonstrates how to register custom processing functions for datasets.
Each processing function is specific to a dataset and handles its unique processing needs.
"""

import logging

import pandas as pd
from src.dataset.registry import DatasetRegistry

logger = logging.getLogger(__name__)


@DatasetRegistry.register_processing_function("huggingface")
def process_huggingface_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process HuggingFace datasets with common transformations.

    Args:
        df: Raw DataFrame from HuggingFace dataset

    Returns:
        Processed DataFrame ready for use
    """
    logger.info("Processing HuggingFace dataset")

    # Common HuggingFace processing
    processed_df = df.copy()

    # Remove any rows with missing critical data
    if "text" in processed_df.columns:
        processed_df = processed_df.dropna(subset=["text"])

    # Standardize column names
    if "input" in processed_df.columns and "text" not in processed_df.columns:
        processed_df = processed_df.rename(columns={"input": "text"})

    # Remove empty strings
    if "text" in processed_df.columns:
        processed_df = processed_df[processed_df["text"].str.strip() != ""]

    # Add metadata
    processed_df["source_type"] = "huggingface"
    processed_df["processed_at"] = pd.Timestamp.now()

    logger.info(f"HuggingFace processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df


@DatasetRegistry.register_processing_function("local_files")
def process_local_files_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process local file datasets with file-specific transformations.

    Args:
        df: Raw DataFrame from local files

    Returns:
        Processed DataFrame ready for use
    """
    logger.info("Processing local files dataset")

    processed_df = df.copy()

    # Handle encoding issues that might occur in local files
    for col in processed_df.select_dtypes(include=["object"]).columns:
        if processed_df[col].dtype == "object":
            # Try to fix encoding issues
            try:
                processed_df[col] = processed_df[col].astype(str)
            except Exception as e:
                logger.warning(f"Could not process column {col}: {e}")

    # Remove completely empty rows
    processed_df = processed_df.dropna(how="all")

    # Add file metadata
    processed_df["source_type"] = "local_files"
    processed_df["processed_at"] = pd.Timestamp.now()

    logger.info(f"Local files processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df


@DatasetRegistry.register_processing_function("smoltalk")
def process_smoltalk_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process SmolTalk dataset with conversation-specific transformations.

    Args:
        df: Raw DataFrame from SmolTalk dataset

    Returns:
        Processed DataFrame with conversation formatting
    """
    logger.info("Processing SmolTalk dataset")

    processed_df = df.copy()

    # SmolTalk specific processing
    if "messages" in processed_df.columns:
        # Extract conversation turns
        processed_df["num_turns"] = processed_df["messages"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Filter out very short conversations
        processed_df = processed_df[processed_df["num_turns"] >= 2]

    # Add conversation metadata
    processed_df["source_type"] = "smoltalk"
    processed_df["processed_at"] = pd.Timestamp.now()
    processed_df["conversation_length"] = processed_df.get("messages", []).apply(
        lambda x: sum(len(str(turn)) for turn in x) if isinstance(x, list) else 0
    )

    logger.info(f"SmolTalk processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df


@DatasetRegistry.register_processing_function("instruction_data")
def process_instruction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process instruction-following datasets.

    Args:
        df: Raw DataFrame with instruction data

    Returns:
        Processed DataFrame with standardized instruction format
    """
    logger.info("Processing instruction dataset")

    processed_df = df.copy()

    # Standardize instruction format columns
    column_mapping = {
        "prompt": "instruction",
        "question": "instruction",
        "input": "instruction",
        "response": "output",
        "answer": "output",
        "completion": "output",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in processed_df.columns and new_col not in processed_df.columns:
            processed_df = processed_df.rename(columns={old_col: new_col})

    # Ensure we have required columns
    required_cols = ["instruction", "output"]
    missing_cols = [col for col in required_cols if col not in processed_df.columns]

    if missing_cols:
        logger.warning(f"Missing required columns for instruction data: {missing_cols}")
    else:
        # Filter out rows with empty instructions or outputs
        processed_df = processed_df.dropna(subset=required_cols)
        processed_df = processed_df[
            (processed_df["instruction"].str.strip() != "")
            & (processed_df["output"].str.strip() != "")
        ]

    # Add instruction-specific metadata
    processed_df["source_type"] = "instruction_data"
    processed_df["processed_at"] = pd.Timestamp.now()

    if "instruction" in processed_df.columns:
        processed_df["instruction_length"] = processed_df["instruction"].str.len()
    if "output" in processed_df.columns:
        processed_df["output_length"] = processed_df["output"].str.len()

    logger.info(f"Instruction data processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df


@DatasetRegistry.register_processing_function("preference_data")
def process_preference_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process preference/ranking datasets.

    Args:
        df: Raw DataFrame with preference data

    Returns:
        Processed DataFrame with standardized preference format
    """
    logger.info("Processing preference dataset")

    processed_df = df.copy()

    # Standardize preference format columns
    column_mapping = {
        "chosen": "preferred",
        "rejected": "dispreferred",
        "winner": "preferred",
        "loser": "dispreferred",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in processed_df.columns and new_col not in processed_df.columns:
            processed_df = processed_df.rename(columns={old_col: new_col})

    # Ensure we have required columns for preference data
    required_cols = ["preferred", "dispreferred"]
    missing_cols = [col for col in required_cols if col not in processed_df.columns]

    if missing_cols:
        logger.warning(f"Missing required columns for preference data: {missing_cols}")
    else:
        # Filter out rows with empty preferences
        processed_df = processed_df.dropna(subset=required_cols)
        processed_df = processed_df[
            (processed_df["preferred"].str.strip() != "")
            & (processed_df["dispreferred"].str.strip() != "")
        ]

    # Add preference-specific metadata
    processed_df["source_type"] = "preference_data"
    processed_df["processed_at"] = pd.Timestamp.now()

    if "preferred" in processed_df.columns:
        processed_df["preferred_length"] = processed_df["preferred"].str.len()
    if "dispreferred" in processed_df.columns:
        processed_df["dispreferred_length"] = processed_df["dispreferred"].str.len()

    logger.info(f"Preference data processing: {len(df)} -> {len(processed_df)} rows")
    return processed_df
